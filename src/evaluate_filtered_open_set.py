
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "lfw_open_set_single_face"

GALLERY_FILE = (
    PROJECT_ROOT / "embeddings" / "filtered_gallery_embeddings.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"

ALL_RESULTS_FILE = (
    RESULTS_DIR / "filtered_open_set_recognition_results.csv"
)

THRESHOLD_FILE = (
    RESULTS_DIR / "filtered_open_set_threshold_analysis.csv"
)

SAVE_EVERY = 25
THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.01), 2)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_identity(value):
    return str(value or "").strip().replace(" ", "_").lower()


# ============================================================
# LOAD GALLERY
# ============================================================

def load_gallery():
    if not GALLERY_FILE.exists():
        raise FileNotFoundError(
            f"Gallery embeddings not found: {GALLERY_FILE}\n"
            "Run create_filtered_gallery_embeddings.py first."
        )

    data = np.load(GALLERY_FILE, allow_pickle=False)

    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    if len(embeddings) != len(labels):
        raise ValueError("Gallery embedding and label counts differ.")

    if len(labels) == 0:
        raise ValueError("Gallery is empty.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Gallery contains a zero-length embedding.")

    embeddings /= norms

    gallery_ids = {normalize_identity(label) for label in labels}

    print("Gallery embeddings:", len(embeddings))
    print("Unique enrolled identities:", len(gallery_ids))
    print("Embedding dimensions:", embeddings.shape[1])

    return embeddings, labels, gallery_ids


# ============================================================
# DISCOVER DATASET IMAGES
# ============================================================

def discover_images():
    records = []

    for split in ("known_test", "unknown_test"):
        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Required split not found: {split_dir}"
            )

        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "split": split,
                    "actual_identity": path.parent.name,
                    "image": str(path),
                })

    if not records:
        raise ValueError("No test images found.")

    df = pd.DataFrame(records)

    print("\nDiscovered test images:", len(df))
    print(df.groupby("split").size().to_string())

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("\nLoading InsightFace buffalo_l on CPU...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Model loaded.")
    return app


# ============================================================
# RECOGNITION
# ============================================================

def evaluate_image(app, image_path, gallery_embeddings, gallery_labels):
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "predicted_identity": "",
            "similarity": np.nan,
            "detection_score": np.nan,
            "detected_faces": 0,
            "status": "IMAGE_LOAD_FAILED",
        }

    faces = app.get(image)

    # Do not silently select one face if the filter and model disagree.
    if len(faces) == 0:
        return {
            "predicted_identity": "",
            "similarity": np.nan,
            "detection_score": np.nan,
            "detected_faces": 0,
            "status": "NO_FACE_DETECTED",
        }

    if len(faces) != 1:
        return {
            "predicted_identity": "",
            "similarity": np.nan,
            "detection_score": np.nan,
            "detected_faces": len(faces),
            "status": "MULTIPLE_FACES",
        }

    face = faces[0]
    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if embedding.shape != (512,) or norm == 0:
        return {
            "predicted_identity": "",
            "similarity": np.nan,
            "detection_score": float(face.det_score),
            "detected_faces": 1,
            "status": "INVALID_EMBEDDING",
        }

    embedding /= norm

    similarities = gallery_embeddings @ embedding
    best_index = int(np.argmax(similarities))

    return {
        "predicted_identity": str(gallery_labels[best_index]),
        "similarity": float(similarities[best_index]),
        "detection_score": float(face.det_score),
        "detected_faces": 1,
        "status": "OK",
    }


# ============================================================
# EVALUATE ALL IMAGES WITH RESUME SUPPORT
# ============================================================

def evaluate_dataset(app, gallery_embeddings, gallery_labels, gallery_ids):
    images_df = discover_images()

    # An identity is enrolled if it appears in the gallery.
    images_df["identity_enrolled"] = images_df["actual_identity"].apply(
        lambda x: normalize_identity(x) in gallery_ids
    )

    # The split name is included in the key to avoid accidental collisions.
    images_df["record_key"] = (
        images_df["split"] + "|" + images_df["image"]
    )

    previous = {}

    if ALL_RESULTS_FILE.exists():
        old = pd.read_csv(ALL_RESULTS_FILE)

        required = {
            "record_key", "status", "predicted_identity", "similarity"
        }

        if required.issubset(old.columns):
            # Reuse only successfully evaluated images.
            completed = old[old["status"] == "OK"]

            for _, row in completed.iterrows():
                previous[str(row["record_key"])] = row.to_dict()

            print("Successful records available to resume:", len(previous))
        else:
            print("Existing results schema differs; starting fresh.")

    pending = [
        row for _, row in images_df.iterrows()
        if str(row["record_key"]) not in previous
    ]

    print("Images remaining to evaluate:", len(pending))

    start_time = time.time()

    try:
        for index, row in enumerate(pending, start=1):
            result = evaluate_image(
                app,
                row["image"],
                gallery_embeddings,
                gallery_labels,
            )

            record = {
                "split": row["split"],
                "actual_identity": row["actual_identity"],
                "identity_enrolled": bool(row["identity_enrolled"]),
                "image": row["image"],
                "record_key": row["record_key"],
                **result,
            }

            previous[str(row["record_key"])] = record

            if index % SAVE_EVERY == 0:
                pd.DataFrame(previous.values()).to_csv(
                    ALL_RESULTS_FILE, index=False
                )

            if index % 25 == 0 or index == len(pending):
                elapsed = time.time() - start_time
                speed = index / elapsed if elapsed > 0 else 0
                remaining = len(pending) - index
                eta = remaining / speed if speed > 0 else 0

                print(
                    f"Progress: {index}/{len(pending)}"
                    f" | Records: {len(previous)}"
                    f" | {speed:.2f} images/sec"
                    f" | ETA: {eta / 60:.1f} min"
                )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        pd.DataFrame(previous.values()).to_csv(
            ALL_RESULTS_FILE, index=False
        )
        print("Progress saved. Run the script again to resume.")
        raise

    finally:
        pd.DataFrame(previous.values()).to_csv(
            ALL_RESULTS_FILE, index=False
        )

    results = pd.DataFrame(previous.values())

    # Retain only paths that still belong to the current dataset.
    valid_keys = set(images_df["record_key"])
    results = results[results["record_key"].isin(valid_keys)].copy()

    # Restore dataset truth from the current folder structure.
    truth = images_df[
        ["record_key", "split", "actual_identity", "identity_enrolled"]
    ]

    results = results.drop(
        columns=["split", "actual_identity", "identity_enrolled"],
        errors="ignore",
    ).merge(truth, on="record_key", how="inner")

    results.to_csv(ALL_RESULTS_FILE, index=False)

    return results


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(results):
    detected = results[results["status"] == "OK"].copy()

    detected["similarity"] = pd.to_numeric(
        detected["similarity"], errors="coerce"
    )

    detected = detected.dropna(subset=["similarity"])

    enrolled = detected[detected["identity_enrolled"]].copy()
    unenrolled = detected[~detected["identity_enrolled"]].copy()

    enrolled["correct_identity"] = (
        enrolled["actual_identity"].apply(normalize_identity)
        == enrolled["predicted_identity"].apply(normalize_identity)
    )

    known_scores = enrolled["similarity"].to_numpy(dtype=float)
    known_correct = enrolled["correct_identity"].to_numpy(dtype=bool)
    unknown_scores = unenrolled["similarity"].to_numpy(dtype=float)

    if len(known_scores) == 0:
        raise ValueError(
            "No successfully evaluated enrolled known-test images."
        )

    if len(unknown_scores) == 0:
        raise ValueError(
            "No successfully evaluated unenrolled images. "
            "Check gallery/test identity overlap."
        )

    rows = []

    for threshold in THRESHOLDS:
        accept_known = known_scores >= threshold
        accept_unknown = unknown_scores >= threshold

        correct_accept = accept_known & known_correct
        wrong_identity_accept = accept_known & ~known_correct

        true_accept = int(correct_accept.sum())
        wrong_accept = int(wrong_identity_accept.sum())
        false_accept = int(accept_unknown.sum())
        true_reject = int((~accept_unknown).sum())

        rows.append({
            "threshold": float(threshold),
            "enrolled_known_images": len(known_scores),
            "unenrolled_images": len(unknown_scores),
            "correct_known_accept": true_accept,
            "wrong_identity_accept": wrong_accept,
            "unknown_false_accept": false_accept,
            "unknown_true_reject": true_reject,
            "correct_known_identification_rate_pct":
                100 * true_accept / len(known_scores),
            "wrong_identity_accept_rate_pct":
                100 * wrong_accept / len(known_scores),
            "unknown_false_accept_rate_pct":
                100 * false_accept / len(unknown_scores),
            "unknown_rejection_rate_pct":
                100 * true_reject / len(unknown_scores),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(THRESHOLD_FILE, index=False)

    print("\n" + "=" * 70)
    print("FILTERED OPEN-SET EVALUATION SUMMARY")
    print("=" * 70)

    print("All images in result file:", len(results))
    print("Successfully evaluated:", len(detected))
    print("Evaluation failures:", len(results) - len(detected))
    print("Enrolled known images evaluated:", len(enrolled))
    print("Unenrolled images evaluated:", len(unenrolled))

    print(
        "Known top-1 accuracy before threshold: "
        f"{100 * known_correct.mean():.2f}%"
    )

    print("\nResults saved:")
    print(ALL_RESULTS_FILE)
    print(THRESHOLD_FILE)

    print(
        "\nChoose a threshold based on the trade-off between "
        "correct known-person identification and unknown-person "
        "false acceptance. Do not select a threshold using "
        "accuracy alone."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FILTERED OPEN-SET FACE RECOGNITION EVALUATION")
    print("InsightFace buffalo_l / ArcFace embeddings / CPU")
    print("=" * 70)

    gallery_embeddings, gallery_labels, gallery_ids = load_gallery()
    app = load_model()

    results = evaluate_dataset(
        app,
        gallery_embeddings,
        gallery_labels,
        gallery_ids,
    )

    analyze_thresholds(results)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()

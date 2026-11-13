
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_FILE = (
    PROJECT_ROOT / "embeddings" / "gallery_embeddings_without_javier.npz"
)

KNOWN_RESULTS_FILE = (
    PROJECT_ROOT / "results" / "lfw_recognition_results.csv"
)

UNKNOWN_TEST_DIR = (
    PROJECT_ROOT / "dataset" / "lfw_open_set" / "unknown_test"
)

RESULTS_DIR = PROJECT_ROOT / "results"

UNKNOWN_RESULTS_FILE = (
    RESULTS_DIR / "unknown_open_set_results.csv"
)

THRESHOLD_RESULTS_FILE = (
    RESULTS_DIR / "open_set_threshold_analysis.csv"
)

SAVE_EVERY = 25
THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.01), 2)


# ============================================================
# 2. LOAD EXISTING GALLERY EMBEDDINGS
# ============================================================

def load_gallery():
    if not GALLERY_FILE.exists():
        raise FileNotFoundError(
            f"Gallery embeddings not found: {GALLERY_FILE}"
        )

    data = np.load(GALLERY_FILE, allow_pickle=True)

    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)

    if len(embeddings) != len(labels):
        raise ValueError("Gallery embedding and label counts differ.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Gallery contains a zero-length embedding.")

    embeddings /= norms

    print(f"Gallery identities loaded: {len(labels)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return embeddings, labels


# ============================================================
# 3. LOAD EXISTING KNOWN-IDENTITY RESULTS
# ============================================================


def normalize_identity(value):
    return str(value or "").strip().replace(" ", "_").lower()


def load_known_results(gallery_labels):
    if not KNOWN_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Known results not found: {KNOWN_RESULTS_FILE}"
        )

    df = pd.read_csv(KNOWN_RESULTS_FILE)

    required = {
        "actual_identity",
        "predicted_identity",
        "face_detected",
        "cosine_similarity",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Known results are missing columns: {sorted(missing)}"
        )

    detected = (
        df["face_detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    detection_failures = int((~detected).sum())
    df = df[detected].copy()

    df["similarity"] = pd.to_numeric(
        df["cosine_similarity"], errors="coerce"
    )
    df = df.dropna(subset=["similarity"])

    gallery_ids = {
        normalize_identity(label) for label in gallery_labels
    }

    df["identity_enrolled"] = df["actual_identity"].apply(
        lambda x: normalize_identity(x) in gallery_ids
    )

    unenrolled = df[~df["identity_enrolled"]].copy()
    known = df[df["identity_enrolled"]].copy()

    known["correct_identity"] = (
        known["actual_identity"].apply(normalize_identity)
        == known["predicted_identity"].apply(normalize_identity)
    )

    print("\n--- Known-set data validation ---")
    print(f"Rows in original CSV: {len(pd.read_csv(KNOWN_RESULTS_FILE))}")
    print(f"Face-detection failures: {detection_failures}")
    print(f"Detected enrolled-identity images: {len(known)}")
    print(f"Detected unenrolled-identity images: {len(unenrolled)}")
    print(
        "Correct top-1 predictions among enrolled images:",
        int(known["correct_identity"].sum())
    )

    if not unenrolled.empty:
        print(
            "Unenrolled identities in known CSV:",
            sorted(unenrolled["actual_identity"].dropna().unique())
        )

    if known.empty:
        raise ValueError("No valid enrolled-identity test results found.")

    return known


# ============================================================
# 4. LOAD INSIGHTFACE ONCE
# ============================================================

def load_model():
    print("\nLoading InsightFace model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )

    print("InsightFace model ready.")
    return app


# ============================================================
# 5. EMBEDDING AND RECOGNITION
# ============================================================

def get_embedding(app, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None, None, "IMAGE_LOAD_FAILED"

    faces = app.get(image)

    if not faces:
        return None, None, "NO_FACE_DETECTED"

    # Consistent with the previous evaluation:
    # select the largest detected face.
    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0])
            * (f.bbox[3] - f.bbox[1])
        )
    )

    embedding = face.embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None, None, "INVALID_EMBEDDING"

    embedding /= norm

    return embedding, float(face.det_score), "OK"


def recognize(embedding, gallery_embeddings, gallery_labels):
    # Normalized embeddings make dot product equivalent
    # to cosine similarity.
    similarities = gallery_embeddings @ embedding

    best_index = int(np.argmax(similarities))

    return (
        str(gallery_labels[best_index]),
        float(similarities[best_index])
    )


# ============================================================
# 6. DISCOVER ALL UNKNOWN IMAGES
# ============================================================

def discover_unknown_images():
    if not UNKNOWN_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Unknown test directory not found: {UNKNOWN_TEST_DIR}"
        )

    images = []

    for path in sorted(UNKNOWN_TEST_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            # The parent directory represents the identity.
            actual_identity = path.parent.name
            images.append((actual_identity, path))

    if not images:
        raise ValueError("No unknown test images were found.")

    print(f"Total unknown images discovered: {len(images)}")

    return images


# ============================================================
# 7. RESUME FROM PREVIOUSLY SAVED RESULTS
# ============================================================

def load_previous_unknown_results():
    if not UNKNOWN_RESULTS_FILE.exists():
        return {}

    df = pd.read_csv(UNKNOWN_RESULTS_FILE)

    required = {
        "image",
        "actual_identity",
        "predicted_identity",
        "similarity",
        "detection_score",
        "face_detected",
    }

    if not required.issubset(df.columns):
        print(
            "Existing unknown results have an unexpected schema. "
            "Starting a new results file."
        )
        return {}

    results = {}

    for _, row in df.iterrows():
        image_path = str(row["image"])

        detected = (
            str(row["face_detected"]).strip().lower()
            in {"true", "1", "yes"}
        )

        # Resume only valid completed records.
        # Failed detections/load attempts can be retried.
        similarity = pd.to_numeric(
            row["similarity"], errors="coerce"
        )

        if detected and pd.isna(similarity):
            continue

        if detected or str(row.get("error", "")) not in {
            "IMAGE_LOAD_FAILED",
            "NO_FACE_DETECTED",
            "INVALID_EMBEDDING",
        }:
            results[image_path] = row.to_dict()

    print(f"Valid completed records to reuse: {len(results)}")

    return results


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results.values())
    df.to_csv(UNKNOWN_RESULTS_FILE, index=False)


# ============================================================
# 8. PROCESS ALL UNKNOWN IMAGES
# ============================================================

def evaluate_unknown_images(
    app,
    gallery_embeddings,
    gallery_labels
):
    unknown_images = discover_unknown_images()

    existing_results = load_previous_unknown_results()

    pending = [
        (identity, path)
        for identity, path in unknown_images
        if str(path) not in existing_results
    ]

    print(f"Images remaining to process: {len(pending)}")

    if not pending:
        print("All unknown images already have saved results.")
        return pd.DataFrame(existing_results.values())

    start_time = time.time()
    newly_processed = 0

    try:
        for index, (actual_identity, image_path) in enumerate(
            pending, start=1
        ):
            embedding, detection_score, status = get_embedding(
                app, image_path
            )

            if embedding is None:
                record = {
                    "actual_identity": actual_identity,
                    "predicted_identity": "",
                    "similarity": np.nan,
                    "detection_score": detection_score,
                    "face_detected": False,
                    "error": status,
                    "image": str(image_path),
                }

            else:
                predicted_identity, similarity = recognize(
                    embedding,
                    gallery_embeddings,
                    gallery_labels,
                )

                record = {
                    "actual_identity": actual_identity,
                    "predicted_identity": predicted_identity,
                    "similarity": similarity,
                    "detection_score": detection_score,
                    "face_detected": True,
                    "error": "",
                    "image": str(image_path),
                }

            existing_results[str(image_path)] = record
            newly_processed += 1

            if newly_processed % SAVE_EVERY == 0:
                save_results(existing_results)

            if index % 25 == 0 or index == len(pending):
                elapsed = time.time() - start_time
                speed = index / elapsed if elapsed > 0 else 0
                remaining = len(pending) - index
                eta = remaining / speed if speed > 0 else 0

                print(
                    f"Progress: {index}/{len(pending)} "
                    f"| Saved records: {len(existing_results)} "
                    f"| Speed: {speed:.2f} images/sec "
                    f"| Estimated remaining: {eta / 60:.1f} min"
                )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed results...")
        save_results(existing_results)
        print("Progress saved. Run the script again to resume.")
        raise

    finally:
        # Save whatever has been completed, including the final batch.
        save_results(existing_results)

    return pd.DataFrame(existing_results.values())


# ============================================================
# 9. ANALYZE THRESHOLDS
# ============================================================


def analyze_thresholds(known_df, unknown_df, gallery_labels):
    def normalize(value):
        return str(value or "").strip().replace(" ", "_").lower()

    # --------------------------------------------------------
    # 1. Build the enrolled identity set
    # --------------------------------------------------------
    gallery_ids = {
        normalize(label) for label in gallery_labels
    }

    # --------------------------------------------------------
    # 2. Load all original known-test results.
    #    Unenrolled identities in this CSV must be evaluated
    #    as unknowns, not as enrolled identities.
    # --------------------------------------------------------
    all_known = pd.read_csv(KNOWN_RESULTS_FILE)

    known_detected = (
        all_known["face_detected"]
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )

    all_known["similarity"] = pd.to_numeric(
        all_known["cosine_similarity"], errors="coerce"
    )

    all_known["identity_enrolled"] = (
        all_known["actual_identity"].apply(normalize)
        .isin(gallery_ids)
    )

    unenrolled_known = all_known[
        (~all_known["identity_enrolled"]) & known_detected
    ].copy()

    unenrolled_known = unenrolled_known.dropna(
        subset=["similarity"]
    )

    # --------------------------------------------------------
    # 3. Validate the separate unknown-test dataset
    # --------------------------------------------------------
    unknown_all = unknown_df.copy()

    unknown_detected = (
        unknown_all["face_detected"]
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )

    unknown_all["similarity"] = pd.to_numeric(
        unknown_all["similarity"], errors="coerce"
    )

    # An identity in the gallery is not genuinely unknown.
    unknown_all["identity_enrolled"] = (
        unknown_all["actual_identity"].apply(normalize)
        .isin(gallery_ids)
    )

    overlap = sorted(
        unknown_all.loc[
            unknown_all["identity_enrolled"], "actual_identity"
        ].dropna().unique()
    )

    if overlap:
        print(
            "\nWARNING: Some identities in unknown_test are "
            "present in the gallery and must not be counted "
            "as unknown:"
        )
        for identity in overlap:
            print(" -", identity)

    valid_unknown = unknown_all[
        (~unknown_all["identity_enrolled"]) & unknown_detected
    ].copy()

    valid_unknown = valid_unknown.dropna(
        subset=["similarity"]
    )

    # Add unenrolled identities from the old known CSV to the
    # unknown evaluation group. Their stored scores are reused.
    unknown_scores = np.concatenate([
        unenrolled_known["similarity"].to_numpy(dtype=float),
        valid_unknown["similarity"].to_numpy(dtype=float),
    ])

    known_scores = known_df["similarity"].to_numpy(dtype=float)
    known_correct = known_df["correct_identity"].to_numpy(dtype=bool)

    if len(known_scores) == 0:
        raise ValueError("No enrolled known-test scores are available.")

    if len(unknown_scores) == 0:
        raise ValueError("No valid unenrolled identity scores are available.")

    # --------------------------------------------------------
    # 4. Calculate threshold metrics
    # --------------------------------------------------------
    rows = []

    for threshold in THRESHOLDS:
        accept_known = known_scores >= threshold
        accept_unknown = unknown_scores >= threshold

        correct_accept = accept_known & known_correct
        wrong_accept = accept_known & ~known_correct

        true_accept = int(correct_accept.sum())
        wrong_identity_accept = int(wrong_accept.sum())
        false_accept = int(accept_unknown.sum())
        true_reject = int((~accept_unknown).sum())

        rows.append({
            "threshold": float(threshold),
            "enrolled_known_images": len(known_scores),
            "unenrolled_images": len(unknown_scores),
            "correct_known_accept": true_accept,
            "known_correct_top1_total": int(known_correct.sum()),
            "wrong_identity_accept": wrong_identity_accept,
            "unknown_false_accept": false_accept,
            "unknown_true_reject": true_reject,
            "correct_known_identification_rate_pct":
                100 * true_accept / len(known_scores),
            "known_correct_top1_acceptance_pct":
                100 * true_accept / int(known_correct.sum())
                if known_correct.sum() else np.nan,
            "unknown_false_accept_rate_pct":
                100 * false_accept / len(unknown_scores),
            "unknown_rejection_rate_pct":
                100 * true_reject / len(unknown_scores),
            "wrong_identity_accept_rate_pct":
                100 * wrong_identity_accept / len(known_scores),
        })

    results = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(THRESHOLD_RESULTS_FILE, index=False)

    # --------------------------------------------------------
    # 5. Report dataset and detection counts
    # --------------------------------------------------------
    known_total_rows = len(all_known)
    known_detection_failures = int((~known_detected).sum())

    unknown_detection_failures = int(
        (~unknown_detected).sum()
    )

    print("\n" + "=" * 65)
    print("CORRECTED OPEN-SET EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Gallery identities: {len(gallery_ids)}")
    print(f"Original known CSV rows: {known_total_rows}")
    print(f"Known CSV detection failures: {known_detection_failures}")
    print(f"Enrolled known images evaluated: {len(known_scores)}")
    print(f"Unenrolled images from known CSV: {len(unenrolled_known)}")
    print(f"Unknown-test rows: {len(unknown_all)}")
    print(f"Unknown-test detection failures: {unknown_detection_failures}")
    print(f"Unknown-test identities overlapping gallery: {len(overlap)}")
    print(f"Valid unenrolled images evaluated: {len(unknown_scores)}")

    print("\nThreshold results saved to:")
    print(THRESHOLD_RESULTS_FILE)

    print(
        "\nNo single best threshold has been selected. "
        "Choose one based on the acceptable false-accept rate "
        "and correct identification rate for your application."
    )

# ============================================================
# 10. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("FULL OPEN-SET FACE RECOGNITION EVALUATION")
    print("InsightFace + ArcFace | CPU")
    print("=" * 65)

    gallery_embeddings, gallery_labels = load_gallery()
    known_df = load_known_results(gallery_labels)
    app = load_model()

    unknown_df = evaluate_unknown_images(
        app,
        gallery_embeddings,
        gallery_labels
    )

    analyze_thresholds(known_df, unknown_df, gallery_labels)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()

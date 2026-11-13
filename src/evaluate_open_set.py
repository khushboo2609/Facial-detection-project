from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "gallery_embeddings.npz"
)

KNOWN_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "lfw_recognition_results.csv"
)

UNKNOWN_TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "lfw_open_set"
    / "unknown_test"
)

RESULTS_DIR = PROJECT_ROOT / "results"

UNKNOWN_RESULTS_FILE = (
    RESULTS_DIR
    / "unknown_open_set_results.csv"
)

THRESHOLD_RESULTS_FILE = (
    RESULTS_DIR
    / "open_set_threshold_analysis.csv"
)

THRESHOLDS = np.arange(0.30, 0.81, 0.01)


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("FAST ARCFACE OPEN-SET EVALUATION")
print("=" * 70)


# =========================================================
# LOAD EXISTING GALLERY
# =========================================================

print("\nLoading existing gallery embeddings...")

gallery_data = np.load(
    GALLERY_FILE,
    allow_pickle=True
)

gallery_embeddings = (
    gallery_data["embeddings"]
    .astype(np.float32)
)

gallery_labels = gallery_data["labels"].astype(str)

# Normalize gallery embeddings
gallery_norms = np.linalg.norm(
    gallery_embeddings,
    axis=1,
    keepdims=True
)

gallery_embeddings = (
    gallery_embeddings / gallery_norms
)

print(
    f"Gallery identities: "
    f"{len(gallery_labels)}"
)

print(
    f"Gallery shape: "
    f"{gallery_embeddings.shape}"
)


# =========================================================
# LOAD EXISTING KNOWN RESULTS
# =========================================================

print("\nLoading existing known-person results...")

known_df = pd.read_csv(
    KNOWN_RESULTS_FILE
)

known_df = known_df[
    known_df["face_detected"] == True
].copy()

known_df["similarity"] = pd.to_numeric(
    known_df["cosine_similarity"],
    errors="coerce"
)

known_df = known_df.dropna(
    subset=["similarity"]
)

known_df["correct_identity"] = (
    known_df["result"] == "CORRECT"
)

print(
    f"Known images available: "
    f"{len(known_df)}"
)

print(
    f"Known correct: "
    f"{known_df['correct_identity'].sum()}"
)

print(
    f"Known incorrect: "
    f"{(~known_df['correct_identity']).sum()}"
)


# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("\nLoading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully!")


# =========================================================
# EMBEDDING FUNCTION
# =========================================================

def get_embedding(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None, None

    faces = app.get(image)

    if len(faces) == 0:
        return None, None

    # Select largest face
    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding.astype(
        np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None, None

    embedding = embedding / norm

    return embedding, float(face.det_score)


# =========================================================
# RECOGNITION
# =========================================================

def recognize(embedding):

    similarities = np.dot(
        gallery_embeddings,
        embedding
    )

    best_index = np.argmax(
        similarities
    )

    return (
        gallery_labels[best_index],
        float(similarities[best_index])
    )


# =========================================================
# FIND UNKNOWN IMAGES
# =========================================================

print("\n" + "=" * 70)
print("FINDING UNKNOWN IMAGES")
print("=" * 70)

unknown_images = []

for identity_dir in sorted(
    UNKNOWN_TEST_DIR.iterdir()
):

    if not identity_dir.is_dir():
        continue

    for image_path in sorted(
        identity_dir.glob("*.jpg")
    ):

        unknown_images.append(
            (
                identity_dir.name,
                image_path
            )
        )

print(
    f"Unknown images found: "
    f"{len(unknown_images)}"
)


# =========================================================
# RESUME SUPPORT
# =========================================================

existing_results = {}

if UNKNOWN_RESULTS_FILE.exists():

    print(
        "\nExisting unknown results found."
    )

    previous_df = pd.read_csv(
        UNKNOWN_RESULTS_FILE
    )

    for _, row in previous_df.iterrows():

        existing_results[
            row["image"]
        ] = row.to_dict()

    print(
        f"Already processed: "
        f"{len(existing_results)}"
    )


# =========================================================
# PROCESS UNKNOWN PEOPLE ONLY
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING UNKNOWN PEOPLE ONLY")
print("=" * 70)

new_results = []

for index, (
    actual_identity,
    image_path
) in enumerate(
    unknown_images,
    start=1
):

    image_string = str(image_path)

    # Skip already processed
    if image_string in existing_results:
        continue

    embedding, detection_score = (
        get_embedding(image_path)
    )

    if embedding is None:

        result = {
            "actual_identity":
                actual_identity,

            "predicted_identity":
                "",

            "similarity":
                np.nan,

            "detection_score":
                np.nan,

            "face_detected":
                False,

            "image":
                image_string
        }

    else:

        predicted_identity, similarity = (
            recognize(embedding)
        )

        result = {
            "actual_identity":
                actual_identity,

            "predicted_identity":
                predicted_identity,

            "similarity":
                similarity,

            "detection_score":
                detection_score,

            "face_detected":
                True,

            "image":
                image_string
        }

    existing_results[image_string] = result
    new_results.append(result)

    # Save every 100 images
    if len(new_results) >= 100:

        temp_df = pd.DataFrame(
            existing_results.values()
        )

        temp_df.to_csv(
            UNKNOWN_RESULTS_FILE,
            index=False
        )

        new_results.clear()

    # Progress
    if index % 100 == 0:

        processed = len(existing_results)

        print(
            f"Progress: "
            f"{index}/{len(unknown_images)} | "
            f"Stored: {processed}"
        )


# Final save
unknown_df = pd.DataFrame(
    existing_results.values()
)

unknown_df.to_csv(
    UNKNOWN_RESULTS_FILE,
    index=False
)

print("\nUnknown results saved:")
print(UNKNOWN_RESULTS_FILE)


# =========================================================
# PREPARE DATA
# =========================================================

unknown_valid = unknown_df[
    unknown_df["face_detected"] == True
].copy()

unknown_similarities = (
    unknown_valid["similarity"]
    .to_numpy()
)


known_similarities = (
    known_df["similarity"]
    .to_numpy()
)

known_correct = (
    known_df["correct_identity"]
    .to_numpy()
)

total_genuine = np.sum(
    known_correct
)


# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

threshold_results = []

for threshold in THRESHOLDS:

    # -----------------------------------------------------
    # KNOWN
    # -----------------------------------------------------

    known_accept = (
        known_similarities >= threshold
    )

    true_accept = np.sum(
        known_accept &
        known_correct
    )

    false_reject = np.sum(
        (~known_accept) &
        known_correct
    )

    wrong_identity_accept = np.sum(
        known_accept &
        (~known_correct)
    )

    # -----------------------------------------------------
    # UNKNOWN
    # -----------------------------------------------------

    unknown_accept = (
        unknown_similarities >= threshold
    )

    false_accept = np.sum(
        unknown_accept
    )

    true_reject = np.sum(
        ~unknown_accept
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    FAR = (
        false_accept
        / len(unknown_similarities)
        * 100
        if len(unknown_similarities) > 0
        else 0
    )

    FRR = (
        false_reject
        / total_genuine
        * 100
        if total_genuine > 0
        else 0
    )

    known_acceptance = (
        true_accept
        / total_genuine
        * 100
        if total_genuine > 0
        else 0
    )

    unknown_rejection = (
        true_reject
        / len(unknown_similarities)
        * 100
        if len(unknown_similarities) > 0
        else 0
    )

    overall_total = (
        total_genuine
        + len(unknown_similarities)
    )

    overall_correct = (
        true_accept
        + true_reject
    )

    overall_accuracy = (
        overall_correct
        / overall_total
        * 100
        if overall_total > 0
        else 0
    )

    threshold_results.append({

        "threshold":
            round(threshold, 2),

        "true_accept":
            int(true_accept),

        "false_reject":
            int(false_reject),

        "wrong_identity_accept":
            int(wrong_identity_accept),

        "false_accept":
            int(false_accept),

        "true_reject":
            int(true_reject),

        "FAR_percent":
            FAR,

        "FRR_percent":
            FRR,

        "known_acceptance_percent":
            known_acceptance,

        "unknown_rejection_percent":
            unknown_rejection,

        "overall_accuracy_percent":
            overall_accuracy
    })


threshold_df = pd.DataFrame(
    threshold_results
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("OPEN-SET SUMMARY")
print("=" * 70)

print(
    f"Known images used          : "
    f"{len(known_df)}"
)

print(
    f"Known correct              : "
    f"{known_correct.sum()}"
)

print(
    f"Unknown images processed   : "
    f"{len(unknown_df)}"
)

print(
    f"Unknown faces detected     : "
    f"{len(unknown_valid)}"
)

print(
    f"Unknown detection failures : "
    f"{len(unknown_df) - len(unknown_valid)}"
)


# =========================================================
# THRESHOLD TABLE
# =========================================================

print("\n" + "=" * 70)
print("OPEN-SET THRESHOLD RESULTS")
print("=" * 70)

columns = [
    "threshold",
    "FAR_percent",
    "FRR_percent",
    "known_acceptance_percent",
    "unknown_rejection_percent",
    "overall_accuracy_percent"
]

print(
    threshold_df[columns].to_string(
        index=False,
        formatters={
            "FAR_percent":
                "{:.2f}".format,

            "FRR_percent":
                "{:.2f}".format,

            "known_acceptance_percent":
                "{:.2f}".format,

            "unknown_rejection_percent":
                "{:.2f}".format,

            "overall_accuracy_percent":
                "{:.2f}".format
        }
    )
)


# =========================================================
# BEST THRESHOLD
# =========================================================

best = threshold_df.loc[
    threshold_df[
        "overall_accuracy_percent"
    ].idxmax()
]

print("\n" + "=" * 70)
print("BEST OPEN-SET THRESHOLD")
print("=" * 70)

print(
    f"Threshold                 : "
    f"{best['threshold']:.2f}"
)

print(
    f"FAR                       : "
    f"{best['FAR_percent']:.2f}%"
)

print(
    f"FRR                       : "
    f"{best['FRR_percent']:.2f}%"
)

print(
    f"Known acceptance          : "
    f"{best['known_acceptance_percent']:.2f}%"
)

print(
    f"Unknown rejection         : "
    f"{best['unknown_rejection_percent']:.2f}%"
)

print(
    f"Overall accuracy          : "
    f"{best['overall_accuracy_percent']:.2f}%"
)


# =========================================================
# SAVE THRESHOLD RESULTS
# =========================================================

threshold_df.to_csv(
    THRESHOLD_RESULTS_FILE,
    index=False
)

print("\nThreshold analysis saved:")
print(THRESHOLD_RESULTS_FILE)

print("\n" + "=" * 70)
print("OPEN-SET EVALUATION COMPLETED")
print("=" * 70)
from pathlib import Path
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "lfw_recognition"
    / "test"
)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "embeddings"
        / "gallery_embeddings_without_javier.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "lfw_recognition_results_without_javier.csv"

# ============================================================
# LOAD GALLERY
# ============================================================

print("=" * 70)
print("FULL LFW ARCFACE RECOGNITION EVALUATION")
print("=" * 70)

data = np.load(EMBEDDING_FILE)

gallery_embeddings = data["embeddings"]
gallery_labels = data["labels"]

print("\nGallery embeddings:", gallery_embeddings.shape)
print("Gallery identities:", len(gallery_labels))

# Normalize gallery embeddings
gallery_embeddings = gallery_embeddings / (
    np.linalg.norm(
        gallery_embeddings,
        axis=1,
        keepdims=True
    ) + 1e-10
)

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("\nLoading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("InsightFace model loaded successfully!")

# ============================================================
# GET TEST IMAGES
# ============================================================

test_images = sorted(
    TEST_DIR.glob("*/*.jpg")
)

total_images = len(test_images)

print("\nTotal test images:", total_images)

# ============================================================
# CSV SETUP
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "actual_identity",
        "predicted_identity",
        "cosine_similarity",
        "detection_score",
        "face_detected",
        "result"
    ])

    # ========================================================
    # COUNTERS
    # ========================================================

    processed = 0
    correct = 0
    incorrect = 0
    detection_failed = 0

    similarity_scores = []

    # ========================================================
    # PROCESS TEST IMAGES
    # ========================================================

    for index, image_path in enumerate(
        test_images,
        start=1
    ):

        actual_identity = image_path.parent.name

        image = cv2.imread(str(image_path))

        # ----------------------------------------------------
        # IMAGE LOAD FAILURE
        # ----------------------------------------------------

        if image is None:

            writer.writerow([
                image_path.name,
                actual_identity,
                "",
                "",
                "",
                False,
                "IMAGE_LOAD_FAILED"
            ])

            detection_failed += 1
            continue

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        faces = app.get(image)

        if len(faces) == 0:

            writer.writerow([
                image_path.name,
                actual_identity,
                "",
                "",
                "",
                False,
                "NO_FACE_DETECTED"
            ])

            detection_failed += 1

        else:

            # Select largest face
            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )

            embedding = face.embedding.astype(
                np.float32
            )

            # Normalize test embedding
            embedding = embedding / (
                np.linalg.norm(embedding)
                + 1e-10
            )

            # ------------------------------------------------
            # COSINE SIMILARITY
            # ------------------------------------------------

            similarities = (
                gallery_embeddings @ embedding
            )

            best_index = np.argmax(similarities)

            predicted_identity = str(
                gallery_labels[best_index]
            )

            best_similarity = float(
                similarities[best_index]
            )

            detection_score = float(
                face.det_score
            )

            similarity_scores.append(
                best_similarity
            )

            # ------------------------------------------------
            # CHECK RESULT
            # ------------------------------------------------

            if predicted_identity == actual_identity:
                result = "CORRECT"
                correct += 1
            else:
                result = "INCORRECT"
                incorrect += 1

            processed += 1

            writer.writerow([
                image_path.name,
                actual_identity,
                predicted_identity,
                round(best_similarity, 6),
                round(detection_score, 6),
                True,
                result
            ])

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == total_images
        ):

            accuracy = (
                correct / processed * 100
                if processed > 0
                else 0
            )

            print(
                f"Progress: {index}/{total_images} | "
                f"Processed: {processed} | "
                f"Correct: {correct} | "
                f"Incorrect: {incorrect} | "
                f"No Face: {detection_failed} | "
                f"Accuracy: {accuracy:.2f}%"
            )

# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print("Total test images:", total_images)
print("Processed:", processed)
print("Correct:", correct)
print("Incorrect:", incorrect)
print("Detection/load failures:", detection_failed)

if processed > 0:

    accuracy = (
        correct / processed
    ) * 100

    print(
        "Top-1 Recognition Accuracy:",
        f"{accuracy:.2f}%"
    )

if similarity_scores:

    similarity_scores = np.array(
        similarity_scores
    )

    print(
        "\nMinimum similarity:",
        f"{similarity_scores.min():.4f}"
    )

    print(
        "Maximum similarity:",
        f"{similarity_scores.max():.4f}"
    )

    print(
        "Mean similarity:",
        f"{similarity_scores.mean():.4f}"
    )

    print(
        "Median similarity:",
        f"{np.median(similarity_scores):.4f}"
    )

print("\nResults saved to:")
print(RESULTS_FILE)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
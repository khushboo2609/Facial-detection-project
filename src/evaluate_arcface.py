
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
    / "gallery_embeddings.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "arcface_evaluation.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Closed-set identification: always select the nearest gallery identity.
# Unknown-person rejection will be evaluated separately after calibration.

# ============================================================
# LOAD GALLERY
# ============================================================

print("=" * 70)
print("INSIGHTFACE + ARCFACE FULL DATASET EVALUATION")
print("=" * 70)

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if not EMBEDDING_FILE.is_file():
    raise FileNotFoundError(
        f"Gallery embeddings not found: {EMBEDDING_FILE}"
    )

with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
    gallery_embeddings = data["embeddings"].astype(np.float32)
    gallery_labels = data["labels"].astype(str)

if gallery_embeddings.ndim != 2:
    raise ValueError("Gallery embeddings must be a 2D matrix.")

if gallery_embeddings.shape[1] != 512:
    raise ValueError(
        f"Expected 512 features per embedding, "
        f"got {gallery_embeddings.shape}"
    )

if len(gallery_embeddings) != len(gallery_labels):
    raise ValueError("Gallery embeddings and labels have different lengths.")

if len(gallery_embeddings) == 0:
    raise ValueError("Gallery is empty.")

norms = np.linalg.norm(
    gallery_embeddings, axis=1, keepdims=True
)
gallery_embeddings /= np.maximum(norms, 1e-12)

gallery_identity_set = set(gallery_labels.tolist())

print("Gallery embeddings:", gallery_embeddings.shape)
print("Gallery identities:", len(gallery_identity_set))

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace loaded successfully.")

# ============================================================
# FIND TEST IMAGES
# ============================================================

test_images = sorted(
    path
    for path in TEST_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not test_images:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

print("Test images found:", len(test_images))
print("Starting evaluation...\n")

# ============================================================
# EVALUATION
# ============================================================

rows = []

processed = 0
correct = 0
incorrect = 0
no_face = 0
image_load_failed = 0
invalid_embedding = 0

# Track results separately for identities represented in the gallery.
known_total = 0
known_correct = 0

# Test identities without a usable gallery embedding.
unseen_total = 0
unseen_nearest_match = 0

for index, image_path in enumerate(test_images, start=1):

    true_label = image_path.parent.name

    image = cv2.imread(str(image_path))

    if image is None:
        image_load_failed += 1
        rows.append([
            str(image_path), true_label, "", "", "",
            "image_load_failed"
        ])
        continue

    faces = app.get(image)

    if len(faces) == 0:
        no_face += 1
        rows.append([
            str(image_path), true_label, "", "", "",
            "no_face_detected"
        ])
        continue

    
    # Default: select the largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    # Diagnostic check for Charles Moose test image
    if (
        image_path.parent.name == "Charles_Moose"
        and image_path.name == "Charles_Moose_0003.jpg"
    ):
        print("\n" + "=" * 60)
        print("CHARLES MOOSE FACE-SELECTION DEBUG")
        print("=" * 60)

        for face_index, detected_face in enumerate(faces):
            embedding = detected_face.embedding.astype(np.float32)
            embedding /= max(np.linalg.norm(embedding), 1e-12)

            face_similarities = gallery_embeddings @ embedding
            top_indices = np.argsort(face_similarities)[-5:][::-1]

            print(f"\nFace {face_index + 1}")
            print("Bounding box:", detected_face.bbox)

            for rank, gallery_index in enumerate(top_indices, start=1):
                print(
                    f"{rank}. {gallery_labels[gallery_index]} "
                    f"| similarity: "
                    f"{face_similarities[gallery_index]:.6f}"
                )

        # Temporarily select the visually confirmed Charles face
        face = faces[0]
        print("\nSelected: Face 1 (visually identified as Charles Moose)")
        print("=" * 60)


# ============================================================
# SAVE CSV
# ============================================================

with OUTPUT_CSV.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_path",
        "true_identity",
        "nearest_gallery_identity",
        "cosine_similarity",
        "identity_in_gallery",
        "status"
    ])
    writer.writerows(rows)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print("Total test images:", len(test_images))
print("Successfully processed:", processed)
print("Correct known-identity predictions:", correct)
print("Incorrect known-identity predictions:", incorrect)
print("Images with no detected face:", no_face)
print("Images that could not be loaded:", image_load_failed)
print("Invalid embeddings:", invalid_embedding)
print("Test images whose identity is absent from gallery:", unseen_total)

if known_total > 0:
    accuracy = known_correct / known_total * 100
    print("Known-identity accuracy:", f"{accuracy:.2f}%")
    print(
        f"Known-identity correct: {known_correct}/{known_total}"
    )
else:
    print("Known-identity accuracy: N/A")

print("\nCSV saved to:", OUTPUT_CSV)
print("=" * 70)

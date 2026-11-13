
from pathlib import Path
import csv

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "dataset" / "lfw_recognition" / "test"

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "gallery_embeddings_single_face.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_CSV = RESULTS_DIR / "arcface_evaluation_clean_gallery.csv"
SUMMARY_FILE = RESULTS_DIR / "arcface_clean_gallery_summary.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================
# VALIDATE PATHS
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if not EMBEDDING_FILE.is_file():
    raise FileNotFoundError(
        f"Clean gallery not found: {EMBEDDING_FILE}"
    )

print("=" * 70)
print("ARCFACE EVALUATION - CLEAN SINGLE-FACE GALLERY")
print("=" * 70)

# ============================================================
# LOAD CLEAN GALLERY
# ============================================================

with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
    gallery_embeddings = data["embeddings"].astype(np.float32)
    gallery_labels = data["labels"].astype(str)

if (
    gallery_embeddings.ndim != 2
    or gallery_embeddings.shape[1] != 512
):
    raise ValueError(
        f"Expected gallery shape (N, 512), "
        f"got {gallery_embeddings.shape}"
    )

if len(gallery_embeddings) != len(gallery_labels):
    raise ValueError("Embedding and label counts do not match.")

if len(gallery_embeddings) == 0:
    raise ValueError("Clean gallery is empty.")

# Normalize embeddings for cosine similarity.
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
app.prepare(ctx_id=-1, det_size=(640, 640))

print("InsightFace loaded successfully.")

# ============================================================
# FIND TEST IMAGES
# ============================================================

test_images = sorted(
    path for path in TEST_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not test_images:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

print("Test images found:", len(test_images))
print("Starting evaluation...\n")

# ============================================================
# COUNTERS
# ============================================================

rows = []

processed = 0
correct = 0
incorrect = 0
no_face = 0
image_load_failed = 0
invalid_embedding = 0

known_total = 0
known_correct = 0

unseen_total = 0
unseen_nearest_match = 0

# ============================================================
# EVALUATION
# ============================================================

for index, image_path in enumerate(test_images, start=1):

    true_label = image_path.parent.name

    if true_label not in gallery_identity_set:
        identity_available = "no"
    else:
        identity_available = "yes"

    image = cv2.imread(str(image_path))

    if image is None:
        image_load_failed += 1
        rows.append([
            str(image_path), true_label, "", "",
            identity_available, "image_load_failed"
        ])
        continue

    faces = app.get(image)

    if len(faces) == 0:
        no_face += 1
        rows.append([
            str(image_path), true_label, "", "",
            identity_available, "no_face_detected"
        ])
        continue

    # Keep the same largest-face selection approach.
    # This does not guarantee that the largest face is the
    # person named by the test folder.
    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0])
            * (f.bbox[3] - f.bbox[1])
        )
    )

    embedding = face.normed_embedding.astype(np.float32)
    embedding_norm = np.linalg.norm(embedding)

    if (
        not np.all(np.isfinite(embedding))
        or not np.isfinite(embedding_norm)
        or embedding_norm < 1e-12
    ):
        invalid_embedding += 1
        rows.append([
            str(image_path), true_label, "", "",
            identity_available, "invalid_embedding"
        ])
        continue

    embedding /= embedding_norm

    # Find nearest gallery identity.
    similarities = gallery_embeddings @ embedding
    best_index = int(np.argmax(similarities))

    predicted_label = gallery_labels[best_index]
    best_similarity = float(similarities[best_index])

    processed += 1

    if true_label in gallery_identity_set:
        known_total += 1

        if predicted_label == true_label:
            correct += 1
            known_correct += 1
            status = "correct"
        else:
            incorrect += 1
            status = "incorrect"
    else:
        unseen_total += 1

        # A nearest identity is always returned in this
        # closed-set evaluation, even for unseen identities.
        unseen_nearest_match += 1
        status = "identity_not_in_gallery"

    rows.append([
        str(image_path),
        true_label,
        predicted_label,
        f"{best_similarity:.6f}",
        identity_available,
        status
    ])

    if index % 100 == 0:
        print(f"Evaluated {index}/{len(test_images)} images")

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

known_accuracy = (
    known_correct / known_total * 100
    if known_total else None
)

summary_lines = [
    "=" * 60,
    "ARCFACE CLEAN GALLERY EVALUATION SUMMARY",
    "=" * 60,
    f"Gallery embeddings: {len(gallery_embeddings)}",
    f"Gallery identities: {len(gallery_identity_set)}",
    f"Total test images: {len(test_images)}",
    f"Successfully evaluated: {processed}",
    f"Correct known-identity predictions: {correct}",
    f"Incorrect known-identity predictions: {incorrect}",
    f"Known identities evaluated: {known_total}",
    f"Test images with identities absent from gallery: {unseen_total}",
    f"Images with no detected face: {no_face}",
    f"Images that could not be loaded: {image_load_failed}",
    f"Invalid embeddings: {invalid_embedding}",
    f"Known-identity accuracy: "
    f"{known_accuracy:.2f}%" if known_accuracy is not None
    else "Known-identity accuracy: N/A",
    "",
    "Note: This is closed-set nearest-neighbor identification.",
    "Unseen identities still receive a nearest gallery match.",
    "No unknown-person rejection threshold is applied.",
    "Largest detected face is selected in each test image.",
    "",
    f"CSV: {OUTPUT_CSV}",
    f"Summary: {SUMMARY_FILE}",
    "=" * 60
]

summary_text = "\n".join(summary_lines)

SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

print("\n" + summary_text)

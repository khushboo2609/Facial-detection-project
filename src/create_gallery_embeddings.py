from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "lfw_recognition"
    / "gallery"
)

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
OUTPUT_FILE = EMBEDDINGS_DIR / "gallery_embeddings.npz"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("CREATING GALLERY ARCFACE EMBEDDINGS")
print("=" * 70)

print("\nGallery:", GALLERY_DIR)

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
# PROCESS GALLERY
# ============================================================

embeddings = []
labels = []
image_paths = []

successful = 0
failed = 0

identity_dirs = sorted(
    folder for folder in GALLERY_DIR.iterdir()
    if folder.is_dir()
)

print("\nGallery identities:", len(identity_dirs))

for index, identity_dir in enumerate(identity_dirs, start=1):

    identity = identity_dir.name

    images = sorted(identity_dir.glob("*.jpg"))

    if not images:
        failed += 1
        continue

    image_path = images[0]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[FAILED] {identity}: image could not be loaded")
        failed += 1
        continue

    faces = app.get(image)

    if len(faces) == 0:
        print(f"[FAILED] {identity}: no face detected")
        failed += 1
        continue

    # Select the largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                       (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding.astype(np.float32)

    if embedding.shape != (512,):
        print(
            f"[FAILED] {identity}: "
            f"unexpected embedding shape {embedding.shape}"
        )
        failed += 1
        continue

    embeddings.append(embedding)
    labels.append(identity)
    image_paths.append(str(image_path))

    successful += 1

    # Progress every 100 identities
    if index % 100 == 0 or index == len(identity_dirs):
        print(
            f"Progress: {index}/{len(identity_dirs)} | "
            f"Successful: {successful} | "
            f"Failed: {failed}"
        )

# ============================================================
# SAVE
# ============================================================

embeddings = np.array(embeddings, dtype=np.float32)
labels = np.array(labels)
image_paths = np.array(image_paths)

np.savez(
    OUTPUT_FILE,
    embeddings=embeddings,
    labels=labels,
    image_paths=image_paths
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GALLERY EMBEDDINGS COMPLETED")
print("=" * 70)

print("Successful:", successful)
print("Failed:", failed)
print("Embedding matrix shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print("Saved to:", OUTPUT_FILE)

print("\n" + "=" * 70)
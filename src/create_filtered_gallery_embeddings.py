
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_DIR = (
    PROJECT_ROOT / "dataset" / "lfw_open_set_single_face" / "gallery"
)

OUTPUT_DIR = PROJECT_ROOT / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "filtered_gallery_embeddings.npz"
REPORT_FILE = PROJECT_ROOT / "results" / "filtered_gallery_build_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BUILDING FILTERED ARCFACE GALLERY")
print("=" * 70)
print("Gallery:", GALLERY_DIR)
print("Output:", OUTPUT_FILE)

if not GALLERY_DIR.exists():
    raise FileNotFoundError(f"Gallery directory not found: {GALLERY_DIR}")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
labels = []
image_paths = []
report = []

identity_dirs = sorted(
    path for path in GALLERY_DIR.iterdir() if path.is_dir()
)

print("Gallery identity folders:", len(identity_dirs))

for index, identity_dir in enumerate(identity_dirs, start=1):
    identity = identity_dir.name

    images = sorted(
        path for path in identity_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        report.append({
            "identity": identity,
            "image": "",
            "status": "NO_IMAGE",
            "detected_faces": 0,
        })
        continue

    # Use the first image that can be verified to contain exactly one face.
    saved = False

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            report.append({
                "identity": identity,
                "image": str(image_path),
                "status": "IMAGE_LOAD_FAILED",
                "detected_faces": 0,
            })
            continue

        faces = app.get(image)

        if len(faces) != 1:
            report.append({
                "identity": identity,
                "image": str(image_path),
                "status": (
                    "NO_FACE" if len(faces) == 0 else "MULTIPLE_FACES"
                ),
                "detected_faces": len(faces),
            })
            continue

        embedding = np.asarray(
            faces[0].embedding, dtype=np.float32
        )

        if embedding.shape != (512,):
            report.append({
                "identity": identity,
                "image": str(image_path),
                "status": "INVALID_EMBEDDING",
                "detected_faces": 1,
            })
            continue

        norm = np.linalg.norm(embedding)

        if norm == 0:
            report.append({
                "identity": identity,
                "image": str(image_path),
                "status": "ZERO_EMBEDDING",
                "detected_faces": 1,
            })
            continue

        embeddings.append(embedding / norm)
        labels.append(identity)
        image_paths.append(str(image_path))

        report.append({
            "identity": identity,
            "image": str(image_path),
            "status": "SAVED",
            "detected_faces": 1,
        })

        saved = True
        break

    if not saved:
        print(f"[FAILED] No usable single-face image: {identity}")

    if index % 100 == 0 or index == len(identity_dirs):
        print(f"Progress: {index}/{len(identity_dirs)}")

if not embeddings:
    raise RuntimeError("No gallery embeddings were generated.")

# Save to a NEW filename. Existing gallery files remain unchanged.
np.savez(
    OUTPUT_FILE,
    embeddings=np.asarray(embeddings, dtype=np.float32),
    labels=np.asarray(labels, dtype=str),
    image_paths=np.asarray(image_paths, dtype=str),
)

import pandas as pd

pd.DataFrame(report).to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 70)
print("FILTERED GALLERY BUILD COMPLETE")
print("=" * 70)
print("Identities with embeddings:", len(labels))
print("Embedding matrix shape:", np.asarray(embeddings).shape)
print("Build report:", REPORT_FILE)
print("Saved embeddings:", OUTPUT_FILE)

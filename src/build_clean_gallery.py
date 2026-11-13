
from pathlib import Path
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_DIR = (
    PROJECT_ROOT / "dataset" / "lfw_recognition" / "gallery"
)

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = EMBEDDING_DIR / "gallery_embeddings_single_face.npz"
REPORT_FILE = RESULTS_DIR / "clean_gallery_report.csv"

# ============================================================
# INITIALIZATION
# ============================================================

EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

embeddings = []
labels = []
image_paths = []
report_rows = []

# ============================================================
# PROCESS GALLERY
# ============================================================

image_files = sorted(
    p for p in GALLERY_DIR.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

print(f"Gallery images found: {len(image_files)}")
print("Keeping images with exactly one detected face.\n")

for index, image_path in enumerate(image_files, start=1):
    label = image_path.parent.name

    image = cv2.imread(str(image_path))

    if image is None:
        report_rows.append([
            str(image_path), label, 0, "Unreadable image"
        ])
        continue

    faces = app.get(image)

    if len(faces) != 1:
        status = (
            "No face detected"
            if len(faces) == 0
            else "Multiple faces detected"
        )
        report_rows.append([
            str(image_path), label, len(faces), status
        ])
        continue

    face = faces[0]
    embedding = face.normed_embedding.astype(np.float32)

    embeddings.append(embedding)
    labels.append(label)
    image_paths.append(str(image_path))

    report_rows.append([
        str(image_path), label, 1, "Included"
    ])

    if index % 100 == 0:
        print(f"Checked {index}/{len(image_files)} images")

# ============================================================
# SAVE NEW EMBEDDINGS
# ============================================================

if embeddings:
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=np.vstack(embeddings),
        labels=np.asarray(labels),
        image_paths=np.asarray(image_paths)
    )

# ============================================================
# SAVE REPORT
# ============================================================

with open(REPORT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_path", "identity_label",
        "detected_faces", "status"
    ])
    writer.writerows(report_rows)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("CLEAN GALLERY BUILD COMPLETE")
print("=" * 55)
print(f"Images examined: {len(image_files)}")
print(f"Images included: {len(embeddings)}")
print(f"Unique identities included: {len(set(labels))}")
print(f"Images skipped: {len(image_files) - len(embeddings)}")
print(f"New embeddings: {OUTPUT_FILE}")
print(f"Report: {REPORT_FILE}")
print("\nOriginal gallery_embeddings.npz was not changed.")

from pathlib import Path
import cv2
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = PROJECT_ROOT / "dataset" / "lfw_split" / "gallery"

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("INSIGHTFACE LFW UPSCALE DETECTION TEST")
print("=" * 70)

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("\nInsightFace model loaded successfully!")

# ============================================================
# FIND 10 IMAGES
# ============================================================

image_files = sorted(GALLERY_DIR.rglob("*.jpg"))[:10]

print(f"\nImages selected: {len(image_files)}")

successful = 0
failed = 0

# ============================================================
# TEST
# ============================================================

for i, image_path in enumerate(image_files, start=1):

    print("\n" + "-" * 70)
    print(f"IMAGE {i}/10")
    print("-" * 70)

    print("Person:", image_path.parent.name)
    print("File:", image_path.name)

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not read image")
        failed += 1
        continue

    print("Original shape:", image.shape)
    print("Original pixel range:", image.min(), "-", image.max())

    # --------------------------------------------------------
    # UPSCALE 4X
    # --------------------------------------------------------

    upscaled = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    print("Upscaled shape:", upscaled.shape)

    # --------------------------------------------------------
    # DETECT FACE
    # --------------------------------------------------------

    faces = app.get(upscaled)

    print("Faces detected:", len(faces))

    if len(faces) > 0:

        successful += 1

        face = faces[0]

        print("Detection Score:", face.det_score)
        print("Embedding Shape:", face.embedding.shape)

    else:

        failed += 1
        print("NO FACE DETECTED")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("UPSCALE TEST SUMMARY")
print("=" * 70)

print("Images tested:", len(image_files))
print("Successful detections:", successful)
print("Failed detections:", failed)

print("=" * 70)
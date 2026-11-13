from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_HOME = PROJECT_ROOT / "dataset" / "lfw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "lfw_split"

SEED = 42
GALLERY_RATIO = 0.70

print("=" * 70)
print("CREATING LFW GALLERY/TEST SPLIT")
print("=" * 70)

lfw = fetch_lfw_people(
    data_home=str(DATA_HOME),
    min_faces_per_person=20,
    resize=None,
    color=True
)

images = lfw.images
targets = lfw.target
target_names = lfw.target_names

print(f"\nTotal images: {len(images)}")
print(f"Identities: {len(target_names)}")
print(f"Image shape: {images.shape}")

print("\nSource pixel range:")
print("Min:", images.min())
print("Max:", images.max())
print("dtype:", images.dtype)

rng = np.random.default_rng(SEED)

gallery_dir = OUTPUT_DIR / "gallery"
test_dir = OUTPUT_DIR / "test"

gallery_count = 0
test_count = 0

for person_id, person_name in enumerate(target_names):

    indices = np.where(targets == person_id)[0]
    rng.shuffle(indices)

    split_point = int(len(indices) * GALLERY_RATIO)

    gallery_indices = indices[:split_point]
    test_indices = indices[split_point:]

    person_gallery = gallery_dir / person_name
    person_test = test_dir / person_name

    person_gallery.mkdir(parents=True, exist_ok=True)
    person_test.mkdir(parents=True, exist_ok=True)

    for number, index in enumerate(gallery_indices, start=1):

        image = images[index]

        # LFW source is 0-1 float32.
        # Convert explicitly to 0-255 uint8.
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

        # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output_path = person_gallery / f"{number:03d}.jpg"

        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        gallery_count += 1

    for number, index in enumerate(test_indices, start=1):

        image = images[index]

        # LFW source is 0-1 float32.
        # Convert explicitly to 0-255 uint8.
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

        # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output_path = person_test / f"{number:03d}.jpg"

        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        test_count += 1

print("\n" + "=" * 70)
print("LFW SPLIT CREATED")
print("=" * 70)

print(f"\nGallery images: {gallery_count}")
print(f"Test images:    {test_count}")
print(f"Total images:   {gallery_count + test_count}")

# ------------------------------------------------------------
# Verify the actual saved file
# ------------------------------------------------------------

sample = gallery_dir / "Alejandro_Toledo" / "001.jpg"

saved = cv2.imread(str(sample))

print("\nSaved-image verification:")
print("Path:", sample)
print("Shape:", saved.shape)
print("dtype:", saved.dtype)
print("Min:", saved.min())
print("Max:", saved.max())
print("Mean:", saved.mean())

if saved.max() > 100:
    print("\n✅ IMAGE PIXEL RANGE IS CORRECT")
else:
    print("\n❌ IMAGE PIXEL RANGE IS STILL INCORRECT")

print("\n" + "=" * 70)
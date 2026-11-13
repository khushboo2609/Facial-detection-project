from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset" / "lfw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "lfw_recognition"

GALLERY_DIR = OUTPUT_DIR / "gallery"
TEST_DIR = OUTPUT_DIR / "test"

# ============================================================
# SETTINGS
# ============================================================

MIN_IMAGES_PER_PERSON = 2

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND IDENTITIES
# ============================================================

identities = sorted(
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
)

print("=" * 70)
print("CREATING LFW RECOGNITION SPLIT")
print("=" * 70)

print("\nSource:", SOURCE_DIR)
print("Output:", OUTPUT_DIR)

print("\nTotal identities found:", len(identities))

# ============================================================
# SPLIT DATA
# ============================================================

gallery_count = 0
test_count = 0
used_identities = 0
skipped_identities = 0

for identity_dir in identities:

    images = sorted(identity_dir.glob("*.jpg"))

    if len(images) < MIN_IMAGES_PER_PERSON:
        skipped_identities += 1
        continue

    identity = identity_dir.name

    # --------------------------------------------------------
    # One image for gallery
    # Remaining images for test
    # --------------------------------------------------------

    gallery_images = images[:1]
    test_images = images[1:]

    # Create identity directories
    gallery_identity_dir = GALLERY_DIR / identity
    test_identity_dir = TEST_DIR / identity

    gallery_identity_dir.mkdir(parents=True, exist_ok=True)
    test_identity_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Copy gallery image
    # --------------------------------------------------------

    for image in gallery_images:
        destination = gallery_identity_dir / image.name
        shutil.copy2(image, destination)
        gallery_count += 1

    # --------------------------------------------------------
    # Copy test images
    # --------------------------------------------------------

    for image in test_images:
        destination = test_identity_dir / image.name
        shutil.copy2(image, destination)
        test_count += 1

    used_identities += 1

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print("Identities used:", used_identities)
print("Identities skipped:", skipped_identities)

print("Gallery images:", gallery_count)
print("Test images:", test_count)

print("\nGallery:", GALLERY_DIR)
print("Test:", TEST_DIR)

print("\n" + "=" * 70)
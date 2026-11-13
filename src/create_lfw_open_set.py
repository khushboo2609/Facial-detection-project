from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = PROJECT_ROOT / "dataset" / "lfw"
OUTPUT = PROJECT_ROOT / "dataset" / "lfw_open_set"

GALLERY = OUTPUT / "gallery"
KNOWN_TEST = OUTPUT / "known_test"
UNKNOWN_TEST = OUTPUT / "unknown_test"

# ---------------------------------------------------------
# CLEAN PREVIOUS SPLIT
# ---------------------------------------------------------

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

GALLERY.mkdir(parents=True)
KNOWN_TEST.mkdir(parents=True)
UNKNOWN_TEST.mkdir(parents=True)

# ---------------------------------------------------------
# FIND ALL IDENTITIES
# ---------------------------------------------------------

identities = sorted(
    [p for p in SOURCE.iterdir() if p.is_dir()]
)

eligible = []
single_image_identities = []

for identity in identities:
    images = sorted(identity.glob("*.jpg"))

    if len(images) >= 2:
        eligible.append((identity, images))
    elif len(images) == 1:
        single_image_identities.append((identity, images))

print("=" * 70)
print("CREATING LFW OPEN-SET SPLIT")
print("=" * 70)

print(f"Total identities              : {len(identities)}")
print(f"Identities with >=2 images    : {len(eligible)}")
print(f"Identities with 1 image       : {len(single_image_identities)}")

# ---------------------------------------------------------
# KNOWN IDENTITIES
# ---------------------------------------------------------

known = eligible[:1680]

print(f"\nKnown identities              : {len(known)}")

gallery_count = 0
known_test_count = 0

for identity, images in known:

    identity_gallery = GALLERY / identity.name
    identity_test = KNOWN_TEST / identity.name

    identity_gallery.mkdir()
    identity_test.mkdir()

    # First image = gallery reference
    shutil.copy2(
        images[0],
        identity_gallery / images[0].name
    )

    gallery_count += 1

    # Remaining images = known test
    for image in images[1:]:
        shutil.copy2(
            image,
            identity_test / image.name
        )

        known_test_count += 1

# ---------------------------------------------------------
# UNKNOWN IDENTITIES
# Single-image identities are NEVER in the gallery
# ---------------------------------------------------------

unknown_count = 0

for identity, images in single_image_identities:

    identity_test = UNKNOWN_TEST / identity.name
    identity_test.mkdir()

    image = images[0]

    shutil.copy2(
        image,
        identity_test / image.name
    )

    unknown_count += 1

# ---------------------------------------------------------
# FINAL INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("OPEN-SET SPLIT CREATED")
print("=" * 70)

print(f"Gallery identities             : {gallery_count}")
print(f"Known test images              : {known_test_count}")
print(f"Unknown identities             : {len(single_image_identities)}")
print(f"Unknown test images            : {unknown_count}")

print("\nOutput:")
print(OUTPUT)

print("\nGallery:")
print(GALLERY)

print("\nKnown test:")
print(KNOWN_TEST)

print("\nUnknown test:")
print(UNKNOWN_TEST)

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)
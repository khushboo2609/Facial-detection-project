
from pathlib import Path
import csv
import shutil

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset" / "lfw_open_set_single_face"
OUTPUT = ROOT / "dataset" / "lfw_open_set_corrected"

GALLERY = SOURCE / "gallery"
KNOWN_TEST = SOURCE / "known_test"
UNKNOWN_TEST = SOURCE / "unknown_test"

OUT_GALLERY = OUTPUT / "gallery"
OUT_KNOWN = OUTPUT / "known_test"
OUT_UNKNOWN = OUTPUT / "unknown_test"

REPORT = ROOT / "results" / "corrected_dataset_report.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Refuse to overwrite an existing corrected dataset.
if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Rename or remove it manually only after checking its contents."
    )

for folder in (GALLERY, KNOWN_TEST, UNKNOWN_TEST):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing source folder: {folder}")


def image_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_identity(source_root, identity, destination_root):
    source_folder = source_root / identity
    destination_folder = destination_root / identity
    copied = 0

    for source_file in image_files(source_folder):
        relative = source_file.relative_to(source_folder)
        destination_file = destination_folder / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied += 1

    return copied


# Gallery identities define the enrolled identities.
gallery_ids = {
    p.name for p in GALLERY.iterdir() if p.is_dir()
}

known_ids = {
    p.name for p in KNOWN_TEST.iterdir() if p.is_dir()
}

unknown_ids = {
    p.name for p in UNKNOWN_TEST.iterdir() if p.is_dir()
}

# Move no files: copy the original gallery.
gallery_images = 0
for identity in sorted(gallery_ids):
    gallery_images += copy_identity(GALLERY, identity, OUT_GALLERY)

# Only enrolled identities belong in known_test.
known_kept = sorted(known_ids & gallery_ids)
known_moved_to_unknown = sorted(known_ids - gallery_ids)

known_images = 0
for identity in known_kept:
    known_images += copy_identity(
        KNOWN_TEST, identity, OUT_KNOWN
    )

# Combine existing unknown identities with unenrolled known_test identities.
# Keep the original unknown_test images if an identity name overlaps.
unknown_combined = unknown_ids | set(known_moved_to_unknown)

unknown_images = 0
for identity in sorted(unknown_combined):
    if identity in unknown_ids:
        unknown_images += copy_identity(
            UNKNOWN_TEST, identity, OUT_UNKNOWN
        )

    # Add images from known_test only when that identity is not
    # already in unknown_test, avoiding duplicate identity folders.
    elif identity in known_moved_to_unknown:
        unknown_images += copy_identity(
            KNOWN_TEST, identity, OUT_UNKNOWN
        )

# Report each identity's source and destination.
REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "identity",
            "source_split",
            "corrected_split",
            "image_count",
        ],
    )
    writer.writeheader()

    for identity in sorted(gallery_ids):
        writer.writerow({
            "identity": identity,
            "source_split": "gallery",
            "corrected_split": "gallery",
            "image_count": len(image_files(GALLERY / identity)),
        })

    for identity in sorted(known_ids):
        is_enrolled = identity in gallery_ids
        writer.writerow({
            "identity": identity,
            "source_split": "known_test",
            "corrected_split": (
                "known_test" if is_enrolled else "unknown_test"
            ),
            "image_count": len(image_files(KNOWN_TEST / identity)),
        })

    for identity in sorted(unknown_ids):
        writer.writerow({
            "identity": identity,
            "source_split": "unknown_test",
            "corrected_split": "unknown_test",
            "image_count": len(image_files(UNKNOWN_TEST / identity)),
        })

print("\nCORRECTED DATASET CREATED")
print("=" * 50)
print(f"Gallery identities: {len(gallery_ids)}")
print(f"Known-test identities retained: {len(known_kept)}")
print(f"Unenrolled identities reassigned: {len(known_moved_to_unknown)}")
print(f"Existing unknown identities: {len(unknown_ids)}")
print(f"Corrected unknown identities: {len(unknown_combined)}")
print(f"Gallery images copied: {gallery_images}")
print(f"Known-test images copied: {known_images}")
print(f"Unknown-test images copied: {unknown_images}")
print(f"Dataset folder: {OUTPUT}")
print(f"Report: {REPORT}")

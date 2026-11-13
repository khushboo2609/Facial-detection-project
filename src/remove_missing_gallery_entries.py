
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# Project root: parent of src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GALLERY_FILE = PROJECT_ROOT / "embeddings" / "gallery_embeddings.npz"
BACKUP_DIR = PROJECT_ROOT / "backups"

# Load the current archive
data = np.load(GALLERY_FILE, allow_pickle=True)

embeddings = data["embeddings"]
labels = data["labels"]
image_paths = data["image_paths"]

if not (
    len(embeddings) == len(labels) == len(image_paths)
):
    raise ValueError("Gallery arrays have inconsistent lengths.")

# Identify entries whose source image no longer exists
keep_mask = np.array([
    Path(str(image_path)).is_file()
    for image_path in image_paths
])

removed_count = int((~keep_mask).sum())

print("Original entries:", len(embeddings))
print("Entries to remove:", removed_count)

for i in np.where(~keep_mask)[0]:
    print("Removing:", labels[i], "|", image_paths[i])

if removed_count == 0:
    print("No missing-image entries found. Nothing changed.")
    raise SystemExit(0)

# Back up the original archive before changing anything
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file = BACKUP_DIR / f"gallery_embeddings_before_cleanup_{timestamp}.npz"

shutil.copy2(GALLERY_FILE, backup_file)
print("Backup created:", backup_file)

# Preserve every existing archive array, filtering only gallery rows
filtered = {
    key: data[key]
    for key in data.files
}

filtered["embeddings"] = embeddings[keep_mask]
filtered["labels"] = labels[keep_mask]
filtered["image_paths"] = image_paths[keep_mask]

# Save to a temporary archive, then replace the original
temporary_file = GALLERY_FILE.with_name(
    "gallery_embeddings.cleaned.tmp.npz"
)

np.savez_compressed(temporary_file, **filtered)
temporary_file.replace(GALLERY_FILE)

print("Cleanup complete.")
print("Remaining entries:", int(keep_mask.sum()))
print("Original archive backup:", backup_file)

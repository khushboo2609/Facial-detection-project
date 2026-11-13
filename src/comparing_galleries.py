import numpy as np
from pathlib import Path

gallery_file = Path("embeddings/gallery_embeddings.npz")

data = np.load(gallery_file, allow_pickle=True)
embeddings = data["embeddings"]
labels = data["labels"].astype(str)
image_paths = data["image_paths"].astype(str)

missing = []

for i, image_path in enumerate(image_paths):
    if not Path(image_path).is_file():
        missing.append((i, labels[i], image_path))

print("Total gallery embeddings:", len(embeddings))
print("Missing source images:", len(missing))

for index, label, path in missing:
    print(f"Index: {index} | Label: {label} | Path: {path}")
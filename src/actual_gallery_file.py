
from pathlib import Path
from collections import Counter

import numpy as np

gallery_file = Path("embeddings/gallery_embeddings.npz")

print("Gallery file exists:", gallery_file.exists())

if not gallery_file.exists():
    raise SystemExit("ERROR: Expected gallery file was not found.")

with np.load(gallery_file, allow_pickle=True) as data:
    print("Archive keys:", data.files)

    if "embeddings" not in data or "labels" not in data:
        raise SystemExit("ERROR: Expected embeddings/labels keys are missing.")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(str).reshape(-1)

print("\n--- Gallery diagnostics ---")
print("Embedding shape:", embeddings.shape)
print("Label count:", len(labels))
print("Embedding dtype:", embeddings.dtype)
print("All embeddings finite:", np.isfinite(embeddings).all())

if embeddings.ndim == 2 and embeddings.shape[0] > 0:
    norms = np.linalg.norm(embeddings, axis=1)

    print("Embedding dimensions:", embeddings.shape[1])
    print("Norm min:", float(norms.min()))
    print("Norm median:", float(np.median(norms)))
    print("Norm max:", float(norms.max()))

print("Unique identities:", len(set(labels)))

counts = Counter(labels)
duplicates = {
    name: count for name, count in counts.items()
    if count > 1
}

print("Identities with multiple embeddings:", len(duplicates))
print("First 15 labels:", labels[:15].tolist())
print("Most repeated labels:", counts.most_common(10))

if embeddings.ndim != 2 or embeddings.shape[1] != 512:
    print("WARNING: Expected embeddings with shape (N, 512).")

if embeddings.shape[0] != len(labels):
    print("ERROR: Embedding count does not match label count.")

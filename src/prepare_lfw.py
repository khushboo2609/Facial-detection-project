from pathlib import Path
from sklearn.datasets import fetch_lfw_people

print("=" * 70)
print("DOWNLOADING / LOADING LFW DATASET")
print("=" * 70)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = PROJECT_ROOT / "dataset" / "lfw"

print("\nDataset location:")
print(DATA_HOME)

lfw = fetch_lfw_people(
    data_home=str(DATA_HOME),
    min_faces_per_person=20,
    resize=None,
    color=True
)

print("\nLFW DATASET LOADED SUCCESSFULLY")

print("\nTotal images:", len(lfw.images))
print("Number of identities:", len(lfw.target_names))
print("Image shape:", lfw.images.shape)
print("Target shape:", lfw.target.shape)

print("\nFirst 20 identities:")
for i, name in enumerate(lfw.target_names[:20]):
    print(f"{i}: {name}")

print("\n" + "=" * 70)
print("LFW DATASET TEST COMPLETED")
print("=" * 70)
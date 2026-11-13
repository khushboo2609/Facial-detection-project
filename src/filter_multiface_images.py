
from pathlib import Path
import csv
import shutil

from insightface.app import FaceAnalysis


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset" / "lfw_open_set"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "lfw_open_set_single_face"

REPORT_DIR = PROJECT_ROOT / "results"

REPORT_FILE = REPORT_DIR / "multiface_filter_report.csv"

SPLITS = ["gallery", "known_test", "unknown_test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# INSIGHTFACE MODEL
# ============================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# MAIN FILTERING FUNCTION
# ============================================================

def main():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {SOURCE_DIR}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_rows = []

    total_images = 0
    kept_images = 0
    excluded_images = 0

    for split in SPLITS:

        split_dir = SOURCE_DIR / split

        if not split_dir.exists():
            print(f"WARNING: Missing split: {split_dir}")
            continue

        print("\n" + "=" * 70)
        print(f"PROCESSING: {split}")
        print("=" * 70)

        for image_path in sorted(split_dir.rglob("*")):

            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            total_images += 1
            relative_path = image_path.relative_to(split_dir)

            output_path = OUTPUT_DIR / split / relative_path

            try:
                import cv2

                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError("Image could not be read")

                faces = app.get(image)
                face_count = len(faces)

                scores = [
                    round(float(face.det_score), 4)
                    for face in faces
                ]

                if face_count == 1:
                    output_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    shutil.copy2(image_path, output_path)

                    decision = "KEEP"
                    kept_images += 1

                else:
                    decision = "EXCLUDE"
                    excluded_images += 1

                reason = (
                    "Exactly one face detected"
                    if face_count == 1
                    else f"{face_count} faces detected"
                )

                print(
                    f"[{decision}] "
                    f"{relative_path} | faces={face_count}"
                )

            except Exception as error:

                face_count = -1
                scores = []
                decision = "ERROR"
                reason = str(error)
                excluded_images += 1

                print(f"[ERROR] {image_path}: {error}")

            report_rows.append({
                "split": split,
                "image": str(image_path),
                "relative_path": str(relative_path),
                "face_count": face_count,
                "detection_scores": scores,
                "decision": decision,
                "reason": reason,
            })

    # Save report
    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "split",
            "image",
            "relative_path",
            "face_count",
            "detection_scores",
            "decision",
            "reason",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 70)
    print("FILTERING COMPLETED")
    print("=" * 70)

    print(f"Total images scanned : {total_images}")
    print(f"Images kept          : {kept_images}")
    print(f"Images excluded      : {excluded_images}")
    print(f"Filtered dataset     : {OUTPUT_DIR}")
    print(f"CSV report           : {REPORT_FILE}")


if __name__ == "__main__":
    main()

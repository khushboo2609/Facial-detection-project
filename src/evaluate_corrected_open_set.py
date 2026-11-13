
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS — preserve all previous experiment results
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_RESULTS = (
    ROOT / "results" / "filtered_open_set_recognition_results.csv"
)

OUTPUT_RESULTS = (
    ROOT / "results" / "corrected_open_set_recognition_results.csv"
)

OUTPUT_THRESHOLDS = (
    ROOT / "results" / "corrected_open_set_threshold_analysis.csv"
)

# ============================================================
# LOAD AND VALIDATE PREVIOUS RESULTS
# ============================================================

if not INPUT_RESULTS.is_file():
    raise FileNotFoundError(
        f"Previous evaluation results not found: {INPUT_RESULTS}"
    )

df = pd.read_csv(INPUT_RESULTS)

required = {
    "actual_identity",
    "identity_enrolled",
    "predicted_identity",
    "similarity",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def normalize_identity(value):
    if pd.isna(value):
        return ""

    return str(value).strip().replace(" ", "_").lower()


df["identity_enrolled"] = df["identity_enrolled"].map(parse_bool)

# A prediction is valid only if it has a predicted identity
# and a usable similarity score.
df["similarity"] = pd.to_numeric(
    df["similarity"], errors="coerce"
)

if "record_key" in df.columns:
    df = df.drop_duplicates(
        subset=["record_key"], keep="last"
    )

valid = (
    df["similarity"].notna()
    & df["predicted_identity"].notna()
    & df["actual_identity"].notna()
)

evaluated = df.loc[valid].copy()

# ============================================================
# CORRECT THE SPLIT LABELS
# ============================================================

# Enrolled identities are known.
# Identities absent from the gallery are unknown, regardless
# of which original folder contained their images.
evaluated["split"] = np.where(
    evaluated["identity_enrolled"],
    "known_test",
    "unknown_test",
)

evaluated["identity_correct"] = (
    evaluated["actual_identity"].map(normalize_identity)
    == evaluated["predicted_identity"].map(normalize_identity)
)

known = evaluated[evaluated["identity_enrolled"]].copy()
unknown = evaluated[~evaluated["identity_enrolled"]].copy()

if known.empty or unknown.empty:
    raise ValueError(
        "Known or unknown evaluation set is empty. "
        "Check identity_enrolled values before continuing."
    )

# Save corrected predictions separately.
OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
evaluated.to_csv(OUTPUT_RESULTS, index=False)

# ============================================================
# THRESHOLD SWEEP
# ============================================================

threshold_rows = []

for threshold in np.round(np.arange(0.30, 0.801, 0.01), 2):
    known_accepted = known["similarity"] >= threshold
    unknown_accepted = unknown["similarity"] >= threshold

    correct_known = (
        known_accepted & known["identity_correct"]
    )

    wrong_known = (
        known_accepted & ~known["identity_correct"]
    )

    unknown_rejected = ~unknown_accepted

    threshold_rows.append({
        "threshold": round(float(threshold), 2),

        "correct_known_identification_rate_pct":
            100 * correct_known.sum() / len(known),

        "wrong_identity_accept_rate_pct":
            100 * wrong_known.sum() / len(known),

        "unknown_false_accept_rate_pct":
            100 * unknown_accepted.sum() / len(unknown),

        "unknown_rejection_rate_pct":
            100 * unknown_rejected.sum() / len(unknown),

        "known_images_evaluated": len(known),
        "unknown_images_evaluated": len(unknown),
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_THRESHOLDS, index=False)

# ============================================================
# SUMMARY
# ============================================================

print("\nCORRECTED OPEN-SET EVALUATION SUMMARY")
print("=" * 55)
print(f"Input prediction rows: {len(df)}")
print(f"Successfully evaluated rows: {len(evaluated)}")
print(f"Rows excluded due to missing values: {len(df) - len(evaluated)}")
print(f"Enrolled known images: {len(known)}")
print(f"Unenrolled unknown images: {len(unknown)}")
print(
    "Known top-1 accuracy before threshold: "
    f"{100 * known['identity_correct'].mean():.4f}%"
)

print("\nThresholds with unknown false acceptance <= 1%:")
columns = [
    "threshold",
    "correct_known_identification_rate_pct",
    "wrong_identity_accept_rate_pct",
    "unknown_false_accept_rate_pct",
    "unknown_rejection_rate_pct",
]

selected = threshold_df[
    threshold_df["unknown_false_accept_rate_pct"] <= 1.0
]

print(selected[columns].to_string(index=False))

print("\nSaved corrected predictions:")
print(OUTPUT_RESULTS)

print("\nSaved threshold analysis:")
print(OUTPUT_THRESHOLDS)

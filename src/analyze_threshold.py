from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "lfw_recognition_results.csv"

print("=" * 70)
print("ARCFACE THRESHOLD ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# Load results
# --------------------------------------------------

df = pd.read_csv(RESULTS_FILE)

# Only use successfully processed images
df = df[df["face_detected"] == True].copy()

df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(subset=["cosine_similarity"])

# Correct/incorrect based on actual recognition result
df["genuine"] = df["result"] == "CORRECT"

print("\nResults loaded:", len(df))
print("Correct:", df["genuine"].sum())
print("Incorrect:", (~df["genuine"]).sum())

# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

thresholds = np.arange(0.40, 0.951, 0.01)

results = []

for threshold in thresholds:

    # If similarity >= threshold, accept the predicted identity
    accepted = df["cosine_similarity"] >= threshold

    true_positive = ((accepted) & (df["genuine"])).sum()
    false_negative = ((~accepted) & (df["genuine"])).sum()

    false_positive = ((accepted) & (~df["genuine"])).sum()
    true_negative = ((~accepted) & (~df["genuine"])).sum()

    total = len(df)

    accuracy = (true_positive + true_negative) / total

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results.append({
        "threshold": round(threshold, 2),
        "TP": true_positive,
        "FN": false_negative,
        "FP": false_positive,
        "TN": true_negative,
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    })

threshold_df = pd.DataFrame(results)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    threshold_df[
        [
            "threshold",
            "TP",
            "FN",
            "FP",
            "TN",
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ]
    ].to_string(
        index=False,
        formatters={
            "accuracy": "{:.2f}".format,
            "precision": "{:.2f}".format,
            "recall": "{:.2f}".format,
            "f1_score": "{:.2f}".format
        }
    )
)

# --------------------------------------------------
# Best thresholds
# --------------------------------------------------

best_f1 = threshold_df.loc[
    threshold_df["f1_score"].idxmax()
]

best_accuracy = threshold_df.loc[
    threshold_df["accuracy"].idxmax()
]

print("\n" + "=" * 70)
print("BEST THRESHOLD BY F1-SCORE")
print("=" * 70)

print(
    f"Threshold : {best_f1['threshold']:.2f}\n"
    f"Accuracy  : {best_f1['accuracy']:.2f}%\n"
    f"Precision : {best_f1['precision']:.2f}%\n"
    f"Recall    : {best_f1['recall']:.2f}%\n"
    f"F1-Score  : {best_f1['f1_score']:.2f}%"
)

print("\n" + "=" * 70)
print("BEST THRESHOLD BY ACCURACY")
print("=" * 70)

print(
    f"Threshold : {best_accuracy['threshold']:.2f}\n"
    f"Accuracy  : {best_accuracy['accuracy']:.2f}%\n"
    f"Precision : {best_accuracy['precision']:.2f}%\n"
    f"Recall    : {best_accuracy['recall']:.2f}%\n"
    f"F1-Score  : {best_accuracy['f1_score']:.2f}%"
)

# --------------------------------------------------
# Save analysis
# --------------------------------------------------

output_file = PROJECT_ROOT / "results" / "threshold_analysis.csv"

threshold_df.to_csv(
    output_file,
    index=False
)

print("\nThreshold analysis saved to:")
print(output_file)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)
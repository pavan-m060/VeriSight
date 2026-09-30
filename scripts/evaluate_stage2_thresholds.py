import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/stage2/stage2_lightgbm.txt"
TEST_PATH = "data/phase2/processed_behavior_v2/test_behavior.csv"

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 THRESHOLD ANALYSIS")
print("=" * 70)

print("\nLoading model...")
model = lgb.Booster(model_file=MODEL_PATH)

print("Loading test dataset...")
df = pd.read_csv(TEST_PATH)

print(f"Test samples: {len(df):,}")

# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "spam"

DROP_COLS = [
    TARGET,
    "user_id",
    "prod_id",
    "date"
]

X = df.drop(columns=DROP_COLS, errors="ignore")
y = df[TARGET].astype(int)

print(f"Features: {X.shape[1]}")
print(f"Spam samples: {y.sum():,}")
print(f"Genuine samples: {(y == 0).sum():,}")

# ============================================================
# PREDICT PROBABILITIES
# ============================================================

print("\nGenerating spam probabilities...")

spam_prob = model.predict(X)

print(
    f"Probability range: "
    f"{spam_prob.min():.4f} - {spam_prob.max():.4f}"
)

# ============================================================
# GLOBAL METRICS
# ============================================================

roc_auc = roc_auc_score(y, spam_prob)
pr_auc = average_precision_score(y, spam_prob)

print("\n" + "=" * 70)
print("PROBABILITY-BASED PERFORMANCE")
print("=" * 70)

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

results = []

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>11} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'Pred Spam':>11}"
)

print("-" * 70)

for threshold in thresholds:

    pred = (spam_prob >= threshold).astype(int)

    accuracy = accuracy_score(y, pred)

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    predicted_spam = pred.sum()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_spam": predicted_spam
    })

    print(
        f"{threshold:10.2f} "
        f"{accuracy:10.4f} "
        f"{precision:11.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f} "
        f"{predicted_spam:11,}"
    )

# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

results_df = pd.DataFrame(results)

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

print(f"Threshold : {best_f1['threshold']:.2f}")
print(f"Accuracy  : {best_f1['accuracy']:.4f}")
print(f"Precision : {best_f1['precision']:.4f}")
print(f"Recall    : {best_f1['recall']:.4f}")
print(f"F1        : {best_f1['f1']:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

best_threshold = best_f1["threshold"]

best_pred = (
    spam_prob >= best_threshold
).astype(int)

cm = confusion_matrix(y, best_pred)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives : {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives : {tp:,}")

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results/stage2", exist_ok=True)

output_path = (
    "results/stage2/"
    "stage2_threshold_analysis.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

summary = {
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_f1_threshold": float(best_threshold),
    "best_accuracy": float(best_f1["accuracy"]),
    "best_precision": float(best_f1["precision"]),
    "best_recall": float(best_f1["recall"]),
    "best_f1": float(best_f1["f1"]),
    "confusion_matrix": cm.tolist()
}

with open(
    "results/stage2/stage2_threshold_summary.json",
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSaved:")
print(output_path)
print("results/stage2/stage2_threshold_summary.json")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
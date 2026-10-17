import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from lightgbm import LGBMClassifier
import joblib


# ============================================================
# PATHS
# ============================================================

INPUT = (
    "data/phase2/page_features/"
    "stage2_page_features.csv"
)

MODEL_DIR = Path("models/stage2")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "rating",

    "word_count",
    "char_count",
    "avg_word_length",

    "uppercase_ratio",

    "punctuation_count",
    "exclamation_count",
    "question_count",

    "digit_ratio",

    "sentiment_score",

    "rating_sentiment_difference",

    "review_date_ordinal",

    "reviews_same_date",

    "product_review_count",

    "product_five_star_ratio",

    "product_one_star_ratio",

    "duplicate_text_count",

    "is_duplicate_text"
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("VERISIGHT STAGE 2 - PAGE AVAILABLE MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# DATA
# ============================================================

X = df[FEATURES].copy()

y = df["spam"].astype(int)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


print(
    "\nFeature count:",
    len(FEATURES)
)

print(
    "Spam distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

print(
    "\nCreating train/validation/test split..."
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(
    f"Train: {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test: {len(X_test):,}"
)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining LightGBM..."
)

model = LGBMClassifier(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=8,

    num_leaves=50,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary",

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

val_prob = model.predict_proba(
    X_val_scaled
)[:, 1]


# Find best F1 threshold

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0.50
best_f1 = 0.0

for threshold in thresholds:

    pred = (
        val_prob >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print(
    "\nBest validation threshold:",
    round(best_threshold, 2)
)

print(
    "Validation F1:",
    round(best_f1, 4)
)


# ============================================================
# TEST
# ============================================================

test_prob = model.predict_proba(
    X_test_scaled
)[:, 1]


test_pred = (
    test_prob >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)

cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("STAGE 2 RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE
# ============================================================

model_path = (
    MODEL_DIR /
    "stage2_page_model.pkl"
)

scaler_path = (
    MODEL_DIR /
    "stage2_page_scaler.pkl"
)

features_path = (
    MODEL_DIR /
    "stage2_page_features.txt"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)


with open(
    features_path,
    "w"
) as f:

    for feature in FEATURES:

        f.write(
            feature + "\n"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    "results/stage2/"
    "stage2_page_model_results.txt"
)

Path(
    "results/stage2"
).mkdir(
    parents=True,
    exist_ok=True
)


with open(
    results_path,
    "w"
) as f:

    f.write(
        "VeriSight Stage 2 - "
        "Page Available Model\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1: {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.4f}\n"
    )

    f.write(
        f"Threshold: {best_threshold:.2f}\n"
    )

    f.write(
        "\nFeatures:\n"
    )

    for feature in FEATURES:

        f.write(
            f"- {feature}\n"
        )


print()
print(
    "Model saved:",
    model_path
)

print(
    "Scaler saved:",
    scaler_path
)

print(
    "Results saved:",
    results_path
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
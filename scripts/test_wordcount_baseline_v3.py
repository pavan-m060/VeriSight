import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_stage1_v3.npy"
)

Y_FILE = (
    PROJECT_ROOT /
    "features" /
    "y_stylometry_stage1_v3.npy"
)


print("=" * 70)
print("VERISIGHT - WORD COUNT ONLY BASELINE")
print("=" * 70)


# ==========================================================
# LOAD
# ==========================================================

X_full = np.load(X_FILE)
y = np.load(Y_FILE)


# Feature 0 = word_count
X = X_full[:, [0]]


print("\nFeature shape:", X.shape)
print("Labels:", y.shape)


# ==========================================================
# SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# SCALE
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# MODEL
# ==========================================================

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    random_state=42,
    max_iter=300
)


print("\nTraining word-count-only classifier...")

model.fit(
    X_train,
    y_train
)


# ==========================================================
# TEST
# ==========================================================

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nWord-count-only accuracy: "
    f"{accuracy:.2%}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Human",
            "AI"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

print("\n" + "=" * 70)
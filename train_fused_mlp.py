# ==========================================================
# VeriSight - Fusion MLP
# MiniLM + Stylometric Features
# ==========================================================

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


# ==========================================================
# Paths
# ==========================================================

X_FILE = "embeddings/X_fused_stage1.npy"
Y_FILE = "embeddings/y_fused_stage1.npy"

MODEL_DIR = "models"
RESULT_DIR = "results/fusion"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "mlp_fusion_stage1.keras"
)

REPORT_FILE = os.path.join(
    RESULT_DIR,
    "mlp_fusion_report.txt"
)


# ==========================================================
# Load Data
# ==========================================================

print("=" * 70)
print("VeriSight - Fusion MLP Training")
print("=" * 70)

print("\nLoading fused features...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("\nFeature Shape:")
print(X.shape)

print("\nLabel Shape:")
print(y.shape)

print("\nClass Distribution:")
print(
    np.bincount(y)
)


# ==========================================================
# Train / Validation / Test Split
# ==========================================================

print("\n" + "=" * 70)
print("DATA SPLITTING")
print("=" * 70)

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

print("\nTraining:")
print(X_train.shape)
print(np.bincount(y_train))

print("\nValidation:")
print(X_val.shape)
print(np.bincount(y_val))

print("\nTesting:")
print(X_test.shape)
print(np.bincount(y_test))


# ==========================================================
# Class Weights
# ==========================================================

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print(class_weights)


# ==========================================================
# Model
# ==========================================================

print("\n" + "=" * 70)
print("BUILDING FUSION MLP")
print("=" * 70)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(X.shape[1],)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================================
# Compile
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ==========================================================
# Callbacks
# ==========================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True
    )
]


# ==========================================================
# Training
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING FUSION MLP")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=64,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ==========================================================
# Evaluation
# ==========================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ==========================================================
# Predictions
# ==========================================================

probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(

    y_test,

    predictions,

    target_names=[
        "Human",
        "AI"
    ],

    digits=4,

    zero_division=0
)

print("\nClassification Report:")
print(report)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:")
print(cm)


# ==========================================================
# Save Report
# ==========================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VeriSight - Fusion MLP\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Features:\n"
    )

    f.write(
        "384 MiniLM + 18 Stylometric = 402\n\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Accuracy (%): "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(report)

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


# ==========================================================
# Save Training History
# ==========================================================

history_file = os.path.join(
    RESULT_DIR,
    "fusion_training_history.csv"
)

import pandas as pd

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    history_file,
    index=False
)


# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("FUSION MLP TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print(MODEL_FILE)

print("\nReport:")
print(REPORT_FILE)

print("\nHistory:")
print(history_file)
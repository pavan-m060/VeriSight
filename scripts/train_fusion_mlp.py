# ==========================================================
# VeriSight
# Stage 1 - MiniLM + Stylometry Fusion MLP
# ==========================================================

import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ==========================================================
# Paths
# ==========================================================

EMBED_DIR = "embeddings/stage1_v2"

STYLO_FILE = "features/stylometry_stage1_v2.npy"
Y_STYLO_FILE = "features/y_stylometry_stage1_v2.npy"

MODEL_FILE = "models/fusion_mlp_stage1_v2.keras"
SCALER_FILE = "features/stylometry_scaler_fusion_stage1_v2.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("features", exist_ok=True)

# ==========================================================
# Load Existing MiniLM Split
# ==========================================================

print("=" * 70)
print("VeriSight - MiniLM + Stylometry Fusion MLP")
print("=" * 70)

print("\nLoading existing MiniLM split...")

X_train_semantic = np.load(
    os.path.join(EMBED_DIR, "X_train.npy")
)

X_val_semantic = np.load(
    os.path.join(EMBED_DIR, "X_val.npy")
)

X_test_semantic = np.load(
    os.path.join(EMBED_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(EMBED_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(EMBED_DIR, "y_val.npy")
)

y_test = np.load(
    os.path.join(EMBED_DIR, "y_test.npy")
)

print("\nMiniLM shapes:")
print("Train:", X_train_semantic.shape)
print("Val  :", X_val_semantic.shape)
print("Test :", X_test_semantic.shape)

# ==========================================================
# Load Stylometric Features
# ==========================================================

print("\nLoading stylometric features...")

X_stylo = np.load(STYLO_FILE)
y_stylo = np.load(Y_STYLO_FILE)

print("Stylometry:", X_stylo.shape)
print("Labels    :", y_stylo.shape)

# ==========================================================
# Verify Complete Dataset Labels
# ==========================================================

if not np.array_equal(
    y_stylo,
    np.load("embeddings/y_stage1_v2.npy")
):
    raise ValueError(
        "Stylometric labels and original Stage 1 v2 "
        "labels are different!"
    )

print("\nComplete dataset label verification: PASSED")

# ==========================================================
# Recreate EXACT same split using indices
# ==========================================================

print("\nRecreating existing Stage 1 v2 split...")

indices = np.arange(
    len(y_stylo)
)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y_stylo
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y_stylo[temp_idx]
)

# ==========================================================
# Extract Stylometric Splits
# ==========================================================

X_train_stylo = X_stylo[train_idx]
X_val_stylo = X_stylo[val_idx]
X_test_stylo = X_stylo[test_idx]

y_train_stylo = y_stylo[train_idx]
y_val_stylo = y_stylo[val_idx]
y_test_stylo = y_stylo[test_idx]

print("\nStylometry split:")
print("Train:", X_train_stylo.shape)
print("Val  :", X_val_stylo.shape)
print("Test :", X_test_stylo.shape)

# ==========================================================
# Verify Labels
# ==========================================================

print("\nChecking label alignment...")

if not np.array_equal(
    y_train,
    y_train_stylo
):
    raise ValueError(
        "Training labels do not match!"
    )

if not np.array_equal(
    y_val,
    y_val_stylo
):
    raise ValueError(
        "Validation labels do not match!"
    )

if not np.array_equal(
    y_test,
    y_test_stylo
):
    raise ValueError(
        "Test labels do not match!"
    )

print("Train labels: PASSED")
print("Validation labels: PASSED")
print("Test labels: PASSED")

# ==========================================================
# Scale Stylometric Features
# ==========================================================

print("\nScaling stylometric features...")

scaler = StandardScaler()

X_train_stylo = scaler.fit_transform(
    X_train_stylo
)

X_val_stylo = scaler.transform(
    X_val_stylo
)

X_test_stylo = scaler.transform(
    X_test_stylo
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    "Scaler saved:",
    SCALER_FILE
)

# ==========================================================
# Fuse MiniLM + Stylometry
# ==========================================================

print("\nCreating fused features...")

X_train = np.concatenate(
    [
        X_train_semantic,
        X_train_stylo
    ],
    axis=1
)

X_val = np.concatenate(
    [
        X_val_semantic,
        X_val_stylo
    ],
    axis=1
)

X_test = np.concatenate(
    [
        X_test_semantic,
        X_test_stylo
    ],
    axis=1
)

print("\nFused feature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ==========================================================
# Build Fusion MLP
# ==========================================================

print("\nBuilding fusion MLP...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(408,)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
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
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ==========================================================
# Training
# ==========================================================

print("\n" + "=" * 70)
print("Training Fusion Model")
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

    callbacks=callbacks,

    verbose=1
)

# ==========================================================
# Load Best Model
# ==========================================================

model = tf.keras.models.load_model(
    MODEL_FILE
)

# ==========================================================
# Test
# ==========================================================

print("\n" + "=" * 70)
print("Fusion Model Test Results")
print("=" * 70)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

# ==========================================================
# Predictions
# ==========================================================

probabilities = model.predict(
    X_test,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        digits=4
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)

# ==========================================================
# Save Fused Features
# ==========================================================

np.save(
    "features/X_fusion_train.npy",
    X_train
)

np.save(
    "features/X_fusion_val.npy",
    X_val
)

np.save(
    "features/X_fusion_test.npy",
    X_test
)

print("\nFused features saved.")

# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("Fusion training completed")
print("=" * 70)

print("\nModel:")
print(MODEL_FILE)

print("\nTest Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("=" * 70)
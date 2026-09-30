# ==========================================================
# VeriSight
# Stage 1 - Stylometry Only MLP
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Paths
# ==========================================================

X_FILE = "features/stylometry_stage1_v3.npy"
Y_FILE = "features/y_stylometry_stage1_v3.npy"

MODEL_FILE = "models/stylometry_mlp_stage1_v2.keras"

os.makedirs("models", exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

print("=" * 70)
print("VeriSight - Stylometry Only MLP")
print("=" * 70)

print("\nLoading stylometric features...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================================
# Train / Validation / Test Split
# Same 80/10/10 structure as Stage 1 v2
# ==========================================================

print("\nCreating train/validation/test split...")

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

print("\nSplit sizes:")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)

print("\nTrain labels:", np.bincount(y_train))
print("Val labels  :", np.bincount(y_val))
print("Test labels :", np.bincount(y_test))

# ==========================================================
# Standardization
# IMPORTANT:
# Fit scaler ONLY on training data
# ==========================================================

print("\nStandardizing stylometric features...")



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "features/stylometry_scaler_stage1_v2.pkl"
)
# ==========================================================
# Build MLP
# ==========================================================

print("\nBuilding MLP...")

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(
        shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
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
# Train
# ==========================================================

print("\n" + "=" * 70)
print("Training")
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
print("Test Results")
print("=" * 70)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

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

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:")
print(cm)

# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("Stylometry-only training completed")
print("=" * 70)

print("\nModel saved to:")
print(MODEL_FILE)

print("\nFinal Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("=" * 70)
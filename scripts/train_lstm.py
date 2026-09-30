# ==========================================================
# VeriSight - Stage 1
# LSTM Model for Human vs AI Review Detection
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================================
# Paths
# ==========================================================

X_PATH = "embeddings/X_stage1.npy"
Y_PATH = "embeddings/y_stage1.npy"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

print("=" * 60)
print("Loading Embeddings...")
print("=" * 60)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Embedding Shape :", X.shape)
print("Labels Shape    :", y.shape)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

print("LSTM Input Shape :", X.shape)

# ==========================================================
# Train / Validation / Test Split
# ==========================================================

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

print("\nTraining :", X_train.shape)
print("Validation:", X_val.shape)
print("Testing   :", X_test.shape)

# ==========================================================
# LSTM Model
# ==========================================================

model = Sequential([
    Input(shape=(384,1)),

    LSTM(128),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "lstm_stage1.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================================================
# Train
# ==========================================================

print("\nTraining LSTM...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==========================================================
# Evaluate
# ==========================================================

print("\nEvaluating...\n")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")

# ==========================================================
# Predictions
# ==========================================================

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=["Human", "AI"]
)

print(report)

with open(os.path.join(RESULT_DIR, "lstm_report.txt"), "w") as f:
    f.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Human", "AI"]
)

disp.plot(cmap="Blues")
plt.title("LSTM Confusion Matrix")

plt.savefig(
    os.path.join(RESULT_DIR, "lstm_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Accuracy Plot
# ==========================================================

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    os.path.join(RESULT_DIR, "lstm_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Loss Plot
# ==========================================================

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    os.path.join(RESULT_DIR, "lstm_loss.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Finished
# ==========================================================

print("\n" + "="*60)
print("LSTM Training Completed")
print("="*60)

print(f"Model Saved : {os.path.join(MODEL_DIR,'lstm_stage1.keras')}")

print("\nResults Saved:")
print("lstm_accuracy.png")
print("lstm_loss.png")
print("lstm_confusion_matrix.png")
print("lstm_report.txt")
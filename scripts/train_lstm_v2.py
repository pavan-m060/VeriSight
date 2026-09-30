# ==========================================================
# VeriSight - Stage 1 v2
# LSTM Model for Human vs AI Review Detection
# ==========================================================

import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "embeddings" / "stage1_v2"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results" / "stage1_v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lstm_stage1_v2.keras"
REPORT_FILE = RESULT_DIR / "lstm_stage1_v2_report.txt"


# ==========================================================
# Load Data
# ==========================================================

print("=" * 65)
print("VeriSight - LSTM Stage 1 v2")
print("=" * 65)

print("\nLoading Stage 1 v2 data...")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("\nOriginal Shapes:")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# ==========================================================
# Reshape for LSTM
# ==========================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nLSTM Input Shapes:")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# ==========================================================
# LSTM Model
# ==========================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(384, 1)
    ),

    tf.keras.layers.LSTM(
        128
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================================
# Compile
# ==========================================================

model.compile(
    optimizer="adam",
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
        monitor="val_loss",
        save_best_only=True
    )
]


# ==========================================================
# Training
# ==========================================================

print("\n" + "=" * 65)
print("Training LSTM...")
print("=" * 65)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# ==========================================================
# Evaluation
# ==========================================================

print("\n" + "=" * 65)
print("Evaluating LSTM...")
print("=" * 65)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")


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
    target_names=["Human", "AI"],
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

    f.write("VeriSight - LSTM Stage 1 v2\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Accuracy : {test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Loss     : {test_loss:.4f}\n\n"
    )

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))


# ==========================================================
# Complete
# ==========================================================

print("\n" + "=" * 65)
print("LSTM v2 Training Completed")
print("=" * 65)

print("\nModel:")
print(MODEL_FILE)

print("\nReport:")
print(REPORT_FILE)
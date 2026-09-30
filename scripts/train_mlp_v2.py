import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==========================================================
# Paths
# ==========================================================

DATA_DIR = Path("embeddings/stage1_v2")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/stage1_v2")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "mlp_stage1_v2.keras"

# ==========================================================
# Load Data
# ==========================================================

print("=" * 60)
print("Loading Stage 1 Data")
print("=" * 60)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)

# ==========================================================
# Class Weights
# ==========================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:")
print(class_weights)

# ==========================================================
# Model
# ==========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(384,)),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

print("\nTraining MLP...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# Evaluation
# ==========================================================

print("\nEvaluating...")

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
    batch_size=64
).ravel()

predictions = (probabilities >= 0.5).astype(int)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_test,
    predictions,
    target_names=["Human", "AI"],
    digits=4
)

print("\n" + report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(cm)

# ==========================================================
# Save Report
# ==========================================================

with open(
    RESULT_DIR / "mlp_stage1_v2_report.txt",
    "w"
) as f:

    f.write("VeriSight Stage 1 - MLP\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")

    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\n" + "=" * 60)
print("MLP Training Completed")
print("=" * 60)

print("Model:")
print(MODEL_FILE)

print("\nReport:")
print(RESULT_DIR / "mlp_stage1_v2_report.txt")
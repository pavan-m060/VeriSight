import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# VERISIGHT
# STAGE 1 - SOURCE + LENGTH CONTROLLED MLP
#
# Complete pipeline:
# CSV
#   ↓
# MiniLM embeddings
#   ↓
# Train / Validation / Test split
#   ↓
# StandardScaler
#   ↓
# MLP
#   ↓
# Internal evaluation
#   ↓
# Gemini external evaluation
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
STAGE1_DIR = DATA_DIR / "stage1"
EMBED_DIR = BASE_DIR / "embeddings" / "stage1_controlled"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results" / "stage1_controlled"

# Controlled training dataset
TRAIN_CSV = (
    STAGE1_DIR /
    "stage1_v3_source_length_controlled.csv"
)

# Frozen Gemini external dataset
GEMINI_CANDIDATES = [
    DATA_DIR / "external" / "gemini_external_balanced.csv",
    DATA_DIR / "external" / "gemini_external_final.csv"
]

# Output paths
X_EMBED_PATH = EMBED_DIR / "X_stage1_controlled.npy"
Y_EMBED_PATH = EMBED_DIR / "y_stage1_controlled.npy"

X_TRAIN_PATH = EMBED_DIR / "X_train.npy"
X_VAL_PATH = EMBED_DIR / "X_val.npy"
X_TEST_PATH = EMBED_DIR / "X_test.npy"

Y_TRAIN_PATH = EMBED_DIR / "y_train.npy"
Y_VAL_PATH = EMBED_DIR / "y_val.npy"
Y_TEST_PATH = EMBED_DIR / "y_test.npy"

SCALER_PATH = (
    MODEL_DIR /
    "mlp_stage1_controlled_scaler.pkl"
)

MODEL_PATH = (
    MODEL_DIR /
    "mlp_stage1_controlled.keras"
)

RESULT_PATH = (
    RESULT_DIR /
    "stage1_controlled_results.txt"
)

PREDICTIONS_PATH = (
    RESULT_DIR /
    "gemini_predictions_stage1_controlled.csv"
)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

EMBED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 75)
print("VERISIGHT - STAGE 1 CONTROLLED MLP PIPELINE")
print("=" * 75)

# ============================================================
# 1. LOAD CONTROLLED DATASET
# ============================================================

print("\n[1/8] Loading controlled dataset...")

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"\nControlled dataset not found:\n{TRAIN_CSV}\n\n"
        "Run create_source_length_controlled.py first."
    )

df = pd.read_csv(TRAIN_CSV)

print(f"Dataset shape: {df.shape}")

required_columns = [
    "review",
    "label",
    "type",
    "source",
    "generator_model"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSource distribution:")
print(df["source"].value_counts())

print("\nGenerator distribution:")
print(df["generator_model"].value_counts())

# ============================================================
# 2. LOAD MINI-LM
# ============================================================

print("\n[2/8] Loading MiniLM...")

print(
    "Model: sentence-transformers/all-MiniLM-L6-v2"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# 3. GENERATE EMBEDDINGS
# ============================================================

print("\n[3/8] Generating MiniLM embeddings...")

if X_EMBED_PATH.exists() and Y_EMBED_PATH.exists():

    print(
        "Existing embeddings found."
    )

    X = np.load(X_EMBED_PATH)
    y = np.load(Y_EMBED_PATH)

    print(
        f"Loaded embeddings: {X.shape}"
    )

else:

    texts = (
        df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Encoding {len(texts)} reviews..."
    )

    X = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    y = df["label"].astype(np.int32).values

    np.save(
        X_EMBED_PATH,
        X
    )

    np.save(
        Y_EMBED_PATH,
        y
    )

    print(
        "Embeddings saved."
    )

print(
    f"\nEmbedding shape: {X.shape}"
)

print(
    f"Label shape: {y.shape}"
)

# ============================================================
# SAFETY CHECK
# ============================================================

if X.shape[0] != len(df):

    raise ValueError(
        "Embedding count does not match dataset rows."
    )

if X.shape[1] != 384:

    raise ValueError(
        f"Expected MiniLM dimension 384, "
        f"got {X.shape[1]}"
    )

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[4/8] Creating stratified splits...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(
    f"\nTrain: {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test: {X_test.shape}"
)

print(
    "\nTrain labels:",
    np.bincount(y_train)
)

print(
    "Validation labels:",
    np.bincount(y_val)
)

print(
    "Test labels:",
    np.bincount(y_test)
)

# Save splits
np.save(
    X_TRAIN_PATH,
    X_train
)

np.save(
    X_VAL_PATH,
    X_val
)

np.save(
    X_TEST_PATH,
    X_test
)

np.save(
    Y_TRAIN_PATH,
    y_train
)

np.save(
    Y_VAL_PATH,
    y_val
)

np.save(
    Y_TEST_PATH,
    y_test
)

# ============================================================
# 5. STANDARDIZATION
# ============================================================

print("\n[5/8] Standardizing embeddings...")

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

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"Scaler saved: {SCALER_PATH}"
)

# ============================================================
# 6. BUILD MLP
# ============================================================

print("\n[6/8] Building MLP...")

# Same basic architecture family used for
# the previous MiniLM MLP experiments.

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(
        shape=(384,)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
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

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting MLP training...")

history = model.fit(

    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=50,

    batch_size=64,

    callbacks=callbacks,

    verbose=1
)

# Save model
model.save(
    MODEL_PATH
)

print(
    f"\nModel saved:\n{MODEL_PATH}"
)

# ============================================================
# 7. INTERNAL TEST EVALUATION
# ============================================================

print("\n[7/8] Evaluating internal test set...")

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

test_probs = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

test_preds = (
    test_probs >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_preds
)

cm = confusion_matrix(
    y_test,
    test_preds
)

report = classification_report(
    y_test,
    test_preds,
    target_names=[
        "Human",
        "AI"
    ],
    digits=4
)

print(
    "\nInternal Test Accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ============================================================
# GEMINI EXTERNAL EVALUATION
# ============================================================

print("\nEvaluating frozen Gemini dataset...")

gemini_path = None

for candidate in GEMINI_CANDIDATES:

    if candidate.exists():

        gemini_path = candidate
        break

if gemini_path is None:

    print(
        "\nWARNING: Gemini dataset not found."
    )

    print(
        "Expected one of:"
    )

    for candidate in GEMINI_CANDIDATES:
        print(candidate)

    gemini_accuracy = None
    gemini_cm = None

else:

    print(
        f"\nGemini dataset: {gemini_path}"
    )

    gemini_df = pd.read_csv(
        gemini_path
    )

    print(
        f"Gemini rows: {len(gemini_df)}"
    )

    # --------------------------------------------------------
    # Generate Gemini embeddings
    # --------------------------------------------------------

    gemini_texts = (
        gemini_df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    X_gemini = embedding_model.encode(

        gemini_texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=False
    )

    # --------------------------------------------------------
    # Scale using TRAINING scaler only
    # --------------------------------------------------------

    X_gemini_scaled = scaler.transform(
        X_gemini
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    gemini_probs = model.predict(
        X_gemini_scaled,
        verbose=0
    ).ravel()

    gemini_preds = (
        gemini_probs >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Determine true labels
    # --------------------------------------------------------

    if "label" in gemini_df.columns:

        gemini_y = (
            gemini_df["label"]
            .astype(int)
            .values
        )

    else:

        # Gemini external dataset is expected
        # to contain AI-generated reviews.
        gemini_y = np.ones(
            len(gemini_df),
            dtype=int
        )

    gemini_accuracy = accuracy_score(
        gemini_y,
        gemini_preds
    )

    gemini_cm = confusion_matrix(
        gemini_y,
        gemini_preds
    )

    print(
        "\nGemini Detection Accuracy:"
    )

    print(
        f"{gemini_accuracy * 100:.2f}%"
    )

    print(
        f"Correct: "
        f"{np.sum(gemini_preds == gemini_y)}"
        f"/{len(gemini_y)}"
    )

    print("\nGemini Confusion Matrix:")
    print(gemini_cm)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    gemini_output = gemini_df.copy()

    gemini_output[
        "predicted_label"
    ] = gemini_preds

    gemini_output[
        "ai_probability"
    ] = gemini_probs

    gemini_output[
        "predicted_type"
    ] = np.where(
        gemini_preds == 1,
        "AI",
        "HUMAN"
    )

    gemini_output.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print(
        f"\nGemini predictions saved:"
    )

    print(
        PREDICTIONS_PATH
    )

# ============================================================
# 8. SAVE COMPLETE REPORT
# ============================================================

print("\n[8/8] Saving final report...")

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VERISIGHT - STAGE 1 "
        "SOURCE + LENGTH CONTROLLED MLP\n"
    )

    f.write("=" * 75 + "\n\n")

    # Dataset
    f.write("DATASET\n")
    f.write("-" * 75 + "\n")

    f.write(
        f"Dataset: {TRAIN_CSV}\n"
    )

    f.write(
        f"Shape: {df.shape}\n"
    )

    f.write(
        f"Embedding shape: {X.shape}\n\n"
    )

    # Splits
    f.write("DATA SPLITS\n")
    f.write("-" * 75 + "\n")

    f.write(
        f"Train: {X_train.shape}\n"
    )

    f.write(
        f"Validation: {X_val.shape}\n"
    )

    f.write(
        f"Test: {X_test.shape}\n\n"
    )

    # Internal results
    f.write(
        "INTERNAL TEST RESULTS\n"
    )

    f.write("-" * 75 + "\n")

    f.write(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write("\n\n")

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )

    f.write("\n")

    # Gemini
    f.write(
        "\nGEMINI EXTERNAL EVALUATION\n"
    )

    f.write("-" * 75 + "\n")

    if gemini_accuracy is not None:

        f.write(
            f"Dataset: {gemini_path}\n"
        )

        f.write(
            f"Samples: "
            f"{len(gemini_y)}\n"
        )

        f.write(
            f"Accuracy / AI detection rate: "
            f"{gemini_accuracy * 100:.4f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(gemini_cm)
        )

        f.write("\n")

    else:

        f.write(
            "Gemini dataset was not found.\n"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("PIPELINE COMPLETE")
print("=" * 75)

print(
    f"\nInternal Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

if gemini_accuracy is not None:

    print(
        f"Gemini Accuracy: "
        f"{gemini_accuracy * 100:.2f}%"
    )

print("\nFiles created:")

print(
    f"\nEmbeddings:\n{EMBED_DIR}"
)

print(
    f"\nScaler:\n{SCALER_PATH}"
)

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nResults:\n{RESULT_PATH}"
)

if gemini_accuracy is not None:

    print(
        f"\nGemini predictions:\n"
        f"{PREDICTIONS_PATH}"
    )

print("\nDone.")
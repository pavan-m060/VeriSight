import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# VERISIGHT
# STAGE 1
# TOKEN-LEVEL MINILM + BiLSTM + ATTENTION
#
# Pipeline:
#
# Review
#   ↓
# MiniLM token embeddings
#   ↓
# BiLSTM
#   ↓
# Attention
#   ↓
# Dense
#   ↓
# Human / AI
#
# Training:
#   Source + length controlled dataset
#
# External evaluation:
#   Frozen Gemini dataset
# ============================================================

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

MAX_LENGTH = 128
EMBEDDING_DIM = 384

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

TRAIN_CSV = (
    DATA_DIR
    / "stage1"
    / "stage1_v3_source_length_controlled.csv"
)

GEMINI_CANDIDATES = [

    DATA_DIR
    / "external"
    / "gemini_external_balanced.csv",

    DATA_DIR
    / "external"
    / "gemini_external_final.csv"

]

TOKEN_DIR = (
    BASE_DIR
    / "embeddings"
    / "stage1_token"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "stage1_token"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "stage1_token"
)

TOKEN_DIR.mkdir(
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
# OUTPUT FILES
# ============================================================

X_TOKEN_PATH = (
    TOKEN_DIR
    / "X_stage1_token.npy"
)

Y_TOKEN_PATH = (
    TOKEN_DIR
    / "y_stage1_token.npy"
)

X_TRAIN_PATH = (
    TOKEN_DIR
    / "X_train_token.npy"
)

X_VAL_PATH = (
    TOKEN_DIR
    / "X_val_token.npy"
)

X_TEST_PATH = (
    TOKEN_DIR
    / "X_test_token.npy"
)

Y_TRAIN_PATH = (
    TOKEN_DIR
    / "y_train_token.npy"
)

Y_VAL_PATH = (
    TOKEN_DIR
    / "y_val_token.npy"
)

Y_TEST_PATH = (
    TOKEN_DIR
    / "y_test_token.npy"
)

MODEL_PATH = (
    MODEL_DIR
    / "bilstm_attention_stage1.keras"
)

RESULT_PATH = (
    RESULT_DIR
    / "token_bilstm_attention_results.txt"
)

GEMINI_PREDICTIONS = (
    RESULT_DIR
    / "gemini_token_bilstm_predictions.csv"
)

# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 80)
print("VERISIGHT - TOKEN LEVEL MINILM + BiLSTM + ATTENTION")
print("=" * 80)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/9] Loading controlled dataset...")

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{TRAIN_CSV}"
    )

df = pd.read_csv(
    TRAIN_CSV
)

print(
    f"Dataset shape: {df.shape}"
)

print("\nColumns:")
print(
    df.columns.tolist()
)

print("\nLabel distribution:")
print(
    df["label"].value_counts()
)

# ============================================================
# 2. LOAD MINILM
# ============================================================

print("\n[2/9] Loading MiniLM...")

print(
    "Model: all-MiniLM-L6-v2"
)

sentence_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# 3. TOKEN-LEVEL EMBEDDINGS
# ============================================================

print("\n[3/9] Generating token-level MiniLM embeddings...")

print(
    f"Maximum sequence length: {MAX_LENGTH}"
)

# ------------------------------------------------------------
# IMPORTANT
#
# SentenceTransformer normally returns one pooled vector.
# We need the underlying transformer to obtain token embeddings.
# ------------------------------------------------------------

transformer = sentence_model[0].auto_model

tokenizer = sentence_model.tokenizer

# Force maximum length
tokenizer.model_max_length = MAX_LENGTH

# ------------------------------------------------------------
# Function for token embeddings
# ------------------------------------------------------------

def encode_token_embeddings(
    texts,
    batch_size=16
):

    all_embeddings = []

    transformer.eval()

    device = transformer.device

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # Move tensors to same device
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with tf.device("/CPU:0"):

            pass

        # Use PyTorch because MiniLM is a
        # PyTorch transformer internally.
        import torch

        with torch.no_grad():

            outputs = transformer(
                **encoded
            )

        # Last hidden state:
        #
        # (batch, sequence_length, 384)
        #
        hidden = (
            outputs.last_hidden_state
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_embeddings.append(
            hidden
        )

        print(
            f"\rEncoded "
            f"{min(start + batch_size, len(texts))}"
            f"/{len(texts)}",
            end=""
        )

    print()

    return np.concatenate(
        all_embeddings,
        axis=0
    )

# ------------------------------------------------------------
# Generate / load
# ------------------------------------------------------------

if (
    X_TOKEN_PATH.exists()
    and Y_TOKEN_PATH.exists()
):

    print(
        "Existing token embeddings found."
    )

    X = np.load(
        X_TOKEN_PATH
    )

    y = np.load(
        Y_TOKEN_PATH
    )

else:

    texts = (
        df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    X = encode_token_embeddings(
        texts,
        batch_size=16
    )

    y = (
        df["label"]
        .astype(np.int32)
        .values
    )

    np.save(
        X_TOKEN_PATH,
        X
    )

    np.save(
        Y_TOKEN_PATH,
        y
    )

print(
    f"\nToken embedding shape: {X.shape}"
)

print(
    f"Expected: "
    f"(samples, {MAX_LENGTH}, {EMBEDDING_DIM})"
)

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[4/9] Creating stratified split...")

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
)

print(
    f"Train: {X_train.shape}"
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

# Save split arrays
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
# 5. ATTENTION LAYER
# ============================================================

print("\n[5/9] Building attention layer...")

class AttentionLayer(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.score_dense = (
            tf.keras.layers.Dense(
                1
            )
        )

    def call(
        self,
        inputs
    ):

        # inputs:
        #
        # (batch, sequence, features)

        scores = self.score_dense(
            inputs
        )

        # Remove final dimension
        scores = tf.squeeze(
            scores,
            axis=-1
        )

        # Attention weights
        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        # Restore dimension
        weights = tf.expand_dims(
            weights,
            axis=-1
        )

        # Weighted sum
        context = tf.reduce_sum(
            inputs * weights,
            axis=1
        )

        return context

# ============================================================
# 6. BUILD BiLSTM + ATTENTION
# ============================================================

print("\n[6/9] Building BiLSTM + Attention model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            MAX_LENGTH,
            EMBEDDING_DIM
        )
    ),

    tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        )

    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        )

    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    AttentionLayer(),

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

    metrics=[
        "accuracy"
    ]

)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
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
# 7. TRAIN
# ============================================================

print("\n[7/9] Training BiLSTM + Attention...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=32,

    callbacks=callbacks,

    verbose=1

)

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved:\n{MODEL_PATH}"
)

# ============================================================
# INTERNAL TEST
# ============================================================

print("\nEvaluating internal test set...")

test_probs = model.predict(
    X_test,
    batch_size=32,
    verbose=1
).ravel()

test_preds = (
    test_probs >= 0.5
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    test_preds
)

test_cm = confusion_matrix(
    y_test,
    test_preds
)

test_report = classification_report(
    y_test,
    test_preds,
    target_names=[
        "Human",
        "AI"
    ],
    digits=4
)

print(
    f"\nInternal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nConfusion Matrix:"
)

print(
    test_cm
)

print(
    "\nClassification Report:"
)

print(
    test_report
)

# ============================================================
# 8. GEMINI EXTERNAL TEST
# ============================================================

print("\n[8/9] Evaluating frozen Gemini dataset...")

gemini_path = None

for candidate in GEMINI_CANDIDATES:

    if candidate.exists():

        gemini_path = candidate

        break

if gemini_path is None:

    print(
        "\nGemini dataset not found."
    )

    gemini_accuracy = None

else:

    gemini_df = pd.read_csv(
        gemini_path
    )

    print(
        f"Gemini dataset: "
        f"{gemini_path}"
    )

    print(
        f"Gemini samples: "
        f"{len(gemini_df)}"
    )

    gemini_texts = (
        gemini_df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "\nGenerating Gemini token embeddings..."
    )

    X_gemini = encode_token_embeddings(
        gemini_texts,
        batch_size=16
    )

    print(
        f"\nGemini embedding shape: "
        f"{X_gemini.shape}"
    )

    if "label" in gemini_df.columns:

        y_gemini = (
            gemini_df["label"]
            .astype(int)
            .values
        )

    else:

        y_gemini = np.ones(
            len(gemini_df),
            dtype=int
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    gemini_probs = model.predict(
        X_gemini,
        batch_size=32,
        verbose=1
    ).ravel()

    gemini_preds = (
        gemini_probs >= 0.5
    ).astype(int)

    gemini_accuracy = accuracy_score(
        y_gemini,
        gemini_preds
    )

    gemini_cm = confusion_matrix(
        y_gemini,
        gemini_preds
    )

    gemini_report = classification_report(
        y_gemini,
        gemini_preds,
        target_names=[
            "Human",
            "AI"
        ],
        digits=4,
        zero_division=0
    )

    print(
        "\nGemini Accuracy:"
    )

    print(
        f"{gemini_accuracy * 100:.2f}%"
    )

    print(
        f"Correct: "
        f"{np.sum(gemini_preds == y_gemini)}"
        f"/{len(y_gemini)}"
    )

    print(
        "\nGemini Confusion Matrix:"
    )

    print(
        gemini_cm
    )

    print(
        "\nGemini Classification Report:"
    )

    print(
        gemini_report
    )

    # --------------------------------------------------------
    # Save Gemini predictions
    # --------------------------------------------------------

    output_df = gemini_df.copy()

    output_df[
        "predicted_label"
    ] = gemini_preds

    output_df[
        "ai_probability"
    ] = gemini_probs

    output_df[
        "predicted_type"
    ] = np.where(
        gemini_preds == 1,
        "AI",
        "HUMAN"
    )

    output_df.to_csv(
        GEMINI_PREDICTIONS,
        index=False
    )

    print(
        f"\nGemini predictions saved:\n"
        f"{GEMINI_PREDICTIONS}"
    )

# ============================================================
# 9. SAVE REPORT
# ============================================================

print("\n[9/9] Saving final report...")

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VERISIGHT STAGE 1\n"
        "TOKEN-LEVEL MINILM + BiLSTM + ATTENTION\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        f"Training dataset:\n{TRAIN_CSV}\n\n"
    )

    f.write(
        f"Dataset shape: {df.shape}\n"
    )

    f.write(
        f"Maximum sequence length: "
        f"{MAX_LENGTH}\n"
    )

    f.write(
        f"Embedding dimension: "
        f"{EMBEDDING_DIM}\n\n"
    )

    f.write(
        "DATA SPLIT\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Train: {len(y_train)}\n"
    )

    f.write(
        f"Validation: {len(y_val)}\n"
    )

    f.write(
        f"Test: {len(y_test)}\n\n"
    )

    f.write(
        "INTERNAL TEST\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Accuracy: "
        f"{test_accuracy * 100:.4f}%\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(test_cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        test_report
    )

    if gemini_accuracy is not None:

        f.write(
            "\n\n"
            "GEMINI EXTERNAL TEST\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Dataset: {gemini_path}\n"
        )

        f.write(
            f"Samples: {len(y_gemini)}\n"
        )

        f.write(
            f"Accuracy: "
            f"{gemini_accuracy * 100:.4f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(gemini_cm)
        )

        f.write(
            "\n\nClassification Report:\n"
        )

        f.write(
            gemini_report
        )

print("\n")
print("=" * 80)
print("TOKEN-LEVEL EXPERIMENT COMPLETE")
print("=" * 80)

print(
    f"\nInternal Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

if gemini_accuracy is not None:

    print(
        f"Gemini Accuracy: "
        f"{gemini_accuracy * 100:.2f}%"
    )

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nReport:\n{RESULT_PATH}"
)

print("\nDone.")
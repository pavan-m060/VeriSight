# ============================================================
# VeriSight - Stage 2
# Fast Deep Learning Text Model
#
# Architecture:
#
# Review Text
#     ↓
# MiniLM (384-D embedding)
#     ↓
# BiLSTM
#     ↓
# Attention
#     ↓
# Dense
#     ↓
# Spam Probability
#
# FAST VERSION:
# - Uses cached MiniLM embeddings when available
# - Uses 150,000 training samples instead of 487k
# - Batch size = 1024
# - Smaller BiLSTM
# - Early stopping
#
# NOTE:
# MiniLM produces a 384-D sentence embedding.
# The BiLSTM processes this embedding as a sequence of
# 384 scalar positions. This is a practical CPU baseline,
# not token-level MiniLM processing.
# ============================================================


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "phase2"
    / "processed_hybrid"
    / "stage2_train_hybrid.csv"
)


VAL_FILE = (
    BASE_DIR
    / "data"
    / "phase2"
    / "processed_hybrid"
    / "stage2_val_hybrid.csv"
)


TEST_FILE = (
    BASE_DIR
    / "data"
    / "phase2"
    / "processed_hybrid"
    / "stage2_test_hybrid.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "stage2"
)


RESULT_DIR = (
    BASE_DIR
    / "results"
    / "stage2"
)


EMBED_DIR = (
    BASE_DIR
    / "embeddings"
    / "stage2_text"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EMBED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = (
    MODEL_DIR
    / "stage2_text_bilstm_attention.keras"
)


REPORT_FILE = (
    RESULT_DIR
    / "stage2_text_bilstm_attention_results.txt"
)


PREDICTION_FILE = (
    RESULT_DIR
    / "stage2_text_bilstm_attention_predictions.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MINILM_MODEL = "all-MiniLM-L6-v2"

EMBEDDING_DIM = 384

MINILM_BATCH_SIZE = 32

# ------------------------------------------------------------
# IMPORTANT:
# We use only 150,000 training samples for the fast baseline.
# Validation and test remain untouched.
# ------------------------------------------------------------

MAX_TRAIN = 150000

TRAIN_BATCH_SIZE = 1024

EPOCHS = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("VERISIGHT - STAGE 2")
print("FAST MiniLM + BiLSTM + ATTENTION")
print("=" * 70)
print()


# ============================================================
# CHECK FILES
# ============================================================

print("Checking datasets...")
print()


for file_path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}"
        )

    print(
        "FOUND:",
        file_path
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)
print()


train_df = pd.read_csv(
    TRAIN_FILE
)


val_df = pd.read_csv(
    VAL_FILE
)


test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"Original training rows : {len(train_df):,}"
)


print(
    f"Validation rows         : {len(val_df):,}"
)


print(
    f"Test rows               : {len(test_df):,}"
)


# ============================================================
# FAST TRAINING SUBSET
# ============================================================
#
# IMPORTANT:
# We select a stratified subset so that the spam/genuine
# ratio remains approximately the same.
#
# Validation and test are NOT reduced.
# ============================================================

if len(train_df) > MAX_TRAIN:

    print()
    print(
        f"Reducing training set to "
        f"{MAX_TRAIN:,} samples..."
    )


    train_df = (
        train_df
        .groupby(
            "spam",
            group_keys=False
        )
        .apply(
            lambda group: group.sample(
                n=round(
                    MAX_TRAIN
                    * len(group)
                    / len(train_df)
                ),
                random_state=SEED
            )
        )
        .reset_index(drop=True)
    )


    # Make sure the final number is exactly MAX_TRAIN.
    if len(train_df) > MAX_TRAIN:

        train_df = train_df.sample(
            n=MAX_TRAIN,
            random_state=SEED
        ).reset_index(drop=True)


print(
    f"\nFinal training rows: "
    f"{len(train_df):,}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)
print()


print("TRAIN:")

print(
    train_df["spam"]
    .value_counts()
    .sort_index()
)


print()
print("VALIDATION:")

print(
    val_df["spam"]
    .value_counts()
    .sort_index()
)


print()
print("TEST:")

print(
    test_df["spam"]
    .value_counts()
    .sort_index()
)


# ============================================================
# EXTRACT TEXT
# ============================================================

print()
print("=" * 70)
print("PREPARING TEXT")
print("=" * 70)
print()


train_texts = (
    train_df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


val_texts = (
    val_df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


test_texts = (
    test_df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# LABELS
# ============================================================

y_train = (
    train_df["spam"]
    .astype(np.float32)
    .values
)


y_val = (
    val_df["spam"]
    .astype(np.float32)
    .values
)


y_test = (
    test_df["spam"]
    .astype(np.float32)
    .values
)


# ============================================================
# EMBEDDING FILES
# ============================================================

TRAIN_EMBED_FILE = (
    EMBED_DIR
    / "train_embeddings_fast.npy"
)


VAL_EMBED_FILE = (
    EMBED_DIR
    / "val_embeddings.npy"
)


TEST_EMBED_FILE = (
    EMBED_DIR
    / "test_embeddings.npy"
)


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("=" * 70)
print("LOADING MiniLM")
print("=" * 70)
print()


print(
    f"Model: {MINILM_MODEL}"
)


print(
    "\nIf embeddings already exist, "
    "they will be loaded from disk."
)


encoder = SentenceTransformer(
    MINILM_MODEL
)


print()
print("MiniLM loaded successfully.")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def generate_or_load_embeddings(
    texts,
    output_file,
    split_name
):

    # --------------------------------------------------------
    # Load existing embeddings
    # --------------------------------------------------------

    if output_file.exists():

        print()
        print(
            f"Loading cached "
            f"{split_name} embeddings..."
        )


        embeddings = np.load(
            output_file
        )


        print(
            f"{split_name} shape: "
            f"{embeddings.shape}"
        )


        return embeddings


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print()
    print(
        f"Generating {split_name} embeddings..."
    )


    print(
        f"Number of reviews: "
        f"{len(texts):,}"
    )


    embeddings = encoder.encode(

        texts,

        batch_size=MINILM_BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=False
    )


    embeddings = (
        embeddings
        .astype(np.float32)
    )


    print(
        f"{split_name} shape: "
        f"{embeddings.shape}"
    )


    np.save(
        output_file,
        embeddings
    )


    print(
        f"Saved embeddings to:"
    )


    print(
        output_file
    )


    return embeddings


# ============================================================
# GENERATE / LOAD TRAIN EMBEDDINGS
# ============================================================

X_train = generate_or_load_embeddings(

    train_texts,

    TRAIN_EMBED_FILE,

    "TRAIN"
)


# ============================================================
# VALIDATION EMBEDDINGS
# ============================================================

X_val = generate_or_load_embeddings(

    val_texts,

    VAL_EMBED_FILE,

    "VALIDATION"
)


# ============================================================
# TEST EMBEDDINGS
# ============================================================

X_test = generate_or_load_embeddings(

    test_texts,

    TEST_EMBED_FILE,

    "TEST"
)


# ============================================================
# VERIFY EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("VERIFYING EMBEDDINGS")
print("=" * 70)
print()


print(
    "Train:",
    X_train.shape
)


print(
    "Validation:",
    X_val.shape
)


print(
    "Test:",
    X_test.shape
)


if X_train.shape[1] != EMBEDDING_DIM:

    raise ValueError(

        f"Expected MiniLM embedding dimension "
        f"{EMBEDDING_DIM}, but received "
        f"{X_train.shape[1]}"
    )


# ============================================================
# RESHAPE FOR BiLSTM
# ============================================================
#
# Original MiniLM embedding:
#
# (samples, 384)
#
# BiLSTM input:
#
# (samples, 384, 1)
#
# ============================================================

print()
print(
    "Converting MiniLM embeddings "
    "into BiLSTM sequences..."
)


X_train = X_train.reshape(

    X_train.shape[0],

    EMBEDDING_DIM,

    1
)


X_val = X_val.reshape(

    X_val.shape[0],

    EMBEDDING_DIM,

    1
)


X_test = X_test.reshape(

    X_test.shape[0],

    EMBEDDING_DIM,

    1
)


print()
print(
    "BiLSTM input:"
)


print(
    "Train:",
    X_train.shape
)


print(
    "Validation:",
    X_val.shape
)


print(
    "Test:",
    X_test.shape
)


# ============================================================
# ATTENTION LAYER
# ============================================================

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


    def build(
        self,
        input_shape
    ):

        hidden_size = (
            input_shape[-1]
        )


        self.W = self.add_weight(

            name="attention_weight",

            shape=(
                hidden_size,
                hidden_size
            ),

            initializer="glorot_uniform",

            trainable=True
        )


        self.b = self.add_weight(

            name="attention_bias",

            shape=(
                hidden_size,
            ),

            initializer="zeros",

            trainable=True
        )


        self.u = self.add_weight(

            name="attention_context",

            shape=(
                hidden_size,
            ),

            initializer="glorot_uniform",

            trainable=True
        )


        super().build(
            input_shape
        )


    def call(
        self,
        inputs
    ):

        # ----------------------------------------------------
        # Calculate attention scores
        # ----------------------------------------------------

        score = tf.tanh(

            tf.tensordot(

                inputs,

                self.W,

                axes=1

            )

            + self.b
        )


        attention_scores = (

            tf.tensordot(

                score,

                self.u,

                axes=1

            )
        )


        # ----------------------------------------------------
        # Convert scores to weights
        # ----------------------------------------------------

        attention_weights = (

            tf.nn.softmax(

                attention_scores,

                axis=1

            )
        )


        # ----------------------------------------------------
        # Weighted sum
        # ----------------------------------------------------

        context = (

            tf.reduce_sum(

                inputs

                * tf.expand_dims(

                    attention_weights,

                    axis=-1

                ),

                axis=1

            )
        )


        return context


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING MODEL")
print("=" * 70)
print()


inputs = tf.keras.Input(

    shape=(

        EMBEDDING_DIM,

        1

    ),

    name="minilm_embedding"
)


# ============================================================
# BiLSTM
# ============================================================

x = (

    tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            32,

            return_sequences=True

        ),

        name="bilstm"

    )

    (inputs)

)


# ============================================================
# ATTENTION
# ============================================================

x = AttentionLayer(

    name="attention"

)(

    x

)


# ============================================================
# DENSE
# ============================================================

x = tf.keras.layers.Dense(

    64,

    activation="relu",

    name="text_dense_64"

)(

    x

)


x = tf.keras.layers.Dropout(

    0.3

)(

    x

)


x = tf.keras.layers.Dense(

    32,

    activation="relu",

    name="text_dense_32"

)(

    x

)


x = tf.keras.layers.Dropout(

    0.2

)(

    x

)


# ============================================================
# OUTPUT
# ============================================================

outputs = tf.keras.layers.Dense(

    1,

    activation="sigmoid",

    name="spam_probability"

)(

    x

)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs

)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.001

    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(

            name="accuracy"

        ),

        tf.keras.metrics.AUC(

            name="auc"

        )

    ]

)


print()
print("MODEL SUMMARY")
print()


model.summary()


# ============================================================
# CLASS WEIGHTS
# ============================================================

genuine_count = np.sum(

    y_train == 0

)


spam_count = np.sum(

    y_train == 1

)


total_count = (

    genuine_count
    + spam_count
)


class_weight = {

    0: total_count
    / (
        2.0
        * genuine_count
    ),

    1: total_count
    / (
        2.0
        * spam_count
    )
}


print()
print(
    "Class weights:"
)


print(
    class_weight
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=1,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            MODEL_FILE
        ),

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1

    )

]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING FAST TRAINING")
print("=" * 70)
print()


print(
    f"Training samples : {len(X_train):,}"
)


print(
    f"Batch size       : {TRAIN_BATCH_SIZE}"
)


print(
    f"Maximum epochs   : {EPOCHS}"
)


print()


history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,

        y_val

    ),

    epochs=EPOCHS,

    batch_size=TRAIN_BATCH_SIZE,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print(
    "Loading best saved model..."
)


model = tf.keras.models.load_model(

    MODEL_FILE,

    custom_objects={

        "AttentionLayer":
        AttentionLayer

    }

)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("RUNNING TEST PREDICTIONS")
print("=" * 70)
print()


probabilities = model.predict(

    X_test,

    batch_size=TRAIN_BATCH_SIZE,

    verbose=1

).ravel()


predictions = (

    probabilities >= 0.5

).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    predictions

)


precision = precision_score(

    y_test,

    predictions,

    zero_division=0

)


recall = recall_score(

    y_test,

    predictions,

    zero_division=0

)


f1 = f1_score(

    y_test,

    predictions,

    zero_division=0

)


roc_auc = roc_auc_score(

    y_test,

    probabilities

)


pr_auc = average_precision_score(

    y_test,

    probabilities

)


cm = confusion_matrix(

    y_test,

    predictions

)


report = classification_report(

    y_test,

    predictions,

    target_names=[

        "Genuine",

        "Spam"

    ],

    digits=4,

    zero_division=0

)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("STAGE 2 TEXT MODEL RESULTS")
print("=" * 70)
print()


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


print()
print("Confusion Matrix:")
print()


print(
    cm
)


print()
print("Classification Report:")
print()


print(
    report
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = (

    test_df[

        [

            "user_id",

            "prod_id",

            "rating",

            "date",

            "text",

            "spam"

        ]

    ]

    .copy()

)


prediction_df[

    "spam_probability"

] = probabilities


prediction_df[

    "prediction"

] = predictions


prediction_df[

    "prediction_label"

] = (

    prediction_df["prediction"]

    .map(

        {

            0: "Genuine",

            1: "Spam"

        }

    )

)


prediction_df.to_csv(

    PREDICTION_FILE,

    index=False

)


# ============================================================
# SAVE REPORT
# ============================================================

with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as f:

    f.write(

        "VERISIGHT - STAGE 2 TEXT DEEP LEARNING\n"

    )

    f.write(

        "MiniLM + BiLSTM + Attention\n"

    )

    f.write(

        "=" * 70 + "\n\n"

    )


    f.write(

        f"MiniLM model: "
        f"{MINILM_MODEL}\n"

    )


    f.write(

        f"Embedding dimension: "
        f"{EMBEDDING_DIM}\n"

    )


    f.write(

        f"Training samples: "
        f"{len(train_df):,}\n"

    )


    f.write(

        f"Validation samples: "
        f"{len(val_df):,}\n"

    )


    f.write(

        f"Test samples: "
        f"{len(test_df):,}\n\n"

    )


    f.write(

        f"Accuracy: "
        f"{accuracy:.6f}\n"

    )


    f.write(

        f"Precision: "
        f"{precision:.6f}\n"

    )


    f.write(

        f"Recall: "
        f"{recall:.6f}\n"

    )


    f.write(

        f"F1: "
        f"{f1:.6f}\n"

    )


    f.write(

        f"ROC-AUC: "
        f"{roc_auc:.6f}\n"

    )


    f.write(

        f"PR-AUC: "
        f"{pr_auc:.6f}\n\n"

    )


    f.write(

        "Confusion Matrix:\n"

    )


    f.write(

        str(cm)

        + "\n\n"

    )


    f.write(

        "Classification Report:\n"

    )


    f.write(

        report

    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TEXT MODEL COMPLETE")
print("=" * 70)
print()


print(
    "Model saved:"
)


print(
    MODEL_FILE
)


print()
print(
    "Report saved:"
)


print(
    REPORT_FILE
)


print()
print(
    "Predictions saved:"
)


print(
    PREDICTION_FILE
)


print()
print("=" * 70)
# ============================================================
# VeriSight - Stage 2 Hybrid Fusion Model
#
# Architecture:
#
#                 REVIEW
#                    |
#          +---------+---------+
#          |                   |
#       MiniLM             Behavioral
#       384-D              Features
#          |                   |
#      Dense 128           Dense 64
#          |                   |
#          +---------+---------+
#                    |
#                  FUSION
#                    |
#                Dense 128
#                    |
#                 Dropout
#                    |
#                 Dense 32
#                    |
#              Spam Probability
#
# IMPORTANT:
# MiniLM embeddings are ALREADY generated.
# This script does NOT regenerate them.
#
# Training:
#   150,000 samples
#
# Validation:
#   61,216 samples
#
# Test:
#   60,106 samples
#
# ============================================================


import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
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


# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------------

TRAIN_EMBED_FILE = (
    BASE_DIR
    / "embeddings"
    / "stage2_text"
    / "train_embeddings_fast.npy"
)

VAL_EMBED_FILE = (
    BASE_DIR
    / "embeddings"
    / "stage2_text"
    / "val_embeddings.npy"
)

TEST_EMBED_FILE = (
    BASE_DIR
    / "embeddings"
    / "stage2_text"
    / "test_embeddings.npy"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

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


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_FILE = (
    MODEL_DIR
    / "stage2_hybrid_fusion.keras"
)

SCALER_FILE = (
    MODEL_DIR
    / "stage2_hybrid_behavior_scaler.pkl"
)

REPORT_FILE = (
    RESULT_DIR
    / "stage2_hybrid_fusion_results.txt"
)

PREDICTION_FILE = (
    RESULT_DIR
    / "stage2_hybrid_fusion_predictions.csv"
)

THRESHOLD_FILE = (
    RESULT_DIR
    / "stage2_hybrid_fusion_thresholds.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

SEED = 42

MAX_TRAIN = 150000

EMBEDDING_DIM = 384

BATCH_SIZE = 1024

EPOCHS = 10


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
print("=" * 75)
print("VERISIGHT - STAGE 2 HYBRID FUSION")
print("=" * 75)
print()

print("Architecture:")
print()
print("MiniLM 384-D")
print("     |")
print(" Dense 128")
print("     |")
print("     +------------------+")
print("                        |")
print("Behavior Features -> Dense 64")
print("                        |")
print("                        v")
print("                     FUSION")
print("                        |")
print("                    Dense 128")
print("                        |")
print("                     Dense 32")
print("                        |")
print("                 Spam Probability")
print()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("=" * 75)
print("CHECKING FILES")
print("=" * 75)
print()


required_files = [

    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,

    TRAIN_EMBED_FILE,
    VAL_EMBED_FILE,
    TEST_EMBED_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print(
        "FOUND:",
        file_path
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 75)
print("LOADING DATASETS")
print("=" * 75)
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
    f"Original train : {len(train_df):,}"
)

print(
    f"Validation     : {len(val_df):,}"
)

print(
    f"Test           : {len(test_df):,}"
)


# ============================================================
# RECREATE EXACT SAME 150K TRAINING SUBSET
# ============================================================
#
# IMPORTANT:
# The MiniLM training embeddings were generated after the
# previous script selected 150,000 rows using this exact
# sampling procedure.
#
# We MUST recreate the same rows so that:
#
# embedding[i] <-> behavior_features[i]
#
# remains correctly aligned.
# ============================================================

if len(train_df) > MAX_TRAIN:

    print()
    print(
        "Recreating the exact 150,000-row "
        "training subset..."
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


    if len(train_df) > MAX_TRAIN:

        train_df = (

            train_df

            .sample(
                n=MAX_TRAIN,
                random_state=SEED
            )

            .reset_index(drop=True)
        )


print()
print(
    f"Final train rows: {len(train_df):,}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print()
print("=" * 75)
print("LOADING MINI-LM EMBEDDINGS")
print("=" * 75)
print()


X_text_train = np.load(
    TRAIN_EMBED_FILE
)

X_text_val = np.load(
    VAL_EMBED_FILE
)

X_text_test = np.load(
    TEST_EMBED_FILE
)


print(
    "Train text embeddings:",
    X_text_train.shape
)

print(
    "Validation text embeddings:",
    X_text_val.shape
)

print(
    "Test text embeddings:",
    X_text_test.shape
)


# ============================================================
# VERIFY EMBEDDING SIZES
# ============================================================

if X_text_train.shape != (
    len(train_df),
    EMBEDDING_DIM
):

    raise ValueError(

        "Training embedding shape does not match "
        "the recreated training dataframe.\n"

        f"Expected: "
        f"({len(train_df)}, {EMBEDDING_DIM})\n"

        f"Received: "
        f"{X_text_train.shape}"
    )


if X_text_val.shape != (
    len(val_df),
    EMBEDDING_DIM
):

    raise ValueError(
        "Validation embedding shape does not match "
        "validation dataframe."
    )


if X_text_test.shape != (
    len(test_df),
    EMBEDDING_DIM
):

    raise ValueError(
        "Test embedding shape does not match "
        "test dataframe."
    )


print()
print(
    "Embedding alignment verified."
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
# BEHAVIOR FEATURE SELECTION
# ============================================================
#
# We exclude:
#
# user_id
# prod_id
# date
# text
# spam
#
# We keep useful numeric information such as:
#
# rating
# user history
# product history
# temporal behavior
# burst activity
# rating consistency
# sentiment information
#
# ============================================================

EXCLUDE_COLUMNS = {

    "user_id",

    "prod_id",

    "date",

    "text",

    "spam",

    "original_index"

}


candidate_features = [

    column

    for column in train_df.columns

    if column not in EXCLUDE_COLUMNS

    and pd.api.types.is_numeric_dtype(
        train_df[column]
    )

]


# ------------------------------------------------------------
# Remove constant feature if present
# ------------------------------------------------------------

behavior_features = []

for column in candidate_features:

    if train_df[column].nunique(
        dropna=False
    ) <= 1:

        print(
            f"Removing constant feature: "
            f"{column}"
        )

    else:

        behavior_features.append(
            column
        )


print()
print("=" * 75)
print("BEHAVIOR FEATURES")
print("=" * 75)
print()


print(
    f"Number of behavioral features: "
    f"{len(behavior_features)}"
)


for index, feature in enumerate(
    behavior_features,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# EXTRACT BEHAVIOR MATRICES
# ============================================================

X_behavior_train = (

    train_df[
        behavior_features
    ]

    .astype(np.float32)

    .values
)


X_behavior_val = (

    val_df[
        behavior_features
    ]

    .astype(np.float32)

    .values
)


X_behavior_test = (

    test_df[
        behavior_features
    ]

    .astype(np.float32)

    .values
)


# ============================================================
# HANDLE NaN / INF
# ============================================================

X_behavior_train = np.nan_to_num(
    X_behavior_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_behavior_val = np.nan_to_num(
    X_behavior_val,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_behavior_test = np.nan_to_num(
    X_behavior_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# STANDARDIZATION
# ============================================================
#
# IMPORTANT:
# Scaler is fitted ONLY on training data.
#
# This prevents validation/test information leakage.
# ============================================================

print()
print("=" * 75)
print("STANDARDIZING BEHAVIOR FEATURES")
print("=" * 75)
print()


scaler = StandardScaler()


X_behavior_train = scaler.fit_transform(
    X_behavior_train
)


X_behavior_val = scaler.transform(
    X_behavior_val
)


X_behavior_test = scaler.transform(
    X_behavior_test
)


joblib.dump(
    scaler,
    SCALER_FILE
)


print(
    "Scaler saved:"
)

print(
    SCALER_FILE
)


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

X_text_train = X_text_train.astype(
    np.float32
)

X_text_val = X_text_val.astype(
    np.float32
)

X_text_test = X_text_test.astype(
    np.float32
)


X_behavior_train = X_behavior_train.astype(
    np.float32
)

X_behavior_val = X_behavior_val.astype(
    np.float32
)

X_behavior_test = X_behavior_test.astype(
    np.float32
)


# ============================================================
# VERIFY FINAL INPUTS
# ============================================================

print()
print("=" * 75)
print("FINAL INPUT SHAPES")
print("=" * 75)
print()


print(
    "Text train     :",
    X_text_train.shape
)

print(
    "Behavior train :",
    X_behavior_train.shape
)

print(
    "Text val       :",
    X_text_val.shape
)

print(
    "Behavior val   :",
    X_behavior_val.shape
)

print(
    "Text test      :",
    X_text_test.shape
)

print(
    "Behavior test  :",
    X_behavior_test.shape
)


# ============================================================
# BUILD TEXT BRANCH
# ============================================================

print()
print("=" * 75)
print("BUILDING HYBRID MODEL")
print("=" * 75)
print()


text_input = tf.keras.Input(

    shape=(
        EMBEDDING_DIM,
    ),

    name="minilm_embedding"
)


text_branch = tf.keras.layers.Dense(

    128,

    activation="relu",

    name="text_dense_128"

)(

    text_input
)


text_branch = tf.keras.layers.Dropout(

    0.30

)(

    text_branch
)


text_branch = tf.keras.layers.Dense(

    64,

    activation="relu",

    name="text_dense_64"

)(

    text_branch
)


# ============================================================
# BUILD BEHAVIOR BRANCH
# ============================================================

behavior_input = tf.keras.Input(

    shape=(

        len(behavior_features),

    ),

    name="behavior_features"
)


behavior_branch = tf.keras.layers.Dense(

    64,

    activation="relu",

    name="behavior_dense_64"

)(

    behavior_input
)


behavior_branch = tf.keras.layers.BatchNormalization()(

    behavior_branch

)


behavior_branch = tf.keras.layers.Dropout(

    0.25

)(

    behavior_branch
)


behavior_branch = tf.keras.layers.Dense(

    32,

    activation="relu",

    name="behavior_dense_32"

)(

    behavior_branch
)


# ============================================================
# FUSION
# ============================================================

fusion = tf.keras.layers.Concatenate(

    name="feature_fusion"

)(

    [

        text_branch,

        behavior_branch

    ]

)


# ============================================================
# FUSION DENSE LAYERS
# ============================================================

fusion = tf.keras.layers.Dense(

    128,

    activation="relu",

    name="fusion_dense_128"

)(

    fusion
)


fusion = tf.keras.layers.BatchNormalization()(

    fusion

)


fusion = tf.keras.layers.Dropout(

    0.35

)(

    fusion
)


fusion = tf.keras.layers.Dense(

    32,

    activation="relu",

    name="fusion_dense_32"

)(

    fusion
)


fusion = tf.keras.layers.Dropout(

    0.20

)(

    fusion
)


# ============================================================
# OUTPUT
# ============================================================

output = tf.keras.layers.Dense(

    1,

    activation="sigmoid",

    name="spam_probability"

)(

    fusion
)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(

    inputs=[

        text_input,

        behavior_input

    ],

    outputs=output

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


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print(
    "MODEL SUMMARY"
)
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

    0:
    total_count
    / (
        2.0
        * genuine_count
    ),

    1:
    total_count
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

        patience=2,

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
print("=" * 75)
print("STARTING HYBRID TRAINING")
print("=" * 75)
print()


print(
    f"Training samples : {len(y_train):,}"
)

print(
    f"Validation       : {len(y_val):,}"
)

print(
    f"Batch size       : {BATCH_SIZE}"
)

print(
    f"Maximum epochs   : {EPOCHS}"
)

print(
    f"Text features    : {EMBEDDING_DIM}"
)

print(
    f"Behavior features: {len(behavior_features)}"
)

print()


history = model.fit(

    [

        X_text_train,

        X_behavior_train

    ],

    y_train,

    validation_data=(

        [

            X_text_val,

            X_behavior_val

        ],

        y_val

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print(
    "Loading best hybrid model..."
)


model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print()
print("=" * 75)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 75)
print()


val_probabilities = model.predict(

    [

        X_text_val,

        X_behavior_val

    ],

    batch_size=BATCH_SIZE,

    verbose=1

).ravel()


# ============================================================
# TEST PROBABILITIES
# ============================================================

print()
print("=" * 75)
print("GENERATING TEST PREDICTIONS")
print("=" * 75)
print()


test_probabilities = model.predict(

    [

        X_text_test,

        X_behavior_test

    ],

    batch_size=BATCH_SIZE,

    verbose=1

).ravel()


# ============================================================
# THRESHOLD SEARCH
# ============================================================
#
# IMPORTANT:
# Threshold is selected using VALIDATION only.
#
# Test remains untouched until final evaluation.
# ============================================================

thresholds = np.arange(

    0.10,

    0.91,

    0.01

)


threshold_results = []


for threshold in thresholds:

    val_predictions = (

        val_probabilities
        >= threshold

    ).astype(int)


    precision = precision_score(

        y_val,

        val_predictions,

        zero_division=0

    )


    recall = recall_score(

        y_val,

        val_predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_val,

        val_predictions,

        zero_division=0

    )


    accuracy = accuracy_score(

        y_val,

        val_predictions

    )


    threshold_results.append(

        {

            "threshold":
            float(threshold),

            "accuracy":
            float(accuracy),

            "precision":
            float(precision),

            "recall":
            float(recall),

            "f1":
            float(f1)

        }

    )


threshold_df = pd.DataFrame(
    threshold_results
)


threshold_df.to_csv(

    THRESHOLD_FILE,

    index=False

)


# ------------------------------------------------------------
# Best validation threshold by F1
# ------------------------------------------------------------

best_row = (

    threshold_df

    .sort_values(

        "f1",

        ascending=False

    )

    .iloc[0]

)


best_threshold = float(

    best_row["threshold"]

)


print()
print(
    "Best validation threshold:"
)


print(
    f"{best_threshold:.2f}"
)


print()
print(
    "Validation metrics at selected threshold:"
)


print(
    f"Accuracy : "
    f"{best_row['accuracy']:.4f}"
)


print(
    f"Precision: "
    f"{best_row['precision']:.4f}"
)


print(
    f"Recall   : "
    f"{best_row['recall']:.4f}"
)


print(
    f"F1       : "
    f"{best_row['f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_predictions = (

    test_probabilities
    >= best_threshold

).astype(int)


accuracy = accuracy_score(

    y_test,

    test_predictions

)


precision = precision_score(

    y_test,

    test_predictions,

    zero_division=0

)


recall = recall_score(

    y_test,

    test_predictions,

    zero_division=0

)


f1 = f1_score(

    y_test,

    test_predictions,

    zero_division=0

)


roc_auc = roc_auc_score(

    y_test,

    test_probabilities

)


pr_auc = average_precision_score(

    y_test,

    test_probabilities

)


cm = confusion_matrix(

    y_test,

    test_predictions

)


report = classification_report(

    y_test,

    test_predictions,

    target_names=[

        "Genuine",

        "Spam"

    ],

    digits=4,

    zero_division=0

)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL HYBRID FUSION RESULTS")
print("=" * 75)
print()


print(
    f"Threshold : {best_threshold:.2f}"
)


print(
    f"Accuracy  : {accuracy:.4f}"
)


print(
    f"Precision : {precision:.4f}"
)


print(
    f"Recall    : {recall:.4f}"
)


print(
    f"F1        : {f1:.4f}"
)


print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print(
    f"PR-AUC    : {pr_auc:.4f}"
)


print()
print(
    "Confusion Matrix:"
)


print(
    cm
)


print()
print(
    "Classification Report:"
)


print(
    report
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[

    [

        "user_id",

        "prod_id",

        "rating",

        "date",

        "text",

        "spam"

    ]

].copy()


prediction_df[
    "spam_probability"
] = test_probabilities


prediction_df[
    "prediction"
] = test_predictions


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
        "VERISIGHT - STAGE 2 HYBRID FUSION\n"
    )

    f.write(
        "MiniLM + Behavioral Features\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Training samples: "
        f"{len(y_train):,}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(y_val):,}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(y_test):,}\n"
    )

    f.write(
        f"MiniLM dimensions: "
        f"{EMBEDDING_DIM}\n"
    )

    f.write(
        f"Behavior features: "
        f"{len(behavior_features)}\n"
    )

    f.write(
        f"Selected threshold: "
        f"{best_threshold:.2f}\n\n"
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
        "Behavior features:\n"
    )

    for feature in behavior_features:

        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        report
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("HYBRID FUSION COMPLETE")
print("=" * 75)
print()


print(
    "Model:"
)

print(
    MODEL_FILE
)


print()
print(
    "Scaler:"
)

print(
    SCALER_FILE
)


print()
print(
    "Report:"
)

print(
    REPORT_FILE
)


print()
print(
    "Predictions:"
)

print(
    PREDICTION_FILE
)


print()
print(
    "Threshold results:"
)

print(
    THRESHOLD_FILE
)


print()
print("=" * 75)
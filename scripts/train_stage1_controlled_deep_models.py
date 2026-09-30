import os
import numpy as np
import pandas as pd
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
# STAGE 1 - CONTROLLED DEEP LEARNING COMPARISON
#
# Models:
#   1. MLP
#   2. CNN
#   3. LSTM
#   4. GRU
#   5. BiLSTM
#
# All models use:
#   Same controlled dataset
#   Same MiniLM embeddings
#   Same train/validation/test split
#   Same frozen Gemini external dataset
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
STAGE1_DIR = DATA_DIR / "stage1"

EMBED_DIR = (
    BASE_DIR /
    "embeddings" /
    "stage1_controlled"
)

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "stage1_controlled_deep"
)

RESULT_DIR = (
    BASE_DIR /
    "results" /
    "stage1_controlled_deep"
)

TRAIN_CSV = (
    STAGE1_DIR /
    "stage1_v3_source_length_controlled.csv"
)

GEMINI_CANDIDATES = [
    DATA_DIR /
    "external" /
    "gemini_external_balanced.csv",

    DATA_DIR /
    "external" /
    "gemini_external_final.csv"
]

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
# SEEDS
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 80)
print("VERISIGHT - STAGE 1 CONTROLLED DEEP LEARNING")
print("=" * 80)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/9] Loading controlled dataset...")

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{TRAIN_CSV}"
    )

df = pd.read_csv(TRAIN_CSV)

print(
    f"Dataset shape: {df.shape}"
)

print("\nLabel distribution:")
print(
    df["label"].value_counts()
)

# ============================================================
# 2. LOAD MINI-LM
# ============================================================

print("\n[2/9] Loading MiniLM...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# 3. GENERATE / LOAD EMBEDDINGS
# ============================================================

print("\n[3/9] Preparing MiniLM embeddings...")

X_EMBED = (
    EMBED_DIR /
    "X_stage1_controlled.npy"
)

Y_EMBED = (
    EMBED_DIR /
    "y_stage1_controlled.npy"
)

if X_EMBED.exists() and Y_EMBED.exists():

    print(
        "Existing embeddings found."
    )

    X = np.load(X_EMBED)
    y = np.load(Y_EMBED)

else:

    print(
        f"Encoding {len(df)} reviews..."
    )

    texts = (
        df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    X = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    y = (
        df["label"]
        .astype(np.int32)
        .values
    )

    np.save(
        X_EMBED,
        X
    )

    np.save(
        Y_EMBED,
        y
    )

print(
    f"Embedding shape: {X.shape}"
)

# ============================================================
# 4. FIXED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[4/9] Creating fixed stratified split...")

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
    f"Train: {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test: {X_test.shape}"
)

# ============================================================
# 5. SCALE EMBEDDINGS
# ============================================================

print("\n[5/9] Scaling embeddings...")

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

# ============================================================
# PREPARE SEQUENCE FORMAT
#
# CNN/LSTM/GRU/BiLSTM:
#
# (samples, 384)
#
# becomes:
#
# (samples, 384, 1)
# ============================================================

X_train_seq = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    X_train_scaled.shape[1],
    1
)

X_val_seq = X_val_scaled.reshape(
    X_val_scaled.shape[0],
    X_val_scaled.shape[1],
    1
)

X_test_seq = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

# ============================================================
# COMMON CALLBACKS
# ============================================================

def get_callbacks():

    return [

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
# MODEL 1 - MLP
# ============================================================

def build_mlp():

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

    return model


# ============================================================
# MODEL 2 - CNN
# ============================================================

def build_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(384, 1)
        ),

        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

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

    return model


# ============================================================
# MODEL 3 - LSTM
# ============================================================

def build_lstm():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(384, 1)
        ),

        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.LSTM(
            64
        ),

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

    return model


# ============================================================
# MODEL 4 - GRU
# ============================================================

def build_gru():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(384, 1)
        ),

        tf.keras.layers.GRU(
            128,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.GRU(
            64
        ),

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

    return model


# ============================================================
# MODEL 5 - BiLSTM
# ============================================================

def build_bilstm():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(384, 1)
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
                64
            )
        ),

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

    return model


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_and_evaluate(
    name,
    model,
    train_x,
    val_x,
    test_x
):

    print("\n")
    print("=" * 80)
    print(f"TRAINING {name}")
    print("=" * 80)

    history = model.fit(

        train_x,
        y_train,

        validation_data=(
            val_x,
            y_val
        ),

        epochs=50,

        batch_size=64,

        callbacks=get_callbacks(),

        verbose=1

    )

    # --------------------------------------------------------
    # Internal test
    # --------------------------------------------------------

    probs = model.predict(
        test_x,
        verbose=0
    ).ravel()

    preds = (
        probs >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    cm = confusion_matrix(
        y_test,
        preds
    )

    report = classification_report(
        y_test,
        preds,
        target_names=[
            "Human",
            "AI"
        ],
        digits=4
    )

    print(
        f"\n{name} Internal Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        f"{name.lower()}_stage1_controlled.keras"
    )

    model.save(
        model_path
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "history": history
    }


# ============================================================
# 6. TRAIN ALL DEEP MODELS
# ============================================================

print("\n[6/9] Training deep-learning models...")

results = []

# ------------------------------------------------------------
# MLP
# ------------------------------------------------------------

mlp = build_mlp()

results.append(
    train_and_evaluate(
        "MLP",
        mlp,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled
    )
)

# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

cnn = build_cnn()

results.append(
    train_and_evaluate(
        "CNN",
        cnn,
        X_train_seq,
        X_val_seq,
        X_test_seq
    )
)

# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

lstm = build_lstm()

results.append(
    train_and_evaluate(
        "LSTM",
        lstm,
        X_train_seq,
        X_val_seq,
        X_test_seq
    )
)

# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

gru = build_gru()

results.append(
    train_and_evaluate(
        "GRU",
        gru,
        X_train_seq,
        X_val_seq,
        X_test_seq
    )
)

# ------------------------------------------------------------
# BiLSTM
# ------------------------------------------------------------

bilstm = build_bilstm()

results.append(
    train_and_evaluate(
        "BiLSTM",
        bilstm,
        X_train_seq,
        X_val_seq,
        X_test_seq
    )
)

# ============================================================
# 7. GEMINI DATASET
# ============================================================

print("\n[7/9] Loading frozen Gemini dataset...")

gemini_path = None

for candidate in GEMINI_CANDIDATES:

    if candidate.exists():

        gemini_path = candidate
        break

if gemini_path is None:

    print(
        "\nGemini dataset not found."
    )

else:

    gemini_df = pd.read_csv(
        gemini_path
    )

    print(
        f"Gemini dataset: {gemini_path}"
    )

    print(
        f"Gemini samples: {len(gemini_df)}"
    )

    gemini_texts = (
        gemini_df["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "\nGenerating Gemini MiniLM embeddings..."
    )

    X_gemini = embedding_model.encode(

        gemini_texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=False

    )

    # IMPORTANT:
    # Scale Gemini using the training scaler only.
    X_gemini_scaled = scaler.transform(
        X_gemini
    )

    X_gemini_seq = X_gemini_scaled.reshape(
        X_gemini_scaled.shape[0],
        X_gemini_scaled.shape[1],
        1
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

    # ========================================================
    # EVALUATE EACH MODEL
    # ========================================================

    for result in results:

        name = result["name"]
        model = result["model"]

        if name == "MLP":

            input_data = X_gemini_scaled

        else:

            input_data = X_gemini_seq

        gemini_probs = model.predict(
            input_data,
            verbose=0
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

        result[
            "gemini_accuracy"
        ] = gemini_accuracy

        result[
            "gemini_cm"
        ] = gemini_cm

        result[
            "gemini_probs"
        ] = gemini_probs

        result[
            "gemini_preds"
        ] = gemini_preds

        print(
            f"\n{name} Gemini Accuracy: "
            f"{gemini_accuracy * 100:.2f}%"
        )

        print(
            f"Correct: "
            f"{np.sum(gemini_preds == y_gemini)}"
            f"/{len(y_gemini)}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            gemini_cm
        )

# ============================================================
# 8. FINAL COMPARISON TABLE
# ============================================================

print("\n[8/9] Creating comparison table...")

comparison = []

for result in results:

    row = {

        "Model":
        result["name"],

        "Internal_Accuracy":
        result["accuracy"] * 100,

        "Gemini_Accuracy":
        result.get(
            "gemini_accuracy",
            np.nan
        ) * 100

    }

    comparison.append(
        row
    )

comparison_df = pd.DataFrame(
    comparison
)

comparison_df = comparison_df.round(
    2
)

print("\n")
print("=" * 80)
print("FINAL DEEP LEARNING COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)

comparison_path = (
    RESULT_DIR /
    "deep_learning_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

# ============================================================
# 9. SAVE COMPLETE REPORT
# ============================================================

print("\n[9/9] Saving final report...")

report_path = (
    RESULT_DIR /
    "deep_learning_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VERISIGHT STAGE 1\n"
        "CONTROLLED DEEP LEARNING COMPARISON\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Dataset: {TRAIN_CSV}\n"
    )

    f.write(
        f"Dataset shape: {df.shape}\n"
    )

    f.write(
        f"Embedding dimension: {X.shape[1]}\n\n"
    )

    f.write(
        "DATA SPLIT\n"
    )

    f.write("-" * 80 + "\n")

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
        "MODEL COMPARISON\n"
    )

    f.write("-" * 80 + "\n\n")

    f.write(
        comparison_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    # --------------------------------------------------------
    # Individual model results
    # --------------------------------------------------------

    for result in results:

        f.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        f.write(
            f"{result['name']}\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Internal Accuracy: "
            f"{result['accuracy'] * 100:.4f}%\n\n"
        )

        f.write(
            "Internal Confusion Matrix:\n"
        )

        f.write(
            str(
                result["confusion_matrix"]
            )
        )

        f.write("\n\n")

        f.write(
            "Classification Report:\n"
        )

        f.write(
            result["report"]
        )

        if (
            "gemini_accuracy"
            in result
        ):

            f.write(
                "\nGemini Accuracy: "
                f"{result['gemini_accuracy'] * 100:.4f}%\n"
            )

            f.write(
                f"Gemini Correct: "
                f"{np.sum(result['gemini_preds'] == y_gemini)}"
                f"/{len(y_gemini)}\n\n"
            )

            f.write(
                "Gemini Confusion Matrix:\n"
            )

            f.write(
                str(
                    result["gemini_cm"]
                )
            )

            f.write("\n")

            # Save individual Gemini predictions
            output_df = gemini_df.copy()

            output_df[
                "predicted_label"
            ] = result["gemini_preds"]

            output_df[
                "ai_probability"
            ] = result["gemini_probs"]

            output_df[
                "predicted_type"
            ] = np.where(
                result["gemini_preds"] == 1,
                "AI",
                "HUMAN"
            )

            prediction_path = (
                RESULT_DIR /
                f"{result['name'].lower()}_gemini_predictions.csv"
            )

            output_df.to_csv(
                prediction_path,
                index=False
            )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)

print(
    "\nComparison CSV:"
)

print(
    comparison_path
)

print(
    "\nDetailed report:"
)

print(
    report_path
)

print(
    "\nModels saved in:"
)

print(
    MODEL_DIR
)

print("\nFinal results:")

print(
    comparison_df.to_string(
        index=False
    )
)

print("\nDone.")
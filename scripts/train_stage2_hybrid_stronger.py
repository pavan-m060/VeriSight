import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# VERISIGHT STAGE 2 - STRONGER HYBRID MODEL
# ============================================================

BASE_DIR = r"C:\Users\ganes\Downloads\VeriSight"

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_train_hybrid.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_val_hybrid.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_test_hybrid.csv"
)

# ------------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------------

TRAIN_EMB = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage2_text",
    "train_embeddings_fast.npy"
)

VAL_EMB = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage2_text",
    "val_embeddings.npy"
)

TEST_EMB = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage2_text",
    "test_embeddings.npy"
)

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "stage2"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "stage2"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stage2_hybrid_stronger.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "stage2_hybrid_stronger_scaler.pkl"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "stage2_hybrid_stronger_results.txt"
)

THRESHOLD_PATH = os.path.join(
    RESULTS_DIR,
    "stage2_hybrid_stronger_thresholds.csv"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "stage2_hybrid_stronger_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_LIMIT = 150000

BATCH_SIZE = 1024

EPOCHS = 20

LEARNING_RATE = 0.0005

RANDOM_SEED = 42


# ============================================================
# EXACT 40 BEHAVIOR FEATURES
# ============================================================

BEHAVIOR_FEATURES = [
    "rating",

    "word_count",
    "char_count",
    "avg_word_length",
    "uppercase_ratio",
    "punctuation_count",
    "punctuation_ratio",
    "exclamation_count",
    "question_count",
    "digit_count",
    "digit_ratio",

    "positive_count",
    "negative_count",
    "positive_ratio",
    "negative_ratio",
    "sentiment_score",

    "user_review_count",
    "user_avg_rating",
    "user_unique_products",

    "product_review_count",
    "product_avg_rating",

    "exact_text_count",

    "rating_sentiment_difference",

    "user_previous_review_count",
    "user_previous_avg_rating",
    "user_previous_rating_std",
    "user_previous_unique_products",

    "time_since_previous_user_review_hours",
    "user_days_since_first_review",
    "user_previous_reviews_per_day",

    "reviews_previous_24h",
    "reviews_previous_7days",

    "product_previous_review_count",
    "product_previous_avg_rating",

    "time_since_previous_product_review_hours",

    "product_reviews_previous_1h",
    "product_reviews_previous_24h",
    "product_reviews_previous_7days",

    "rating_distance_from_user_history",
    "rating_distance_from_product_history"
]


# ============================================================
# SETUP
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print("\n" + "=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    required_files = [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        TRAIN_EMB,
        VAL_EMB,
        TEST_EMB
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile not found:\n{path}"
            )

        print("OK:", path)


# ============================================================
# LOAD BEHAVIOR DATA
# ============================================================

def load_behavior_data(
    csv_path,
    limit=None
):

    print("\nLoading:")
    print(csv_path)

    df = pd.read_csv(
        csv_path
    )

    if limit is not None:

        df = df.iloc[
            :limit
        ].copy()

    print(
        f"Rows: {len(df):,}"
    )

    missing = [
        col
        for col in BEHAVIOR_FEATURES
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing behavior features:\n"
            + "\n".join(missing)
        )

    X = df[
        BEHAVIOR_FEATURES
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    y = df[
        "spam"
    ].astype(
        np.int32
    ).values

    return (
        X.values.astype(np.float32),
        y
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # MINILM BRANCH
    # --------------------------------------------------------

    minilm_input = Input(
        shape=(384,),
        name="minilm_embedding"
    )

    text = Dense(
        256,
        activation="relu",
        name="text_dense_256"
    )(minilm_input)

    text = BatchNormalization(
        name="text_bn_256"
    )(text)

    text = Dropout(
        0.20,
        name="text_dropout_1"
    )(text)

    text = Dense(
        128,
        activation="relu",
        name="text_dense_128"
    )(text)

    text = Dropout(
        0.15,
        name="text_dropout_2"
    )(text)

    # --------------------------------------------------------
    # BEHAVIOR BRANCH
    # --------------------------------------------------------

    behavior_input = Input(
        shape=(40,),
        name="behavior_features"
    )

    behavior = Dense(
        128,
        activation="relu",
        name="behavior_dense_128"
    )(behavior_input)

    behavior = BatchNormalization(
        name="behavior_bn_128"
    )(behavior)

    behavior = Dropout(
        0.20,
        name="behavior_dropout_1"
    )(behavior)

    behavior = Dense(
        64,
        activation="relu",
        name="behavior_dense_64"
    )(behavior)

    behavior = Dropout(
        0.15,
        name="behavior_dropout_2"
    )(behavior)

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fused = Concatenate(
        name="feature_fusion"
    )([
        text,
        behavior
    ])

    fused = Dense(
        128,
        activation="relu",
        name="fusion_dense_128"
    )(fused)

    fused = BatchNormalization(
        name="fusion_bn_128"
    )(fused)

    fused = Dropout(
        0.30,
        name="fusion_dropout_1"
    )(fused)

    fused = Dense(
        64,
        activation="relu",
        name="fusion_dense_64"
    )(fused)

    fused = Dropout(
        0.20,
        name="fusion_dropout_2"
    )(fused)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = Dense(
        1,
        activation="sigmoid",
        name="spam_probability"
    )(fused)

    model = Model(
        inputs=[
            minilm_input,
            behavior_input
        ],
        outputs=output
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def search_thresholds(
    y_true,
    probabilities
):

    results = []

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int32)

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        results.append([
            round(
                float(threshold),
                2
            ),
            accuracy,
            precision,
            recall,
            f1
        ])

    return pd.DataFrame(
        results,
        columns=[
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "predictions": predictions
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("VERISIGHT STAGE 2 - STRONGER HYBRID MODEL")
    print("=" * 70)

    print("\nConfiguration:")
    print(
        f"Training samples : {TRAIN_LIMIT:,}"
    )
    print(
        f"Batch size       : {BATCH_SIZE}"
    )
    print(
        f"Maximum epochs   : {EPOCHS}"
    )
    print(
        f"Learning rate    : {LEARNING_RATE}"
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    check_files()

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING MINILM EMBEDDINGS")
    print("=" * 70)

    X_train_text = np.load(
        TRAIN_EMB
    )

    X_val_text = np.load(
        VAL_EMB
    )

    X_test_text = np.load(
        TEST_EMB
    )

    print(
        "\nTrain embeddings:",
        X_train_text.shape
    )

    print(
        "Validation embeddings:",
        X_val_text.shape
    )

    print(
        "Test embeddings:",
        X_test_text.shape
    )

    # --------------------------------------------------------
    # EMBEDDING CHECKS
    # --------------------------------------------------------

    if X_train_text.shape != (
        TRAIN_LIMIT,
        384
    ):

        raise ValueError(
            "\nTraining embeddings have wrong shape.\n"
            f"Expected: ({TRAIN_LIMIT}, 384)\n"
            f"Found: {X_train_text.shape}"
        )

    if X_val_text.shape[1] != 384:

        raise ValueError(
            "Validation embeddings are not 384-dimensional."
        )

    if X_test_text.shape[1] != 384:

        raise ValueError(
            "Test embeddings are not 384-dimensional."
        )

    # ========================================================
    # LOAD BEHAVIOR FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING BEHAVIOR FEATURES")
    print("=" * 70)

    X_train_behavior, y_train = load_behavior_data(
        TRAIN_CSV,
        TRAIN_LIMIT
    )

    X_val_behavior, y_val = load_behavior_data(
        VAL_CSV
    )

    X_test_behavior, y_test = load_behavior_data(
        TEST_CSV
    )

    # ========================================================
    # ALIGNMENT CHECK
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VERIFYING DATA ALIGNMENT")
    print("=" * 70)

    print(
        "\nTrain text rows     :",
        len(X_train_text)
    )

    print(
        "Train behavior rows :",
        len(X_train_behavior)
    )

    print(
        "Train labels        :",
        len(y_train)
    )

    print(
        "\nValidation text rows     :",
        len(X_val_text)
    )

    print(
        "Validation behavior rows :",
        len(X_val_behavior)
    )

    print(
        "Validation labels        :",
        len(y_val)
    )

    print(
        "\nTest text rows     :",
        len(X_test_text)
    )

    print(
        "Test behavior rows :",
        len(X_test_behavior)
    )

    print(
        "Test labels        :",
        len(y_test)
    )

    if not (
        len(X_train_text)
        == len(X_train_behavior)
        == len(y_train)
    ):

        raise ValueError(
            "\nTraining alignment FAILED."
        )

    if not (
        len(X_val_text)
        == len(X_val_behavior)
        == len(y_val)
    ):

        raise ValueError(
            "\nValidation alignment FAILED."
        )

    if not (
        len(X_test_text)
        == len(X_test_behavior)
        == len(y_test)
    ):

        raise ValueError(
            "\nTest alignment FAILED."
        )

    print(
        "\nAlignment: PASS"
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")

    print(
        "Genuine:",
        np.sum(y_train == 0)
    )

    print(
        "Spam   :",
        np.sum(y_train == 1)
    )

    print("\nValidation:")

    print(
        "Genuine:",
        np.sum(y_val == 0)
    )

    print(
        "Spam   :",
        np.sum(y_val == 1)
    )

    print("\nTest:")

    print(
        "Genuine:",
        np.sum(y_test == 0)
    )

    print(
        "Spam   :",
        np.sum(y_test == 1)
    )

    # ========================================================
    # SCALE BEHAVIOR FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SCALING BEHAVIOR FEATURES")
    print("=" * 70)

    scaler = StandardScaler()

    X_train_behavior = scaler.fit_transform(
        X_train_behavior
    ).astype(np.float32)

    X_val_behavior = scaler.transform(
        X_val_behavior
    ).astype(np.float32)

    X_test_behavior = scaler.transform(
        X_test_behavior
    ).astype(np.float32)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(
        "\nScaler saved:"
    )

    print(
        SCALER_PATH
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

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

    weight_genuine = (
        total_count
        / (
            2.0
            * genuine_count
        )
    )

    weight_spam = (
        total_count
        / (
            2.0
            * spam_count
        )
    )

    class_weights = {
        0: weight_genuine,
        1: weight_spam
    }

    print("\nClass weights:")

    print(
        f"Genuine (0): {weight_genuine:.4f}"
    )

    print(
        f"Spam    (1): {weight_spam:.4f}"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUILDING STRONGER HYBRID MODEL")
    print("=" * 70)

    model = build_model()

    print("\nModel inputs:")

    for tensor in model.inputs:

        print(
            " -",
            tensor.name,
            tensor.shape
        )

    print("\nModel summary:\n")

    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        "\nUsing existing 150,000 MiniLM embeddings."
    )

    print(
        "No new embedding generation is required."
    )

    history = model.fit(

        {
            "minilm_embedding": X_train_text,
            "behavior_features": X_train_behavior
        },

        y_train,

        validation_data=(
            {
                "minilm_embedding": X_val_text,
                "behavior_features": X_val_behavior
            },
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ],

        verbose=1
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    best_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Best model loaded successfully."
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VALIDATION PREDICTIONS")
    print("=" * 70)

    val_probs = best_model.predict(
        {
            "minilm_embedding": X_val_text,
            "behavior_features": X_val_behavior
        },
        batch_size=BATCH_SIZE,
        verbose=1
    ).ravel()

    val_auc = roc_auc_score(
        y_val,
        val_probs
    )

    val_pr_auc = average_precision_score(
        y_val,
        val_probs
    )

    print(
        f"\nValidation ROC-AUC: {val_auc:.4f}"
    )

    print(
        f"Validation PR-AUC : {val_pr_auc:.4f}"
    )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 70)

    threshold_df = search_thresholds(
        y_val,
        val_probs
    )

    # --------------------------------------------------------
    # BEST ACCURACY
    # --------------------------------------------------------

    best_accuracy_row = threshold_df.loc[
        threshold_df["accuracy"].idxmax()
    ]

    best_accuracy_threshold = float(
        best_accuracy_row["threshold"]
    )

    # --------------------------------------------------------
    # BEST F1
    # --------------------------------------------------------

    best_f1_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nTOP 10 THRESHOLDS BY ACCURACY:"
    )

    print(
        threshold_df
        .sort_values(
            "accuracy",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nTOP 10 THRESHOLDS BY F1:"
    )

    print(
        threshold_df
        .sort_values(
            "f1",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nBest validation accuracy threshold:")
    print(
        f"{best_accuracy_threshold:.2f}"
    )

    print(
        f"Validation accuracy: "
        f"{best_accuracy_row['accuracy'] * 100:.2f}%"
    )

    print("\nBest validation F1 threshold:")
    print(
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1_row['f1']:.4f}"
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST PREDICTIONS")
    print("=" * 70)

    test_probs = best_model.predict(
        {
            "minilm_embedding": X_test_text,
            "behavior_features": X_test_behavior
        },
        batch_size=BATCH_SIZE,
        verbose=1
    ).ravel()

    test_auc = roc_auc_score(
        y_test,
        test_probs
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probs
    )

    # ========================================================
    # ACCURACY THRESHOLD TEST
    # ========================================================

    accuracy_result = evaluate(
        y_test,
        test_probs,
        best_accuracy_threshold
    )

    print("\n")
    print("=" * 70)
    print("FINAL TEST - ACCURACY THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold : "
        f"{accuracy_result['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{accuracy_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{accuracy_result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{accuracy_result['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{accuracy_result['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : "
        f"{test_auc:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        accuracy_result["confusion_matrix"]
    )

    # ========================================================
    # F1 THRESHOLD TEST
    # ========================================================

    f1_result = evaluate(
        y_test,
        test_probs,
        best_f1_threshold
    )

    print("\n")
    print("=" * 70)
    print("FINAL TEST - F1 THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold : "
        f"{f1_result['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{f1_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{f1_result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{f1_result['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{f1_result['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : "
        f"{test_auc:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f1_result["confusion_matrix"]
    )

    # ========================================================
    # MAJORITY BASELINE
    # ========================================================

    majority_accuracy = max(
        np.mean(y_test == 0),
        np.mean(y_test == 1)
    )

    difference = (
        accuracy_result["accuracy"]
        - majority_accuracy
    ) * 100

    print("\n")
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"\nMajority baseline : "
        f"{majority_accuracy * 100:.2f}%"
    )

    print(
        f"Model accuracy    : "
        f"{accuracy_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Difference        : "
        f"{difference:.2f} percentage points"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({

        "true_label":
            y_test,

        "spam_probability":
            test_probs,

        "prediction_accuracy_threshold":
            (
                test_probs
                >= best_accuracy_threshold
            ).astype(int),

        "prediction_f1_threshold":
            (
                test_probs
                >= best_f1_threshold
            ).astype(int)
    })

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VERISIGHT STAGE 2 - STRONGER HYBRID MODEL\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "ARCHITECTURE\n"
        )

        f.write(
            "MiniLM 384-D embedding branch\n"
        )

        f.write(
            "40-dimensional behavior branch\n"
        )

        f.write(
            "Dense fusion network\n\n"
        )

        f.write(
            f"Training samples: {TRAIN_LIMIT:,}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Maximum epochs: {EPOCHS}\n"
        )

        f.write(
            f"Learning rate: {LEARNING_RATE}\n\n"
        )

        f.write(
            "VALIDATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"ROC-AUC: {val_auc:.6f}\n"
        )

        f.write(
            f"PR-AUC: {val_pr_auc:.6f}\n"
        )

        f.write(
            f"Best accuracy threshold: "
            f"{best_accuracy_threshold:.2f}\n"
        )

        f.write(
            f"Best F1 threshold: "
            f"{best_f1_threshold:.2f}\n\n"
        )

        f.write(
            "TEST - ACCURACY THRESHOLD\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{accuracy_result['threshold']:.2f}\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy_result['accuracy']:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{accuracy_result['precision']:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{accuracy_result['recall']:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{accuracy_result['f1']:.6f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{test_auc:.6f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{test_pr_auc:.6f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(
                accuracy_result[
                    "confusion_matrix"
                ]
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "TEST - F1 THRESHOLD\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{f1_result['threshold']:.2f}\n"
        )

        f.write(
            f"Accuracy: "
            f"{f1_result['accuracy']:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{f1_result['precision']:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{f1_result['recall']:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{f1_result['f1']:.6f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{test_auc:.6f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{test_pr_auc:.6f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(
                f1_result[
                    "confusion_matrix"
                ]
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "MAJORITY BASELINE\n"
        )

        f.write(
            f"{majority_accuracy:.6f}\n"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nModel:")
    print(
        MODEL_PATH
    )

    print("\nScaler:")
    print(
        SCALER_PATH
    )

    print("\nThreshold table:")
    print(
        THRESHOLD_PATH
    )

    print("\nPredictions:")
    print(
        PREDICTIONS_PATH
    )

    print("\nResults:")
    print(
        RESULTS_PATH
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
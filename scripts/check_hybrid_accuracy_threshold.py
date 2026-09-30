import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\ganes\Downloads\VeriSight"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stage2",
    "stage2_hybrid_fusion.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stage2",
    "stage2_hybrid_behavior_scaler.pkl"
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


# ============================================================
# EXACT 40 BEHAVIOR FEATURES USED BY HYBRID MODEL
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
    "rating_distance_from_product_history",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(csv_path, embedding_path):

    print("\nLoading:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("Rows:", len(df))

    embeddings = np.load(embedding_path)

    print("Embeddings:", embeddings.shape)

    if len(df) != len(embeddings):
        raise ValueError(
            "\nROW MISMATCH!\n"
            f"CSV rows       : {len(df)}\n"
            f"Embedding rows : {len(embeddings)}"
        )

    # Check all required features
    missing_features = [
        col for col in BEHAVIOR_FEATURES
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nMISSING BEHAVIOR FEATURES:\n"
            + "\n".join(missing_features)
        )

    X_behavior = df[BEHAVIOR_FEATURES].copy()

    X_behavior = X_behavior.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_behavior = X_behavior.fillna(0)

    y = df["spam"].astype(int).values

    return embeddings, X_behavior.values, y


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    return accuracy, precision, recall, f1, predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VERISIGHT STAGE 2 - HYBRID THRESHOLD ANALYSIS")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        VAL_CSV,
        TEST_CSV,
        VAL_EMB,
        TEST_EMB
    ]

    print("\nChecking required files...")

    for path in required_files:

        if not os.path.exists(path):

            print("\nERROR: FILE NOT FOUND")
            print(path)

            return

        print("OK:", path)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING HYBRID MODEL")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nModel loaded successfully.")

    # --------------------------------------------------------
    # IMPORTANT:
    # Show the exact input names expected by the model
    # --------------------------------------------------------

    print("\nModel inputs:")

    for input_tensor in model.inputs:

        print(
            " -",
            input_tensor.name,
            input_tensor.shape
        )

    # ========================================================
    # LOAD SCALER
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING BEHAVIOR SCALER")
    print("=" * 70)

    scaler = joblib.load(
        SCALER_PATH
    )

    print("Scaler loaded successfully.")

    # ========================================================
    # VALIDATION DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING VALIDATION DATA")
    print("=" * 70)

    val_embeddings, val_behavior, y_val = load_data(
        VAL_CSV,
        VAL_EMB
    )

    val_behavior = scaler.transform(
        val_behavior
    )

    print("\nValidation distribution:")

    print(
        "Genuine:",
        np.sum(y_val == 0)
    )

    print(
        "Spam   :",
        np.sum(y_val == 1)
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_embeddings, test_behavior, y_test = load_data(
        TEST_CSV,
        TEST_EMB
    )

    test_behavior = scaler.transform(
        test_behavior
    )

    print("\nTest distribution:")

    print(
        "Genuine:",
        np.sum(y_test == 0)
    )

    print(
        "Spam   :",
        np.sum(y_test == 1)
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION PREDICTIONS")
    print("=" * 70)

    # IMPORTANT:
    # These are the exact names expected by your saved model.
    val_probs = model.predict(
        {
            "minilm_embedding": val_embeddings,
            "behavior_features": val_behavior
        },
        batch_size=1024,
        verbose=1
    ).ravel()

    val_auc = roc_auc_score(
        y_val,
        val_probs
    )

    print(
        "\nValidation ROC-AUC:",
        f"{val_auc:.4f}"
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    test_probs = model.predict(
        {
            "minilm_embedding": test_embeddings,
            "behavior_features": test_behavior
        },
        batch_size=1024,
        verbose=1
    ).ravel()

    test_auc = roc_auc_score(
        y_test,
        test_probs
    )

    print(
        "\nTest ROC-AUC:",
        f"{test_auc:.4f}"
    )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print("\n" + "=" * 70)
    print("SEARCHING THRESHOLDS")
    print("=" * 70)

    results = []

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    for threshold in thresholds:

        accuracy, precision, recall, f1, _ = evaluate_threshold(
            y_val,
            val_probs,
            threshold
        )

        results.append([
            round(float(threshold), 2),
            accuracy,
            precision,
            recall,
            f1
        ])

    results_df = pd.DataFrame(
        results,
        columns=[
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    # ========================================================
    # BEST ACCURACY THRESHOLD
    # ========================================================

    best_accuracy_row = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    best_accuracy_threshold = float(
        best_accuracy_row["threshold"]
    )

    # ========================================================
    # BEST F1 THRESHOLD
    # ========================================================

    best_f1_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # ========================================================
    # TOP 15 ACCURACY RESULTS
    # ========================================================

    print(
        "\nTOP 15 THRESHOLDS BY VALIDATION ACCURACY"
    )

    top_accuracy = results_df.sort_values(
        "accuracy",
        ascending=False
    ).head(15)

    print(
        top_accuracy.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # BEST ACCURACY
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"BEST VALIDATION ACCURACY THRESHOLD: "
        f"{best_accuracy_threshold:.2f}"
    )

    print(
        f"Validation Accuracy : "
        f"{best_accuracy_row['accuracy']:.4f} "
        f"({best_accuracy_row['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Validation Precision: "
        f"{best_accuracy_row['precision']:.4f}"
    )

    print(
        f"Validation Recall   : "
        f"{best_accuracy_row['recall']:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{best_accuracy_row['f1']:.4f}"
    )

    # ========================================================
    # BEST F1
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"BEST VALIDATION F1 THRESHOLD: "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Validation Accuracy : "
        f"{best_f1_row['accuracy']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Validation Recall   : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{best_f1_row['f1']:.4f}"
    )

    # ========================================================
    # FINAL TEST USING VALIDATION-SELECTED ACCURACY THRESHOLD
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        f"\nLocked threshold from validation: "
        f"{best_accuracy_threshold:.2f}"
    )

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_predictions
    ) = evaluate_threshold(
        y_test,
        test_probs,
        best_accuracy_threshold
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\nFINAL TEST RESULTS")
    print("-" * 50)

    print(
        f"Threshold : "
        f"{best_accuracy_threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : "
        f"{test_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{test_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(test_cm)

    # ========================================================
    # MAJORITY CLASS BASELINE
    # ========================================================

    majority_accuracy = max(
        np.mean(y_test == 0),
        np.mean(y_test == 1)
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"Majority-class accuracy : "
        f"{majority_accuracy * 100:.2f}%"
    )

    print(
        f"Hybrid model accuracy   : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Difference              : "
        f"{(test_accuracy - majority_accuracy) * 100:.2f} "
        f"percentage points"
    )

    # ========================================================
    # SAVE THRESHOLD TABLE
    # ========================================================

    output_dir = os.path.join(
        BASE_DIR,
        "results",
        "stage2"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    threshold_file = os.path.join(
        output_dir,
        "stage2_hybrid_accuracy_thresholds.csv"
    )

    results_df.to_csv(
        threshold_file,
        index=False
    )

    # ========================================================
    # SAVE FINAL RESULT
    # ========================================================

    final_file = os.path.join(
        output_dir,
        "stage2_hybrid_accuracy_threshold_final.txt"
    )

    with open(
        final_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VERISIGHT STAGE 2 HYBRID THRESHOLD ANALYSIS\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Validation accuracy threshold: "
            f"{best_accuracy_threshold:.2f}\n"
        )

        f.write(
            f"Validation accuracy: "
            f"{best_accuracy_row['accuracy']:.6f}\n"
        )

        f.write(
            f"Validation precision: "
            f"{best_accuracy_row['precision']:.6f}\n"
        )

        f.write(
            f"Validation recall: "
            f"{best_accuracy_row['recall']:.6f}\n"
        )

        f.write(
            f"Validation F1: "
            f"{best_accuracy_row['f1']:.6f}\n\n"
        )

        f.write(
            "BEST F1 THRESHOLD\n"
        )

        f.write("-" * 70 + "\n")

        f.write(
            f"Threshold: "
            f"{best_f1_threshold:.2f}\n"
        )

        f.write(
            f"Validation accuracy: "
            f"{best_f1_row['accuracy']:.6f}\n"
        )

        f.write(
            f"Validation precision: "
            f"{best_f1_row['precision']:.6f}\n"
        )

        f.write(
            f"Validation recall: "
            f"{best_f1_row['recall']:.6f}\n"
        )

        f.write(
            f"Validation F1: "
            f"{best_f1_row['f1']:.6f}\n\n"
        )

        f.write(
            "FINAL TEST RESULT\n"
        )

        f.write("-" * 70 + "\n")

        f.write(
            f"Threshold: "
            f"{best_accuracy_threshold:.2f}\n"
        )

        f.write(
            f"Accuracy: "
            f"{test_accuracy:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{test_precision:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{test_recall:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{test_f1:.6f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{test_auc:.6f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(test_cm)
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        threshold_file
    )

    print(
        final_file
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
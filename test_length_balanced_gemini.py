# ==========================================================
# VeriSight
# Length-Balanced MLP - External Gemini Evaluation
#
# Features:
# 384 MiniLM + 18 Stylometric
#
# IMPORTANT:
# The same StandardScaler used during training is applied
# to the 18 stylometric features.
# ==========================================================

import os
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "external",
    "gemini_unseen_balanced.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "mlp_length_balanced.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "stylometric_scaler.pkl"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "external",
    "length_balanced"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

PREDICTION_FILE = os.path.join(
    RESULT_DIR,
    "gemini_predictions.csv"
)

REPORT_FILE = os.path.join(
    RESULT_DIR,
    "gemini_report.txt"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Length Balanced MLP")
    print("External Gemini Evaluation")
    print("=" * 70)


    # ======================================================
    # STEP 1 - Load Dataset
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 - Loading Gemini Dataset")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "\nDataset Shape:",
        df.shape
    )

    print(
        "\nLabel Distribution:"
    )

    print(
        df["label"].value_counts()
    )

    print(
        "\nGenerator Distribution:"
    )

    if "generator_model" in df.columns:

        print(
            df[
                "generator_model"
            ].value_counts()
        )


    # ======================================================
    # Prepare Reviews
    # ======================================================

    df["review"] = (
        df["review"]
        .astype(str)
        .str.strip()
    )

    texts = df[
        "review"
    ].tolist()

    y_true = df[
        "label"
    ].astype(int).values

    print(
        f"\nTotal Reviews: "
        f"{len(texts)}"
    )


    # ======================================================
    # STEP 2 - MiniLM
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 - MiniLM Embeddings")
    print("=" * 70)

    print(
        "\nLoading MiniLM..."
    )

    encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "\nGenerating embeddings..."
    )

    X_semantic = encoder.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True
    )

    X_semantic = X_semantic.astype(
        np.float32
    )

    print(
        "\nMiniLM Shape:",
        X_semantic.shape
    )


    # ======================================================
    # STEP 3 - Stylometric Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 - Stylometric Features")
    print("=" * 70)

    # Import your existing feature extractor
    from features.stylometric_features import (
        extract_stylometric_features
    )

    style_features = []

    for i, review in enumerate(texts):

        features = extract_stylometric_features(
            review
        )

        style_features.append(
            features
        )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1:,} / "
                f"{len(texts):,}"
            )


    style_df = pd.DataFrame(
        style_features
    )

    X_style = style_df.values.astype(
        np.float32
    )

    print(
        "\nStylometric Shape:",
        X_style.shape
    )


    # ======================================================
    # STEP 4 - Verify Feature Count
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 4 - Feature Fusion")
    print("=" * 70)

    print(
        "\nMiniLM Features:",
        X_semantic.shape[1]
    )

    print(
        "Stylometric Features:",
        X_style.shape[1]
    )

    if X_semantic.shape[1] != 384:

        raise ValueError(
            "MiniLM feature count is not 384."
        )

    if X_style.shape[1] != 18:

        raise ValueError(
            "Stylometric feature count is not 18."
        )


    # ======================================================
    # STEP 5 - Apply Training Scaler
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 5 - Applying Training Scaler")
    print("=" * 70)

    if not os.path.exists(SCALER_FILE):

        raise FileNotFoundError(
            f"\nScaler not found:\n{SCALER_FILE}"
        )

    scaler = joblib.load(
        SCALER_FILE
    )

    print(
        "\n✓ Training scaler loaded."
    )

    # IMPORTANT:
    # Do NOT fit the scaler again.
    #
    # Use transform() only.

    X_style_scaled = scaler.transform(
        X_style
    )

    print(
        "✓ Gemini stylometric features scaled."
    )


    # ======================================================
    # STEP 6 - Combine
    # ======================================================

    X_fused = np.concatenate(

        [
            X_semantic,
            X_style_scaled
        ],

        axis=1
    )

    print(
        "\nFinal Feature Shape:",
        X_fused.shape
    )

    if X_fused.shape[1] != 402:

        raise ValueError(
            f"Expected 402 features, "
            f"got {X_fused.shape[1]}"
        )

    print(
        "✓ 402 features verified."
    )


    # ======================================================
    # STEP 7 - Validation
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 6 - Data Validation")
    print("=" * 70)

    print(
        "\nNaN values:",
        np.isnan(X_fused).sum()
    )

    print(
        "Infinite values:",
        np.isinf(X_fused).sum()
    )


    # ======================================================
    # STEP 8 - Load Model
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 7 - Loading Length-Balanced MLP")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_FILE}"
        )

    model = load_model(
        MODEL_FILE
    )

    print(
        "\n✓ Length-balanced MLP loaded."
    )


    # ======================================================
    # STEP 9 - Prediction
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 8 - Predicting Gemini Dataset")
    print("=" * 70)

    probabilities = model.predict(

        X_fused,

        batch_size=64,

        verbose=1
    ).ravel()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # ======================================================
    # STEP 10 - Metrics
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    report = classification_report(

        y_true,

        predictions,

        labels=[
            0,
            1
        ],

        target_names=[
            "Human",
            "AI"
        ],

        digits=4,

        zero_division=0
    )

    cm = confusion_matrix(

        y_true,

        predictions,

        labels=[
            0,
            1
        ]
    )


    # ======================================================
    # Human / Gemini Detection
    # ======================================================

    human_mask = (
        y_true == 0
    )

    gemini_mask = (
        y_true == 1
    )

    human_correct = np.sum(
        predictions[human_mask] == 0
    )

    gemini_correct = np.sum(
        predictions[gemini_mask] == 1
    )

    human_total = np.sum(
        human_mask
    )

    gemini_total = np.sum(
        gemini_mask
    )

    human_accuracy = (
        human_correct /
        human_total
    )

    gemini_detection = (
        gemini_correct /
        gemini_total
    )


    # ======================================================
    # Probability Statistics
    # ======================================================

    human_probs = (
        probabilities[human_mask]
    )

    gemini_probs = (
        probabilities[gemini_mask]
    )


    # ======================================================
    # Print Results
    # ======================================================

    print("\n" + "=" * 70)
    print("LENGTH-BALANCED GEMINI RESULTS")
    print("=" * 70)

    print(
        f"\nOverall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"\nHuman Accuracy: "
        f"{human_accuracy * 100:.2f}%"
    )

    print(
        f"\nGemini Detection Rate: "
        f"{gemini_detection * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


    # ======================================================
    # Probability Analysis
    # ======================================================

    print("\n" + "=" * 70)
    print("PROBABILITY ANALYSIS")
    print("=" * 70)

    print(
        "\nHuman:"
    )

    print(
        f"Mean AI Probability: "
        f"{human_probs.mean():.4f}"
    )

    print(
        f"Median AI Probability: "
        f"{np.median(human_probs):.4f}"
    )

    print(
        "\nGemini:"
    )

    print(
        f"Mean AI Probability: "
        f"{gemini_probs.mean():.4f}"
    )

    print(
        f"Median AI Probability: "
        f"{np.median(gemini_probs):.4f}"
    )


    # ======================================================
    # Save Predictions
    # ======================================================

    output_df = df.copy()

    output_df[
        "ai_probability"
    ] = probabilities

    output_df[
        "prediction"
    ] = predictions

    output_df[
        "prediction_text"
    ] = np.where(

        predictions == 1,

        "AI Generated",

        "Human Written"
    )

    output_df.to_csv(

        PREDICTION_FILE,

        index=False
    )


    # ======================================================
    # Save Report
    # ======================================================

    with open(

        REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "VeriSight - Length Balanced Gemini Evaluation\n"
        )

        f.write(
            "=" * 65 + "\n\n"
        )

        f.write(
            "Model: mlp_length_balanced.keras\n"
        )

        f.write(
            "Features: 384 MiniLM + 18 Stylometric\n"
        )

        f.write(
            "Stylometric preprocessing: "
            "Training StandardScaler\n\n"
        )

        f.write(
            f"Overall Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Overall Accuracy (%): "
            f"{accuracy * 100:.2f}%\n\n"
        )

        f.write(
            f"Human Accuracy: "
            f"{human_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Gemini Detection Rate: "
            f"{gemini_detection * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(
            report
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nProbability Statistics:\n"
        )

        f.write(
            f"Human Mean: "
            f"{human_probs.mean():.4f}\n"
        )

        f.write(
            f"Human Median: "
            f"{np.median(human_probs):.4f}\n"
        )

        f.write(
            f"Gemini Mean: "
            f"{gemini_probs.mean():.4f}\n"
        )

        f.write(
            f"Gemini Median: "
            f"{np.median(gemini_probs):.4f}\n"
        )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("EXTERNAL GEMINI EVALUATION COMPLETED")
    print("=" * 70)

    print(
        "\nPredictions:"
    )

    print(
        PREDICTION_FILE
    )

    print(
        "\nReport:"
    )

    print(
        REPORT_FILE
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()
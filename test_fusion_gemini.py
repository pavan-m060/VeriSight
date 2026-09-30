# ==========================================================
# VeriSight - Balanced Unseen Gemini Evaluation
# MiniLM (384) + Stylometry (18) = 402 Features
# ==========================================================

import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features.stylometric_features import (
    extract_stylometric_features
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_unseen_balanced.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "mlp_fusion_stage1.keras"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "external"
    / "fusion"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    RESULT_DIR
    / "gemini_unseen_balanced_report.txt"
)

PREDICTION_FILE = (
    RESULT_DIR
    / "gemini_unseen_balanced_predictions.csv"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Balanced Unseen Gemini Evaluation")
    print("=" * 70)


    # ======================================================
    # STEP 1 - Load Dataset
    # ======================================================

    print("\nLoading balanced Gemini dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(
        f"\nDataset Shape: {df.shape}"
    )

    print("\nLabel Distribution:")
    print(df["label"].value_counts())

    print("\nGenerator Distribution:")
    print(df["generator_model"].value_counts())


    # ======================================================
    # STEP 2 - Prepare Data
    # ======================================================

    df["review"] = (
        df["review"]
        .astype(str)
        .str.strip()
    )

    texts = df["review"].tolist()

    y_true = (
        df["label"]
        .astype(int)
        .values
    )

    print(
        f"\nTotal Reviews: {len(texts)}"
    )


    # ======================================================
    # STEP 3 - Load MiniLM
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 - MiniLM Embeddings")
    print("=" * 70)

    encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("\nGenerating MiniLM embeddings...")

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
    # STEP 4 - Stylometric Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 - Stylometric Features")
    print("=" * 70)

    style_features = []

    for i, review in enumerate(texts):

        features = extract_stylometric_features(
            review
        )

        style_features.append(features)

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1:,} / {len(texts):,}"
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
    # STEP 5 - Feature Fusion
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 - Feature Fusion")
    print("=" * 70)

    X_fused = np.concatenate(
        [
            X_semantic,
            X_style
        ],
        axis=1
    )

    print(
        "\nMiniLM Features:",
        X_semantic.shape[1]
    )

    print(
        "Stylometric Features:",
        X_style.shape[1]
    )

    print(
        "Total Features:",
        X_fused.shape[1]
    )


    # ======================================================
    # Verify Feature Count
    # ======================================================

    if X_fused.shape[1] != 402:

        raise ValueError(
            f"Expected 402 features, "
            f"got {X_fused.shape[1]}"
        )

    print(
        "\n✓ 402 feature shape verified"
    )


    # ======================================================
    # STEP 6 - Load Model
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 4 - Loading Fusion MLP")
    print("=" * 70)

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_FILE}"
        )

    model = load_model(
        MODEL_FILE
    )

    print(
        "\n✓ Fusion MLP loaded"
    )


    # ======================================================
    # STEP 7 - Prediction
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 5 - Prediction")
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
    # STEP 8 - Accuracy
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    print("\n" + "=" * 70)
    print("UNSEEN GEMINI RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


    # ======================================================
    # STEP 9 - Classification Report
    # ======================================================

    report = classification_report(

        y_true,

        predictions,

        labels=[0, 1],

        target_names=[
            "Human",
            "AI"
        ],

        digits=4,

        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(report)


    # ======================================================
    # STEP 10 - Confusion Matrix
    # ======================================================

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)


    # ======================================================
    # STEP 11 - Separate Human / Gemini Results
    # ======================================================

    human_mask = (
        y_true == 0
    )

    gemini_mask = (
        y_true == 1
    )

    human_accuracy = np.mean(
        predictions[human_mask] == 0
    )

    gemini_detection_rate = np.mean(
        predictions[gemini_mask] == 1
    )

    print(
        "\nHuman Correctly Identified:"
    )

    print(
        f"{np.sum(predictions[human_mask] == 0)} "
        f"/ {np.sum(human_mask)}"
    )

    print(
        f"Human Accuracy: "
        f"{human_accuracy * 100:.2f}%"
    )

    print(
        "\nGemini AI Correctly Detected:"
    )

    print(
        f"{np.sum(predictions[gemini_mask] == 1)} "
        f"/ {np.sum(gemini_mask)}"
    )

    print(
        f"Gemini Detection Rate: "
        f"{gemini_detection_rate * 100:.2f}%"
    )


    # ======================================================
    # STEP 12 - Probability Analysis
    # ======================================================

    human_probabilities = (
        probabilities[human_mask]
    )

    gemini_probabilities = (
        probabilities[gemini_mask]
    )

    print("\n" + "=" * 70)
    print("PROBABILITY ANALYSIS")
    print("=" * 70)

    print(
        "\nHuman Reviews:"
    )

    print(
        f"Mean AI Probability: "
        f"{human_probabilities.mean():.4f}"
    )

    print(
        f"Median AI Probability: "
        f"{np.median(human_probabilities):.4f}"
    )

    print(
        "\nGemini Reviews:"
    )

    print(
        f"Mean AI Probability: "
        f"{gemini_probabilities.mean():.4f}"
    )

    print(
        f"Median AI Probability: "
        f"{np.median(gemini_probabilities):.4f}"
    )


    # ======================================================
    # STEP 13 - Save Predictions
    # ======================================================

    results = df.copy()

    results[
        "ai_probability"
    ] = probabilities

    results[
        "prediction"
    ] = predictions

    results[
        "prediction_text"
    ] = np.where(
        predictions == 1,
        "AI Generated",
        "Human Written"
    )

    results.to_csv(
        PREDICTION_FILE,
        index=False
    )

    print(
        "\nPredictions saved:"
    )

    print(
        PREDICTION_FILE
    )


    # ======================================================
    # STEP 14 - Save Report
    # ======================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VeriSight - Unseen Gemini Evaluation\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Features: "
            "384 MiniLM + 18 Stylometric = 402\n\n"
        )

        f.write(
            f"Dataset Size: "
            f"{len(df)}\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Accuracy (%): "
            f"{accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(report)

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nHuman Accuracy: "
            f"{human_accuracy * 100:.2f}%\n"
        )

        f.write(
            "Gemini Detection Rate: "
            f"{gemini_detection_rate * 100:.2f}%\n"
        )

        f.write(
            "\nProbability Analysis:\n"
        )

        f.write(
            f"Human Mean: "
            f"{human_probabilities.mean():.4f}\n"
        )

        f.write(
            f"Human Median: "
            f"{np.median(human_probabilities):.4f}\n"
        )

        f.write(
            f"Gemini Mean: "
            f"{gemini_probabilities.mean():.4f}\n"
        )

        f.write(
            f"Gemini Median: "
            f"{np.median(gemini_probabilities):.4f}\n"
        )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("UNSEEN GEMINI EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nFinal Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Human Accuracy: "
        f"{human_accuracy * 100:.2f}%"
    )

    print(
        f"Gemini Detection: "
        f"{gemini_detection_rate * 100:.2f}%"
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
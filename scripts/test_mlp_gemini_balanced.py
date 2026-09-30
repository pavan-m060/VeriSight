import pandas as pd
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_external_balanced.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "mlp_stage1_v2.keras"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 65)
    print("VeriSight - MLP Gemini External Evaluation")
    print("=" * 65)

    # ------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print("\nGemini Dataset Shape:")
    print(df.shape)

    # ------------------------------------------------------
    # Load Sentence Transformer
    # ------------------------------------------------------

    print("\nLoading SentenceTransformer...")

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ------------------------------------------------------
    # Generate embeddings
    # ------------------------------------------------------

    print("\nGenerating Gemini Embeddings...")

    X = encoder.encode(
        df["review"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("\nEmbedding Shape:")
    print(X.shape)

    # ------------------------------------------------------
    # Load trained MLP
    # ------------------------------------------------------

    print("\nLoading trained MLP...")

    model = load_model(MODEL_FILE)

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    print("\nPredicting...")

    probabilities = model.predict(
        X,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------

    y_true = df["label"].astype(int).values

    # ------------------------------------------------------
    # Detection rate
    # ------------------------------------------------------

    detection_rate = np.mean(
        predictions == 1
    ) * 100

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("GEMINI EXTERNAL RESULTS")
    print("=" * 65)

    print(
        f"\nGemini AI Detection Rate: "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Gemini Reviews Correctly "
        f"Detected as AI: "
        f"{np.sum(predictions == 1)} / {len(predictions)}"
    )

    # ------------------------------------------------------
    # Classification report
    # ------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Human",
                "AI"
            ],
            zero_division=0
        )
    )

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(cm)

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------

    output_dir = (
        BASE_DIR
        / "results"
        / "external"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df = df.copy()

    result_df["ai_probability"] = probabilities

    result_df["prediction"] = predictions

    result_df["prediction_text"] = np.where(
        predictions == 1,
        "AI Generated",
        "Human Written"
    )

    prediction_file = (
        output_dir
        / "mlp_gemini_balanced_predictions.csv"
    )

    result_df.to_csv(
        prediction_file,
        index=False
    )

    # ------------------------------------------------------
    # Save report
    # ------------------------------------------------------

    report_file = (
        output_dir
        / "mlp_gemini_balanced_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VeriSight - MLP Gemini External Evaluation\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Dataset Size: {len(df)}\n"
        )

        f.write(
            f"AI Detection Rate: "
            f"{detection_rate:.2f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm) + "\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(
            classification_report(
                y_true,
                predictions,
                target_names=[
                    "Human",
                    "AI"
                ],
                zero_division=0
            )
        )

    # ------------------------------------------------------
    # Complete
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("External Evaluation Completed")
    print("=" * 65)

    print("\nPredictions saved to:")
    print(prediction_file)

    print("\nReport saved to:")
    print(report_file)


if __name__ == "__main__":
    main()
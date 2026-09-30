import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model


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

MODEL_DIR = BASE_DIR / "models"

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "external"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Models
# ==========================================================

MODELS = {
    "MLP": MODEL_DIR / "mlp_stage1_v2.keras",
    "CNN": MODEL_DIR / "cnn_stage1_v2.keras",
    "LSTM": MODEL_DIR / "lstm_stage1_v2.keras",
    "GRU": MODEL_DIR / "gru_stage1_v2.keras",
    "BiLSTM": MODEL_DIR / "bilstm_stage1_v2.keras",
}


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Cross-Generator Gemini Evaluation")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Gemini dataset
    # ------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print("\nGemini Dataset:")
    print(f"Reviews : {len(df)}")

    # ------------------------------------------------------
    # Load SentenceTransformer
    # ------------------------------------------------------

    print("\nLoading SentenceTransformer...")

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ------------------------------------------------------
    # Generate embeddings ONCE
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
    # Results
    # ------------------------------------------------------

    results = []

    detailed_predictions = df.copy()

    # ------------------------------------------------------
    # Test each model
    # ------------------------------------------------------

    for model_name, model_path in MODELS.items():

        print("\n" + "=" * 70)
        print(f"Testing {model_name}")
        print("=" * 70)

        if not model_path.exists():

            print(
                f"WARNING: Model not found:\n"
                f"{model_path}"
            )

            continue

        print("Loading model...")

        model = load_model(model_path)

        # --------------------------------------------------
        # Prepare input
        # --------------------------------------------------

        model_input = X

        # CNN expects (samples, 384, 1)
        if model_name == "CNN":

            model_input = X.reshape(
                X.shape[0],
                X.shape[1],
                1
            )

        # LSTM / GRU / BiLSTM
        #
        # IMPORTANT:
        # These models were trained using the same sequence
        # representation as their Stage-1 training scripts.
        #
        # If your recurrent models were trained directly on
        # embeddings with shape (384, 1), use this shape.
        elif model_name in ["LSTM", "GRU", "BiLSTM"]:

            model_input = X.reshape(
                X.shape[0],
                X.shape[1],
                1
            )

        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        probabilities = model.predict(
            model_input,
            verbose=0
        ).flatten()

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        # --------------------------------------------------
        # Gemini detection
        # --------------------------------------------------

        detected = np.sum(
            predictions == 1
        )

        detection_rate = (
            detected / len(predictions)
        ) * 100

        mean_probability = (
            probabilities.mean()
        )

        median_probability = (
            np.median(probabilities)
        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        results.append({
            "model": model_name,
            "total_gemini_reviews": len(df),
            "detected_as_ai": int(detected),
            "classified_as_human": int(
                len(predictions) - detected
            ),
            "ai_detection_rate": detection_rate,
            "mean_ai_probability": mean_probability,
            "median_ai_probability": median_probability
        })

        # --------------------------------------------------
        # Add detailed predictions
        # --------------------------------------------------

        detailed_predictions[
            f"{model_name}_ai_probability"
        ] = probabilities

        detailed_predictions[
            f"{model_name}_prediction"
        ] = np.where(
            predictions == 1,
            "AI Generated",
            "Human Written"
        )

        print(
            f"\nGemini AI Detection Rate: "
            f"{detection_rate:.2f}%"
        )

        print(
            f"Detected as AI: "
            f"{detected} / {len(df)}"
        )

        print(
            f"Classified as Human: "
            f"{len(df) - detected} / {len(df)}"
        )

        print(
            f"Mean AI Probability: "
            f"{mean_probability:.4f}"
        )

        print(
            f"Median AI Probability: "
            f"{median_probability:.4f}"
        )

    # ======================================================
    # Comparison
    # ======================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("GEMINI CROSS-GENERATOR COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "detected_as_ai",
                "classified_as_human",
                "ai_detection_rate",
                "mean_ai_probability",
                "median_ai_probability"
            ]
        ].to_string(index=False)
    )

    # ======================================================
    # Save comparison
    # ======================================================

    comparison_file = (
        RESULT_DIR
        / "gemini_all_models_comparison.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )

    # ======================================================
    # Save detailed predictions
    # ======================================================

    detailed_file = (
        RESULT_DIR
        / "gemini_all_models_predictions.csv"
    )

    detailed_predictions.to_csv(
        detailed_file,
        index=False
    )

    # ======================================================
    # Save text report
    # ======================================================

    report_file = (
        RESULT_DIR
        / "gemini_all_models_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VeriSight - Gemini Cross-Generator Evaluation\n"
        )

        f.write("=" * 65 + "\n\n")

        f.write(
            f"Gemini Reviews: {len(df)}\n\n"
        )

        f.write(
            results_df.to_string(index=False)
        )

        f.write("\n")

    # ======================================================
    # Complete
    # ======================================================

    print("\n" + "=" * 70)
    print("Evaluation Completed")
    print("=" * 70)

    print("\nComparison saved to:")
    print(comparison_file)

    print("\nDetailed predictions saved to:")
    print(detailed_file)

    print("\nReport saved to:")
    print(report_file)


if __name__ == "__main__":
    main()
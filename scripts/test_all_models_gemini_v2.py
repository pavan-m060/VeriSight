import numpy as np
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_external_balanced.csv"
)

MODEL_DIR = BASE_DIR / "models"

RESULT_DIR = BASE_DIR / "results" / "external"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "MLP": MODEL_DIR / "mlp_stage1_v2.keras",
    "CNN": MODEL_DIR / "cnn_stage1_v2.keras",
    "LSTM": MODEL_DIR / "lstm_stage1_v2.keras",
    "GRU": MODEL_DIR / "gru_stage1_v2.keras",
    "BiLSTM": MODEL_DIR / "bilstm_stage1_v2.keras",
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VeriSight - All Models Gemini External Evaluation")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nGemini Dataset Shape:")
print(df.shape)

reviews = df["review"].astype(str).tolist()

# All Gemini reviews have label 1 = AI
y_true = df["label"].astype(int).values

# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading SentenceTransformer...")

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating Gemini Embeddings...")

X = encoder.encode(
    reviews,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding Shape:")
print(X.shape)

# ============================================================
# PREPARE INPUTS
# ============================================================

# CNN/LSTM/GRU/BiLSTM expect 3D input
X_sequence = X.reshape(X.shape[0], X.shape[1], 1)

# ============================================================
# RESULTS
# ============================================================

results = []

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    if not model_path.exists():
        print(f"ERROR: Model not found:")
        print(model_path)
        continue

    print("Loading model...")

    model = load_model(model_path)

    # --------------------------------------------------------
    # Select correct input format
    # --------------------------------------------------------

    if model_name == "MLP":
        X_input = X
    else:
        X_input = X_sequence

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("Predicting...")

    probabilities = model.predict(
        X_input,
        verbose=0
    ).reshape(-1)

    predictions = (probabilities >= 0.5).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    ai_detected = int(np.sum(predictions == 1))

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nResults:")
    print(f"Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    print("\nGemini AI Reviews Detected:")
    print(f"{ai_detected} / {len(y_true)}")

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    output_df = df.copy()

    output_df["ai_probability"] = probabilities
    output_df["prediction"] = predictions

    output_df["prediction_text"] = np.where(
        predictions == 1,
        "AI Generated",
        "Human Written"
    )

    prediction_file = (
        RESULT_DIR
        / f"{model_name.lower()}_gemini_v2_predictions.csv"
    )

    output_df.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ai_detected": ai_detected,
        "total": len(y_true),
    })

# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL GEMINI EXTERNAL COMPARISON")
print("=" * 70)

if len(results_df) > 0:

    results_display = results_df.copy()

    results_display["accuracy"] = (
        results_display["accuracy"] * 100
    ).round(2)

    results_display["precision"] = (
        results_display["precision"] * 100
    ).round(2)

    results_display["recall"] = (
        results_display["recall"] * 100
    ).round(2)

    results_display["f1_score"] = (
        results_display["f1_score"] * 100
    ).round(2)

    results_display = results_display.rename(
        columns={
            "accuracy": "Accuracy %",
            "precision": "Precision %",
            "recall": "Recall %",
            "f1_score": "F1 %",
            "ai_detected": "AI Detected",
            "total": "Total"
        }
    )

    print(
        results_display[
            [
                "model",
                "Accuracy %",
                "Precision %",
                "Recall %",
                "F1 %",
                "AI Detected",
                "Total"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save final comparison
    # --------------------------------------------------------

    comparison_file = (
        RESULT_DIR
        / "all_models_gemini_v2_comparison.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )

    print("\nComparison saved to:")
    print(comparison_file)

print("\n")
print("=" * 70)
print("External Gemini Evaluation Completed")
print("=" * 70)
import numpy as np
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "external" / "ai_generated_reviews_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "mlp_stage1_v2.keras"

RESULT_DIR = BASE_DIR / "results" / "external"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Gemini Dataset
# ==========================================================

print("=" * 65)
print("VeriSight - MLP External Gemini Evaluation")
print("=" * 65)

df = pd.read_csv(DATA_FILE)

texts = df["review"].astype(str).tolist()
y_true = df["label"].astype(int).values

print("\nGemini Dataset Shape:", df.shape)

# ==========================================================
# Load Sentence Transformer
# ==========================================================

print("\nLoading SentenceTransformer...")

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# Generate Embeddings
# ==========================================================

print("\nGenerating Gemini Embeddings...")

X = encoder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding Shape:", X.shape)

# ==========================================================
# Load MLP
# ==========================================================

print("\nLoading MLP...")

model = load_model(MODEL_FILE)

# ==========================================================
# Predict
# ==========================================================

print("\nPredicting...")

probabilities = model.predict(
    X,
    batch_size=64,
    verbose=1
).ravel()

predictions = (probabilities >= 0.5).astype(int)

# ==========================================================
# Results
# ==========================================================

print("\n" + "=" * 65)
print("EXTERNAL GEMINI RESULTS")
print("=" * 65)

accuracy = np.mean(predictions == y_true)

print(f"\nGemini AI Detection Accuracy: {accuracy:.4f}")
print(f"Gemini AI Detection Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

report = classification_report(
    y_true,
    predictions,
    target_names=["Human", "AI"],
    labels=[0, 1],
    zero_division=0,
    digits=4
)

print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_true, predictions, labels=[0, 1]))

# ==========================================================
# Gemini Detection Rate
# ==========================================================

detected_as_ai = np.sum(predictions == 1)

print("\nGemini Reviews Correctly Detected as AI:")
print(f"{detected_as_ai} / {len(y_true)}")

print(
    f"Detection Rate: "
    f"{detected_as_ai / len(y_true) * 100:.2f}%"
)

# ==========================================================
# Save Detailed Predictions
# ==========================================================

results = df.copy()

results["ai_probability"] = probabilities
results["prediction"] = predictions

results["prediction_text"] = results["prediction"].map({
    0: "Human Written",
    1: "AI Generated"
})

OUTPUT_FILE = RESULT_DIR / "mlp_gemini_predictions.csv"

results.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# Save Report
# ==========================================================

REPORT_FILE = RESULT_DIR / "mlp_gemini_report.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("VeriSight - MLP External Gemini Evaluation\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset Size: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Accuracy (%): {accuracy * 100:.2f}%\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(
        str(
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            )
        )
    )

    f.write("\n\nGemini AI Detection Rate:\n")
    f.write(
        f"{detected_as_ai / len(y_true) * 100:.2f}%\n"
    )

print("\n" + "=" * 65)
print("External Evaluation Completed")
print("=" * 65)

print("\nPredictions saved to:")
print(OUTPUT_FILE)

print("\nReport saved to:")
print(REPORT_FILE)
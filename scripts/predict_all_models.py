# ==========================================================
# VeriSight
# Compare All Trained Models
# ==========================================================

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

print("=" * 65)
print("        VeriSight - AI Review Detection")
print("=" * 65)

review = input("\nEnter Review:\n\n")

print("\nLoading Sentence Transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

embedding = embedder.encode([review], convert_to_numpy=True)

# ----------------------------------------------------------
# Model Information
# ----------------------------------------------------------

models = {
    "MLP": {
        "path": "models/mlp_stage1.keras",
        "reshape": False
    },
    "CNN": {
        "path": "models/cnn_stage1.keras",
        "reshape": True
    },
    "LSTM": {
        "path": "models/lstm_stage1.keras",
        "reshape": True
    },
    "GRU": {
        "path": "models/gru_stage1.keras",
        "reshape": True
    },
    "BiLSTM": {
        "path": "models/bilstm_stage1.keras",
        "reshape": True
    }
}

print("\nLoading Models...\n")

print("=" * 65)
print(f"{'Model':<12}{'Prediction':<20}{'Confidence'}")
print("=" * 65)

for model_name, info in models.items():

    if not os.path.exists(info["path"]):
        print(f"{model_name:<12} Model Not Found")
        continue

    model = load_model(info["path"])

    if info["reshape"]:
        x = embedding.reshape((1, 384, 1))
    else:
        x = embedding

    probability = model.predict(x, verbose=0)[0][0]

    if probability >= 0.5:
        prediction = "AI Generated"
        confidence = probability
    else:
        prediction = "Human Written"
        confidence = 1 - probability

    print(f"{model_name:<12}{prediction:<20}{confidence*100:.2f}%")

print("=" * 65)
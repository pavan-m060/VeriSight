import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================

DATA_FILE = "data/stage1/stage1_final_v2.csv"

X_OUTPUT = "embeddings/X_stage1_v2.npy"
Y_OUTPUT = "embeddings/y_stage1_v2.npy"

os.makedirs("embeddings", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Final Stage 1 Dataset")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset Shape :", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())

# ==========================================================
# Prepare Text and Labels
# ==========================================================

texts = df["review"].astype(str).tolist()
labels = df["label"].astype(int).values

# ==========================================================
# Load SentenceTransformer
# ==========================================================

print("\nLoading SentenceTransformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# Generate Embeddings
# ==========================================================

print("\nGenerating Embeddings...")

X = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ==========================================================
# Save
# ==========================================================

np.save(X_OUTPUT, X)
np.save(Y_OUTPUT, labels)

print("\n" + "=" * 60)
print("Embedding Generation Completed")
print("=" * 60)

print("Embedding Shape :", X.shape)
print("Labels Shape    :", labels.shape)

print("\nSaved:")
print(X_OUTPUT)
print(Y_OUTPUT)
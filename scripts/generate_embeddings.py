# ==========================================================
# VeriSight
# Stage 1 - Generate Sentence Embeddings
# ==========================================================

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================

DATASET = "data/stage1/stage1_final.csv"

OUTPUT_DIR = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_FILE = os.path.join(OUTPUT_DIR, "X_stage1.npy")
Y_FILE = os.path.join(OUTPUT_DIR, "y_stage1.npy")

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Stage 1 Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET)

print(f"Dataset Shape : {df.shape}")

reviews = df["review"].astype(str).tolist()
labels = df["label"].values

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
    reviews,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

y = np.array(labels)

# ==========================================================
# Save
# ==========================================================

np.save(X_FILE, X)
np.save(Y_FILE, y)

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Embedding Generation Completed")
print("=" * 60)

print(f"Embedding Shape : {X.shape}")
print(f"Labels Shape    : {y.shape}")

print(f"\nSaved:")
print(f"  {X_FILE}")
print(f"  {Y_FILE}")

print("\nClass Distribution:")
print(df["label"].value_counts())
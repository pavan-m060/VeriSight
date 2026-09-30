import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "stage1" / "stage1_final_v3.csv"

OUTPUT_DIR = PROJECT_ROOT / "embeddings" / "stage1_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_FILE = OUTPUT_DIR / "X_stage1_v3.npy"
Y_FILE = OUTPUT_DIR / "y_stage1_v3.npy"

print("=" * 70)
print("VERISIGHT - STAGE 1 V3 MINILM EMBEDDINGS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

reviews = df["review"].astype(str).tolist()
labels = df["label"].astype(int).to_numpy()

print("\nLabels:")
print(pd.Series(labels).value_counts().sort_index())

print("\nLoading MiniLM...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating embeddings...")

X = model.encode(
    reviews,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding shape:", X.shape)
print("Label shape:", labels.shape)

np.save(X_FILE, X)
np.save(Y_FILE, labels)

print("\nSaved:")
print(X_FILE)
print(Y_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
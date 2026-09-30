import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HUMAN_FILE = PROJECT_ROOT / "data" / "processed" / "human_reviews.csv"
AI_FILE = PROJECT_ROOT / "data" / "processed" / "ai_reviews.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "stage1_final.csv"

# ==========================================================
# Load datasets
# ==========================================================

human = pd.read_csv(HUMAN_FILE)
ai = pd.read_csv(AI_FILE)

print(f"Human Reviews : {len(human):,}")
print(f"AI Reviews    : {len(ai):,}")

# ==========================================================
# Keep only review column
# ==========================================================

human = human[["review"]].copy()
ai = ai[["review"]].copy()

# ==========================================================
# Labels
# ==========================================================

human["label"] = 0      # Human
ai["label"] = 1         # AI

# ==========================================================
# Merge
# ==========================================================

df = pd.concat([human, ai], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================================
# Save
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ==========================================================
# Summary
# ==========================================================

print("\nStage 1 Dataset Created Successfully!")
print("-" * 40)
print(f"Total Reviews : {len(df):,}")
print(f"Human Reviews : {(df['label']==0).sum():,}")
print(f"AI Reviews    : {(df['label']==1).sum():,}")
print(f"\nSaved to:\n{OUTPUT_FILE}")
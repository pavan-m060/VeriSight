import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HUMAN_FILE = BASE_DIR / "data" / "processed" / "human_reviews.csv"
OLD_AI_FILE = BASE_DIR / "data" / "processed" / "ai_reviews_original.csv"
NEW_AI_FILE = BASE_DIR / "data" / "processed" / "ai_reviews_v2_clean.csv"

OUTPUT_DIR = BASE_DIR / "data" / "stage1"
OUTPUT_FILE = OUTPUT_DIR / "stage1_final_v2.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load datasets
# ==========================================================

print("=" * 65)
print("Creating Final Stage 1 Dataset")
print("=" * 65)

human = pd.read_csv(HUMAN_FILE)
old_ai = pd.read_csv(OLD_AI_FILE)
new_ai = pd.read_csv(NEW_AI_FILE)

print(f"\nHuman dataset : {len(human):,}")
print(f"Old AI dataset: {len(old_ai):,}")
print(f"New AI dataset: {len(new_ai):,}")

# ==========================================================
# Prepare HUMAN
# ==========================================================

human_final = pd.DataFrame({
    "review": human["review"],
    "label": 0,
    "type": "Human",
    "source": human["source"],
    "generator_model": "Human"
})

# ==========================================================
# Prepare OLD AI
# ==========================================================

old_ai_final = pd.DataFrame({
    "review": old_ai["review"],
    "label": 1,
    "type": "AI",
    "source": old_ai["source"],
    "generator_model": old_ai["generator_model"]
        if "generator_model" in old_ai.columns
        else "Old_AI"
})

# ==========================================================
# Prepare NEW AI
# ==========================================================

new_ai_final = pd.DataFrame({
    "review": new_ai["review"],
    "label": 1,
    "type": "AI",
    "source": new_ai["source"],
    "generator_model": new_ai["generator_model"]
})

# ==========================================================
# Merge
# ==========================================================

df = pd.concat(
    [human_final, old_ai_final, new_ai_final],
    ignore_index=True
)

print(f"\nBefore Cleaning: {len(df):,}")

# ==========================================================
# Clean text
# ==========================================================

df["review"] = df["review"].astype(str).str.strip()

# Remove empty reviews
before = len(df)

df = df[
    df["review"].notna() &
    (df["review"] != "") &
    (df["review"].str.lower() != "nan")
].copy()

print(f"Empty reviews removed: {before - len(df):,}")

# ==========================================================
# Remove exact duplicate reviews
# ==========================================================

before = len(df)

df = df.drop_duplicates(
    subset=["review"],
    keep="first"
).reset_index(drop=True)

print(f"Duplicate reviews removed: {before - len(df):,}")

# ==========================================================
# Shuffle
# ==========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================================
# Save
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Statistics
# ==========================================================

print("\n" + "=" * 65)
print("FINAL DATASET")
print("=" * 65)

print(f"\nFinal Shape: {df.shape}")

print("\nType Distribution:")
print(df["type"].value_counts())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nSource Distribution:")
print(df["source"].value_counts())

print("\nGenerator Distribution:")
print(df["generator_model"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)
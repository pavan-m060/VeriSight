import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "ai_reviews_v2.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ai_reviews_v2_clean.csv"

# ==========================================================
# Load
# ==========================================================

print("=" * 60)
print("Cleaning New AI Reviews")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Original Shape : {df.shape}")

# ==========================================================
# Remove "Title:" prefix
# ==========================================================

df["review"] = (
    df["review"]
    .astype(str)
    .str.replace(r"^\s*Title\s*:\s*", "", regex=True)
    .str.strip()
)

# ==========================================================
# Remove empty reviews
# ==========================================================

before_empty = len(df)

df = df[
    df["review"].notna() &
    (df["review"].str.strip() != "")
].copy()

print(f"Removed Empty Reviews : {before_empty - len(df)}")

# ==========================================================
# Remove exact duplicate reviews
# ==========================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["review"],
    keep="first"
).reset_index(drop=True)

print(f"Removed Duplicates : {before_duplicates - len(df)}")

# ==========================================================
# Save
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("Cleaning Completed")
print("=" * 60)

print(f"Final Shape : {df.shape}")
print(f"Saved to    : {OUTPUT_FILE}")

print("\nGenerator Distribution:")
print(df["generator_model"].value_counts())

print("\nSource Distribution:")
print(df["source"].value_counts())
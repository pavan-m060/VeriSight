import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "external" / "ai_generated_reviews_batch_4.csv"

# ==========================================================
# Load
# ==========================================================

print("=" * 65)
print("Verifying Gemini External Dataset")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

# ==========================================================
# Basic checks
# ==========================================================

print("\n" + "=" * 65)
print("BASIC CHECKS")
print("=" * 65)

print("\nEmpty reviews:")
print(df["review"].isna().sum())

print("\nBlank reviews:")
print((df["review"].astype(str).str.strip() == "").sum())

print("\nDuplicate reviews:")
print(df["review"].duplicated().sum())

# ==========================================================
# Source distribution
# ==========================================================

print("\n" + "=" * 65)
print("SOURCE DISTRIBUTION")
print("=" * 65)

print(df["source"].value_counts())

# ==========================================================
# Rating distribution
# ==========================================================

print("\n" + "=" * 65)
print("RATING DISTRIBUTION")
print("=" * 65)

print(df["rating"].value_counts().sort_index())

print("\nSource + Rating:")
print(
    pd.crosstab(
        df["source"],
        df["rating"]
    )
)

# ==========================================================
# Generator
# ==========================================================

print("\n" + "=" * 65)
print("GENERATOR DISTRIBUTION")
print("=" * 65)

print(df["generator_model"].value_counts())

# ==========================================================
# Labels
# ==========================================================

print("\n" + "=" * 65)
print("LABEL DISTRIBUTION")
print("=" * 65)

print(df["label"].value_counts())

# ==========================================================
# Review lengths
# ==========================================================

df["word_count"] = df["review"].astype(str).str.split().str.len()

print("\n" + "=" * 65)
print("REVIEW LENGTH")
print("=" * 65)

print(df["word_count"].describe())

print("\nReviews shorter than 10 words:")
print((df["word_count"] < 10).sum())

# ==========================================================
# Sample reviews
# ==========================================================

print("\n" + "=" * 65)
print("SAMPLE REVIEWS")
print("=" * 65)

print(
    df[
        ["review", "source", "category", "rating",
         "generator_model", "label"]
    ].head(10).to_string(index=False)
)

# ==========================================================
# Final checks
# ==========================================================

print("\n" + "=" * 65)
print("FINAL CHECKS")
print("=" * 65)

expected_sources = {
    "Amazon": 250,
    "Yelp": 250,
    "TripAdvisor": 250,
    "GooglePlay": 250
}

for source, expected in expected_sources.items():

    actual = (df["source"] == source).sum()

    print(
        f"{source:15s}: {actual:4d} "
        f"(expected {expected})"
    )

print("\nAll generator values should be Gemini:")
print(
    df["generator_model"].eq("Gemini").all()
)

print("\nAll labels should be 1:")
print(
    df["label"].eq(1).all()
)

print("\nVerification completed.")
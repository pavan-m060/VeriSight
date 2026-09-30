import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BATCH_DIR = BASE_DIR / "data" / "external" / "gemini_batches"

OUTPUT_DIR = BASE_DIR / "data" / "external"
OUTPUT_FILE = OUTPUT_DIR / "gemini_external_final.csv"

# ==========================================================
# Load batches
# ==========================================================

print("=" * 65)
print("Creating Final Gemini External Dataset")
print("=" * 65)

files = sorted(BATCH_DIR.glob("*.csv"))

print(f"\nBatch files found: {len(files)}")

if len(files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {BATCH_DIR}"
    )

frames = []

for file in files:

    df = pd.read_csv(file)

    print(f"{file.name}: {len(df)} reviews")

    frames.append(df)

# ==========================================================
# Combine
# ==========================================================

df = pd.concat(
    frames,
    ignore_index=True
)

print("\nCombined Shape:", df.shape)

# ==========================================================
# Clean column names
# ==========================================================

df.columns = df.columns.str.strip()

# ==========================================================
# Remove empty reviews
# ==========================================================

before = len(df)

df["review"] = df["review"].astype(str).str.strip()

df = df[
    df["review"].notna() &
    (df["review"] != "")
]

print(
    "Empty reviews removed:",
    before - len(df)
)

# ==========================================================
# Remove exact duplicates
# ==========================================================

before = len(df)

df = df.drop_duplicates(
    subset=["review"],
    keep="first"
)

print(
    "Duplicate reviews removed:",
    before - len(df)
)

# ==========================================================
# Keep required columns
# ==========================================================

required_columns = [
    "review",
    "source",
    "category",
    "rating",
    "generator_model",
    "label"
]

df = df[required_columns]

# ==========================================================
# Validation
# ==========================================================

print("\n" + "=" * 65)
print("COMBINED DATASET")
print("=" * 65)

print("\nShape:")
print(df.shape)

print("\nSource Distribution:")
print(df["source"].value_counts())

print("\nRating Distribution:")
print(df["rating"].value_counts().sort_index())

print("\nGenerator:")
print(df["generator_model"].value_counts())

print("\nLabel:")
print(df["label"].value_counts())

# ==========================================================
# Save
# ==========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 65)
print("Final Gemini Dataset Saved")
print("=" * 65)

print("\nFinal Shape:", df.shape)

print("Saved to:")
print(OUTPUT_FILE)
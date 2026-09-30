import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HUMAN_FILE = PROJECT_ROOT / "data" / "processed" / "human_reviews_rawcase.csv"
AI_ORIGINAL_FILE = PROJECT_ROOT / "data" / "processed" / "ai_reviews_original.csv"
AI_V2_FILE = PROJECT_ROOT / "data" / "processed" / "ai_reviews_v2_clean.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "stage1" / "stage1_final_v3.csv"


print("=" * 70)
print("VERISIGHT STAGE 1 V3 - CORRECTED DATASET")
print("=" * 70)


# ==========================================================
# HUMAN
# ==========================================================

print("\nLoading corrected Human dataset...")

human = pd.read_csv(HUMAN_FILE)

print("Human shape:", human.shape)

human["label"] = 0
human["type"] = "human"
human["generator_model"] = "human"

human = human[
    [
        "review",
        "label",
        "type",
        "source",
        "generator_model"
    ]
].copy()


# ==========================================================
# ORIGINAL AI
# ==========================================================

print("\nLoading original AI dataset...")

ai_original = pd.read_csv(AI_ORIGINAL_FILE)

print("Original AI shape:", ai_original.shape)

ai_original["label"] = 1
ai_original["type"] = "ai"

ai_original = ai_original[
    [
        "review",
        "label",
        "type",
        "source",
        "generator_model"
    ]
].copy()


# ==========================================================
# NEW AI V2
# ==========================================================

print("\nLoading new AI v2 dataset...")

ai_v2 = pd.read_csv(AI_V2_FILE)

print("AI v2 shape:", ai_v2.shape)

ai_v2["label"] = 1
ai_v2["type"] = "ai"

ai_v2 = ai_v2[
    [
        "review",
        "label",
        "type",
        "source",
        "generator_model"
    ]
].copy()


# ==========================================================
# COMBINE AI
# ==========================================================

print("\nCombining original AI + AI v2...")

ai = pd.concat(
    [
        ai_original,
        ai_v2
    ],
    ignore_index=True
)

print("Combined AI shape:", ai.shape)


# ==========================================================
# COMBINE HUMAN + AI
# ==========================================================

print("\nCombining Human + AI...")

df = pd.concat(
    [
        human,
        ai
    ],
    ignore_index=True
)


# ==========================================================
# CHECK DUPLICATES
# ==========================================================

duplicates = df["review"].duplicated().sum()

print("\nDuplicate reviews:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates(
        subset=["review"]
    ).reset_index(drop=True)

    print(
        "After duplicate removal:",
        df.shape
    )


# ==========================================================
# SHUFFLE
# ==========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================================
# SAVE
# ==========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("STAGE 1 V3 CREATED")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nLabel distribution:")
print(
    df["label"].value_counts().sort_index()
)

print("\nType distribution:")
print(
    df["type"].value_counts()
)

print("\nSource distribution:")
print(
    df["source"].value_counts()
)

print("\nGenerator distribution:")
print(
    df["generator_model"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 70)
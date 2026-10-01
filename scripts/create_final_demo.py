import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE
    / "data"
    / "phase2"
    / "processed_hybrid"
    / "stage2_test_hybrid.csv"
)

PRED_FILE = (
    BASE
    / "results"
    / "stage2"
    / "stage2_hybrid_stronger_predictions.csv"
)

OUT_FILE = (
    BASE
    / "data"
    / "phase2"
    / "verisight_demo_reviews.csv"
)

print("=" * 70)
print("VERISIGHT - CREATING FINAL DEMO SET")
print("=" * 70)

# ----------------------------------------------------------
# Load test data
# ----------------------------------------------------------

print("\nLoading test dataset...")

df = pd.read_csv(TEST_FILE)

print("Test rows:", len(df))


# ----------------------------------------------------------
# Load model predictions
# ----------------------------------------------------------

print("\nLoading predictions...")

pred = pd.read_csv(PRED_FILE)

print("Prediction rows:", len(pred))


# ----------------------------------------------------------
# Verify alignment
# ----------------------------------------------------------

if len(df) != len(pred):
    raise RuntimeError(
        f"Row mismatch: test={len(df)}, predictions={len(pred)}"
    )

print("Prediction alignment: PASS")


# ----------------------------------------------------------
# Find probability column
# ----------------------------------------------------------

probability_column = None

for column in [
    "spam_probability",
    "probability",
    "prediction_probability",
    "y_prob"
]:
    if column in pred.columns:
        probability_column = column
        break

if probability_column is None:
    raise RuntimeError(
        "Could not find spam probability column in prediction file."
    )

print("Probability column:", probability_column)

df["spam_probability"] = pred[probability_column].astype(float)


# ----------------------------------------------------------
# Ground-truth label
# ----------------------------------------------------------

if "spam" not in df.columns:
    raise RuntimeError("spam column not found.")

df["spam"] = df["spam"].astype(int)


# ----------------------------------------------------------
# Model threshold
# ----------------------------------------------------------

THRESHOLD = 0.62

df["prediction"] = (
    df["spam_probability"] >= THRESHOLD
).astype(int)

df["correct"] = (
    df["prediction"] == df["spam"]
)


# ----------------------------------------------------------
# Find demo cases
# ----------------------------------------------------------

cases = {}


# ==========================================================
# 1. STRONG GENUINE
# ==========================================================

x = df[
    (df["spam"] == 0) &
    (df["correct"])
].sort_values(
    "spam_probability"
)

if len(x) > 0:
    cases["GENUINE"] = x.iloc[0]


# ==========================================================
# 2. STRONG SPAM
# ==========================================================

x = df[
    (df["spam"] == 1) &
    (df["correct"])
].sort_values(
    "spam_probability",
    ascending=False
)

if len(x) > 0:
    cases["SPAM"] = x.iloc[0]


# ==========================================================
# 3. BORDERLINE GENUINE
# ==========================================================

x = df[
    (df["spam"] == 0) &
    (df["spam_probability"].between(0.40, 0.75))
]

if len(x) > 0:
    cases["BORDERLINE_GENUINE"] = x.iloc[0]


# ==========================================================
# 4. BORDERLINE SPAM
# ==========================================================

x = df[
    (df["spam"] == 1) &
    (df["spam_probability"].between(0.40, 0.75))
]

if len(x) > 0:
    cases["BORDERLINE_SPAM"] = x.iloc[0]


# ----------------------------------------------------------
# Create final demo dataframe
# ----------------------------------------------------------

rows = []

for name, row in cases.items():

    r = row.copy()

    r["demo_case"] = name

    rows.append(r)


if len(rows) == 0:
    raise RuntimeError(
        "No suitable demo cases were found."
    )


demo = pd.DataFrame(rows)


# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

demo.to_csv(
    OUT_FILE,
    index=False
)


# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("DEMO CASES CREATED")
print("=" * 70)

print(
    demo[
        [
            "demo_case",
            "spam",
            "spam_probability",
            "prediction",
            "correct"
        ]
    ].to_string(index=False)
)


print("\nOutput file:")
print(OUT_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
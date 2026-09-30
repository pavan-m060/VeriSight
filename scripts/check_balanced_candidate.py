import pandas as pd

FILE = "data/stage1/audit/stage1_v3_length_balanced_candidate.csv"

df = pd.read_csv(FILE)

print("\n" + "=" * 80)
print("BALANCED CANDIDATE DATASET CHECK")
print("=" * 80)

print("\nShape:")
print(df.shape)

# ------------------------------------------------------------
# LABEL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

print(df["label"].value_counts())

# ------------------------------------------------------------
# SOURCE × LABEL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE × LABEL")
print("=" * 80)

print(
    pd.crosstab(
        df["source"],
        df["label"]
    )
)

# ------------------------------------------------------------
# GENERATOR × LABEL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATOR × LABEL")
print("=" * 80)

print(
    pd.crosstab(
        df["generator_model"],
        df["label"]
    )
)

# ------------------------------------------------------------
# SOURCE × LENGTH
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE × LENGTH")
print("=" * 80)

print(
    pd.crosstab(
        df["source"],
        df["length_bin"]
    )
)

# ------------------------------------------------------------
# GENERATOR × LENGTH
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATOR × LENGTH")
print("=" * 80)

print(
    pd.crosstab(
        df["generator_model"],
        df["length_bin"]
    )
)

# ------------------------------------------------------------
# GENERATOR × LENGTH × LABEL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATOR × LENGTH × LABEL")
print("=" * 80)

for generator in sorted(
    df["generator_model"].dropna().unique()
):

    print(
        f"\n--- {generator} ---"
    )

    subset = df[
        df["generator_model"] == generator
    ]

    print(
        pd.crosstab(
            subset["length_bin"],
            subset["label"]
        )
    )

# ------------------------------------------------------------
# WORD COUNT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WORD COUNT")
print("=" * 80)

print(
    df.groupby("label")["word_count"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
    .round(2)
)

# ------------------------------------------------------------
# GENERATOR WORD COUNT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATOR WORD COUNT")
print("=" * 80)

print(
    df.groupby(
        "generator_model"
    )["word_count"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
    .round(2)
)

# ------------------------------------------------------------
# SOURCE WORD COUNT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE WORD COUNT")
print("=" * 80)

print(
    df.groupby(
        ["source", "label"]
    )["word_count"]
    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )
    .round(2)
)

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)
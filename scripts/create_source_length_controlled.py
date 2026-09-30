import pandas as pd
from pathlib import Path

# ============================================================
# VeriSight Stage 1 V3
# SOURCE + LENGTH + GENERATOR CONTROLLED DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "stage1" / "stage1_final_v3.csv"

OUTPUT = (
    BASE_DIR
    / "data"
    / "stage1"
    / "stage1_v3_source_length_controlled.csv"
)

REPORT = (
    BASE_DIR
    / "data"
    / "stage1"
    / "stage1_v3_source_length_controlled_report.txt"
)

RANDOM_STATE = 42

print("=" * 70)
print("VERISIGHT - STAGE 1 V3 SOURCE + LENGTH CONTROL")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT)

print(f"\nOriginal dataset: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# 2. WORD COUNT
# ============================================================

df["word_count"] = (
    df["review"]
    .astype(str)
    .str.split()
    .str.len()
)

# ============================================================
# 3. KEEP COMMON HUMAN/AI RANGE
# ============================================================

df = df[
    (df["word_count"] >= 20) &
    (df["word_count"] <= 300)
].copy()

print(
    f"\nAfter 20-300 word filtering: "
    f"{df.shape}"
)

# ============================================================
# 4. LENGTH BINS
# ============================================================

bins = [
    20,
    41,
    61,
    81,
    101,
    121,
    151,
    201,
    301
]

labels = [
    "20-40",
    "41-60",
    "61-80",
    "81-100",
    "101-120",
    "121-150",
    "151-200",
    "201-300"
]

df["length_bin"] = pd.cut(
    df["word_count"],
    bins=bins,
    labels=labels,
    right=False
)

# ============================================================
# 5. SOURCE × LENGTH HUMAN/AI BALANCING
# ============================================================

parts = []

for (source, length_bin), group in df.groupby(
    ["source", "length_bin"],
    observed=True
):

    human = group[group["label"] == 0]
    ai = group[group["label"] == 1]

    # Skip cells where one class doesn't exist
    if len(human) == 0 or len(ai) == 0:
        continue

    n = min(
        len(human),
        len(ai)
    )

    human_sample = human.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    ai_sample = ai.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    parts.append(human_sample)
    parts.append(ai_sample)

balanced = pd.concat(
    parts,
    ignore_index=True
)

print(
    f"\nAfter SOURCE × LENGTH balancing: "
    f"{balanced.shape}"
)

print("\nLabel distribution:")
print(
    balanced["label"].value_counts()
)

# ============================================================
# 6. AI GENERATOR BALANCING
#
# Column is generator_model
# ============================================================

ai = balanced[
    balanced["label"] == 1
].copy()

human = balanced[
    balanced["label"] == 0
].copy()

generator_counts = (
    ai["generator_model"]
    .value_counts()
)

print(
    "\nAI generator distribution BEFORE "
    "generator balancing:"
)

print(generator_counts)

# Remove human if it somehow appears
ai_generators = [
    g
    for g in generator_counts.index
    if str(g).lower() != "human"
]

# Equal global target
target_per_generator = min(
    generator_counts[g]
    for g in ai_generators
)

print(
    f"\nTarget per AI generator: "
    f"{target_per_generator}"
)

# ============================================================
# 7. STRATIFIED GENERATOR SAMPLING
#
# Preserve SOURCE × LENGTH diversity
# ============================================================

selected_ai_parts = []

for generator in ai_generators:

    gen_df = ai[
        ai["generator_model"] == generator
    ].copy()

    # --------------------------------------------------------
    # If already <= target, keep everything
    # --------------------------------------------------------

    if len(gen_df) <= target_per_generator:

        selected = gen_df.copy()

    else:

        groups = list(
            gen_df.groupby(
                ["source", "length_bin"],
                observed=True
            )
        )

        # --------------------------------------------
        # Proportional allocation
        # --------------------------------------------

        allocations = {}

        exact_values = {}

        for key, group in groups:

            exact = (
                len(group)
                / len(gen_df)
                * target_per_generator
            )

            exact_values[key] = exact
            allocations[key] = int(exact)

        # --------------------------------------------
        # Distribute remaining samples
        # --------------------------------------------

        remaining = (
            target_per_generator
            - sum(allocations.values())
        )

        remainders = sorted(
            exact_values.items(),
            key=lambda x: (
                x[1] - int(x[1])
            ),
            reverse=True
        )

        for i in range(remaining):

            key = remainders[i][0]

            allocations[key] += 1

        # --------------------------------------------
        # Sample each cell
        # --------------------------------------------

        sampled_parts = []

        for key, group in groups:

            n = min(
                allocations[key],
                len(group)
            )

            if n > 0:

                sampled_parts.append(
                    group.sample(
                        n=n,
                        random_state=RANDOM_STATE
                    )
                )

        selected = pd.concat(
            sampled_parts,
            ignore_index=True
        )

    selected_ai_parts.append(
        selected
    )

controlled_ai = pd.concat(
    selected_ai_parts,
    ignore_index=True
)

print(
    "\nAI generator distribution AFTER "
    "generator balancing:"
)

print(
    controlled_ai[
        "generator_model"
    ].value_counts()
)

# ============================================================
# 8. REBALANCE HUMAN vs AI
#
# Generator balancing can slightly alter
# SOURCE × LENGTH cells.
# ============================================================

final_parts = []

for (source, length_bin), human_group in human.groupby(
    ["source", "length_bin"],
    observed=True
):

    ai_group = controlled_ai[
        (controlled_ai["source"] == source) &
        (
            controlled_ai["length_bin"]
            == length_bin
        )
    ]

    if len(ai_group) == 0:
        continue

    n = min(
        len(human_group),
        len(ai_group)
    )

    final_parts.append(
        human_group.sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

    final_parts.append(
        ai_group.sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

final_df = pd.concat(
    final_parts,
    ignore_index=True
)

# ============================================================
# 9. SHUFFLE
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ============================================================
# 10. SAVE
# ============================================================

final_df.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# 11. VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"\nFinal shape: {final_df.shape}"
)

# ------------------------------------------------------------
# Label
# ------------------------------------------------------------

print("\nLABEL DISTRIBUTION")
print("-" * 70)

print(
    final_df["label"].value_counts()
)

# ------------------------------------------------------------
# Source × Label
# ------------------------------------------------------------

print("\nSOURCE × LABEL")
print("-" * 70)

source_label = pd.crosstab(
    final_df["source"],
    final_df["label"]
)

print(source_label)

# ------------------------------------------------------------
# Generator × Label
# ------------------------------------------------------------

print("\nGENERATOR × LABEL")
print("-" * 70)

generator_label = pd.crosstab(
    final_df["generator_model"],
    final_df["label"]
)

print(generator_label)

# ------------------------------------------------------------
# Source × Length × Label
# ------------------------------------------------------------

print("\nSOURCE × LENGTH × LABEL")
print("-" * 70)

source_length_label = pd.crosstab(
    [
        final_df["source"],
        final_df["length_bin"]
    ],
    final_df["label"]
)

print(source_length_label)

# ------------------------------------------------------------
# Generator × Length
# ------------------------------------------------------------

print("\nGENERATOR × LENGTH")
print("-" * 70)

generator_length = pd.crosstab(
    final_df["generator_model"],
    final_df["length_bin"]
)

print(generator_length)

# ------------------------------------------------------------
# Word statistics
# ------------------------------------------------------------

print("\nWORD COUNT STATISTICS")
print("-" * 70)

word_stats = (
    final_df
    .groupby("label")["word_count"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .round(2)
)

print(word_stats)

# ------------------------------------------------------------
# Source word statistics
# ------------------------------------------------------------

print("\nSOURCE × LABEL WORD STATISTICS")
print("-" * 70)

source_stats = (
    final_df
    .groupby(
        ["source", "label"]
    )["word_count"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .round(2)
)

print(source_stats)

# ============================================================
# 12. SAVE REPORT
# ============================================================

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VERISIGHT STAGE 1 V3 "
        "SOURCE + LENGTH CONTROLLED DATASET\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Input: {INPUT}\n"
    )

    f.write(
        f"Output: {OUTPUT}\n\n"
    )

    f.write(
        f"Final shape: {final_df.shape}\n\n"
    )

    f.write(
        "LABEL DISTRIBUTION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(
            final_df["label"]
            .value_counts()
        )
    )

    f.write("\n\n")

    f.write(
        "SOURCE × LABEL\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(source_label)
    )

    f.write("\n\n")

    f.write(
        "GENERATOR × LABEL\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(generator_label)
    )

    f.write("\n\n")

    f.write(
        "SOURCE × LENGTH × LABEL\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(source_length_label)
    )

    f.write("\n\n")

    f.write(
        "GENERATOR × LENGTH\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(generator_length)
    )

    f.write("\n\n")

    f.write(
        "WORD COUNT STATISTICS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(word_stats)
    )

    f.write("\n\n")

    f.write(
        "SOURCE × LABEL WORD STATISTICS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(source_stats)
    )

print("\n")
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"\nCreated:\n{OUTPUT}"
)

print(
    f"\nReport:\n{REPORT}"
)
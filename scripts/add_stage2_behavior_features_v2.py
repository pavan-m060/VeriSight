import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/phase2/processed"
OUTPUT_DIR = "data/phase2/processed_behavior_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "train": "stage2_train.csv",
    "val": "stage2_val.csv",
    "test": "stage2_test.csv"
}


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, filename):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()} SET")
    print("=" * 70)

    input_file = os.path.join(
        INPUT_DIR,
        filename
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_behavior.csv"
    )

    df = pd.read_csv(input_file)

    print(
        f"Loaded {len(df):,} rows"
    )

    # ========================================================
    # DATE
    # ========================================================

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    # Preserve original row identity
    df["_original_order"] = np.arange(len(df))

    # Sort chronologically by user
    df = df.sort_values(
        ["user_id", "date", "_original_order"]
    ).reset_index(drop=True)

    # ========================================================
    # USER-LEVEL CUMULATIVE FEATURES
    # ========================================================

    print("\nCalculating leakage-safe user behavior...")

    user_groups = df.groupby(
        "user_id",
        sort=False
    )

    # --------------------------------------------------------
    # Number of PREVIOUS reviews
    # --------------------------------------------------------

    df["user_previous_review_count"] = (
        user_groups.cumcount()
    )

    # --------------------------------------------------------
    # Previous rating statistics
    # --------------------------------------------------------

    df["_rating_sum_previous"] = (
        user_groups["rating"]
        .cumsum()
        - df["rating"]
    )

    df["user_previous_avg_rating"] = np.where(
        df["user_previous_review_count"] > 0,
        df["_rating_sum_previous"]
        / df["user_previous_review_count"],
        0
    )

    # --------------------------------------------------------
    # Previous rating squared sum
    # --------------------------------------------------------

    df["_rating_squared"] = (
        df["rating"] ** 2
    )

    df["_rating_squared_sum_previous"] = (
        user_groups["_rating_squared"]
        .cumsum()
        - df["_rating_squared"]
    )

    n = df["user_previous_review_count"]

    variance = np.where(
        n > 1,
        (
            df["_rating_squared_sum_previous"]
            - (
                df["_rating_sum_previous"] ** 2
                / np.maximum(n, 1)
            )
        )
        / np.maximum(n - 1, 1),
        0
    )

    df["user_previous_rating_std"] = np.sqrt(
        np.maximum(variance, 0)
    )

    # --------------------------------------------------------
    # Previous unique products
    #
    # Number of distinct products reviewed BEFORE
    # the current review.
    # --------------------------------------------------------

    previous_product_count = (
        df.groupby(
            ["user_id", "prod_id"]
        )
        .cumcount()
    )

    # A product is new for the user when its previous count is 0
    new_product_flag = (
        previous_product_count == 0
    ).astype(int)

    df["user_previous_unique_products"] = (
        new_product_flag
        .groupby(df["user_id"])
        .cumsum()
        - new_product_flag
    )

    # --------------------------------------------------------
    # Previous review date
    # --------------------------------------------------------

    df["previous_review_date"] = (
        user_groups["date"]
        .shift(1)
    )

    time_difference = (
        df["date"]
        - df["previous_review_date"]
    )

    df["time_since_previous_review_hours"] = (
        time_difference
        .dt.total_seconds()
        / 3600
    )

    df["time_since_previous_review_hours"] = (
        df["time_since_previous_review_hours"]
        .fillna(999999)
        .clip(lower=0)
    )

    # ========================================================
    # USER ACTIVE PERIOD
    # ========================================================

    first_review_date = (
        user_groups["date"]
        .transform("min")
    )

    df["user_days_since_first_review"] = (
        (
            df["date"]
            - first_review_date
        )
        .dt.total_seconds()
        / 86400
    )

    df["user_days_since_first_review"] = (
        df["user_days_since_first_review"]
        .fillna(0)
        .clip(lower=0)
    )

    # Reviews per day BEFORE current review
    df["user_previous_reviews_per_day"] = np.where(
        df["user_days_since_first_review"] > 0,
        df["user_previous_review_count"]
        / df["user_days_since_first_review"],
        0
    )

    # ========================================================
    # TEMPORAL BURST FEATURES
    # ========================================================

    print("\nCalculating temporal burst features...")

    burst_24h = np.zeros(
        len(df),
        dtype=np.int32
    )

    burst_7d = np.zeros(
        len(df),
        dtype=np.int32
    )

    grouped_indices = (
        df.groupby(
            "user_id",
            sort=False
        )
        .indices
    )

    window_24h = (
        24 * 60 * 60 * 1_000_000_000
    )

    window_7d = (
        7 * 24 * 60 * 60 * 1_000_000_000
    )

    for indices in grouped_indices.values():

        indices = np.asarray(
            indices,
            dtype=np.int64
        )

        dates_ns = (
            df.loc[indices, "date"]
            .astype("int64")
            .values
        )

        # Invalid dates → 0
        dates_ns = np.where(
            dates_ns < 0,
            0,
            dates_ns
        )

        # ----------------------------------------------------
        # Previous 24 hours
        # ----------------------------------------------------

        left_24 = np.searchsorted(
            dates_ns,
            dates_ns - window_24h,
            side="left"
        )

        positions = np.arange(
            len(indices)
        )

        burst_24h[indices] = (
            positions - left_24
        )

        # ----------------------------------------------------
        # Previous 7 days
        # ----------------------------------------------------

        left_7d = np.searchsorted(
            dates_ns,
            dates_ns - window_7d,
            side="left"
        )

        burst_7d[indices] = (
            positions - left_7d
        )

    df["reviews_previous_24h"] = burst_24h
    df["reviews_previous_7days"] = burst_7d

    # ========================================================
    # PRODUCT FEATURES
    # ========================================================

    print("\nCalculating product behavior...")

    product_groups = df.groupby(
        "prod_id",
        sort=False
    )

    # --------------------------------------------------------
    # Previous product review count
    # --------------------------------------------------------

    df["product_previous_review_count"] = (
        product_groups.cumcount()
    )

    # --------------------------------------------------------
    # Previous product rating average
    # --------------------------------------------------------

    product_rating_sum = (
        product_groups["rating"]
        .cumsum()
        - df["rating"]
    )

    df["product_previous_avg_rating"] = np.where(
        df["product_previous_review_count"] > 0,
        product_rating_sum
        / df["product_previous_review_count"],
        0
    )

    # ========================================================
    # USER-PRODUCT RELATIONSHIP
    # ========================================================

    df["user_product_previous_reviews"] = (
        df.groupby(
            ["user_id", "prod_id"]
        )
        .cumcount()
    )

    # ========================================================
    # RATING BEHAVIOR
    # ========================================================

    df["rating_distance_from_user_history"] = (
        (
            df["rating"]
            - df["user_previous_avg_rating"]
        )
        .abs()
    )

    df["rating_distance_from_product_history"] = np.where(
        df["product_previous_review_count"] > 0,
        (
            df["rating"]
            - df["product_previous_avg_rating"]
        ).abs(),
        0
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    df = df.drop(
        columns=[
            "_rating_sum_previous",
            "_rating_squared",
            "_rating_squared_sum_previous",
            "previous_review_date",
            "_original_order"
        ],
        errors="ignore"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    # ========================================================
    # RESTORE ORIGINAL ORDER
    # ========================================================

    # The original order column was dropped above, so the
    # dataset remains chronological within users. That is
    # acceptable because the model does not depend on row order.

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Important sanity checks
    # --------------------------------------------------------

    print("\nSanity checks:")

    print(
        "NaN values:",
        int(df.isna().sum().sum())
    )

    print(
        "Infinite values:",
        int(
            np.isinf(
                df.select_dtypes(
                    include=[np.number]
                )
            ).sum().sum()
        )
    )

    print(
        "Max previous 24h reviews:",
        int(
            df["reviews_previous_24h"].max()
        )
    )

    print(
        "Max previous 7d reviews:",
        int(
            df["reviews_previous_7days"].max()
        )
    )

    print(
        "Max previous user reviews:",
        int(
            df["user_previous_review_count"].max()
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 LEAKAGE-SAFE BEHAVIOR FEATURES")
print("=" * 70)

print("""
Features are calculated using information available BEFORE
each review whenever the feature is historical.

The spam/fake label is NEVER used to construct features.
""")

for split_name, filename in FILES.items():

    process_split(
        split_name,
        filename
    )


print("\n" + "=" * 70)
print("LEAKAGE-SAFE FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("""
Output:

data/phase2/processed_behavior_v2/

    train_behavior.csv
    val_behavior.csv
    test_behavior.csv
""")
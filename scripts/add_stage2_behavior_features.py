import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/phase2/processed"
OUTPUT_DIR = "data/phase2/processed_behavior"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# FILES
# ============================================================

FILES = {
    "train": "stage2_train.csv",
    "val": "stage2_val.csv",
    "test": "stage2_test.csv"
}


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_entropy(values):
    """
    Calculate Shannon entropy of a discrete distribution.
    """

    if len(values) == 0:
        return 0.0

    counts = pd.Series(values).value_counts()

    probabilities = (
        counts / counts.sum()
    ).values

    return float(
        -np.sum(
            probabilities *
            np.log2(probabilities + 1e-12)
        )
    )


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

    print("Input:", input_file)

    df = pd.read_csv(input_file)

    print(
        f"Loaded {len(df):,} rows"
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    # Sort chronologically within users
    df = df.sort_values(
        ["user_id", "date"]
    ).reset_index(drop=True)

    # ========================================================
    # USER-LEVEL FEATURES
    # ========================================================

    print("\nCalculating user behavior...")

    user_group = df.groupby(
        "user_id",
        sort=False
    )

    # --------------------------------------------------------
    # Number of reviews
    # --------------------------------------------------------

    df["user_review_count"] = (
        user_group["rating"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Rating standard deviation
    # --------------------------------------------------------

    df["user_rating_std"] = (
        user_group["rating"]
        .transform("std")
        .fillna(0)
    )

    # --------------------------------------------------------
    # Rating entropy
    # --------------------------------------------------------

    rating_entropy = (
        user_group["rating"]
        .apply(calculate_entropy)
        .rename("user_rating_entropy")
    )

    df = df.merge(
        rating_entropy,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Unique products reviewed
    # --------------------------------------------------------

    unique_products = (
        user_group["prod_id"]
        .nunique()
        .rename("user_unique_products")
    )

    df = df.merge(
        unique_products,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Active days
    # --------------------------------------------------------

    active_days = (
        user_group["date"]
        .apply(
            lambda x: (
                x.max() - x.min()
            ).days + 1
        )
        .rename("user_active_days")
    )

    df = df.merge(
        active_days,
        on="user_id",
        how="left"
    )

    df["user_active_days"] = (
        df["user_active_days"]
        .clip(lower=1)
    )

    # --------------------------------------------------------
    # Reviews per active day
    # --------------------------------------------------------

    df["user_reviews_per_day"] = (
        df["user_review_count"]
        / df["user_active_days"]
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    print("\nCalculating temporal behavior...")

    # --------------------------------------------------------
    # Time since previous review by same user
    # --------------------------------------------------------

    df["previous_review_date"] = (
        df.groupby("user_id")["date"]
        .shift(1)
    )

    time_difference = (
        df["date"]
        - df["previous_review_date"]
    )

    df["time_since_previous_review_hours"] = (
        time_difference
        .dt.total_seconds()
        / 3600.0
    )

    # First review has no previous review
    df["time_since_previous_review_hours"] = (
        df["time_since_previous_review_hours"]
        .fillna(999999)
    )

    # Prevent negative values from malformed dates
    df["time_since_previous_review_hours"] = (
        df["time_since_previous_review_hours"]
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Burst score
    #
    # Number of reviews made by the same user within
    # the previous 24 hours.
    # --------------------------------------------------------

    print("Calculating 24-hour review burst scores...")

    burst_scores = np.zeros(
        len(df),
        dtype=np.int32
    )

    # Work with positional indices after sorting
    user_indices = (
        df.groupby("user_id", sort=False)
        .indices
    )

    dates = df["date"].values

    for counter, (user_id, indices) in enumerate(
        user_indices.items(),
        start=1
    ):

        indices = np.asarray(
            indices,
            dtype=np.int64
        )

        user_dates = (
            df.loc[indices, "date"]
            .astype("int64")
            .values
        )

        # NaT becomes the minimum int64 value.
        # Replace invalid dates with zero.
        user_dates = np.where(
            user_dates < 0,
            0,
            user_dates
        )

        # 24 hours in nanoseconds
        window = (
            24 * 60 * 60 * 1_000_000_000
        )

        left_positions = np.searchsorted(
            user_dates,
            user_dates - window,
            side="left"
        )

        counts = (
            np.arange(len(indices))
            - left_positions
        )

        burst_scores[indices] = counts

        if counter % 50000 == 0:

            print(
                f"  Processed "
                f"{counter:,} users..."
            )

    df["reviews_previous_24h"] = (
        burst_scores
    )

    # --------------------------------------------------------
    # 7-day burst
    # --------------------------------------------------------

    burst_7day = np.zeros(
        len(df),
        dtype=np.int32
    )

    window_7d = (
        7 * 24 * 60 * 60 * 1_000_000_000
    )

    for indices in user_indices.values():

        indices = np.asarray(
            indices,
            dtype=np.int64
        )

        user_dates = (
            df.loc[indices, "date"]
            .astype("int64")
            .values
        )

        user_dates = np.where(
            user_dates < 0,
            0,
            user_dates
        )

        left_positions = np.searchsorted(
            user_dates,
            user_dates - window_7d,
            side="left"
        )

        counts = (
            np.arange(len(indices))
            - left_positions
        )

        burst_7day[indices] = counts

    df["reviews_previous_7days"] = (
        burst_7day
    )

    # ========================================================
    # PRODUCT-LEVEL FEATURES
    # ========================================================

    print("\nCalculating product behavior...")

    product_group = df.groupby(
        "prod_id",
        sort=False
    )

    df["product_review_count"] = (
        product_group["rating"]
        .transform("count")
    )

    df["product_avg_rating"] = (
        product_group["rating"]
        .transform("mean")
    )

    df["product_rating_std"] = (
        product_group["rating"]
        .transform("std")
        .fillna(0)
    )

    # ========================================================
    # USER-PRODUCT RELATIONSHIP
    # ========================================================

    print("\nCalculating user-product behavior...")

    # Number of reviews this user wrote for this product
    user_product_count = (
        df.groupby(
            ["user_id", "prod_id"]
        )["rating"]
        .transform("count")
    )

    df["user_product_review_count"] = (
        user_product_count
    )

    # ========================================================
    # RATING BEHAVIOR
    # ========================================================

    df["rating_distance_from_user_mean"] = (
        (
            df["rating"]
            - df["user_avg_rating"]
        )
        .abs()
    )

    df["rating_distance_from_product_mean"] = (
        (
            df["rating"]
            - df["product_avg_rating"]
        )
        .abs()
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    df = df.drop(
        columns=[
            "previous_review_date"
        ],
        errors="ignore"
    )

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numerical missing values
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

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

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 BEHAVIORAL FEATURE ENGINEERING")
print("=" * 70)

print("""
IMPORTANT:
These features do NOT use the spam/fake label to construct
user, temporal, product, or behavioral signals.

The original user-grouped train/validation/test split
is preserved.
""")

for split_name, filename in FILES.items():

    process_split(
        split_name,
        filename
    )


print("\n" + "=" * 70)
print("BEHAVIOR FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("""
Output directory:

data/phase2/processed_behavior/

Files:

train_behavior.csv
val_behavior.csv
test_behavior.csv
""")
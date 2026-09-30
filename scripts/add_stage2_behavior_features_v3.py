import os
import numpy as np
import pandas as pd


INPUT_DIR = "data/phase2/processed"
OUTPUT_DIR = "data/phase2/processed_behavior_v3"

FILES = {
    "train": "stage2_train.csv",
    "val": "stage2_val.csv",
    "test": "stage2_test.csv"
}


# ============================================================
# STRICTLY CHRONOLOGICAL COUNT
# ============================================================

def previous_window_count(df, group_col, window):

    """
    For every row, count how many previous events from the
    same group occurred within the specified time window.

    Current review is NOT included.

    Example:
        window = 24 hours

    For review at T:
        count reviews with:
            T-window <= previous_time < T
    """

    result = np.zeros(len(df), dtype=np.int32)

    # Work on integer nanoseconds for speed
    timestamps = (
        df["date"]
        .astype("int64")
        .to_numpy()
    )

    group_values = (
        df[group_col]
        .to_numpy()
    )

    # Process each group independently
    groups = pd.Series(
        np.arange(len(df))
    ).groupby(
        group_values,
        sort=False
    )

    window_ns = pd.Timedelta(
        window
    ).value

    for _, indices in groups:

        indices = indices.to_numpy()

        times = timestamps[indices]

        # Data is already globally chronological.
        # Within each group it is therefore chronological too.

        left = np.searchsorted(
            times,
            times - window_ns,
            side="left"
        )

        right = np.arange(
            len(times)
        )

        counts = right - left

        result[indices] = counts

    return result


# ============================================================
# MAIN FEATURE FUNCTION
# ============================================================

def add_behavior_features(df):

    print(
        f"\nProcessing {len(df):,} reviews..."
    )

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Original row ID for restoring original order
    df["_original_index"] = np.arange(
        len(df)
    )

    # Sort chronologically
    df = df.sort_values(
        ["date", "_original_index"],
        kind="mergesort"
    ).reset_index(drop=True)

    # ========================================================
    # USER HISTORY
    # ========================================================

    df["user_previous_review_count"] = (
        df.groupby("user_id")
        .cumcount()
    )

    # Previous user rating sum
    user_rating_sum = (
        df.groupby("user_id")["rating"]
        .cumsum()
        - df["rating"]
    )

    previous_user_count = (
        df["user_previous_review_count"]
    )

    df["user_previous_avg_rating"] = (
        user_rating_sum /
        previous_user_count.replace(
            0,
            np.nan
        )
    )

    # Previous user rating standard deviation
    rating_squared = (
        df["rating"] ** 2
    )

    previous_squared_sum = (
        rating_squared.groupby(
            df["user_id"]
        ).cumsum()
        - rating_squared
    )

    previous_variance = (
        previous_squared_sum /
        previous_user_count.replace(
            0,
            np.nan
        )
        -
        df["user_previous_avg_rating"] ** 2
    )

    previous_variance = (
        previous_variance
        .clip(lower=0)
    )

    df["user_previous_rating_std"] = (
        np.sqrt(previous_variance)
    )

    # ========================================================
    # USER UNIQUE PRODUCTS
    # ========================================================

    # A simple cumulative unique-product counter.
    # The current product is excluded.

    user_product_seen = (
        df.groupby(
            ["user_id", "prod_id"]
        ).cumcount()
    )

    # Number of distinct products previously reviewed
    # by each user.
    df["_new_user_product"] = (
        user_product_seen == 0
    ).astype(int)

    df["user_previous_unique_products"] = (
        df.groupby("user_id")[
            "_new_user_product"
        ].cumsum()
        - df["_new_user_product"]
    )

    # ========================================================
    # USER PREVIOUS REVIEW TIME
    # ========================================================

    previous_user_date = (
        df.groupby("user_id")["date"]
        .shift(1)
    )

    df[
        "time_since_previous_user_review_hours"
    ] = (
        (
            df["date"] -
            previous_user_date
        )
        .dt.total_seconds()
        / 3600
    )

    # ========================================================
    # USER LIFETIME
    # ========================================================

    first_user_date = (
        df.groupby("user_id")["date"]
        .transform("min")
    )

    df["user_days_since_first_review"] = (
        (
            df["date"] -
            first_user_date
        )
        .dt.total_seconds()
        / 86400
    )

    df["user_previous_reviews_per_day"] = (
        df["user_previous_review_count"] /
        df[
            "user_days_since_first_review"
        ].clip(lower=1)
    )

    # ========================================================
    # USER BURST FEATURES
    # ========================================================

    print("Calculating user 24-hour burst...")

    df["reviews_previous_24h"] = (
        previous_window_count(
            df,
            "user_id",
            "24h"
        )
    )

    print("Calculating user 7-day burst...")

    df["reviews_previous_7days"] = (
        previous_window_count(
            df,
            "user_id",
            "7D"
        )
    )

    # ========================================================
    # PRODUCT HISTORY
    # ========================================================

    df["product_previous_review_count"] = (
        df.groupby("prod_id")
        .cumcount()
    )

    product_rating_sum = (
        df.groupby("prod_id")["rating"]
        .cumsum()
        - df["rating"]
    )

    previous_product_count = (
        df["product_previous_review_count"]
    )

    df["product_previous_avg_rating"] = (
        product_rating_sum /
        previous_product_count.replace(
            0,
            np.nan
        )
    )

    # ========================================================
    # PREVIOUS PRODUCT REVIEW TIME
    # ========================================================

    previous_product_date = (
        df.groupby("prod_id")["date"]
        .shift(1)
    )

    df[
        "time_since_previous_product_review_hours"
    ] = (
        (
            df["date"] -
            previous_product_date
        )
        .dt.total_seconds()
        / 3600
    )

    # ========================================================
    # PRODUCT BURST
    # ========================================================

    print(
        "Calculating product 1-hour burst..."
    )

    df["product_reviews_previous_1h"] = (
        previous_window_count(
            df,
            "prod_id",
            "1h"
        )
    )

    print(
        "Calculating product 24-hour burst..."
    )

    df["product_reviews_previous_24h"] = (
        previous_window_count(
            df,
            "prod_id",
            "24h"
        )
    )

    print(
        "Calculating product 7-day burst..."
    )

    df["product_reviews_previous_7days"] = (
        previous_window_count(
            df,
            "prod_id",
            "7D"
        )
    )

    # ========================================================
    # USER × PRODUCT HISTORY
    # ========================================================

    df[
        "user_product_previous_reviews"
    ] = (
        df.groupby(
            ["user_id", "prod_id"]
        ).cumcount()
    )

    # ========================================================
    # RATING DEVIATION
    # ========================================================

    df[
        "rating_distance_from_user_history"
    ] = (
        (
            df["rating"] -
            df["user_previous_avg_rating"]
        )
        .abs()
        .fillna(0)
    )

    df[
        "rating_distance_from_product_history"
    ] = (
        (
            df["rating"] -
            df["product_previous_avg_rating"]
        )
        .abs()
        .fillna(0)
    )

    # ========================================================
    # CLEAN NUMERIC VALUES
    # ========================================================

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_cols] = (
        df[numeric_cols]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # ========================================================
    # RESTORE ORIGINAL ORDER
    # ========================================================

    df = df.sort_values(
        "_original_index"
    ).reset_index(drop=True)

    # ========================================================
    # CLEANUP
    # ========================================================

    df["date"] = (
        df["date"]
        .dt.strftime("%Y-%m-%d")
    )

    df = df.drop(
        columns=[
            "_original_index",
            "_new_user_product"
        ],
        errors="ignore"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 75)
    print(
        "VERISIGHT - STAGE 2 "
        "BEHAVIOR FEATURES V3"
    )
    print("=" * 75)

    for split, filename in FILES.items():

        print("\n" + "-" * 75)
        print(
            f"Processing {split.upper()}"
        )
        print("-" * 75)

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{split}_behavior.csv"
        )

        print(
            f"Input : {input_path}"
        )

        df = pd.read_csv(
            input_path
        )

        original_rows = len(df)

        print(
            f"Rows before: {original_rows:,}"
        )

        result = add_behavior_features(
            df
        )

        # ====================================================
        # SANITY CHECK
        # ====================================================

        print(
            f"\nRows after : {len(result):,}"
        )

        print(
            f"Features   : {result.shape[1]}"
        )

        if len(result) != original_rows:

            raise RuntimeError(
                "ROW COUNT CHANGED! "
                "Feature generation is invalid."
            )

        nan_count = (
            result.isna()
            .sum()
            .sum()
        )

        numeric = result.select_dtypes(
            include=[np.number]
        )

        inf_count = np.isinf(
            numeric.to_numpy()
        ).sum()

        print(
            f"NaN values : {nan_count:,}"
        )

        print(
            f"Inf values : {inf_count:,}"
        )

        # ====================================================
        # FEATURE CHECK
        # ====================================================

        check_features = [
            "user_previous_review_count",
            "product_previous_review_count",
            "reviews_previous_24h",
            "reviews_previous_7days",
            "product_reviews_previous_1h",
            "product_reviews_previous_24h",
            "product_reviews_previous_7days",
            "user_product_previous_reviews"
        ]

        print("\nFeature ranges:")

        for col in check_features:

            if col in result.columns:

                print(
                    f"{col:45s} "
                    f"min={result[col].min():.2f} "
                    f"max={result[col].max():.2f} "
                    f"mean={result[col].mean():.2f}"
                )

        # ====================================================
        # SAVE
        # ====================================================

        result.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nSaved: {output_path}"
        )

    print("\n" + "=" * 75)
    print(
        "V3 FEATURE GENERATION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
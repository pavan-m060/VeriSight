import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\ganes\Downloads\VeriSight"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_train_hybrid.csv"
)

TRAIN_EMB = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage2_text",
    "train_embeddings_fast.npy"
)

# Possible alternative embedding locations
POSSIBLE_EMBEDDINGS = [
    TRAIN_EMB,

    os.path.join(
        BASE_DIR,
        "embeddings",
        "stage2_text",
        "train_embeddings.npy"
    ),

    os.path.join(
        BASE_DIR,
        "embeddings",
        "stage2_text",
        "train_embeddings_full.npy"
    ),
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VERISIGHT STAGE 2 - TRAINING EMBEDDING AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN CSV
    # --------------------------------------------------------

    print("\nChecking training CSV...")

    if not os.path.exists(TRAIN_CSV):

        print("ERROR: Training CSV not found:")
        print(TRAIN_CSV)
        return

    print("OK:")
    print(TRAIN_CSV)

    print("\nReading training CSV...")

    df = pd.read_csv(
        TRAIN_CSV,
        usecols=[
            "user_id",
            "prod_id",
            "rating",
            "date",
            "spam"
        ]
    )

    print(
        f"Training CSV rows: {len(df):,}"
    )

    print("\nTraining label distribution:")

    print(
        df["spam"].value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # EXPECTED FULL TRAIN SIZE
    # --------------------------------------------------------

    expected_rows = len(df)

    print("\nExpected full training embeddings:")
    print(
        f"{expected_rows:,} rows × 384 dimensions"
    )

    # --------------------------------------------------------
    # SEARCH EMBEDDING FILES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING EMBEDDING FILES")
    print("=" * 70)

    found = []

    for path in POSSIBLE_EMBEDDINGS:

        if os.path.exists(path):

            size_mb = (
                os.path.getsize(path)
                / (1024 * 1024)
            )

            print("\nFOUND:")
            print(path)

            print(
                f"File size: {size_mb:.2f} MB"
            )

            found.append(path)

        else:

            print("\nNot found:")
            print(path)

    if not found:

        print("\nNo training embedding files found.")

        print(
            "\nRESULT:"
        )

        print(
            "We need to generate the full training embeddings."
        )

        return

    # --------------------------------------------------------
    # INSPECT EACH FILE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EMBEDDING SHAPE ANALYSIS")
    print("=" * 70)

    for path in found:

        print("\n" + "-" * 70)

        print(
            "File:",
            path
        )

        try:

            embeddings = np.load(
                path,
                mmap_mode="r"
            )

            print(
                "Shape:",
                embeddings.shape
            )

            print(
                "Dtype:",
                embeddings.dtype
            )

            print(
                "Rows:",
                f"{embeddings.shape[0]:,}"
            )

            print(
                "Dimensions:",
                embeddings.shape[1]
                if len(embeddings.shape) > 1
                else "N/A"
            )

            if (
                len(embeddings.shape) == 2
                and embeddings.shape[1] == 384
            ):

                print(
                    "Dimension check: PASS"
                )

            else:

                print(
                    "Dimension check: FAIL"
                )

            if embeddings.shape[0] == expected_rows:

                print(
                    "Row count check: PASS"
                )

                print(
                    "\n*** THIS LOOKS LIKE A FULL TRAINING "
                    "EMBEDDING FILE ***"
                )

            else:

                difference = (
                    expected_rows
                    - embeddings.shape[0]
                )

                print(
                    "Row count check: FAIL"
                )

                print(
                    f"Difference: {difference:,} rows"
                )

        except Exception as e:

            print(
                "ERROR loading embedding file:"
            )

            print(e)

    # --------------------------------------------------------
    # SPECIFIC CURRENT CACHE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CURRENT FAST EMBEDDING CACHE")
    print("=" * 70)

    if os.path.exists(TRAIN_EMB):

        embeddings = np.load(
            TRAIN_EMB,
            mmap_mode="r"
        )

        print(
            "\nCurrent cache:"
        )

        print(TRAIN_EMB)

        print(
            "\nShape:",
            embeddings.shape
        )

        print(
            "CSV rows:",
            f"{expected_rows:,}"
        )

        print(
            "Embedding rows:",
            f"{embeddings.shape[0]:,}"
        )

        if embeddings.shape[0] == expected_rows:

            print(
                "\nSTATUS: FULL TRAINING EMBEDDINGS"
            )

        else:

            percentage = (
                embeddings.shape[0]
                / expected_rows
                * 100
            )

            print(
                "\nSTATUS: SUBSET EMBEDDINGS"
            )

            print(
                f"Coverage: {percentage:.2f}%"
            )

            print(
                f"Missing rows: "
                f"{expected_rows - embeddings.shape[0]:,}"
            )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT RESULT")
    print("=" * 70)

    full_found = False
    full_path = None

    for path in found:

        try:

            emb = np.load(
                path,
                mmap_mode="r"
            )

            if (
                len(emb.shape) == 2
                and emb.shape[0] == expected_rows
                and emb.shape[1] == 384
            ):

                full_found = True
                full_path = path
                break

        except Exception:

            pass

    if full_found:

        print(
            "\nFULL TRAINING EMBEDDINGS FOUND."
        )

        print(
            "Use:"
        )

        print(full_path)

        print(
            "\nWe can proceed to full 487k hybrid training."
        )

    else:

        print(
            "\nFULL TRAINING EMBEDDINGS NOT FOUND."
        )

        print(
            "The existing training embedding file "
            "is only a subset."
        )

        print(
            "\nNEXT STEP:"
        )

        print(
            "Generate MiniLM embeddings for the full "
            "training dataset."
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
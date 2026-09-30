import os
import time
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\ganes\Downloads\VeriSight"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_train_hybrid.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage2_text"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "train_embeddings_full.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32

# Save progress every N rows
SAVE_EVERY = 10000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VERISIGHT - FULL STAGE 2 TRAINING EMBEDDINGS")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not os.path.exists(TRAIN_CSV):

        print("\nERROR: Training CSV not found:")
        print(TRAIN_CSV)

        return

    print("\nTraining CSV:")
    print(TRAIN_CSV)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training data...")

    df = pd.read_csv(
        TRAIN_CSV,
        usecols=["text"]
    )

    print(
        f"Training rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    del df

    # --------------------------------------------------------
    # LOAD MINILM
    # --------------------------------------------------------

    print("\nLoading MiniLM:")

    print(
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "\nMiniLM loaded successfully."
    )

    print(
        "Embedding dimension:",
        model.get_sentence_embedding_dimension()
    )

    # --------------------------------------------------------
    # CHECK EXISTING FULL FILE
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        print("\nExisting full embedding file found:")

        print(
            OUTPUT_FILE
        )

        try:

            existing = np.load(
                OUTPUT_FILE,
                mmap_mode="r"
            )

            print(
                "Existing shape:",
                existing.shape
            )

            if (
                existing.shape[0] == len(texts)
                and existing.shape[1] == 384
            ):

                print(
                    "\nFull embeddings already exist."
                )

                print(
                    "Nothing to generate."
                )

                return

        except Exception as e:

            print(
                "Existing file could not be used:"
            )

            print(e)

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    total = len(texts)

    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    print(
        f"Total reviews : {total:,}"
    )

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Output        : {OUTPUT_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This may take a while on CPU."
    )

    print(
        "Do not close the terminal while it is running."
    )

    # --------------------------------------------------------
    # PREALLOCATE
    # --------------------------------------------------------

    embeddings = np.zeros(
        (total, 384),
        dtype=np.float32
    )

    start_time = time.time()

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_texts = texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        embeddings[
            start:end
        ] = batch_embeddings.astype(
            np.float32
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        processed = end

        elapsed = time.time() - start_time

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total - processed
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else 0
        )

        percent = (
            processed / total * 100
        )

        print(
            f"\r"
            f"Progress: {processed:,}/{total:,} "
            f"({percent:6.2f}%) | "
            f"Speed: {rate:.1f} reviews/s | "
            f"ETA: {eta_seconds / 60:.1f} min",
            end="",
            flush=True
        )

        # ----------------------------------------------------
        # PERIODIC BACKUP
        # ----------------------------------------------------

        if (
            processed % SAVE_EVERY < BATCH_SIZE
            or processed == total
        ):

            temp_file = OUTPUT_FILE + ".tmp.npy"

            np.save(
                temp_file,
                embeddings
            )

    print("\n")

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    print(
        "Saving final embeddings..."
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    print(
        "\nVerifying output..."
    )

    saved = np.load(
        OUTPUT_FILE,
        mmap_mode="r"
    )

    print(
        "Output shape:",
        saved.shape
    )

    print(
        "Output dtype:",
        saved.dtype
    )

    expected_shape = (
        total,
        384
    )

    if saved.shape != expected_shape:

        print(
            "\nERROR: Shape mismatch!"
        )

        print(
            "Expected:",
            expected_shape
        )

        print(
            "Actual:",
            saved.shape
        )

        return

    # --------------------------------------------------------
    # CHECK NaN / INF
    # --------------------------------------------------------

    # Check a sample first to avoid unnecessarily loading
    # the entire large array into RAM again.

    sample = np.asarray(
        saved[:1000]
    )

    if not np.isfinite(sample).all():

        print(
            "\nERROR: NaN or Inf detected."
        )

        return

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    file_size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"\nReviews embedded : {total:,}"
    )

    print(
        f"Embedding shape  : {saved.shape}"
    )

    print(
        f"Embedding dtype  : {saved.dtype}"
    )

    print(
        f"File size        : {file_size_mb:.2f} MB"
    )

    print(
        f"Time taken       : {elapsed / 60:.2f} minutes"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nVerification: PASS")

    print("=" * 70)


if __name__ == "__main__":
    main()
"""
download_dataset_part2.py

Downloads the NEXT 150M characters of clean educational web text
after skipping the first 100M characters.

Output:
    data/corpus_part2.txt
"""

import os
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SPLIT = "train"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_part2.txt")

# Skip the first 100M characters (already used)
SKIP_CHARACTERS = 100_000_000

# Collect the next 150M characters
TARGET_CHARACTERS = 150_000_000


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading FineWeb-Edu...")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=SPLIT,
        streaming=True,
    )

    skipped_chars = 0
    collected_chars = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        pbar = tqdm(total=TARGET_CHARACTERS, unit="chars")

        for sample in dataset:

            text = sample.get("text", "").strip()

            if not text:
                continue

            # Skip the first 100M characters
            if skipped_chars < SKIP_CHARACTERS:
                skipped_chars += len(text)
                continue

            # Save the next 150M characters
            f.write(text)
            f.write("\n\n")

            collected_chars += len(text)
            pbar.update(len(text))

            if collected_chars >= TARGET_CHARACTERS:
                break

        pbar.close()

    print("\nDone!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Characters skipped:   {skipped_chars:,}")
    print(f"Characters collected: {collected_chars:,}")


if __name__ == "__main__":
    main()
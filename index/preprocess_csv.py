import math

import pandas as pd
import utilities_v3 as utils
import argparse
import sys
import csv
import os
import re


CHUNK_SIZE = 100000
SEED = 404
VALID_LABELS = ["fake", "reliable"]  # TODO: adapt for multiple labels
csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--balanced-size", type=int, default=3e5)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-balancing", action="store_true")

    args = parser.parse_args()
    utils.save_args(args, args.out_dir, "preprocess_csv")
    return args


def normalize_string(s):
    s = str(s or "")
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = "\n".join(paragraph.strip() for paragraph in s.split("\n"))
    s = re.sub(r"[\t\s]+", " ", s)
    return s.lower().strip()


def preprocess_chunk(chunk):
    new_chunk = chunk[["id", "type", "title", "content"]]
    new_chunk = new_chunk[new_chunk["type"].isin(VALID_LABELS)]
    new_chunk.rename(columns={"type": "label"}, inplace=True)
    new_chunk["content"] = new_chunk["content"].map(normalize_string)
    new_chunk["title"] = new_chunk["title"].map(normalize_string)
    # TODO: remove bad content, eg. truncated content
    # TODO: remove illegal characters,
    # TODO: remove photo alts, mentions of publication
    # TODO: drop nulls, NA
    return new_chunk


def check_exists(path: str) -> bool:
    if os.path.exists(path):
        print("File exists: {}".format(path))
        return True
    return False


def preprocess(csv_path, output_path):
    if check_exists(csv_path):
        return
    row_count = 0
    with pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        engine="python",
    ) as reader:
        for chunk in reader:
            new_chunk = preprocess_chunk(chunk)
            row_count += new_chunk.shape[0]
            if os.path.isfile(output_path):
                new_chunk.to_csv(output_path, mode="a", header=False, index=False)
            else:
                new_chunk.to_csv(output_path, index=False, header=True)
            del chunk, new_chunk


def balanced_sample(
    csv_path, output_path, output_path_fake, output_path_reliable, max_rows
):
    df = pd.read_csv(csv_path, engine="python")
    counts = df["label"].value_counts(sort=False)
    rows_per_label = max_rows // len(VALID_LABELS)
    rows_per_label = int(min(rows_per_label, counts.min()))
    sampled = df.groupby("label", group_keys=False).sample(
        n=rows_per_label, random_state=SEED
    )
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled_fake = sampled[sampled["label"] == "fake"]
    sampled_reliable = sampled[sampled["label"] == "reliable"]
    sampled_fake.to_csv(output_path_fake, index=False)
    sampled_reliable.to_csv(output_path_reliable, index=False)
    sampled.to_csv(output_path, index=False)
    return sampled


def split(csv_path, output_dir, test_split, validation_split, split_label=None):
    if split_label is None:
        train_out = os.path.join(output_dir, "train.csv")
        test_out = os.path.join(output_dir, "test.csv")
        val_out = os.path.join(output_dir, "val.csv")
    else:
        train_out = os.path.join(output_dir, f"train_{split_label}.csv")
        test_out = os.path.join(output_dir, f"test_{split_label}.csv")
        val_out = os.path.join(output_dir, f"val_{split_label}.csv")

    if os.path.isfile(train_out) or os.path.isfile(test_out) or os.path.isfile(val_out):
        return
    with pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        engine="python",
    ) as reader:
        for chunk in reader:
            test = chunk.sample(frac=test_split, replace=False, random_state=SEED)
            rest = chunk.drop(test.index)
            val = rest.sample(
                frac=validation_split / (1 - test_split),
                replace=False,
                random_state=SEED,
            )
            train = rest.drop(val.index)

            if os.path.isfile(train_out):
                train.to_csv(train_out, mode="a", header=False, index=False)
            else:
                train.to_csv(train_out, index=False, header=True)
            if os.path.isfile(val_out):
                val.to_csv(val_out, mode="a", header=False, index=False)
            else:
                val.to_csv(val_out, index=False, header=True)
            if os.path.isfile(test_out):
                test.to_csv(test_out, mode="a", header=False, index=False)
            else:
                test.to_csv(test_out, index=False, header=True)


def main():
    args = parse_args()
    filepath = os.path.join(args.out_dir, "news_preprocessed.csv")
    filepath_2 = os.path.join(args.out_dir, "news_balanced.csv")
    filepath_2_fake = os.path.join(args.out_dir, "news_balanced_fake.csv")
    filepath_2_reliable = os.path.join(args.out_dir, "news_balanced_reliable.csv")
    if not args.skip_preprocessing:
        preprocess(args.input, filepath)
    if not args.skip_balancing:
        balanced_sample(
            filepath,
            filepath_2,
            filepath_2_fake,
            filepath_2_reliable,
            args.balanced_size,
        )
    split(filepath_2, args.out_dir, args.test_split, args.val_split)
    split(filepath_2_fake, args.out_dir, args.test_split, args.val_split, "fake")
    split(
        filepath_2_reliable, args.out_dir, args.test_split, args.val_split, "reliable"
    )


if __name__ == "__main__":
    main()

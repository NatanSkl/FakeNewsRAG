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

    args = parser.parse_args()
    utils.save_args(args, args.out_dir)
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


def preprocess(csv_path, output_path):
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
    return row_count

def split(csv_path, output_dir, test_split, validation_split):
    row_count = 0
    train_out = os.path.join(output_dir, "train.csv")
    test_out = os.path.join(output_dir, "test.csv")
    val_out = os.path.join(output_dir, "val.csv")
    with pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        engine="python",
    ) as reader:
        for chunk in reader:
            test = chunk.sample(frac=test_split, replace=False, random_state=SEED)
            val = chunk.sample(frac=validation_split, replace=False, random_state=SEED)
            train = chunk.drop(test.index).drop(val.index)

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
    row_count = preprocess(args.input, os.path.join(args.out_dir, "/news_preprocessed.csv"))

if __name__ == "__main__":
    main()

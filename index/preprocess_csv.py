import os
import pandas as pd
import sys
import csv
import re


CHUNK_SIZE = 1000
VALID_LABELS = ["fake", "reliable"]  # TODO: adapt for multiple labels
csv.field_size_limit(sys.maxsize)


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
    if os.path.isfile(output_path):
        os.remove(output_path)
        print("Preprocessed CSV already exists")
        # return
        # TODO: switch to not overwriting once preprocess is stable
    with pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        engine="python",
    ) as reader:
        for chunk in reader:
            new_chunk = preprocess_chunk(chunk)
            if os.path.isfile(output_path):
                new_chunk.to_csv(output_path, mode="a", header=False, index=False)
            else:
                new_chunk.to_csv(output_path, index=False, header=True)
            del chunk, new_chunk


preprocess("data/news.csv", "data/news_preprocessed.csv")

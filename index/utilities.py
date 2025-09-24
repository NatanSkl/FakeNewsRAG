import json
import faiss
import os.path
import argparse
import pandas as pd
from typing import Any, List, Iterable


def load_dataframe(path: str, limit: int = 0, chunksize: int = 0) -> pd.DataFrame:
    if chunksize <= 0:
        df = pd.read_csv(path)
        if limit and limit > 0:
            df = df.head(limit)
        return df

    frames: List[pd.DataFrame] = []
    read_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if limit and limit > 0:
            if limit - read_rows <= 0:
                break
            chunk = chunk.head(limit - read_rows)
        frames.append(chunk)
        read_rows += chunksize
        if limit and read_rows >= limit:
            break
    return pd.concat(frames, ignore_index=True)


def validate_columns(columns: Iterable[str], required: List[str]):
    if [col for col in required if col not in columns]:
        raise ValueError("Missing required columns in CSV")


def cast_string(candidate: Any) -> str:
    if pd.isna(candidate):
        return ""
    return str(candidate)


def concat_text(title: str, content: str) -> str:
    if title and content:
        return f"{title}\n{content}"
    return title or content


def save_args(args: argparse.Namespace, path: str) -> None:
    dict_args = vars(args)
    if "input" in dict_args and dict_args["input"] is not None:
        dict_args["input"] = os.path.abspath(dict_args["input"])
    with open(os.path.join(path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(dict_args, f, ensure_ascii=False, indent=4)


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, os.path.join(path, "index.faiss"))

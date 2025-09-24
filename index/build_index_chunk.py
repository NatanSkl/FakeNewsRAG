import os
import gc
import argparse
from typing import List

from utilities import (
    load_dataframe,
    validate_columns,
    cast_string,
    concat_text,
    save_args,
    save_index,
)

while True:
    try:
        import faiss
        import tiktoken
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from sentence_transformers import SentenceTransformer
    except ImportError:
        os.system(
            "pip install faiss-cpu tiktoken sentence-transformers pandas numpy tqdm"
        )
        # TODO: Switch to gpu
        # TODO: handle import errors when package already installed, avoid infinite loops
        continue
    break


COLUMNS = ["id", "label", "title", "content"]


def parse_args() -> argparse.Namespace:
    # TODO: add help to all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="index/store")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--index-type", default="IndexFlatIP", choices=["IndexFlatIP", "IndexFlatL2"]
    )
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument(
        "--save-metadata-as", default="parquet", choices=["parquet", "csv", "jsonl"]
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    save_args(args, args.outdir)
    return args


def build_index(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    build_index(args)
    print(args)

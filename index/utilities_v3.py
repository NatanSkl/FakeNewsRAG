import os
import json
from typing import Any, List, Iterable, Dict

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse


def load_dataframe_chunks(
    path: str, limit: int, chunksize: int
) -> Iterable[pd.DataFrame]:
    """
    Steams data from a CSV file in chunks as DataFrames.
    """
    read_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if limit and limit > 0:
            remaining = limit - read_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)
        yield chunk
        read_rows += len(chunk)
        if limit and read_rows >= limit:
            break


def validate_columns(columns: Iterable[str], required: List[str]):
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns in CSV: {missing}")


def save_args(args: argparse.Namespace, path: str) -> None:
    """
    Save the provided command-line arguments to a JSON file in a given directory.
    """
    args_dict = vars(args).copy()
    if "input" in args_dict and args_dict["input"] is not None:
        args_dict["input"] = os.path.abspath(args_dict["input"])
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)


def _hash64(text: str) -> np.int64:
    h = np.int64(1469598103934665603)
    for char in text.encode("utf-8"):
        h = np.int64(h ^ np.int64(char))
        h = np.int64(h * np.int64(1099511628211))
    return h


def make_vector_id(db_id: str, chunk_id: int) -> np.int64:
    try:
        base = np.int64(int(db_id))
        return np.int64((base << 16) | (chunk_id & 0xFFFF))
    except Exception:
        base = _hash64(str(db_id))
        return np.int64(base ^ np.int64(chunk_id & 0xFFFF))


class MetadataSink:
    def __init__(self, path: str, append: bool = False) -> None:
        self.path = path
        self.append = append

    def write(self, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def get_path(self) -> str:
        return self.path


class ParquetSink(MetadataSink):
    def __init__(self, path: str, append: bool = False) -> None:
        super().__init__(path, append)
        self.append = append
        self.path = os.path.join(path, "metadata.parquet")
        self.writer = None

        if self.append and os.path.exists(self.path):
            # TODO: maybe switch to true appends, unclear if necessary yet
            base, ext = os.path.splitext(self.path)
            self.path = f"{base}_append{ext}"

    def write(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class CSVSink(MetadataSink):
    def __init__(self, path: str, append: bool = False) -> None:
        super().__init__(path, append)
        self.append = append
        self.path = os.path.join(path, "metadata.csv")
        self.header_written = False
        if append and os.path.exists(self.path):
            self.header_written = True

    def write(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(self.path, mode="a", header=not self.header_written, index=False)
        self.header_written = True

    def close(self) -> None:
        pass


class JSONLSink(MetadataSink):
    def __init__(self, path: str, append: bool = False) -> None:
        super().__init__(path, append)
        self.append = append
        self.path = os.path.join(path, "metadata.jsonl")
        self.file = open(self.path, "a" if self.append else "w", encoding="utf-8")

    def write(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.file.close()


def get_metadata_sink(path: str, kind: str, append: bool = False) -> MetadataSink:
    """
    Retrieves a metadata sink based on the specified file format.
    """
    if kind == "parquet":
        return ParquetSink(path, append)
    if kind == "csv":
        return CSVSink(path, append)
    if kind == "jsonl":
        return JSONLSink(path, append)
    else:
        raise ValueError(f"[ERROR] Unsupported metadata format: {kind}")

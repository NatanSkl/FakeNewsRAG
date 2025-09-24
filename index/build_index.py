import os
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
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument(
        "--save-metadata-as", default="parquet", choices=["parquet", "csv", "jsonl"]
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    save_args(args, args.outdir)
    return args


def get_column_indexes(df: pd.DataFrame):
    id_index = df.columns.get_loc("id")
    label_index = df.columns.get_loc("label")
    title_index = df.columns.get_loc("title")
    content_index = df.columns.get_loc("content")
    return id_index, label_index, title_index, content_index


def chunk_tokens(
    text: str, encoding: tiktoken.Encoding, chunk_size: int, overlap: int
) -> List[str]:

    if chunk_size <= 0:
        return [text]
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = encoding.encode(text or "")
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        piece = tokens[start:end]
        chunks.append(encoding.decode(piece))
        if end == len(tokens):
            break
        start += step
    return chunks


def build_embeddings(
    model_name: str, texts: List[str], batch_size: int, normalize: bool
) -> np.ndarray:

    model = SentenceTransformer(model_name)
    vectors_list: List[np.ndarray] = []
    iterable = tqdm(
        range(0, len(texts), batch_size), desc="Embedding vectors", unit="batch"
    )

    for i in iterable:
        batch = texts[i : i + batch_size]
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        vectors_list.append(vectors)

    if not vectors_list:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors_list)


def get_index(dimension: int, index_type: str, path: str, append: bool) -> faiss.Index:

    if append:
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)

            if index.d != dimension:
                raise ValueError(
                    f"Existing index dim ({index.d}) doesn't match new embeddings dim ({dimension})"
                )
            if index_type == "IndexFlatIP" and not isinstance(index, faiss.IndexFlatIP):
                raise ValueError(f"Existing index type is not IndexFlatIP")
            if index_type == "IndexFlatL2" and not isinstance(index, faiss.IndexFlatL2):
                raise ValueError(f"Existing index type is not IndexFlatL2")

            return index

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(
            f"Unsupported index type: {index_type}. Probably an issue with args loading"
        )
    return index


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_dataframe(args.input, limit=args.limit)
    validate_columns(df, COLUMNS)

    encoding = tiktoken.get_encoding(args.encoding)
    rows = []
    rows_iterator = tqdm(
        df.itertuples(index=False), total=len(df), desc="Preparing encoding", unit="row"
    )

    # TODO: replace with chunk processing to reduce memory footprint

    id_index, label_index, title_index, content_index = get_column_indexes(df)
    for row in rows_iterator:
        row_id = cast_string(row[id_index])
        row_label = cast_string(row[label_index])
        row_title = cast_string(row[title_index])
        row_content = cast_string(row[content_index])
        row_text = concat_text(row_title, row_content)

        if args.chunk_tokens and args.chunk_tokens > 0:
            chunks = chunk_tokens(
                row_text, encoding, args.chunk_tokens, args.chunk_overlap
            )
            for index, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                new_row = {
                    "db_id": row_id,
                    "label": row_label,
                    "title": row_title,
                    "content": row_content,
                    "chunk_text": chunk,
                    "chunk_id": index,
                    "token_count": len(encoding.encode(chunk)),
                }
                rows.append(new_row)
        else:
            new_row = {
                "db_id": row_id,
                "label": row_label,
                "title": row_title,
                "content": row_content,
                "chunk_text": row_text,
                "chunk_id": 0,
                "token_count": len(encoding.encode(row_text)),
            }
            rows.append(new_row)

    if not rows:
        raise RuntimeError("No valid rows to index.")

    chunk_texts = [row["chunk_text"] for row in rows]
    embeddings = build_embeddings(
        args.model, chunk_texts, args.batch_size, args.normalize
    )
    dimension = embeddings.shape[1] if embeddings.size else 0

    if embeddings.shape[0] != len(rows):
        raise RuntimeError("Embeddings count does not match rows count.")
    if dimension == 0:
        raise RuntimeError("No embeddings produced.")

    index = get_index(dimension, args.index_type, args.outdir, args.append)
    index.add(embeddings)

    # TODO: Build metadata table aligned to vector order to preserve metadata, and save it

    save_index(index, args.outdir)


if __name__ == "__main__":
    main()

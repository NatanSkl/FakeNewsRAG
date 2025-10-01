import os
import argparse
import pickle
from typing import List, Tuple, Dict, Any, Union

import utilities_v3 as utils

try:
    import faiss
    import torch
    import tiktoken
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[WARN] Missing dependencies. Installing now.")
    os.system(
        "pip install torch rank-bm25 faiss-cpt tiktoken sentence_transformers pandas numpy tqdm pyarrow"
    )
    import faiss
    import torch
    import tiktoken
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer


COLUMNS = ["id", "label", "title", "content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--b25-out", type=str, default="bm25.pkl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=2e5)

    # Model parameters
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--normalize", action="store_true")

    # Tokenization and chunking parameters
    parser.add_argument("--chunk-size", type=int, default=1e5)
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--use-encoding", action="store_true")
    parser.add_argument("--encoding", default="cl100k_base")

    # Index parameters
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--index-type",
        default="IVFPQ-IP",
        choices=[
            "FlatIP",
            "FlatL2",
            "IVF-IP",
            "IVF-L2",
            "IVFPQ-IP",
            "IVFPQ-L2",
            "HNSW-IP",
            "HNSW-L2",
        ],
    )

    # IVF / PQ / HNSW parameters
    parser.add_argument(
        "--train-sample", type=int, default=10**6
    )  # sample size for training
    parser.add_argument("--nlist", type=int, default=16384)  # IVF: number of centroids
    parser.add_argument("--nprobe", type=int, default=32)  # IVF: search probes
    parser.add_argument("--pq-m", type=int, default=32)  # IVFPQ: sub-vectors
    parser.add_argument("--pq-bits", type=int, default=8)  # IVFPQ: bits per code
    parser.add_argument("--hnsw-m", type=int, default=32)  # HNSW: Neighbor degree

    # Metadata parameters
    parser.add_argument(
        "--save-metadata-as", default="parquet", choices=["parquet", "csv", "jsonl"]
    )

    args = parser.parse_args()
    utils.save_args(args, args.out_dir)
    return args


def embed_batches(
    texts: List[str], model: SentenceTransformer, batch_size: int, normalize: bool
) -> np.ndarray:
    vectors_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if not batch:
            continue
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        ).astype(np.float32)
        vectors_list.append(vectors)
    if not vectors_list:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors_list)


def make_index(dim: int, args: argparse.Namespace) -> faiss.Index:
    use_ip = "IP" in args.index_type
    metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2
    flat = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)

    if args.index_type.startswith("Flat"):
        return flat

    if args.index_type.startswith("IVF") and "PQ" not in args.index_type:
        return faiss.IndexIVFFlat(flat, dim, args.nlist, metric)

    if args.index_type.startswith("IVFPQ"):
        return faiss.IndexIVFPQ(flat, dim, args.nlist, args.pq_m, args.pq_bits, metric)

    if args.index_type.startswith("HNSW") and metric == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)

    if args.index_type.startswith("HNSW"):
        return faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_L2)

    raise ValueError(f"[ERROR] Unsupported index type: {args.index_type}")


def infer_dim(args: argparse.Namespace, model: SentenceTransformer) -> int:
    for df in utils.load_dataframe_chunks(
        args.input, limit=args.limit, chunksize=args.chunk_size
    ):
        utils.validate_columns(df.columns, COLUMNS)
        texts = (df["title"].astype(str) + "\n" + df["content"].astype(str)).tolist()
        if not texts:
            continue
        batch = texts[: min(len(texts), args.batch_size)]
        batch_embeddings = embed_batches(batch, model, args.batch_size, args.normalize)
        if batch_embeddings.size > 0:
            return int(batch_embeddings.shape[1])
    raise RuntimeError("[ERROR] Failed to infer embedding dimensions.")


def build_index(
    args: argparse.Namespace, model: SentenceTransformer
) -> Tuple[faiss.Index, int]:
    dim = infer_dim(args, model)
    pre_index = make_index(dim, args)

    # IVF / IVFPQ, if picked as index type
    if isinstance(pre_index, faiss.IndexIVF):
        # Collect training sample
        sample_vectors: List[np.ndarray] = []
        total_sampled = 0

        for df in utils.load_dataframe_chunks(
            args.input, limit=args.limit, chunksize=args.chunk_size
        ):
            texts = (
                df["title"].astype(str) + "\n" + df["content"].astype(str)
            ).tolist()
            if not texts:
                continue

            # Calc sample fraction
            frac = min(1.0, max(0.01, args.train_sample / max(len(texts), 1)))
            if 0 < frac < 1.0:
                samp = df.sample(frac=frac, random_state=1332)
                texts = (
                    samp["title"].astype(str) + "\n" + samp["content"].astype(str)
                ).tolist()
            batch_embeddings = embed_batches(
                texts, model, args.batch_size, args.normalize
            )
            if batch_embeddings.size == 0:
                continue
            sample_vectors.append(batch_embeddings)
            total_sampled += batch_embeddings.shape[0]
            if total_sampled >= args.train_sample:
                break
        if not sample_vectors:
            raise RuntimeError(
                "[ERROR] Failed to sample vectors for IVF / IVFPQ training."
            )
        train_matrix = np.vstack(sample_vectors).astype(np.float32)
        pre_index.train(train_matrix)

    # wrap pre_index with IDMap
    index = faiss.IndexIDMap2(pre_index)
    if isinstance(pre_index, faiss.IndexIVF):
        pre_index.nprobe = args.nprobe
    return index, dim


def chunk_tokens(
    text: str, encoder: tiktoken.Encoding, chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, List[int]]]:
    tokens = encoder.encode(text or "")
    if chunk_size <= 0:
        return [(text, tokens)]
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "[ERROR] overlap must be non-negative and smaller than chunk size"
        )
    if len(tokens) <= chunk_size:
        return [(text, tokens)]
    chunks: List[Tuple[str, List[int]]] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        piece = tokens[start:end]
        chunks.append((encoder.decode(piece),piece))
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_words(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, List[str]]]:
    words = text.split()
    if chunk_size <= 0:
        return [(text, words)]
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "[ERROR] overlap must be non-negative and smaller than chunk size"
        )
    if len(words) <= chunk_size:
        return [(text, words)]
    chunks: List[Tuple[str, List[str]]] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        piece_words = words[start : start + chunk_size]
        if not piece_words:
            break
        chunks.append((" ".join(piece_words), piece_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def add_vectors_streaming(
    args: argparse.Namespace,
    index: faiss.Index,
    model: SentenceTransformer,
    encoder: tiktoken.Encoding,
    metadata_sink: utils.MetadataSink,
) -> int:
    added = 0
    out_path = os.path.join(args.out_dir, "index.faiss")

    bm25_corpus: List[List[str]] = []
    bm25_ids: List[int] = []

    for df in utils.load_dataframe_chunks(
        args.input, limit=args.limit, chunksize=args.chunk_size
    ):
        utils.validate_columns(df.columns, COLUMNS)
        df = df[COLUMNS].dropna()

        texts_list = (
            df["title"].astype(str) + "\n" + df["content"].astype(str)
        ).tolist()

        # row-by-row chunking
        if args.chunk_tokens <= 0:
            # no chunking
            db_ids = df["id"].astype(str).tolist()
            labels = df["label"].astype(str).tolist()
            titles = df["title"].astype(str).tolist()
            contents = df["content"].astype(str).tolist()
            texts = texts_list

            ids = [utils.make_vector_id(db_id, 0) for db_id in db_ids]
            word_counts = [len(t.split()) for t in texts_list]
            meta_rows = [
                {
                    "vector_id": v_id,
                    "db_id": db_id,
                    "chunk_id": 0,
                    "label": l,
                    "title": t,
                    "content": c,
                    "token_count": wc,
                }
                for v_id, db_id, l, t, c, wc in zip(
                    ids, db_ids, labels, titles, contents, word_counts
                )
            ]
        else:
            # chunk the text of articles based on words / tokens
            texts: List[str] = []
            ids: List[np.int64] = []
            meta_rows: List[Dict[str, Any]] = []

            for db_id, label, title, content, text in zip(
                df["id"].astype(str).tolist(),
                df["label"].astype(str).tolist(),
                df["title"].astype(str).tolist(),
                df["content"].astype(str).tolist(),
                texts_list,
            ):
                if args.use_encoding:
                    chunks_tuples = chunk_tokens(
                        text, encoder, args.chunk_tokens, args.chunk_overlap
                    )
                else:
                    chunks_tuples = chunk_words(text, args.chunk_tokens, args.chunk_overlap)
                for chunk_id, (chunk_text, tokens) in enumerate(chunks_tuples):
                    if not chunk_text.strip():
                        continue
                    v_id = utils.make_vector_id(db_id, chunk_id)
                    ids.append(v_id)
                    texts.append(chunk_text)
                    meta_rows.append(
                        {
                            "vector_id": v_id,
                            "db_id": db_id,
                            "chunk_id": chunk_id,
                            "label": label,
                            "title": title,
                            "content": content,
                            "token_count": len(tokens),
                        }
                    )
                    if args.use_encoding:
                        bm25_corpus.append([str(token) for token in tokens])
                    else:
                        bm25_corpus.append(tokens)
                    bm25_ids.append(v_id)


        if not texts:
            continue

        batch_embeddings = embed_batches(texts, model, args.batch_size, args.normalize)
        ids_array = np.array(ids, dtype=np.int64)
        index.add_with_ids(batch_embeddings, ids_array)
        added += len(ids_array)
        metadata_sink.write(meta_rows)
        if added % args.checkpoint_every < len(ids_array):
            faiss.write_index(index, out_path)
            print(f"[INFO] Checkpointed at {added} vectors")

    faiss.write_index(index, out_path)
    print(f"[INFO] Done adding vectors. Total added: {added}")
    print(f"[INFO] Training BM25 on {len(bm25_corpus)} documents]")
    bm25 = BM25Okapi(bm25_corpus)  # ,tokenizer=encoder)
    bm25.doc_ids = bm25_ids
    bm25_path = os.path.join(args.out_dir, args.bm25_out)
    with open (bm25_path, "wb") as bm25_file:
        pickle.dump(bm25, bm25_file)
    print(f"[INFO] Done saving BM25 vectors to {bm25_path}]")


def main() -> None:
    args = parse_args()
    index_path = os.path.join(args.out_dir, "index.faiss")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize tokenizer and model
    encoder = tiktoken.get_encoding(args.encoding)
    model = SentenceTransformer(args.model, device=device)
    print(
        f"[INFO] Model loaded: {args.model} | Device: {device} | Encoding: {args.encoding if args.use_encoding else 'words'}"
    )

    # If set to append, load the existing index. If now, train a new one.
    if args.append and os.path.exists(index_path):
        index = faiss.read_index(index_path)
        try:
            # set correct index nprobe if necessary
            if isinstance(index, faiss.IndexIDMap2) and isinstance(
                index.index, faiss.IndexIVF
            ):
                index.index.nprobe = args.nprobe
            elif isinstance(index, faiss.IndexIVF):
                index.nprobe = args.nprobe
        except Exception:
            pass
        dim = index.d
        print(f"[INFO] Index loaded: {index_path}")

    else:
        index, dim = build_index(args, model)
        print(f"[INFO] Index built: {index_path}")

    # Open metadata sink
    metadata_sink = utils.get_metadata_sink(
        args.out_dir, args.save_metadata_as, append=args.append
    )
    print(f"[INFO] Metadata sink opened: {metadata_sink.get_path()}")

    add_vectors_streaming(args, index, model, encoder, metadata_sink)

    metadata_sink.close()
    print(f"[INFO] Metadata sink closed.")


if __name__ == "__main__":
    main()

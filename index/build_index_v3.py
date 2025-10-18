import os
import time
import logging
import faiss
import torch
import pickle
import argparse
import tiktoken
import numpy as np
import utilities_v3 as utils
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


COLUMNS = ["id", "label", "title", "content"]


def setup_logging() -> None:
    """Configure logging to output to output.log with timestamps and appropriate levels."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('output.log'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    logging.info("Logging initialized - output will be written to output.log")


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    try:
        # GPU information
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.debug(f"GPU memory: {gpu_memory:.1f} GB")
        
    except Exception as e:
        logging.warning(f"Could not retrieve system information: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--bm25-out", type=str, default="bm25.pkl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=2e5)

    # Model parameters
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
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
        "--train-sample", type=int, default=7.5e4
    )  # sample size for training
    parser.add_argument("--nlist", type=int, default=1024)  # IVF: number of centroids
    parser.add_argument("--nprobe", type=int, default=16)  # IVF: search probes
    parser.add_argument("--pq-m", type=int, default=16)  # IVFPQ: sub-vectors
    parser.add_argument("--pq-bits", type=int, default=8)  # IVFPQ: bits per code
    parser.add_argument("--hnsw-m", type=int, default=32)  # HNSW: Neighbor degree

    # Metadata parameters
    parser.add_argument(
        "--save-metadata-as", default="parquet", choices=["parquet", "csv", "jsonl"]
    )

    args = parser.parse_args()
    utils.save_args(args, args.out_dir, "build_index")
    return args


def embed_batches(
    texts: List[str], model: SentenceTransformer, batch_size: int, normalize: bool
) -> np.ndarray:
    logging.debug(
        f"embed_batches called with {len(texts)} texts, batch_size={batch_size}"
    )
    vectors_list: List[np.ndarray] = []
    encode_times = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if not batch:
            continue
        logging.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} texts")
        logging.debug("About to call model.encode()")

        # Time the model.encode() call
        start_time = time.time()
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        ).astype(np.float32)
        end_time = time.time()

        encode_time = end_time - start_time
        encode_times.append(encode_time)

        logging.debug(
            f"model.encode() completed in {encode_time:.3f}s, shape: {vectors.shape}"
        )
        vectors_list.append(vectors)

    if not vectors_list:
        logging.debug("No vectors generated, returning empty array")
        return np.zeros((0, 0), dtype=np.float32)

    # Calculate and report timing statistics
    if encode_times:
        avg_time = sum(encode_times) / len(encode_times)
        total_time = sum(encode_times)
        logging.info(f"Model.encode() statistics:")
        logging.info(f"  - Total batches: {len(encode_times)}")
        logging.info(f"  - Average time per batch: {avg_time:.3f}s")
        logging.info(f"  - Total encoding time: {total_time:.3f}s")
        logging.info(f"  - Min time: {min(encode_times):.3f}s")
        logging.info(f"  - Max time: {max(encode_times):.3f}s")

    result = np.vstack(vectors_list)
    logging.debug(f"Final embeddings shape: {result.shape}")
    
    return result


def make_index(dim: int, args: argparse.Namespace) -> faiss.Index:
    """
    Create a FAISS index, automatically using GPU if available.
    Falls back to CPU if GPU is not available.
    """
    use_ip = "IP" in args.index_type
    metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

    # Helper for multi-GPU options
    def _multi_gpu_opts():
        co = faiss.GpuMultipleClonerOptions()
        co.shard = bool(getattr(args, "shard", False))
        co.useFloat16 = bool(getattr(args, "use_float16", True))
        return co

    # ---------- HNSW (CPU only) ----------
    if args.index_type.startswith("HNSW"):
        return faiss.IndexHNSWFlat(dim, args.hnsw_m, metric)

    flat = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    ngpu = faiss.get_num_gpus()

    # ---------- FLAT ----------
    if args.index_type.startswith("Flat"):
        # Move to GPU if available
        if ngpu > 0:
            logging.info("Moving index to GPU")
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, flat)
        return flat

    # ---------- IVF / PQ----------
    if args.index_type.startswith("IVF"):
        if "PQ" not in args.index_type:
            cpu_index = faiss.IndexIVFFlat(flat, dim, args.nlist, metric)
        else:
            cpu_index = faiss.IndexIVFPQ(
                flat, dim, args.nlist, args.pq_m, args.pq_bits, metric
            )

        # Move to GPU if available
        if ngpu > 0:
            logging.info("Moving index to GPU")
            if ngpu == 1:
                res = faiss.StandardGpuResources()
                return faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                return faiss.index_cpu_to_all_gpus(cpu_index, co=_multi_gpu_opts())
        return cpu_index

    raise ValueError(f"[ERROR] Unsupported index type: {args.index_type}")


def infer_dim(args: argparse.Namespace, model: SentenceTransformer) -> int:
    logging.debug("infer_dim function started")
    for df in utils.load_dataframe_chunks(
        args.input, limit=args.limit, chunksize=args.chunk_size
    ):
        logging.debug(f"Processing chunk for dimension inference with {len(df)} rows")
        utils.validate_columns(df.columns, COLUMNS)
        texts = (df["title"].astype(str) + "\n" + df["content"].astype(str)).tolist()
        if not texts:
            logging.debug("No texts in chunk for dimension inference, skipping")
            continue
        batch = texts[: min(len(texts), args.batch_size)]
        logging.debug(
            f"About to embed batch of {len(batch)} texts for dimension inference"
        )
        batch_embeddings = embed_batches(batch, model, args.batch_size, args.normalize)
        if batch_embeddings.size > 0:
            dim = int(batch_embeddings.shape[1])
            logging.debug(f"Successfully inferred dimension: {dim}")
            return dim
    raise RuntimeError("Failed to infer embedding dimensions.")


def build_index(
    args: argparse.Namespace, model: SentenceTransformer
) -> Tuple[faiss.Index, int]:
    logging.debug("build_index function started")

    # Check GPU availability
    ngpu = faiss.get_num_gpus()
    if ngpu > 0:
        logging.info(f"FAISS GPU support detected: {ngpu} GPU(s) available")
    else:
        logging.info("FAISS GPU support not available, using CPU")

    logging.debug("About to call infer_dim")
    dim = infer_dim(args, model)
    logging.debug(f"infer_dim completed, dimension: {dim}")

    logging.debug("About to call make_index")
    pre_index = make_index(dim, args)
    logging.debug("make_index completed successfully")

    # Determine if we have an IVF-based index that needs training
    needs_training = False
    if isinstance(pre_index, faiss.IndexIVF):
        needs_training = True
    elif hasattr(pre_index, "index") and isinstance(pre_index.index, faiss.IndexIVF):
        # GPU-wrapped IVF index
        needs_training = True

    # IVF / IVFPQ training (automatically happens on GPU if index is on GPU)
    if needs_training:
        logging.debug("Index is IVF type, starting training sample collection")
        # Collect training sample
        sample_vectors: List[np.ndarray] = []
        total_sampled = 0

        logging.debug("About to start loading dataframe chunks for training")
        for df in utils.load_dataframe_chunks(
            args.input, limit=args.limit, chunksize=args.chunk_size
        ):
            logging.debug(f"Processing training chunk with {len(df)} rows")
            texts = (
                df["title"].astype(str) + "\n" + df["content"].astype(str)
            ).tolist()
            if not texts:
                logging.debug("No texts in training chunk, skipping")
                continue

            # Calc sample fraction
            frac = min(1.0, max(0.01, args.train_sample / max(len(texts), 1)))
            if 0 < frac < 1.0:
                logging.debug(f"Sampling {frac:.2%} of {len(texts)} texts for training")
                samp = df.sample(frac=frac, random_state=1332)
                texts = (
                    samp["title"].astype(str) + "\n" + samp["content"].astype(str)
                ).tolist()
            logging.debug(f"About to embed {len(texts)} texts for training")
            batch_embeddings = embed_batches(
                texts, model, args.batch_size, args.normalize
            )
            if batch_embeddings.size == 0:
                logging.debug("No embeddings generated for training, skipping")
                continue
            sample_vectors.append(batch_embeddings)
            total_sampled += batch_embeddings.shape[0]
            logging.debug(f"Total sampled for training so far: {total_sampled}")
            if total_sampled >= args.train_sample:
                logging.debug("Reached training sample limit, breaking")
                break

        if not sample_vectors:
            raise RuntimeError(
                "Failed to sample vectors for IVF / IVFPQ training."
            )

        logging.debug(f"Training matrix shape: {len(sample_vectors)} vectors")
        train_matrix = np.vstack(sample_vectors).astype(np.float32)
        logging.debug(f"About to train index with matrix shape: {train_matrix.shape}")

        # Train (automatically on GPU if index is on GPU)
        pre_index.train(train_matrix)
        training_location = "GPU" if ngpu > 0 else "CPU"
        logging.info(f"Index training completed on {training_location}")

    # wrap pre_index with IDMap
    logging.debug("Wrapping index with IDMap")
    index = faiss.IndexIDMap2(pre_index)

    # Set nprobe for IVF indices
    if needs_training:
        if isinstance(pre_index, faiss.IndexIVF):
            pre_index.nprobe = args.nprobe
        elif hasattr(pre_index, "index") and isinstance(
            pre_index.index, faiss.IndexIVF
        ):
            pre_index.index.nprobe = args.nprobe
        logging.debug(f"Set nprobe={args.nprobe}")

    logging.debug("build_index function completed successfully")
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
        chunks.append((encoder.decode(piece), piece))
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_words(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, List[str]]]:
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
) -> None:
    logging.debug("add_vectors_streaming function started")
    added = 0
    counter = [0]
    out_path = os.path.join(args.out_dir, "index.faiss")

    bm25_corpus: List[List[str]] = []
    bm25_ids: List[int] = []

    logging.debug("About to start processing dataframe chunks")
    chunk_count = 0
    for df in utils.load_dataframe_chunks(
        args.input, limit=args.limit, chunksize=args.chunk_size
    ):
        chunk_count += 1
        logging.debug(f"Processing chunk {chunk_count} with {len(df)} rows")
        utils.validate_columns(df.columns, COLUMNS)
        df = df[COLUMNS].dropna()

        texts_list = (
            df["title"].astype(str) + "\n" + df["content"].astype(str)
        ).tolist()
        logging.debug(f"Generated {len(texts_list)} texts from chunk")
        vid_to_dbid = {}

        # row-by-row chunking
        if args.chunk_tokens <= 0:
            # no chunking
            db_ids = df["id"].astype(str).tolist()
            texts = texts_list

            v_ids = [utils.make_vector_id(db_id, counter) for db_id in db_ids]
            tokens_list = [text.split() for text in texts_list]
            for v_id, db_id, toks in zip(v_ids, db_ids, tokens_list):
                bm25_corpus.append(toks)
                bm25_ids.append(v_id)
                vid_to_dbid[v_id] = db_id

        else:
            # chunk the text of articles based on words / tokens
            texts: List[str] = []
            v_ids: List[int] = []

            for db_id, text in zip(
                df["id"].astype(str).tolist(),
                texts_list,
            ):
                if args.use_encoding:
                    chunks_tuples = chunk_tokens(
                        text, encoder, args.chunk_tokens, args.chunk_overlap
                    )
                else:
                    chunks_tuples = chunk_words(
                        text, args.chunk_tokens, args.chunk_overlap
                    )
                for chunk_id, (chunk_text, tokens) in enumerate(chunks_tuples):
                    if not chunk_text.strip():
                        continue
                    texts.append(chunk_text)
                    v_id = utils.make_vector_id(db_id, counter)
                    v_ids.append(v_id)
                    vid_to_dbid[v_id] = db_id
                    bm25_corpus.append([str(token) for token in tokens])
                    bm25_ids.append(v_id)

        if not texts:
            logging.debug("No texts in chunk, skipping")
            continue

        logging.debug(f"About to embed {len(texts)} texts")
        batch_embeddings = embed_batches(texts, model, args.batch_size, args.normalize)
        logging.debug(f"Embeddings generated, shape: {batch_embeddings.shape}")
        ids_array = np.array(v_ids, dtype=int)
        logging.debug(f"About to add {len(ids_array)} vectors to index")
        index.add_with_ids(batch_embeddings, ids_array)
        added += len(ids_array)
        logging.debug(f"Added {len(ids_array)} vectors, total: {added}")
        
        metadata_sink.write(vid_to_dbid)
        if added % args.checkpoint_every < len(ids_array):
            logging.debug(f"Checkpointing at {added} vectors")
            faiss.write_index(index, out_path)
            logging.info(f"Checkpointed at {added} vectors")

    faiss.write_index(index, out_path)
    logging.info(f"Done adding vectors. Total added: {added}")
    logging.info(f"Training BM25 on {len(bm25_corpus)} documents")
    bm25 = BM25Okapi(bm25_corpus)  # ,tokenizer=encoder)
    bm25.doc_ids = bm25_ids
    bm25_path = os.path.join(args.out_dir, args.bm25_out)
    with open(bm25_path, "wb") as bm25_file:
        pickle.dump(bm25, bm25_file)
    logging.info(f"Done saving BM25 vectors to {bm25_path}")


def main() -> None:
    # Setup logging first
    setup_logging()
    
    # Log start time
    start_time = time.time()
    logging.info(f"=== INDEX BUILDING STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Log system information
    log_system_info()
    
    args = parse_args()
    index_path = os.path.join(args.out_dir, "index.faiss")

    # Use GPU if available (Tesla M60 should work with PyTorch 1.11.0)
    if torch.cuda.is_available():
        device = "cuda"
        logging.debug(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logging.debug("Using MPS (Apple Silicon) - FAISS will use CPU")
    else:
        device = "cpu"

    # Initialize tokenizer and model
    encoder = tiktoken.get_encoding(args.encoding)
    model = SentenceTransformer(args.model, device=device)
    logging.info(
        f"Model loaded: {args.model} | Device: {device} | Encoding: {args.encoding if args.use_encoding else 'words'}"
    )
    logging.debug("Model loading completed successfully")

    # If set to append, load the existing index. If now, train a new one.
    logging.debug("Checking if index exists and append mode")
    if args.append and os.path.exists(index_path):
        logging.debug("Loading existing index")
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
        logging.info(f"Index loaded: {index_path}")
        logging.debug(f"Index dimension: {dim}")

    else:
        logging.debug("Building new index - calling build_index function")
        index, dim = build_index(args, model)
        logging.info(f"Index built: {index_path}")
        logging.debug(f"Index dimension: {dim}")

    # Open metadata sink
    logging.debug("Opening metadata sink")
    metadata_sink = utils.get_metadata_sink(
        args.out_dir, args.save_metadata_as, append=args.append
    )
    logging.info(f"Metadata sink opened: {metadata_sink.get_path()}")

    logging.debug("Starting add_vectors_streaming function")
    add_vectors_streaming(args, index, model, encoder, metadata_sink)
    logging.debug("add_vectors_streaming completed successfully")

    logging.debug("Closing metadata sink")
    metadata_sink.close()
    logging.info("Metadata sink closed.")
    
    # Log end time and total duration
    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"=== INDEX BUILDING COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logging.info(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


if __name__ == "__main__":
    main()

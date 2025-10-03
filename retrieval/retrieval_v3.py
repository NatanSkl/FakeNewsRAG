"""
Retrieval functions for index_v3 format.

This module contains functions specifically designed to work with indices
built using build_index_v3.py.
"""

import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


def embed_and_normalize(texts, model):
    """
    Embed and normalize text using the given model.
    
    Args:
        texts: List of strings to embed
        model: SentenceTransformer model
        
    Returns:
        Normalized embeddings as float32 numpy array
    """
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # If you didn't normalize at add time, normalize here:
    # faiss.normalize_L2(X)  # normalize in-place
    return X.astype(np.float32)


def query_once(index, query, args, k=10, nprobe=16):
    """
    Perform a single query against the FAISS index.
    
    Args:
        index: FAISS index
        query: Query string
        args: Dictionary containing 'out_dir', 'model', 'nprobe'
        k: Number of results to return
        nprobe: Number of clusters to search (for IVF indices)
        
    Returns:
        List of result dictionaries with metadata
    """
    # Tune nprobe for recall/latency; typical IVF values: 8–64
    index.nprobe = args["nprobe"]
    model = SentenceTransformer(args["model"])
    
    # Load metadata from CSV
    metadata_path = Path(args["out_dir"]) / "metadata.csv"
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        meta_rows = df.to_dict('records')
    else:
        print(f"Warning: metadata.csv not found at {metadata_path}")
        meta_rows = []
    
    id_to_meta = {m["vector_id"]: m for m in meta_rows}

    # 1) Embed + normalize
    q = embed_and_normalize([query], model)

    # 2) Search
    distances, ids = index.search(q, k)  # distances: (1,k), ids: (1,k)

    # 3) Map results to metadata
    results = []
    for d, vid in zip(distances[0], ids[0]):
        if vid == -1:
            continue  # no hit
        m = id_to_meta.get(int(vid))
        if not m:
            continue
        results.append({
            "score": float(d),             # inner product; higher is better
            "vector_id": int(vid),
            "db_id": m["db_id"],
            "chunk_id": m["chunk_id"],
            "label": m["label"],
            "title": m["title"],
            "content": m["content"],
            "token_count": m["token_count"],
        })
    return results


def hybrid_query(
    query: str,
    faiss_index: faiss.Index, 
    bm25: BM25Okapi,
    tokenized_corpus: list[list[str]],
    meta_rows: list[dict],
    k: int = 10,
    k_fa: int = 10,
    k_bm: int = 50,
    alpha: float = 0.5,
    nprobe: int = 16
):
    # 1) FAISS part
    faiss_index.nprobe = nprobe
    q_emb = embed_and_normalize([query])
    D_fa, I_fa = faiss_index.search(q_emb, k_fa)
    faiss_scores = {int(vid): float(score) 
                    for vid, score in zip(I_fa[0], D_fa[0]) if vid != -1}

    # 2) BM25 part
    # Your same tokenization routine
    query_tokens = [str(tok) for tok in some_tokenizer(query)]
    bm25_scores_array = bm25.get_scores(query_tokens)
    top_bm25_idxs = np.argsort(bm25_scores_array)[::-1][:k_bm]
    bm25_scores = {
        meta_rows[i]["vector_id"]: float(bm25_scores_array[i])
        for i in top_bm25_idxs
    }

    # 3) Normalize scores to [0,1]
    max_fa = max(faiss_scores.values()) if faiss_scores else 1.0
    max_bm = max(bm25_scores.values()) if bm25_scores else 1.0

    # 4) Merge candidates
    candidates = set(faiss_scores) | set(bm25_scores)

    # 5) Compute hybrid score & assemble results
    results = []
    for vid in candidates:
        norm_fa = faiss_scores.get(vid, 0.0) / max_fa
        norm_bm = bm25_scores.get(vid, 0.0) / max_bm
        hybrid_score = alpha * norm_fa + (1 - alpha) * norm_bm

        meta = id_to_meta[vid]
        results.append({
            "vector_id": vid,
            "hybrid_score": hybrid_score,
            "faiss_score": faiss_scores.get(vid, 0.0),
            "bm25_score": bm25_scores.get(vid, 0.0),
            **meta
        })

    # 6) Return top-k by hybrid_score
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:k]

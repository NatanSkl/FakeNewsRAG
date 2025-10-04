"""
Retrieval functions for index_v3 format.

This module contains functions specifically designed to work with indices
built using build_index_v3.py.
"""

import json
import pickle
import numpy as np
import faiss
import pandas as pd
import re
import tiktoken
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Store:
    emb: SentenceTransformer
    index: Any
    bm25: Optional[BM25Okapi]
    v2d: Dict[int, str]  # vector_id -> db_id mapping
    original: pd.DataFrame  # Original CSV data
    json_path: str  # Path to build_index_args.json


def simple_tokenizer(text: str) -> List[str]:
    """Simple tokenizer that matches the BM25 corpus building process.
    
    Uses text.split() to match the tokenization used in build_index_v3.py
    """
    return text.split() if text else []


def create_encoding_tokenizer(encoding_name: str = "cl100k_base"):
    """Create tokenizer that matches chunk_tokens function from build_index_v3.py"""
    encoder = tiktoken.get_encoding(encoding_name)
    
    def tokenizer(text: str) -> List[str]:
        tokens = encoder.encode(text or "")
        return [str(token) for token in tokens]
    
    return tokenizer


def create_words_tokenizer():
    """Create tokenizer that matches chunk_words function from build_index_v3.py"""
    def tokenizer(text: str) -> List[str]:
        words = text.split()
        return [str(word) for word in words]
    
    return tokenizer


def get_appropriate_tokenizer(store: Store) -> callable:
    """Examine build args and return the appropriate tokenizer"""
    # Load build args from store's json_path
    if not Path(store.json_path).exists():
        return simple_tokenizer
    
    with open(store.json_path, 'r') as f:
        args = json.load(f)
    
    chunk_tokens = args.get("chunk_tokens", 0)
    use_encoding = args.get("use_encoding", False)
    encoding = args.get("encoding", "cl100k_base")
    
    if chunk_tokens <= 0:
        # No chunking - use simple tokenizer (text.split())
        return simple_tokenizer
    elif use_encoding:
        # Use encoding-based tokenizer (matches chunk_tokens)
        return create_encoding_tokenizer(encoding)
    else:
        # Use word-based tokenizer (matches chunk_words)
        return create_words_tokenizer()


def load_store(store_dir: str, verbose: bool = False) -> Store:
    """
    Load a store built with build_index_v3.py.
    
    Args:
        store_dir: Directory containing the index files
        verbose: If True, print progress information for each step
        
    Returns:
        Store object with loaded components
    """
    if verbose:
        print(f"Step 1: Checking store directory: {store_dir}")
    
    store_path = Path(store_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"Store directory not found: {store_dir}")
    
    if verbose:
        print(f"Store directory exists: {store_path}")
    
    # Load build args to get model name
    if verbose:
        print("Step 2: Loading build arguments...")
    
    args_path = store_path / "build_index_args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"build_index_args.json not found in {store_dir}")
    
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    if verbose:
        print(f"Build arguments loaded, model: {args.get('model', 'unknown')}")
    
    # Load embedding model
    if verbose:
        print("Step 3: Loading embedding model...")
    
    model_name = args.get("model", None)
    emb = SentenceTransformer(model_name)
    
    if verbose:
        print(f"Embedding model loaded: {model_name}")
    
    # Load FAISS index
    if verbose:
        print("Step 4: Loading FAISS index...")
    
    index_path = store_path / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss not found in {store_dir}")
    
    index = faiss.read_index(str(index_path))
    
    if verbose:
        print(f"FAISS index loaded: {index.ntotal} vectors, dimension {index.d}")
    
    # Load BM25 (optional)
    if verbose:
        print("Step 5: Loading BM25 index (optional)...")
    
    bm25_path = store_path / "bm25.pkl"
    bm25 = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        if verbose:
            print("BM25 index loaded")
    else:
        if verbose:
            print("BM25 index not found, skipping")
    
    # Load metadata.csv and create v2d mapping
    if verbose:
        print("Step 6: Loading metadata and creating v2d mapping...")
    
    metadata_path = store_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {store_dir}")
    
    df_meta = pd.read_csv(metadata_path)
    v2d = {int(row["vector_id"]): str(row["db_id"]) for _, row in df_meta.iterrows()}
    
    if verbose:
        print(f"Metadata loaded: {len(v2d)} vector_id -> db_id mappings")
    
    # Load original CSV data
    if verbose:
        print("Step 7: Loading original CSV data...")
    
    original_path = Path("/StudentData/preprocessed/train.csv")
    if not original_path.exists():
        raise FileNotFoundError(f"Original CSV not found at {original_path}")
    
    original = pd.read_csv(original_path)
    
    if verbose:
        print(f"Original CSV loaded: {original.shape[0]} rows, {original.shape[1]} columns")
    
    if verbose:
        print("Step 8: Creating Store object...")
    
    store = Store(
        emb=emb,
        index=index,
        bm25=bm25,
        v2d=v2d,
        original=original,
        json_path=str(args_path)
    )
    
    if verbose:
        print("Store object created successfully!")
    
    return store


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


def query_once(store: Store, query: str, k: int = 10, nprobe: int = 16):
    """
    Perform a single query against the FAISS index using a Store object.
    
    Args:
        store: Store object containing index, model, and metadata
        query: Query string
        k: Number of results to return
        nprobe: Number of clusters to search (for IVF indices)
        
    Returns:
        List of result dictionaries with metadata
    """
    # Tune nprobe for recall/latency; typical IVF values: 8–64
    store.index.nprobe = nprobe

    # 1) Embed + normalize using store's model
    q = embed_and_normalize([query], store.emb)

    # 2) Search using store's index
    distances, ids = store.index.search(q, k)  # distances: (1,k), ids: (1,k)

    # 3) Map results to metadata (only for search results)
    results = []
    for d, vid in zip(distances[0], ids[0]):
        if vid == -1:
            continue  # no hit
        
        # Get db_id from v2d mapping
        db_id = store.v2d.get(int(vid))
        if not db_id:
            continue
            
        # Find the row in original data with matching db_id
        matching_rows = store.original[store.original['id'] == int(db_id)]
        if matching_rows.empty:
            continue
            
        row = matching_rows.iloc[0]
        results.append({
            "score": float(d),             # inner product; higher is better
            "vector_id": int(vid),
            "db_id": db_id,
            "chunk_id": 0,  # Default chunk_id since we don't have chunking info in original
            "label": row['label'],
            "title": row['title'],
            "content": row['content'],
            "token_count": len(row['content'].split()) if pd.notna(row['content']) else 0,
        })
    return results


def hybrid_query(
    store: Store,
    query: str,
    k: int = 10,
    k_fa: int = 10,
    k_bm: int = 50,
    alpha: float = 0.5,
    nprobe: int = 16,
    tokenizer_fn = None
):
    """
    Hybrid query using Store object.
    
    Args:
        store: Store object containing index, BM25, and metadata
        query: Query string
        k: Number of final results to return
        k_fa: Number of FAISS results to retrieve
        k_bm: Number of BM25 results to retrieve
        alpha: Weight for FAISS vs BM25 (0.5 = equal weight)
        nprobe: Number of clusters to search for IVF indices
        tokenizer_fn: Tokenizer function (auto-detected from build args if None)
        
    Returns:
        List of result dictionaries with hybrid scores
    """
    if tokenizer_fn is None:
        # Auto-detect tokenizer based on build args stored in store
        tokenizer_fn = get_appropriate_tokenizer(store)

    if store.bm25 is None:
        raise ValueError("BM25 is not available in the store")

    # FAISS retrieval
    store.index.nprobe = nprobe
    q_emb = embed_and_normalize([query], store.emb)
    D_fa, I_fa = store.index.search(q_emb, k_fa)
    faiss_scores = {int(vid): float(score) for vid, score in zip(I_fa[0], D_fa[0]) if vid != -1}

    # BM25 retrieval using bm25.get_scores
    q_tokens = tokenizer_fn(query)
    bm25_scores_array = store.bm25.get_scores(q_tokens)       # scores per document index
    # pick top document indices from BM25
    top_bm25_idxs = np.argsort(bm25_scores_array)[::-1][:k_bm]
    bm25_scores = {}
    for doc_idx in top_bm25_idxs:
        # We need to map doc_idx to vector_id
        # Since BM25 was built with the same order as the index, we can use the v2d mapping
        # But we need to find the vector_id that corresponds to this doc_idx
        # This is a bit tricky - we need to find which vector_id maps to this document position
        # For now, let's assume the BM25 corpus order matches the vector order
        if doc_idx < len(store.v2d):
            # Get the vector_id at this position (assuming order is preserved)
            vector_id = list(store.v2d.keys())[doc_idx]
            bm25_scores[int(vector_id)] = float(bm25_scores_array[doc_idx])

    # Normalize each score type to [0,1]
    max_fa = max(faiss_scores.values()) if faiss_scores else 1.0
    max_bm = max(bm25_scores.values()) if bm25_scores else 1.0

    candidates = set(faiss_scores) | set(bm25_scores)
    results = []
    for vid in candidates:
        norm_fa = faiss_scores.get(vid, 0.0) / max_fa
        norm_bm = bm25_scores.get(vid, 0.0) / max_bm
        hybrid_score = alpha * norm_fa + (1 - alpha) * norm_bm
        
        # Get metadata for this vector_id
        db_id = store.v2d.get(int(vid))
        if not db_id:
            continue
            
        # Find the row in original data with matching db_id
        matching_rows = store.original[store.original['id'] == int(db_id)]
        if matching_rows.empty:
            continue
            
        row = matching_rows.iloc[0]
        meta = {
            "vector_id": int(vid),
            "db_id": db_id,
            "chunk_id": 0,  # Default chunk_id since we don't have chunking info in original
            "label": row['label'],
            "title": row['title'],
            "content": row['content'],
            "token_count": len(row['content'].split()) if pd.notna(row['content']) else 0,
        }
        
        results.append({
            "hybrid_score": hybrid_score,
            "faiss_score": faiss_scores.get(vid, 0.0),
            "bm25_score": bm25_scores.get(vid, 0.0),
            **meta
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:k]


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
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import chunking functions from build_index_v3
import sys
sys.path.append(str(Path(__file__).parent.parent / "index"))
from build_index_v3 import chunk_tokens, chunk_words


@dataclass
class Store:
    emb: SentenceTransformer
    index: Any
    bm25: Optional[BM25Okapi]
    v2d: Dict[int, int]  # vector_id -> db_id mapping
    original: pd.DataFrame  # Original CSV data
    json_path: str  # Path to build_index_args.json
    ce_model: Optional[CrossEncoder] = None  # Optional cross-encoder model


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations - compatibility class for v3 interface."""
    k: int = 10  # Number of results to return
    ce_model: Optional[CrossEncoder] = None  # Cross-encoder model for reranking
    diversity_type: Optional[str] = None  # Diversity method ("mmr" or None)
    verbose: bool = False  # Verbose output


def simple_tokenizer(text: str) -> List[str]:
    """Simple tokenizer that matches the BM25 corpus building process.
    
    Uses text.split() to match the tokenization used in build_index_v3.py
    """
    return text.split() if text else []


def create_encoding_tokenizer(encoding_name: str = "cl100k_base"):
    """Create tokenizer that matches chunk_tokens function from build_index_v3.py"""
    encoder = tiktoken.get_encoding(encoding_name)
    
    def tokenizer(text: str) -> List[str]:
        # Use chunk_tokens with chunk_size=0 to get all tokens without chunking
        chunks = chunk_tokens(text, encoder, chunk_size=0, chunk_overlap=0)
        # Get the first (and only) chunk's tokens
        tokens = chunks[0][1]  # chunks[0] is (text, tokens), we want the tokens
        return [str(token) for token in tokens]
    
    return tokenizer


def create_words_tokenizer():
    """Create tokenizer that matches chunk_words function from build_index_v3.py"""
    def tokenizer(text: str) -> List[str]:
        # Use chunk_words with chunk_size=0 to get all words without chunking
        chunks = chunk_words(text, chunk_size=0, chunk_overlap=0)
        # Get the first (and only) chunk's words
        words = chunks[0][1]  # chunks[0] is (text, words), we want the words
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


def load_store(store_dir: str, verbose: bool = False, ce_model_name: Optional[str] = None, load_bm25: bool = False) -> Store:
    """
    Load a store built with build_index_v3.py.
    
    Args:
        store_dir: Directory containing the index files
        verbose: If True, print progress information for each step
        ce_model_name: Optional cross-encoder model name to load (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
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
        
        # Check GPU availability and move index to GPU if possible
        try:
            ngpu = faiss.get_num_gpus()
            if ngpu > 0:
                print(f"FAISS GPU support: {ngpu} GPU(s) available")
                
                # Try to move index to GPU
                try:
                    if verbose:
                        print("Attempting to move index to GPU...")
                    
                    # Create GPU resources
                    res = faiss.StandardGpuResources()
                    
                    # Move index to GPU
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                    index = gpu_index
                    
                    if verbose:
                        print("Index successfully moved to GPU")
                        print(f"Index device: GPU 0")
                        
                except Exception as gpu_error:
                    if verbose:
                        print(f"Could not move index to GPU: {gpu_error}")
                        print("Continuing with CPU index...")
                        print("Index device: CPU")
            else:
                print("Index device: CPU (no GPU support)")
        except Exception as e:
            print(f"Could not determine GPU availability: {e}")
            print("Index device: CPU")
    
    # Load BM25 (optional)
    if verbose:
        print("Step 5: Loading BM25 index (optional)...")
    
    bm25_path = store_path / "bm25.pkl"
    bm25 = None
    if load_bm25 and bm25_path.exists():
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
    
    df_meta = pd.read_csv(metadata_path)  # TODO replace v2d with search function
    v2d = {int(row["vector_id"]): int(row["db_id"]) for _, row in df_meta.iterrows()}
    
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
    
    # Load cross-encoder model (optional)
    if verbose:
        print("Step 8: Loading cross-encoder model (optional)...")
    
    ce_model = None
    if ce_model_name:
        try:
            ce_model = CrossEncoder(ce_model_name)
            if verbose:
                print(f"Cross-encoder model loaded: {ce_model_name}")
        except Exception as e:
            if verbose:
                print(f"Failed to load cross-encoder model {ce_model_name}: {e}")
            ce_model = None
    else:
        if verbose:
            print("No cross-encoder model specified, skipping")
    
    if verbose:
        print("Step 9: Creating Store object...")
    
    store = Store(
        emb=emb,
        index=index,
        bm25=bm25,
        v2d=v2d,
        original=original,
        json_path=str(args_path),
        ce_model=ce_model
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


def deduplicate(results: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """
    Remove duplicate db_id entries from results, keeping only the one with the highest score.
    
    Args:
        results: List of result dictionaries
        score_key: Key to use for score comparison (default: "score")
        
    Returns:
        List of deduplicated results, sorted by score (highest first)
    """
    if not results:
        return results
    
    # Group results by db_id
    db_id_groups = {}
    for result in results:
        db_id = result.get("db_id")
        if db_id is not None:
            if db_id not in db_id_groups:
                db_id_groups[db_id] = []
            db_id_groups[db_id].append(result)
    
    # For each db_id group, keep only the result with the highest score
    deduplicated = []
    for db_id, group in db_id_groups.items():
        if len(group) == 1:
            # Only one result for this db_id, keep it
            deduplicated.append(group[0])
        else:
            # Multiple results for this db_id, keep the one with highest score
            best_result = max(group, key=lambda x: x.get(score_key, float('-inf')))
            deduplicated.append(best_result)
    
    # Sort by score (highest first)
    deduplicated.sort(key=lambda x: x.get(score_key, float('-inf')), reverse=True)
    
    return deduplicated


def filter_label(results: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """
    Filter results to keep only those with the specified label.
    
    Args:
        results: List of result dictionaries from query_once or other retrieval functions
        label: Label to filter by (e.g., "real", "fake")
        
    Returns:
        List of results that have the specified label
    """
    if not results:
        return results
    
    filtered = []
    for result in results:
        if result.get("label") == label:
            filtered.append(result)
    
    return filtered


def retrieve_evidence(store: Store,
                      article_text: str,
                      label_name: str,
                      diversity_type,
                      k: int = 10,
                      minimum_k: int = 100,
                      verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve evidence for an article with optional cross-encoder reranking and diversity.
    
    Args:
        store: Store object containing index, model, and metadata
        article_text: Article text to find evidence for
        label_name: Label to filter by (e.g., "real", "fake")
        diversity_type: Diversity method ("mmr" or None to skip)
        k: Number of final results to return
        verbose: If True, print progress information
        
    Returns:
        List of evidence results with optional reranking and diversification
    """
    if verbose:
        print(f"Starting evidence retrieval for label: {label_name}")
        print(f"Article text: {article_text[:100]}...")
    
    # Step 1: Perform initial query
    if verbose:
        print("Step 1: Performing initial query...")
    
    # Use a larger k for initial retrieval to account for filtering and diversification
    initial_k = max(k * 3, minimum_k)  # Retrieve 3x more than needed, minimum minimum_k
    results = query_once(store, article_text, k=initial_k)
    
    if verbose:
        print(f"Retrieved {len(results)} initial results")
    
    # Step 2: Deduplicate results
    if verbose:
        print("Step 2: Deduplicating results...")
    
    results = deduplicate(results)
    
    if verbose:
        print(f"After deduplication: {len(results)} results")
    
    # Step 3: Filter by label
    if verbose:
        print(f"Step 3: Filtering by label '{label_name}'...")
    
    results = filter_label(results, label_name)
    
    if verbose:
        print(f"After label filtering: {len(results)} results")
    
    if not results:
        if verbose:
            print("No results found after filtering, returning empty list")
        return []

    ce_model = store.ce_model
    
    # Step 4: Cross-encoder reranking (if model provided)
    if ce_model is not None:
        if verbose:
            print("Step 4: Applying cross-encoder reranking...")
        
        try:
            results = cross_encoder_rerank(
                cross_enc=ce_model,
                query_text=article_text,
                results=results,
                ce_topk=min(len(results), k * 2),  # Rerank up to 2x final k
                ce_weight=1.0,
                batch_size=8
            )
            
            if verbose:
                print(f"After cross-encoder reranking: {len(results)} results")
                
        except Exception as e:
            if verbose:
                print(f"Cross-encoder reranking failed: {e}")
            # Continue without reranking if it fails
    
    # Step 5: Diversity (if requested)
    if diversity_type is not None and diversity_type.lower() == "mmr":
        if verbose:
            print("Step 5: Applying MMR diversification...")
        
        try:
            results = mmr_diversify(
                store=store,
                query_text=article_text,
                results=results,
                top_k=k,
                lambda_mmr=0.5,
                content_key="content"
            )
            
            if verbose:
                print(f"After MMR diversification: {len(results)} results")
                
        except Exception as e:
            if verbose:
                print(f"MMR diversification failed: {e}")
            # Continue without diversification if it fails
    
    # Step 6: Return top k results
    final_results = results[:k]
    
    if verbose:
        print(f"Final results: {len(final_results)} evidence items")
        print("Evidence retrieval completed successfully!")
    
    return final_results

# ----------------------------
# Utility helpers
# ----------------------------

def _normalize_scores(d: Dict[int, float]) -> Dict[int, float]:
    """Robust 0–1 normalization that also handles zeros/negatives."""
    if not d:
        return {}
    mx = max(d.values())
    if mx <= 0:
        mn = min(d.values())
        rng = mx - mn if mx != mn else 1.0
        return {k: (v - mn) / rng for k, v in d.items()}
    return {k: v / mx for k, v in d.items()}

def _v2d_list(store: Store) -> List[int]:
    """Stable vector_id list used to align BM25 corpus order with vector_ids."""
    return list(store.v2d.keys())

def _bm25_scores_for_query(store: Store, query: str, tokenizer_fn) -> Dict[int, float]:
    """
    Return BM25 scores as {vector_id: score}.
    Assumes BM25 corpus order == order of list(store.v2d.keys()) from build step.
    """
    if store.bm25 is None:
        return {}
    q_tokens = tokenizer_fn(query)
    arr = store.bm25.get_scores(q_tokens)
    vlist = _v2d_list(store)
    n = min(len(arr), len(vlist))
    return {int(vlist[i]): float(arr[i]) for i in range(n)}

def _embed_texts_norm(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return X.astype(np.float32)

def _cos_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized rows (dot product)."""
    return A @ B.T

def _default_aspects_from_query(query: str, max_aspects: int = 4) -> List[str]:
    """Very light heuristic for xQuAD aspects when none are provided."""
    toks = re.findall(r"[A-Za-z0-9\-]+", query.lower())
    uniq = []
    for t in toks:
        if len(t) >= 4 and t not in uniq:
            uniq.append(t)
        if len(uniq) >= max_aspects:
            break
    return uniq or [query]

# ----------------------------
# MMR diversity re-ranking
# ----------------------------

def mmr_diversify(
    store: Store,
    query_text: str,
    results: List[Dict[str, Any]],
    top_k: int,
    lambda_mmr: float = 0.5,
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection over the candidate results.
    score(d) = λ * sim(q,d) - (1-λ) * max_{s in S} sim(d,s)
    """
    if not results:
        return results

    cand_texts = [r.get(content_key, "") or "" for r in results]
    doc_embs = _embed_texts_norm(store.emb, cand_texts)     # (N, d)
    q_emb = _embed_texts_norm(store.emb, [query_text])[0:1] # (1, d)

    q2d = _cos_sim_matrix(q_emb, doc_embs)[0]  # (N,)
    d2d = _cos_sim_matrix(doc_embs, doc_embs)  # (N, N)

    selected: List[int] = []
    remaining = set(range(len(results)))

    while len(selected) < min(top_k, len(results)):
        best_idx, best_score = None, -1e9
        for i in remaining:
            if not selected:
                score = lambda_mmr * float(q2d[i])
            else:
                max_sim = float(np.max(d2d[i, selected]))
                score = lambda_mmr * float(q2d[i]) - (1.0 - lambda_mmr) * max_sim
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    out = []
    for rank, idx in enumerate(selected):
        item = dict(results[idx])
        item["mmr_rank"] = rank + 1
        out.append(item)
    return out


def xquad_diversify(
    store: Store,
    query_text: str,
    results: List[Dict[str, Any]],
    top_k: int,
    aspects: Optional[List[str]] = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    content_key: str = "content",
    tokenizer_fn = None
) -> List[Dict[str, Any]]:
    """
    xQuAD-style selection:
      score(d) = α * P(d|q) + (1-α) * Σ_i [ (1 - C_i(S)) * P(d|a_i) ]
    where C_i(S) = β * max_{s∈S} P(s|a_i), P(d|q) = normalized base 'score',
    P(d|a_i) from BM25 (normalized).
    """
    if not results:
        return results

    if tokenizer_fn is None:
        tokenizer_fn = get_appropriate_tokenizer(store)

    aspects = aspects or _default_aspects_from_query(query_text)

    # Base relevance from 'score' (already hybrid); normalize
    vid_list = [int(r["vector_id"]) for r in results if r.get("vector_id") is not None]
    base_rel = {int(r["vector_id"]): float(r.get("score", 0.0)) for r in results if r.get("vector_id") is not None}
    base_rel = _normalize_scores(base_rel)

    # Aspect relevances via BM25
    if store.bm25 is not None:
        aspect_rel: Dict[str, Dict[int, float]] = {
            a: _normalize_scores(_bm25_scores_for_query(store, a, tokenizer_fn)) for a in aspects
        }
    else:
        aspect_rel = {a: {} for a in aspects}

    selected: List[int] = []
    selected_vids: List[int] = []
    remaining = list(range(len(results)))
    vector_ids = [int(r.get("vector_id") or -1) for r in results]

    while len(selected) < min(top_k, len(results)):
        # Coverage now
        coverage = {
            a: 0.0 if not selected_vids else beta * max(aspect_rel.get(a, {}).get(vid, 0.0) for vid in selected_vids)
            for a in aspects
        }
        best_idx, best_score = None, -1e9

        for i in remaining:
            vid = vector_ids[i]
            rel_q = base_rel.get(vid, 0.0)
            div_bonus = 0.0
            for a in aspects:
                p_da = aspect_rel.get(a, {}).get(vid, 0.0)
                div_bonus += (1.0 - coverage[a]) * p_da
            xq = alpha * rel_q + (1.0 - alpha) * div_bonus
            if xq > best_score:
                best_score, best_idx = xq, i

        selected.append(best_idx)
        selected_vids.append(vector_ids[best_idx])
        remaining.remove(best_idx)

    out = []
    for rank, idx in enumerate(selected):
        item = dict(results[idx])
        item["xquad_rank"] = rank + 1
        item["xquad_aspects"] = aspects
        out.append(item)
    return out


def cross_encoder_rerank(
    cross_enc,
    query_text: str,
    results: List[Dict[str, Any]],
    ce_topk: int = 200,
    ce_weight: float = 1.0,
    batch_size: int = 32,
    content_key: str = "content"
) -> List[Dict[str, Any]]:
    # TODO add gpu support
    """
    Rerank with a cross-encoder over the top-K by base_score.
    Blends z-scored CE into 'score' and returns the re-sorted head only.
    """
    if not results or cross_enc is None or ce_topk <= 0:
        return results

    # 1) Order by your base scorer (stable sort → stable ties)
    ordered = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)

    k = min(ce_topk, len(ordered))
    head = ordered[:k]

    # 2) Build query-passage pairs (tolerate missing/None content)
    pairs = [(query_text, (h.get(content_key) or "")) for h in head]

    try:
        ce_raw = cross_enc.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        ce_raw = np.asarray(ce_raw, dtype=float).reshape(-1)
    except Exception:
        # On CE failure, return original top-k by base_score
        return head

    # 3) Z-score normalize within the head
    mu = float(ce_raw.mean())
    sd = float(ce_raw.std())
    if sd == 0.0:
        ce_norm = np.zeros_like(ce_raw)
    else:
        ce_norm = (ce_raw - mu) / sd

    # 4) Blend into 'score' and record debug fields
    for res, raw, ns in zip(head, ce_raw.tolist(), ce_norm.tolist()):
        b = float(res.get("score", 0.0))
        res["base_score"] = b
        res["ce_score"] = float(raw)
        res["score"] = b + ce_weight * ns

    # 5) Return the top-k head re-sorted by blended score
    head.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return head



def get_data_from_vector_id(store: Store, vector_id: int) -> Optional[Dict[str, Any]]:
    """
    Get data from a vector_id using the store's mappings.
    
    Args:
        store: Store object containing mappings and data
        vector_id: The vector ID to look up
        
    Returns:
        Dictionary with metadata for the vector_id, or None if not found
    """
    # Get db_id from v2d mapping
    db_id = store.v2d.get(int(vector_id))
    if not db_id:
        return None
        
    # Find the row in original data with matching db_id
    matching_rows = store.original[store.original['id'] == int(db_id)]
    if matching_rows.empty:
        return None
        
    row = matching_rows.iloc[0]
    return {
        "vector_id": int(vector_id),
        "db_id": db_id,
        "chunk_id": 0,  # Default chunk_id since we don't have chunking info in original
        "label": row['label'],
        "title": row['title'],
        "content": row['content'],
        "token_count": len(row['content'].split()) if pd.notna(row['content']) else 0,
    }


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
        
        # Get data from vector_id using the extracted function
        data = get_data_from_vector_id(store, int(vid))
            
        results.append({
            "score": float(d),  # inner product; higher is better
            **data
        })
    return results


# def hybrid_query(
#     store: Store,
#     query: str,
#     k: int = 10,
#     k_fa: int = 10,
#     k_bm: int = 50,
#     alpha: float = 0.5,
#     nprobe: int = 16,
#     tokenizer_fn = None
# ):
#     """
#     Hybrid query using Store object.
#
#     Args:
#         store: Store object containing index, BM25, and metadata
#         query: Query string
#         k: Number of final results to return
#         k_fa: Number of FAISS results to retrieve
#         k_bm: Number of BM25 results to retrieve
#         alpha: Weight for FAISS vs BM25 (0.5 = equal weight)
#         nprobe: Number of clusters to search for IVF indices
#         tokenizer_fn: Tokenizer function (auto-detected from build args if None)
#
#     Returns:
#         List of result dictionaries with hybrid scores
#     """
#     if tokenizer_fn is None:
#         # Auto-detect tokenizer based on build args stored in store
#         tokenizer_fn = get_appropriate_tokenizer(store)
#
#     if store.bm25 is None:
#         raise ValueError("BM25 is not available in the store")
#
#     # FAISS retrieval
#     store.index.nprobe = nprobe
#     q_emb = embed_and_normalize([query], store.emb)
#     D_fa, I_fa = store.index.search(q_emb, k_fa)
#     faiss_scores = {int(vid): float(score) for vid, score in zip(I_fa[0], D_fa[0]) if vid != -1}
#
#     # BM25 retrieval using bm25.get_scores
#     q_tokens = tokenizer_fn(query)
#     bm25_scores_array = store.bm25.get_scores(q_tokens)       # scores per document index
#     # pick top document indices from BM25
#     top_bm25_idxs = np.argsort(bm25_scores_array)[::-1][:k_bm]
#     bm25_scores = {}
#
#     # TODO optimize, create these once in load_store
#     v2d_len = len(store.v2d)
#     v2d_list = list(store.v2d.keys())
#
#     for doc_idx in top_bm25_idxs:
#         # We need to map doc_idx to vector_id
#         # Since BM25 was built with the same order as the index, we can use the v2d mapping
#         # But we need to find the vector_id that corresponds to this doc_idx
#         # This is a bit tricky - we need to find which vector_id maps to this document position
#         # For now, let's assume the BM25 corpus order matches the vector order
#         if doc_idx < v2d_len:
#             # Get the vector_id at this position (assuming order is preserved)
#             vector_id = v2d_list[doc_idx]
#             bm25_scores[int(vector_id)] = float(bm25_scores_array[doc_idx])
#
#     # Normalize each score type to [0,1]
#     max_fa = max(faiss_scores.values()) if faiss_scores else 1.0
#     max_bm = max(bm25_scores.values()) if bm25_scores else 1.0
#
#     candidates = set(faiss_scores) | set(bm25_scores)
#     results = []
#     for vid in candidates:
#         norm_fa = faiss_scores.get(vid, 0.0) / max_fa
#         norm_bm = bm25_scores.get(vid, 0.0) / max_bm
#         hybrid_score = alpha * norm_fa + (1 - alpha) * norm_bm
#
#         # Get data from vector_id using the extracted function
#         data = get_data_from_vector_id(store, int(vid))
#
#         results.append({
#             "score": hybrid_score,
#             "faiss_score": faiss_scores.get(vid, 0.0),
#             "bm25_score": bm25_scores.get(vid, 0.0),
#             **data
#         })
#
#     results.sort(key=lambda x: x["score"], reverse=True)
#     return results[:k]
def hybrid_query(
    store: Store,
    query: str,
    k: int = 10,
    k_fa: int = 10,
    k_bm: int = 50,
    alpha: float = 0.5,
    nprobe: int = 16,
    tokenizer_fn = None,
    # NEW:
    diversify: Optional[str] = None,     # None | "mmr" | "xquad" | "both"
    diversify_k: Optional[int] = None,   # pool to diversify from (defaults to k)
    mmr_lambda: float = 0.5,
    xquad_alpha: float = 0.7,
    xquad_beta: float = 0.3,
    xquad_aspects: Optional[List[str]] = None,
    both_mode: str = "sequential",       # for method="both"
    both_order: str = "mmr->xquad",      # for method="both" & sequential
    both_gamma: float = 0.5              # for method="both" & blended
):
    """
    Hybrid query using Store object, with optional diversity-aware re-ranking.
    """
    if tokenizer_fn is None:
        tokenizer_fn = get_appropriate_tokenizer(store)

    if store.bm25 is None:
        raise ValueError("BM25 is not available in the store")

    # FAISS retrieval
    store.index.nprobe = nprobe
    q_emb = embed_and_normalize([query], store.emb)
    D_fa, I_fa = store.index.search(q_emb, k_fa)
    faiss_scores = {int(vid): float(score) for vid, score in zip(I_fa[0], D_fa[0]) if vid != -1}

    # BM25 retrieval
    q_tokens = tokenizer_fn(query)
    bm25_scores_array = store.bm25.get_scores(q_tokens)
    top_bm25_idxs = np.argsort(bm25_scores_array)[::-1][:k_bm]

    v2d_list = _v2d_list(store)
    bm25_scores = {}
    for doc_idx in top_bm25_idxs:
        if doc_idx < len(v2d_list):
            vector_id = v2d_list[doc_idx]
            bm25_scores[int(vector_id)] = float(bm25_scores_array[doc_idx])

    # Normalize each score type robustly
    norm_fa = _normalize_scores(faiss_scores)
    norm_bm = _normalize_scores(bm25_scores)

    # Merge candidates; compute hybrid score
    candidates = set(faiss_scores) | set(bm25_scores)
    results = []
    for vid in candidates:
        nf = norm_fa.get(vid, 0.0)
        nb = norm_bm.get(vid, 0.0)
        hybrid_score = alpha * nf + (1.0 - alpha) * nb
        data = get_data_from_vector_id(store, int(vid))
        results.append({
            "score": hybrid_score,
            "faiss_score": faiss_scores.get(vid, 0.0),
            "bm25_score": bm25_scores.get(vid, 0.0),
            **(data or {"vector_id": int(vid), "db_id": None, "content": ""})
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # Keep a larger pool before diversification (helps both MMR and xQuAD)
    pool_size = max(k, diversify_k or k)
    results = results[:pool_size]

    # Dedup → (optional) diversity
    results = diversify_results(
        store=store,
        query_text=query,
        results=results,
        method=diversify,           # None | "mmr" | "xquad" | "both"
        top_k=k,
        mmr_lambda=mmr_lambda,
        xquad_alpha=xquad_alpha,
        xquad_beta=xquad_beta,
        xquad_aspects=xquad_aspects,
        tokenizer_fn=tokenizer_fn,
        content_key="content",
        both_mode=both_mode,
        both_order=both_order,
        both_gamma=both_gamma
    )

    return results[:k]



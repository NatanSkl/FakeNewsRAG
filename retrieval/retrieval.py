"""
Core retrieval functions for the FakeNewsRAG system.

This module contains the fundamental retrieval algorithms and data structures.
"""

import numpy as np
import faiss
import json
import re
import pickle
import math
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch
import pandas as pd


# Retrieval helpers

def _get_timestamp() -> str:
    """Get current time in HH:MM format."""
    return dt.datetime.now().strftime("%H:%M")

def rrf(rank: int, k: int = 60) -> float:  # Reciprocal Rank Fusion
    return 1.0 / (k + rank)


def mmr(query_vec: np.ndarray,
        doc_vecs: np.ndarray,
        candidates: List[int],
        lam: float = 0.4,
        k: int = 120) -> List[int]:
    """Maximal Marginal Relevance (cosine with normalized vectors)."""
    sel: List[int] = []
    cand = candidates.copy()
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    # doc_vecs assumed normalized
    while cand and len(sel) < k:
        best_i, best_s = None, -1e9
        for i in cand:
            rel = float(np.dot(q, doc_vecs[i]))
            div = 0.0 if not sel else max(float(np.dot(doc_vecs[i], doc_vecs[j])) for j in sel)
            s = lam * rel - (1.0 - lam) * div
            if s > best_s:
                best_s, best_i = s, i
        sel.append(best_i)  # type: ignore[arg-type]
        cand.remove(best_i)  # type: ignore[arg-type]
    return sel


def claimify(text: str, emb_model: SentenceTransformer, max_sents: int = 6) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text or "")
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return (text or "").strip()
    vecs = emb_model.encode(sents, normalize_embeddings=True).astype("float32")
    centroid = vecs.mean(axis=0, keepdims=True)
    sims = (vecs @ centroid.T).ravel()
    idx = sorted(np.argsort(-sims)[:max_sents])
    return " ".join([sents[i] for i in idx])


def make_mqe_variants(article_text: str,
                      title_hint: str | None,
                      emb_model: SentenceTransformer) -> List[str]:
    base = (article_text or "").strip()
    if not base:
        return []
    variants: List[str] = []
    # v1: claim-focused
    variants.append(claimify(base, emb_model))
    # v2: lead sentences
    sents = re.split(r'(?<=[.!?])\s+', base)
    variants.append(" ".join(sents[:3]))
    # v3: title-boosted
    if title_hint:
        variants.append((title_hint.strip() + " ") * 3 + variants[0])
    # dedupe & length
    out, seen = [], set()
    for v in variants:
        v2 = v.strip()
        if len(v2) >= 40 and v2 not in seen:
            out.append(v2)
            seen.add(v2)
    return out[:3]


@dataclass
class Store:
    emb: SentenceTransformer
    index: Any
    bm25: BM25Okapi
    chunks: List[Dict[str, Any]]
    meta: Dict[str, Any]


def load_store(store_dir: str = "mini_index/store") -> Store:
    out = Path(store_dir)
    if not out.exists():
        raise FileNotFoundError(f"Index store not found: {store_dir}")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    with open(out / "bm25.pkl", "rb") as f:
        obj = pickle.load(f)
    bm25: BM25Okapi = obj["bm25"]
    chunks: List[Dict[str, Any]] = obj["chunks_meta"]
    index = faiss.read_index(str(out / "faiss.index"))
    emb = SentenceTransformer(meta["embedding_model"])
    return Store(emb=emb, index=index, bm25=bm25, chunks=chunks, meta=meta)


def load_index_v3_to_gpu(store_dir: str = "/StudentData/data") -> Store:
    """Load an index built with build_index_v3.py and move to GPU."""
    print(f"Loading index from {store_dir}...")
    
    import os
    import pandas as pd
    
    # Load build args
    args_path = Path(store_dir) / "build_index_args.json"
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    print(f"Model: {args['model']}")
    print(f"Index type: {args['index_type']}")
    
    # Load the FAISS index
    index_path = Path(store_dir) / "index.faiss"
    index = faiss.read_index(str(index_path))
    
    print(f"Original index type: {type(index)}")
    print(f"Index dimension: {index.d}")
    print(f"Number of vectors: {index.ntotal}")
    
    # Load metadata (try CSV first, then parquet as fallback)
    metadata_path = Path(store_dir) / "metadata.csv"
    if metadata_path.exists():
        print("Loading metadata from CSV...")
        df = pd.read_csv(metadata_path)
    else:
        metadata_path = Path(store_dir) / "metadata.parquet"
        print("Loading metadata from Parquet...")
        df = pd.read_parquet(metadata_path)
    
    print(f"Loaded {len(df)} metadata records")
    print(f"Index has {index.ntotal} vectors")
    
    # Check for mismatch
    if len(df) != index.ntotal:
        print(f"WARNING: Metadata records ({len(df)}) != Index vectors ({index.ntotal})")
        print("This might cause issues. Using only the available metadata.")
    
    # Load the embedding model
    emb = SentenceTransformer(args['model'])
    
    # Load BM25 from pickle file (like load_store does)
    bm25_path = os.path.join(store_dir, "bm25.pkl")
    if os.path.exists(bm25_path):
        print(f"Loading BM25 from pickle file... {bm25_path}")
        with open(bm25_path, "rb") as f:
            bm25: BM25Okapi = pickle.load(f)
    
    """# Create metadata
    meta = {
        "embedding_model": args['model'],
        "total_chunks": len(chunks),
        "embedding_dim": index.d,
        "index_type": args['index_type']
    }"""
    
    # Create store
    store = Store(emb=emb, index=index, bm25=bm25, chunks=None, meta=None)
    
    # Check if CUDA is available and move to GPU
    if torch.cuda.is_available():
        print("CUDA available, moving index to GPU...")
        try:
            # Check if we have GPU support
            if hasattr(faiss, 'StandardGpuResources'):
                # Get GPU resources
                res = faiss.StandardGpuResources()
                
                # Move index to GPU
                gpu_index = faiss.index_cpu_to_gpu(res, 0, store.index)
                store.index = gpu_index
                
                print("✓ Successfully moved index to GPU")
                print(f"GPU index type: {type(store.index)}")
            else:
                print("FAISS GPU support not available, using CPU index")
            
        except Exception as e:
            print(f"Failed to move index to GPU: {e}")
            print("Using CPU index instead")
    else:
        print("WARNING: CUDA not available, using CPU index")
    
    return store, index


def encode(emb: SentenceTransformer, text: str) -> np.ndarray:
    v = emb.encode([text], normalize_embeddings=True)
    return v.astype("float32")


# Retrieval config

@dataclass
class RetrievalConfig:
    k_dense: int = 600
    k_bm25: int = 600
    w_dense: float = 1.35
    w_lex: float = 1.0
    mmr_k: int = 180
    mmr_lambda: float = 0.45
    topn: int = 24
    topn_per_label: int = 12
    domain_cap: int = 2

    # lost-in-the-middle: sentence re-scoring
    sent_maxpool: bool = True
    sent_max_sents: int = 12          # max sentences to score per chunk
    sent_min_len: int = 20            # skip tiny sentences
    sent_bonus: float = 0.25          # how much to mix into the fused score

    # cross-encoder reranking
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_topk: int = 40                 # rerank only the final top-K after MMR
    ce_weight: float = 1.0            # linear mix with fused score

    # xQuAD diversification
    use_xquad: bool = True
    xquad_k: int = 16                 # how many to select with xQuAD
    xquad_lambda: float = 0.6
    xquad_aspects: int = 3            # #sub-queries from MQE variants

    # metadata filters
    label_filter: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    lang_whitelist: set[str] = field(default_factory=set)
    domain_whitelist: set[str] = field(default_factory=set)
    domain_blacklist: set[str] = field(default_factory=set)
    min_source_score: float | None = None
    min_chunk_chars: int = 200


def _parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    return dt.date.fromisoformat(s)


def filter_by_metadata(hits: List[Dict[str, Any]], cfg: RetrievalConfig) -> List[Dict[str, Any]]:
    d_from = _parse_date(cfg.date_from)
    d_to = _parse_date(cfg.date_to)

    out: List[Dict[str, Any]] = []
    for h in hits:
        if cfg.label_filter and h.get("label") != cfg.label_filter:
            continue
        if len(h.get("chunk_text", "")) < cfg.min_chunk_chars:
            continue

        pd_s = h.get("published_date")
        if pd_s and (d_from or d_to):
            try:
                pd = dt.date.fromisoformat(pd_s[:10])
                if d_from and pd < d_from:
                    continue
                if d_to and pd > d_to:
                    continue
            except Exception:
                pass

        lang = (h.get("language") or "").lower()
        dom = (h.get("source_domain") or "").lower()

        if cfg.lang_whitelist and lang not in cfg.lang_whitelist:
            continue
        if cfg.domain_whitelist and dom not in cfg.domain_whitelist:
            continue
        if cfg.domain_blacklist and dom in cfg.domain_blacklist:
            continue

        if cfg.min_source_score is not None:
            if float(h.get("source_score", 1.0)) < cfg.min_source_score:
                continue

        out.append(h)
    return out


def sentence_maxpool_boost(store: Store,
                           qv: np.ndarray,
                           hits: List[Dict[str, Any]],
                           cfg: RetrievalConfig) -> List[Dict[str, Any]]:
    """Mitigate 'lost in the middle' by scoring sentences and max-pooling."""
    if not cfg.sent_maxpool or not hits:
        return hits

    qv_norm = qv / (np.linalg.norm(qv) + 1e-9)
    for h in hits:
        text = h.get("chunk_text", "")
        sents = re.split(r'(?<=[.!?])\s+', text)
        sents = [s.strip() for s in sents if len(s.strip()) >= cfg.sent_min_len][:cfg.sent_max_sents]
        if not sents:
            h["_sent_max"] = 0.0
            continue

        sv = store.emb.encode(sents, normalize_embeddings=True).astype("float32")
        sims = (sv @ qv_norm).ravel()
        h["_sent_max"] = float(np.max(sims)) if sims.size else 0.0

    # min-max normalize sentence scores and blend
    mvals = [h.get("_sent_max", 0.0) for h in hits]
    if mvals:
        lo, hi = min(mvals), max(mvals)
        rng = (hi - lo) or 1e-6
        for h in hits:
            z = (h.get("_sent_max", 0.0) - lo) / rng
            h["_score"] = h.get("_score", h.get("rrf", 0.0)) + cfg.sent_bonus * z
    return hits


def xquad_diversify(store: Store,
                    qv: np.ndarray,
                    hits: List[Dict[str, Any]],
                    aspects: List[np.ndarray],
                    cfg: RetrievalConfig) -> List[Dict[str, Any]]:
    """Simple xQuAD: trade-off between relevance to aspects and novelty wrt selected set."""
    if not cfg.use_xquad or not hits:
        return hits

    dvs = store.emb.encode([h["chunk_text"] for h in hits],
                           normalize_embeddings=True).astype("float32")
    selected: List[int] = []
    remaining: List[int] = list(range(len(hits)))
    qn = qv / (np.linalg.norm(qv) + 1e-9)

    while remaining and len(selected) < min(cfg.xquad_k, len(hits)):
        best_i, best_val = None, -1e9
        for i in remaining:
            rel = float(dvs[i] @ qn)
            cov = 0.0
            for a in aspects:
                an = a / (np.linalg.norm(a) + 1e-9)
                cov += float(dvs[i] @ an)
            cov /= max(len(aspects), 1)
            nov = 0.0 if not selected else max(float(dvs[i] @ dvs[j]) for j in selected)
            score = cfg.xquad_lambda * (rel + cov) - (1.0 - cfg.xquad_lambda) * nov
            if score > best_val:
                best_val, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)

    sel_hits = [hits[i] for i in selected]
    return sel_hits


def cross_encoder_rerank(cross_enc: CrossEncoder,
                         query_text: str,
                         hits: List[Dict[str, Any]],
                         cfg: RetrievalConfig) -> List[Dict[str, Any]]:
    if not cfg.use_cross_encoder or not hits:
        return hits

    subset = hits[:cfg.ce_topk]
    pairs = [(query_text, h.get("chunk_text", "")) for h in subset]
    try:
        ce_scores = cross_enc.predict(pairs).tolist()
    except Exception:
        # be robust if model missing
        return hits

    # min-max normalize CE scores and blend
    lo, hi = min(ce_scores), max(ce_scores)
    rng = (hi - lo) or 1e-6
    for h, s in zip(subset, ce_scores):
        z = (s - lo) / rng
        base = h.get("_score", h.get("rrf", 0.0))
        h["_score"] = base + cfg.ce_weight * z

    head_sorted = sorted(subset, key=lambda h: h.get("_score", h.get("rrf", 0.0)), reverse=True)
    tail = hits[cfg.ce_topk:]
    return head_sorted + tail


def hybrid_once(store: Store,
                query_text: str,
                cfg: RetrievalConfig,
                *,
                label_filter: str | None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # Dense
    qv = encode(store.emb, query_text)
    v, I = store.index.search(qv, cfg.k_dense)
    dense_ids = I[0].tolist()

    print("DEBUG: dense_ids")
    print(v)
    print(I)

    # BM25
    q_tokens = re.findall(r"\w+", query_text.lower())
    try:
        scores_bm = store.bm25.get_scores(q_tokens)
    except Exception:
        tokenized = [re.findall(r"\w+", m["chunk_text"].lower()) for m in store.chunks]
        store.bm25 = BM25Okapi(tokenized)
        scores_bm = store.bm25.get_scores(q_tokens)
    bm_top = np.argsort(-scores_bm)[:cfg.k_bm25].tolist()

    # Weighted RRF (dense + lex)
    fused: Dict[int, float] = defaultdict(float)
    for r, idx in enumerate(dense_ids, 1):
        fused[idx] += cfg.w_dense * rrf(r)
    for r, idx in enumerate(bm_top, 1):
        fused[idx] += cfg.w_lex * rrf(r)

    cand = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    # label pre-filter and bounds checking
    if label_filter:
        cand = [i for i in cand if i < len(store.chunks) and store.chunks[i]["label"] == label_filter]
    else:
        # Just bounds checking
        cand = [i for i in cand if i < len(store.chunks)]

    # Diversity (MMR) on the candidates
    texts = [store.chunks[i]["chunk_text"] for i in cand]
    if texts:
        doc_vecs = store.emb.encode(texts, normalize_embeddings=True).astype("float32")
    else:
        doc_vecs = np.zeros((0, qv.shape[-1]), dtype="float32")

    loc2glob = {li: gi for li, gi in enumerate(cand)}
    sel_local = mmr(qv[0], doc_vecs, list(range(len(cand))),
                    lam=cfg.mmr_lambda, k=min(cfg.mmr_k, len(cand)))
    sel_global = [loc2glob[i] for i in sel_local]

    hits: List[Dict[str, Any]] = []
    for gi in sel_global:
        print("DEBUG: gi")
        print(gi)
        print(store.chunks[gi])
        h = store.chunks[gi].copy()
        h["rrf"] = fused[gi]
        h["_score"] = fused[gi]
        hits.append(h)

    return hits, qv[0]




def apply_domain_cap(hits: List[Dict[str, Any]], cap: int = 2) -> List[Dict[str, Any]]:
    if cap <= 0:
        return hits
    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for h in hits:
        d = (h.get("source_domain") or "").lower()
        seen[d] = seen.get(d, 0) + 1
        if seen[d] <= cap:
            out.append(h)
    return out


def retrieve_evidence(store: Store,
                      article_text: str,
                      title_hint: str | None,
                      *,
                      label_name: str,
                      cfg: RetrievalConfig,
                      verbose: bool = False) -> List[Dict[str, Any]]:
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Starting retrieval for: '{article_text[:100]}...'")
        print(f"[{_get_timestamp()}] [retrieve_evidence] Title hint: {title_hint}")
        print(f"[{_get_timestamp()}] [retrieve_evidence] Label filter: {label_name}")
        print(f"[{_get_timestamp()}] [retrieve_evidence] Config: k_dense={cfg.k_dense}, k_bm25={cfg.k_bm25}, topn={cfg.topn}")
    
    # multi-query expansion
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Generating query variants...")
    variants = make_mqe_variants(article_text, title_hint, store.emb)
    if not variants:
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence] No variants generated, returning empty results")
        return []
    
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Generated {len(variants)} variants:")
        for i, variant in enumerate(variants, 1):
            print(f"  [{i}] {variant[:100]}...")

    pooled: Dict[Tuple[str, str], Dict[str, Any]] = {}
    fuse_scores: Dict[Tuple[str, str], float] = defaultdict(float)

    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Processing {len(variants)} variants...")

    for i, v in enumerate(variants, 1):
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence] Processing variant {i}/{len(variants)}: '{v[:80]}...'")
        
        hits, qv = hybrid_once(store, v, cfg, label_filter=label_name)
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence]   hybrid_once returned {len(hits)} hits")
        
        hits = sentence_maxpool_boost(store, qv, hits, cfg)
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence]   sentence_maxpool_boost returned {len(hits)} hits")

        for r, h in enumerate(hits, 1):
            key = (h.get("doc_id", h.get("id", "unknown")), h.get("chunk_id", 0))
            fuse_scores[key] += rrf(r)
            if key not in pooled:
                pooled[key] = h
            else:
                pooled[key]["_score"] = max(
                    pooled[key].get("_score", 0.0),
                    h.get("_score", 0.0)
                )
        
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence]   Added {len(hits)} hits to pool (total unique: {len(pooled)})")

    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Merging results from {len(pooled)} unique documents...")
    
    merged = [pooled[k] for k in sorted(fuse_scores, key=fuse_scores.get, reverse=True)]
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Merged to {len(merged)} results")

    cfg_local = dataclass_replace(cfg, label_filter=label_name)
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Applying metadata filter...")
    merged = filter_by_metadata(merged, cfg_local)
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] After metadata filter: {len(merged)} results")
    
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Applying domain cap (cap={cfg.domain_cap})...")
    merged = apply_domain_cap(merged, cap=cfg.domain_cap)
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] After domain cap: {len(merged)} results")

    if cfg.use_xquad and merged:
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence] Applying xQuAD diversification...")
        aspects_text = variants[:cfg.xquad_aspects]
        aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
        merged = xquad_diversify(store, encode(store.emb, article_text)[0], merged, list(aspects_vecs), cfg)
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence] After xQuAD: {len(merged)} results")

    if cfg.use_cross_encoder and merged:
        if verbose:
            print(f"[{_get_timestamp()}] [retrieve_evidence] Applying cross-encoder reranking...")
        try:
            ce = CrossEncoder(cfg.cross_encoder_model)
            merged = cross_encoder_rerank(ce, article_text, merged, cfg)
            if verbose:
                print(f"[{_get_timestamp()}] [retrieve_evidence] After cross-encoder: {len(merged)} results")
        except Exception as e:
            if verbose:
                print(f"[{_get_timestamp()}] [retrieve_evidence] Cross-encoder failed: {e}")

    final_results = merged[:cfg.topn]
    if verbose:
        print(f"[{_get_timestamp()}] [retrieve_evidence] Final results: {len(final_results)} (limited to topn={cfg.topn})")
        print(f"[{_get_timestamp()}] [retrieve_evidence] Retrieval completed successfully!")
    
    return final_results


def dataclass_replace(cfg: RetrievalConfig, **kw) -> RetrievalConfig:
    """Shallow clone with overrides (like dataclasses.replace but explicit to avoid import)."""
    d = cfg.__dict__.copy()
    d.update(kw)
    return RetrievalConfig(**d)
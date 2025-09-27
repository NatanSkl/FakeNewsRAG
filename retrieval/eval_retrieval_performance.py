# eval_retrieval_quick.py
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from retrieval import load_store, RetrievalConfig, claimify, retrieve_evidence

def keymap(chunks: List[Dict[str, Any]]) -> Dict[Tuple[str,int], int]:
    m = {}
    for i, ch in enumerate(chunks):
        m[(ch["doc_id"], ch["chunk_id"])] = i
    return m

def build_qrels_by_faiss(store, m_queries=30, nn=60, same_label=True) -> Tuple[List[int], Dict[int, set]]:
    """Use FAISS to fetch neighbors for each sampled chunk; filter by label; return qrels sets."""
    ids = random.sample(range(len(store.chunks)), min(m_queries, len(store.chunks)))
    qrels: Dict[int, set] = {}
    for i in ids:
        text = store.chunks[i]["chunk_text"]
        qv = store.emb.encode([text], normalize_embeddings=True).astype("float32")
        _, I = store.index.search(qv, nn + 1)
        lab_i = store.chunks[i].get("label")
        rel = []
        for j in I[0]:
            if j == i:
                continue
            if same_label and store.chunks[j].get("label") != lab_i:
                continue
            rel.append(j)
        qrels[i] = set(rel)
    return ids, qrels

def metrics_at_k(ranklist: List[int], relset: set, k=10):
    top = ranklist[:k]
    # Recall
    recall = len(set(top) & relset) / max(1, len(relset))
    # MRR
    mrr = 0.0
    for r, idx in enumerate(top, 1):
        if idx in relset:
            mrr = 1.0 / r
            break
    # NDCG
    dcg = 0.0
    for r, idx in enumerate(top, 1):
        gain = 1.0 if idx in relset else 0.0
        dcg += gain / np.log2(r + 1)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, max(1, len(relset))) + 1))
    ndcg = dcg / (ideal or 1e-9)
    return recall, mrr, ndcg

def retrieve_global_ids(store, query_text: str, label: str, cfg: RetrievalConfig, kmap: Dict[Tuple[str,int], int]) -> List[int]:
    hits = retrieve_evidence(store, query_text, title_hint=None, label_name=label, cfg=cfg)
    out = []
    for h in hits:
        g = kmap.get((h["doc_id"], h["chunk_id"]))
        if g is not None:
            out.append(g)
    return out

if __name__ == "__main__":
    random.seed(42)
    store = load_store("../index_tmp/store")
    kmap = keymap(store.chunks)

    # FAST config: first-stage only
    cfg = RetrievalConfig(
        topn=40,
        k_dense=400, k_bm25=400,
        mmr_k=120, mmr_lambda=0.45,
        w_dense=1.35, w_lex=1.0,
        domain_cap=3,
        sent_maxpool=False,
        use_cross_encoder=False,
        use_xquad=False,
        min_chunk_chars=100
    )

    # Build pseudo-qrels via FAISS (very fast)
    q_ids, qrels = build_qrels_by_faiss(store, m_queries=30, nn=60, same_label=True)

    R10 = M10 = N10 = 0.0
    n = 0
    for qi in tqdm(q_ids):
        qchunk = store.chunks[qi]
        qtext = claimify(qchunk["chunk_text"], store.emb)
        qlabel = qchunk.get("label") or "credible"
        ranks = retrieve_global_ids(store, qtext, qlabel, cfg, kmap)
        r, m, nD = metrics_at_k(ranks, qrels[qi], k=10)
        R10 += r; M10 += m; N10 += nD; n += 1

    print(f"Queries={n}  Recall@10={R10/n:.3f}  MRR@10={M10/n:.3f}  NDCG@10={N10/n:.3f}")

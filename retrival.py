import argparse, json, re, pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def rrf(rank, k=60):  # Reciprocal Rank Fusion
    return 1.0 / (k + rank)

def apply_domain_cap(hits, cap=2):
    seen, out = {}, []
    for h in hits:
        d = h.get("source_domain", "")
        seen[d] = seen.get(d, 0) + 1
        if seen[d] <= cap:
            out.append(h)
    return out

def mmr(query_vec, doc_vecs, candidates, lam=0.3, k=60):
    sel, cand = [], candidates.copy()
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    while cand and len(sel) < k:
        best_i, best_s = None, -1e9
        for i in cand:
            rel = float(np.dot(q, doc_vecs[i]))
            div = 0.0 if not sel else max(float(np.dot(doc_vecs[i], doc_vecs[j])) for j in sel)
            s = lam*rel - (1.0-lam)*div
            if s > best_s:
                best_s, best_i = s, i
        sel.append(best_i); cand.remove(best_i)
    return sel

def claimify(text, emb_model, max_sents=6):
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents: return text
    vecs = emb_model.encode(sents, normalize_embeddings=True).astype("float32")
    centroid = vecs.mean(axis=0, keepdims=True)
    sims = (vecs @ centroid.T).ravel()
    idx = sorted(np.argsort(-sims)[:max_sents])  # keep original order
    return " ".join([sents[i] for i in idx])

def make_mqe_variants(article_text, title_hint=None, emb_model=None):
    """Return 2-3 query variants for multi-query expansion."""
    base = article_text.strip()
    variants = []
    # v1: claim-focused
    if emb_model is not None:
        variants.append(claimify(base, emb_model))
    else:
        variants.append(base)
    # v2: first ~3 sentences (lead bias)
    sents = re.split(r'(?<=[.!?])\s+', base)
    variants.append(" ".join(sents[:3]))
    # v3: title-boosted (if provided)
    if title_hint:
        variants.append((title_hint + " ") * 3 + variants[0])  # crude but effective boost
    # dedupe short
    seen = set(); out = []
    for v in variants:
        v2 = v.strip()
        if len(v2) >= 40 and v2 not in seen:
            out.append(v2); seen.add(v2)
    return out[:3]

def load_store(outdir: str):
    out = Path(outdir)
    with open(out/"meta.json") as f: meta = json.load(f)
    with open(out/"bm25.pkl","rb") as f: obj = pickle.load(f)
    bm25: BM25Okapi = obj["bm25"]
    chunks_meta: List[Dict[str,Any]] = obj["chunks_meta"]
    index = faiss.read_index(str(out/"faiss.index"))
    emb = SentenceTransformer(meta["embedding_model"])
    return emb, index, bm25, chunks_meta, meta

def encode(emb, text: str):
    vec = emb.encode([text], normalize_embeddings=True)
    return vec.astype("float32")

def hybrid_once(emb, index, bm25, chunks_meta, query_text,
                k_dense=400, k_bm25=400, mmr_k=120, lam=0.4,
                label_filter=None, w_dense=1.3, w_lex=1.0, domain_cap=2, topn=20):

    qv = encode(emb, query_text)
    _, I = index.search(qv, k_dense)
    dense_ids = I[0].tolist()


    q_tokens = re.findall(r"\w+", query_text.lower())
    try:
        scores_bm = bm25.get_scores(q_tokens)
    except Exception:

        tokenized = [re.findall(r"\w+", m["chunk_text"].lower()) for m in chunks_meta]
        bm25 = BM25Okapi(tokenized)
        scores_bm = bm25.get_scores(q_tokens)
    bm_top = np.argsort(-scores_bm)[:k_bm25].tolist()

    # Weighted RRF
    fused = {}
    for r, idx in enumerate(dense_ids, 1): fused[idx] = fused.get(idx, 0.0) + w_dense * rrf(r)
    for r, idx in enumerate(bm_top,    1): fused[idx] = fused.get(idx, 0.0) + w_lex   * rrf(r)
    cand = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    if label_filter:
        cand = [i for i in cand if chunks_meta[i]["label"] == label_filter]

    # MMR diversity
    texts = [chunks_meta[i]["chunk_text"] for i in cand]
    doc_vecs = np.asarray(emb.encode(texts, normalize_embeddings=True)).astype("float32")
    loc2glob = {li: gi for li, gi in enumerate(cand)}
    sel_local = mmr(qv[0], doc_vecs, list(range(len(cand))), lam=lam, k=min(mmr_k, len(cand)))
    sel_global = [loc2glob[i] for i in sel_local]

    results = [chunks_meta[i] | {"rrf": fused[i]} for i in sel_global[:topn*3]]  # generous before cap
    if domain_cap and domain_cap > 0:
        results = apply_domain_cap(results, cap=domain_cap)
    return results[:topn], qv[0]

def hybrid_search(store_dir, article_text, title_hint=None, label_filter=None,
                  k_dense=400, k_bm25=400, mmr_k=120, lam=0.4, w_dense=1.3, w_lex=1.0,
                  domain_cap=2, topn=20, use_mqe=True):
    emb, index, bm25, chunks_meta, meta = load_store(store_dir)
    # Build 1–3 query variants
    variants = make_mqe_variants(article_text, title_hint=title_hint, emb_model=emb) if use_mqe else [article_text]
    # Run each and fuse by RRF over result ranks
    fuse_scores, pooled = {}, {}
    for v in variants:
        hits, _ = hybrid_once(emb, index, bm25, chunks_meta, v, k_dense, k_bm25, mmr_k, lam,
                              label_filter, w_dense, w_lex, domain_cap=0, topn=topn*3)
        for r, h in enumerate(hits, 1):
            did = (h["doc_id"], h["chunk_id"])
            fuse_scores[did] = fuse_scores.get(did, 0.0) + rrf(r)  # equal across variants
            pooled[did] = h
    # Final sort and domain cap
    merged = [pooled[k] for k in sorted(fuse_scores, key=fuse_scores.get, reverse=True)]
    if domain_cap and domain_cap > 0:
        merged = apply_domain_cap(merged, cap=domain_cap)
    return merged[:topn]

def load_reranker(name="BAAI/bge-reranker-base"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device

def rerank(tok, model, device, query_text, docs, topn):
    pairs = [(query_text, d["chunk_text"]) for d in docs]
    if not pairs: return docs[:topn]
    batch = tok([p[0] for p in pairs], [p[1] for p in pairs],
                padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        s = model(**batch).logits.squeeze(-1).float().cpu().numpy()
    order = np.argsort(-s)
    return [docs[i] for i in order[:topn]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="mini_index/store")
    ap.add_argument("--query_file", help="Path to article text file")
    ap.add_argument("--query_text", help="Raw article text (overrides file)", default=None)
    ap.add_argument("--title_hint", help="Optional title to boost", default=None)
    ap.add_argument("--label-filter", choices=["fake","credible","other"], default=None)

    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--k_bm25", type=int, default=400)
    ap.add_argument("--mmr_k", type=int, default=120)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--w_dense", type=float, default=1.3)
    ap.add_argument("--w_lex", type=float, default=1.0)
    ap.add_argument("--domain_cap", type=int, default=2)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--no_mqe", action="store_true")

    ap.add_argument("--use_reranker", action="store_true")
    ap.add_argument("--rerank_pool", type=int, default=80)

    args = ap.parse_args()
    # Read query
    if args.query_text:
        q_text = args.query_text
    else:
        if not args.query_file:
            raise SystemExit("Provide --query_text or --query_file")
        q_text = Path(args.query_file).read_text(encoding="utf-8")

    hits = hybrid_search(
        args.store, q_text, title_hint=args.title_hint, label_filter=args.label_filter,
        k_dense=args.k, k_bm25=args.k_bm25, mmr_k=args.mmr_k, lam=args.lam,
        w_dense=args.w_dense, w_lex=args.w_lex, domain_cap=args.domain_cap,
        topn=max(args.topn, args.rerank_pool if args.use_reranker else args.topn),
        use_mqe=(not args.no_mqe)
    )

    if args.use_reranker:
        tok, model, device = load_reranker()
        hits = rerank(tok, model, device, q_text, hits[:args.rerank_pool], topn=args.topn)
    else:
        hits = hits[:args.topn]

    for j,h in enumerate(hits,1):
        print(f"[{j}] {h['label']:<9} {h.get('source_domain',''):<25} {h.get('title','')[:80]}")
        print(f"    doc_id={h['doc_id']}  chunk_id={h['chunk_id']}  rrf={h.get('rrf',0):.4f}")
        print(f"    snippet: {h.get('chunk_text','')[:220].replace('\\n',' ')}…\n")

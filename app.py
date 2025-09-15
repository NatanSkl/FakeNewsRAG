
import json, re, pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

STORE_DIR = "mini_index/store"

def rrf(rank, k=60):
    return 1.0 / (k + rank)

def apply_domain_cap(hits, cap=2):
    if cap is None or cap <= 0:
        return hits
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
    if not sents:
        return text
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
        variants.append((title_hint + " ") * 3 + variants[0])  # simple title boost
    # dedupe
    seen = set(); out = []
    for v in variants:
        v2 = v.strip()
        if len(v2) >= 40 and v2 not in seen:
            out.append(v2); seen.add(v2)
    return out[:3]

@st.cache_resource(show_spinner=False)
def load_store(store_dir: str):
    out = Path(store_dir)
    if not out.exists():
        raise FileNotFoundError(f"Store not found at: {store_dir}")
    meta = json.loads((out/"meta.json").read_text(encoding="utf-8"))
    with open(out/"bm25.pkl","rb") as f:
        obj = pickle.load(f)
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


    fused = {}
    for r, idx in enumerate(dense_ids, 1): fused[idx] = fused.get(idx, 0.0) + w_dense * rrf(r)
    for r, idx in enumerate(bm_top,    1): fused[idx] = fused.get(idx, 0.0) + w_lex   * rrf(r)
    cand = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    if label_filter:
        cand = [i for i in cand if chunks_meta[i]["label"] == label_filter]

    texts = [chunks_meta[i]["chunk_text"] for i in cand]
    doc_vecs = np.asarray(emb.encode(texts, normalize_embeddings=True)).astype("float32")
    loc2glob = {li: gi for li, gi in enumerate(cand)}
    sel_local = mmr(qv[0], doc_vecs, list(range(len(cand))), lam=lam, k=min(mmr_k, len(cand)))
    sel_global = [loc2glob[i] for i in sel_local]

    results = [chunks_meta[i] | {"rrf": fused[i]} for i in sel_global[:topn*3]]  # generous before cap
    results = apply_domain_cap(results, cap=domain_cap)
    return results[:topn], qv[0]

def hybrid_search(store_dir, article_text, title_hint=None, label_filter=None,
                  k_dense=400, k_bm25=400, mmr_k=120, lam=0.4, w_dense=1.3, w_lex=1.0,
                  domain_cap=2, topn=15, use_mqe=True):
    emb, index, bm25, chunks_meta, meta = load_store(store_dir)
    # 1–3 query variants
    variants = make_mqe_variants(article_text, title_hint=title_hint, emb_model=emb) if use_mqe else [article_text]
    # Run each and fuse over result ranks
    fuse_scores, pooled = {}, {}
    for v in variants:
        hits, _ = hybrid_once(emb, index, bm25, chunks_meta, v, k_dense, k_bm25, mmr_k, lam,
                              label_filter, w_dense, w_lex, domain_cap=0, topn=topn*3)
        for r, h in enumerate(hits, 1):
            did = (h["doc_id"], h["chunk_id"])
            fuse_scores[did] = fuse_scores.get(did, 0.0) + rrf(r)  # equal per variant
            pooled[did] = h
    merged = [pooled[k] for k in sorted(fuse_scores, key=fuse_scores.get, reverse=True)]
    merged = apply_domain_cap(merged, cap=domain_cap)
    return merged[:topn]

def to_evidence_pack(label, article_text, hits):
    return {
        "label": label,
        "query_text": article_text,
        "items": [{
            "doc_id": h["doc_id"],
            "chunk_id": h["chunk_id"],
            "title": h.get("title",""),
            "domain": h.get("source_domain",""),
            "date": h.get("published_at",""),
            "snippet": h.get("chunk_text","")[:450]
        } for h in hits]
    }


st.set_page_config(page_title="FakeNews RAG – Retrieval", layout="wide")
st.title("📰 FakeNews RAG")

with st.sidebar:
    st.header("Index")
    store = st.text_input("Store directory", STORE_DIR)
    st.divider()
    st.header("Retrieval knobs")
    k_dense = st.slider("Dense topK (FAISS)", 100, 1200, 600, 50)
    k_bm25  = st.slider("BM25 topK", 100, 1200, 600, 50)
    mmr_k   = st.slider("MMR diversify K", 20, 300, 180, 10)
    lam     = st.slider("MMR λ (relevance vs diversity)", 0.1, 0.9, 0.45, 0.05)
    w_dense = st.slider("Dense weight (RRF)", 0.8, 2.0, 1.35, 0.05)
    w_lex   = st.slider("Lexical weight (RRF)", 0.5, 2.0, 1.0, 0.05)
    domcap  = st.number_input("Domain cap (per label)", min_value=0, max_value=10, value=2, step=1)
    topn    = st.slider("Show top N", 5, 50, 15, 1)
    use_mqe = st.checkbox("Multi-Query Expansion (claimify/lead/title)", value=True)

st.write("Paste the full article (or upload a file). Optional: add a **title** to boost title terms.")

colA, colB = st.columns([3, 1])
with colA:
    article_text = st.text_area("Article text", height=220, placeholder="Paste article here...")
with colB:
    title_hint = st.text_input("Optional title boost")
    uploaded = st.file_uploader("…or upload .txt", type=["txt"])
    if uploaded is not None:
        article_text = uploaded.read().decode("utf-8", errors="ignore")

run = st.button("🔎 Retrieve")

if run:
    if not article_text or len(article_text.strip()) < 20:
        st.warning("Please provide at least ~20 characters of article text.")
    else:
        try:
            with st.spinner("Retrieving evidence…"):
                # Credible and Fake separately
                hits_cred = hybrid_search(
                    store, article_text, title_hint=title_hint, label_filter="credible",
                    k_dense=k_dense, k_bm25=k_bm25, mmr_k=mmr_k, lam=lam,
                    w_dense=w_dense, w_lex=w_lex, domain_cap=domcap,
                    topn=topn, use_mqe=use_mqe
                )
                hits_fake = hybrid_search(
                    store, article_text, title_hint=title_hint, label_filter="fake",
                    k_dense=k_dense, k_bm25=k_bm25, mmr_k=mmr_k, lam=lam,
                    w_dense=w_dense, w_lex=w_lex, domain_cap=domcap,
                    topn=topn, use_mqe=use_mqe
                )

            tab1, tab2 = st.tabs(["✅ Credible", "🚩 Fake"])
            for label, hits, tab in [
                ("credible", hits_cred, tab1),
                ("fake", hits_fake, tab2)
            ]:
                with tab:
                    st.subheader(f"Top {len(hits)} {label} matches")
                    for j, h in enumerate(hits, 1):
                        with st.container(border=True):
                            st.markdown(f"**[{j}] {h.get('title','').strip() or '(no title)'}**")
                            st.caption(f"{h.get('source_domain','')} — {h.get('published_at','')}")
                            st.write(h.get("chunk_text","")[:600])
                            st.code(f"doc_id={h['doc_id']} | chunk_id={h['chunk_id']}", language="text")
                    # Download evidence pack
                    pack = to_evidence_pack(label, article_text, hits)
                    st.download_button(
                        f"⬇️ Download {label} evidence pack (JSON)",
                        data=json.dumps(pack, ensure_ascii=False, indent=2),
                        file_name=f"{label}_evidence.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.exception(e)

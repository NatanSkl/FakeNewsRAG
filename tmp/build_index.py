import argparse, json, os, re, uuid, math
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle
def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    return s

def chunk_text(text: str, target_tokens=700, overlap_tokens=120, enc=None):
    """Token-aware chunking with overlap; returns list of strings."""
    enc = enc or tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + target_tokens)
        chunk = enc.decode(toks[i:j])
        chunks.append(chunk)
        if j == len(toks): break
        i = max(0, j - overlap_tokens)
    return chunks

@dataclass
class DocRow:
    doc_id: str
    source_domain: str
    published_at: str
    label: str
    title: str
    url: str
    text: str
    language: str = "en"

@dataclass
class ChunkRow:
    doc_id: str
    chunk_id: int
    label: str
    title: str
    source_domain: str
    published_at: str
    url: str
    chunk_text: str

def detect_domain(url: str) -> str:
    if not url: return ""
    m = re.search(r'://([^/]+)/?', url)
    return m.group(1).lower() if m else ""

def build(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading CSV…")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)

    # Map your columns
    id_col    = args.id_col or "id"
    text_col  = args.text_col or "content"
    title_col = args.title_col or "title"
    label_col = args.label_col or "label"
    url_col   = args.url_col or "url"
    date_col  = args.date_col or "published_at"

    keep = df[[c for c in [id_col, text_col, title_col, label_col, url_col, date_col] if c in df.columns]].copy()
    keep.rename(columns={
        id_col:"doc_id", text_col:"text", title_col:"title", label_col:"label",
        url_col:"url", date_col:"published_at"
    }, inplace=True)

    keep["doc_id"] = keep["doc_id"].fillna("").astype(str)
    # Fallback ids
    keep.loc[keep["doc_id"].eq(""), "doc_id"] = [str(uuid.uuid4()) for _ in range(keep["doc_id"].eq("").sum())]
    keep["text"] = keep["text"].astype(str).map(clean_text)
    keep["title"] = keep.get("title", pd.Series([""]*len(keep))).astype(str).map(clean_text)
    keep["published_at"] = keep.get("published_at", pd.Series([""]*len(keep))).astype(str)
    keep["url"] = keep.get("url", pd.Series([""]*len(keep))).astype(str)
    keep["source_domain"] = keep["url"].map(detect_domain)
    keep["label"] = keep["label"].str.lower().map(lambda x: "fake" if "fake" in x else ("credible" if "credible" in x else "other"))

    # Save docs parquet (not chunked)
    docs_path = out/"docs.parquet"
    keep.to_parquet(docs_path, index=False)
    print(f"Saved docs → {docs_path}")
    print("Chunking…")
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    for _,r in keep.iterrows():
        if not r["text"]: continue
        chs = chunk_text(r["text"], target_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens, enc=enc)
        for i,ct in enumerate(chs):
            chunks.append(ChunkRow(
                doc_id=r["doc_id"], chunk_id=i, label=r["label"], title=r["title"],
                source_domain=r["source_domain"], published_at=r["published_at"], url=r["url"],
                chunk_text=ct
            ))

    chunks_path = out/"chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False)+"\n")
    print(f"Saved chunks → {chunks_path} ({len(chunks)} chunks)")

    # Build BM25
    print("Building BM25…")
    tokenized = []
    for ch in chunks:
        toks = re.findall(r"\w+", ch.chunk_text.lower())
        tokenized.append(toks)
    bm25 = BM25Okapi(tokenized)


    with open(out/"bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks_meta": [asdict(c) for c in chunks]}, f)
    print(f"Saved BM25 → {out/'bm25.pkl'}")

    # Build FAISS
    print("Embedding for FAISS… (bge-small-en-v1.5)")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 384-d
    texts = [c.chunk_text for c in chunks]
    vecs = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine via normalized vectors
    index.add(vecs)
    faiss.write_index(index, str(out/"faiss.index"))
    print(f"Saved FAISS → {out/'faiss.index'}")

    # Meta
    meta = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "dim": int(vecs.shape[1]),
        "num_chunks": int(vecs.shape[0]),
        "bm25_docs": int(len(tokenized)),
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap_tokens
    }
    with open(out/"meta.json","w") as f:
        json.dump(meta, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="mini_index/store")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--url-col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--chunk-tokens", type=int, default=700)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    args = ap.parse_args()
    build(args)

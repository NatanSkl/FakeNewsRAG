
import json, re, random, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, pickle

STORE="mini_index/store"

def load():
    out = Path(STORE)
    with open(out/"meta.json") as f: meta = json.load(f)
    with open(out/"bm25.pkl","rb") as f: obj = pickle.load(f)
    bm25, chunks = obj["bm25"], obj["chunks_meta"]
    index = faiss.read_index(str(out/"faiss.index"))
    emb = SentenceTransformer(meta["embedding_model"])
    return emb, index, bm25, chunks

def encode(m, texts): return m.encode(texts, normalize_embeddings=True).astype("float32")

def claimify(text, m, max_sents=6):
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s]
    if not sents: return text
    vecs = encode(m, sents)
    centroid = vecs.mean(axis=0, keepdims=True)
    sims = (vecs @ centroid.T).ravel()
    top = sorted(np.argsort(-sims)[:max_sents])
    return " ".join([sents[i] for i in top])

def retrieve(m, index, bm25, chunks, qtext, K=200, KB=200, topn=20):
    qv = encode(m, [qtext])
    D,I = index.search(qv, K); dense_ids=I[0].tolist()
    qtok = re.findall(r"\w+", qtext.lower())
    try: scores_bm = bm25.get_scores(qtok)
    except:
        tokenized = [re.findall(r"\w+", c["chunk_text"].lower()) for c in chunks]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized)
        scores_bm = bm25.get_scores(qtok)
    bm_top = np.argsort(-scores_bm)[:KB].tolist()
    # simple RRF
    fused={}
    def add(lst, w=1.0):
        for r,idx in enumerate(lst,1): fused[idx]=fused.get(idx,0.0)+w*(1/(60+r))
    add(dense_ids,1.2); add(bm_top,1.0)
    cand = sorted(fused.keys(), key=lambda i:fused[i], reverse=True)
    return cand[:topn]

def build_pseudo_qrels(m, chunks, thresh=0.75, pool=2000):
    # sample some queries; for each, consider items with cosine>=thresh as relevant
    ids = random.sample(range(len(chunks)), min(pool, len(chunks)))
    texts = [chunks[i]["chunk_text"] for i in ids]
    vecs = encode(m, texts)
    # cosine via dot (already normalized)
    sim = vecs @ vecs.T
    qrels = {}
    for qi,i in enumerate(ids):
        rel = [ids[j] for j,s in enumerate(sim[qi]) if s>=thresh and j!=qi]
        qrels[i]=set(rel)
    return ids, qrels

def metrics_at_k(ranklist, relset, k=10):
    top = ranklist[:k]
    hit = [i for i,x in enumerate(top,1) if x in relset]
    recall = len(set(top) & relset)/max(1,len(relset))
    mrr = (1.0/hit[0]) if hit else 0.0
    ndcg = 0.0
    denom=0.0
    for i,x in enumerate(top,1):
        gain = 1.0 if x in relset else 0.0
        ndcg += gain/np.log2(i+1)
        denom += 1.0/np.log2(i+1)  # ideal DCG if at least 1 rel per rank; simple proxy
    ndcg = ndcg/max(1e-9,denom)
    return recall, mrr, ndcg

if __name__=="__main__":
    random.seed(42)
    emb, index, bm25, chunks = load()
    # sample 100 queries
    q_ids, qrels = build_pseudo_qrels(emb, chunks, thresh=0.78, pool=min(100,len(chunks)))
    R10=MRR10=N10=0.0; n=0
    for qi in q_ids:
        qtext = claimify(chunks[qi]["chunk_text"], emb)
        ranks = retrieve(emb, index, bm25, chunks, qtext, K=400, KB=400, topn=50)
        r,m,nDCG = metrics_at_k(ranks, qrels[qi], k=10)
        R10+=r; MRR10+=m; N10+=nDCG; n+=1
    print(f"Queries={n}  Recall@10={R10/n:.3f}  MRR@10={MRR10/n:.3f}  NDCG@10={N10/n:.3f}")

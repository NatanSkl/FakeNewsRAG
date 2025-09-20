
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

from rag_pipeline import load_store, retrieve_evidence, RetrievalConfig

def rows(label: str, hits: List[Dict[str, Any]]):
    out = []
    for h in hits:
        out.append({
            "label": label,
            "doc_id": h.get("doc_id",""),
            "chunk_id": h.get("chunk_id",""),
            "domain": h.get("source_domain",""),
            "date": h.get("published_date") or h.get("published_at",""),
            "title": (h.get("title","") or "")[:200],
            "snippet": (h.get("chunk_text","") or "").replace("\n"," ")[:500],
            "score": round(float(h.get("_score", h.get("rrf", 0.0))), 4),
        })
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show retrieval results only (credible & fake).")
    ap.add_argument("--store", default="mini_index/store")
    ap.add_argument("--file", required=True, help="Path to article .txt")
    ap.add_argument("--title", default="", help="Optional title to boost")
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--domain_cap", type=int, default=2)
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    store = load_store(args.store)

    cfg = RetrievalConfig(
        topn=args.topn,
        domain_cap=args.domain_cap,
        use_cross_encoder=False,
        use_xquad=False,

    )

    cred = retrieve_evidence(store, text, title_hint=args.title, label_name="credible", cfg=cfg)
    fake = retrieve_evidence(store, text, title_hint=args.title, label_name="fake",     cfg=cfg)

    rows_cred, rows_fake = rows("credible", cred), rows("fake", fake)

    print(f"\n=== CREDIBLE (top {len(rows_cred)}) ===")
    for i,r in enumerate(rows_cred,1):
        print(f"[{i}] {r['domain']} | {r['date']} | {r['title']}")
        print(f"     doc_id={r['doc_id']}  chunk_id={r['chunk_id']}  score={r['score']}")
        print(f"     {r['snippet']}\n")

    print(f"\n=== FAKE (top {len(rows_fake)}) ===")
    for i,r in enumerate(rows_fake,1):
        print(f"[{i}] {r['domain']} | {r['date']} | {r['title']}")
        print(f"     doc_id={r['doc_id']}  chunk_id={r['chunk_id']}  score={r['score']}")
        print(f"     {r['snippet']}\n")

    if args.save_json:
        payload = {"credible": rows_cred, "fake": rows_fake}
        Path(args.save_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON → {args.save_json}")

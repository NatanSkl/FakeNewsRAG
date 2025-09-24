"""
Simple testing tool for retrieval functionality.

This script tests the retrieval system without requiring LLMs.
It takes an article file and retrieves relevant fake and credible evidence.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from retrieval import load_store, retrieve_evidence, RetrievalConfig


def rows(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format retrieved hits for display."""
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append({
            "label": h.get("label", "unknown"),
            "doc_id": h.get("doc_id", ""),
            "chunk_id": h.get("chunk_id", 0),
            "domain": h.get("source_domain", ""),
            "date": h.get("published_at", ""),
            "title": h.get("title", "")[:80] + "..." if len(h.get("title", "")) > 80 else h.get("title", ""),
            "snippet": h.get("chunk_text", "")[:200] + "..." if len(h.get("chunk_text", "")) > 200 else h.get("chunk_text", ""),
            "score": h.get("_score", h.get("rrf", 0.0))
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Show retrieval results only (credible & fake).")
    parser.add_argument("--store", default="index/store", help="Path to index store")
    parser.add_argument("--file", required=True, help="Path to article .txt")
    parser.add_argument("--title", default="", help="Optional title to boost")
    parser.add_argument("--topn", type=int, default=12, help="Number of results to show")
    parser.add_argument("--domain_cap", type=int, default=2, help="Max results per domain")
    parser.add_argument("--save_json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load article text
    text = Path(args.file).read_text(encoding="utf-8")
    
    # Load index store
    store = load_store(args.store)

    # Configure retrieval
    cfg = RetrievalConfig(
        topn=args.topn,
        domain_cap=args.domain_cap,
        use_cross_encoder=False,
        use_xquad=False,
    )

    # Retrieve evidence for both labels
    print("Retrieving credible evidence...")
    cred = retrieve_evidence(store, text, title_hint=args.title, label_name="credible", cfg=cfg)
    
    print("Retrieving fake evidence...")
    fake = retrieve_evidence(store, text, title_hint=args.title, label_name="fake", cfg=cfg)

    # Format results
    rows_cred, rows_fake = rows(cred), rows(fake)

    # Display results
    print(f"\n=== CREDIBLE (top {len(rows_cred)}) ===")
    for i, r in enumerate(rows_cred, 1):
        print(f"[{i}] {r['domain']} | {r['date']} | {r['title']}")
        print(f"     doc_id={r['doc_id']}  chunk_id={r['chunk_id']}  score={r['score']:.3f}")
        print(f"     {r['snippet']}\n")

    print(f"\n=== FAKE (top {len(rows_fake)}) ===")
    for i, r in enumerate(rows_fake, 1):
        print(f"[{i}] {r['domain']} | {r['date']} | {r['title']}")
        print(f"     doc_id={r['doc_id']}  chunk_id={r['chunk_id']}  score={r['score']:.3f}")
        print(f"     {r['snippet']}\n")

    # Save to JSON if requested
    if args.save_json:
        payload = {"credible": rows_cred, "fake": rows_fake}
        Path(args.save_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON → {args.save_json}")


if __name__ == "__main__":
    main()
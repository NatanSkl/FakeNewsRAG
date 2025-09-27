#!/usr/bin/env python3
"""
Focused test for Lost in the Middle mechanism.

This test verifies that sentence max-pooling correctly addresses the "lost in the middle"
problem by scoring individual sentences and boosting documents with relevant middle content.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, sentence_maxpool_boost
)


def test_lost_in_middle(store, query_text: str, verbose: bool = True):
    """Test Lost in the Middle - Sentence Max-Pooling"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Lost in the Middle (Sentence Max-Pooling)")
        print("="*60)
    
    # Create config with sentence max-pooling enabled
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        sent_maxpool=True,
        sent_bonus=0.5,  # High bonus to see effect
        sent_max_sents=5,
        sent_min_len=15,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,  # No domain capping
        topn=5
    )
    
    # Get initial results without sentence max-pooling
    hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    
    # Apply sentence max-pooling boost
    hits_with_sent = sentence_maxpool_boost(store, qv, hits, cfg)
    
    if verbose:
        print(f"Query: '{query_text}'")
        print(f"Results with sentence max-pooling: {len(hits_with_sent)}")
        
        # Show difference in scoring
        print("\nTop 3 results (with sentence boost):")
        for i, h in enumerate(hits_with_sent[:3], 1):
            original_score = h.get('rrf', 0.0)
            sent_score = h.get('_sent_max', 0.0)
            final_score = h.get('_score', original_score)
            boost = final_score - original_score
            
            print(f"[{i}] Score: {final_score:.4f} (orig: {original_score:.4f}, sent: {sent_score:.4f}, boost: {boost:.4f})")
            print(f"     Text: {h.get('chunk_text', '')}...")
            print(f"     Domain: {h.get('source_domain', 'unknown')}")
            print()
    
    return hits_with_sent


def test_sentence_scoring_effectiveness(store, query_text: str):
    """Test how effectively sentence scoring identifies relevant content"""
    print("\n" + "="*60)
    print("SENTENCE SCORING EFFECTIVENESS TEST")
    print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        sent_maxpool=True,
        sent_bonus=0.5,
        sent_max_sents=8,
        sent_min_len=10,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=10
    )
    
    hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    hits_with_sent = sentence_maxpool_boost(store, qv, hits, cfg)
    
    # Analyze sentence scores
    print(f"Query: '{query_text}'")
    print(f"Analyzing {len(hits_with_sent)} results...")
    
    # Sort by sentence score to see which documents have high sentence relevance
    hits_by_sent_score = sorted(hits_with_sent, key=lambda h: h.get('_sent_max', 0.0), reverse=True)
    
    print("\nTop 5 results by sentence score:")
    for i, h in enumerate(hits_by_sent_score[:5], 1):
        sent_score = h.get('_sent_max', 0.0)
        original_score = h.get('rrf', 0.0)
        final_score = h.get('_score', original_score)
        
        print(f"[{i}] Sentence Score: {sent_score:.4f} | Final: {final_score:.4f} | Orig: {original_score:.4f}")
        print(f"     Text: {h.get('chunk_text', '')[:150]}...")
        print(f"     Domain: {h.get('source_domain', 'unknown')}")
        print()
    
    # Show documents that got the biggest boost
    boosts = [(h, h.get('_score', h.get('rrf', 0.0)) - h.get('rrf', 0.0)) for h in hits_with_sent]
    boosts.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 3 documents with biggest sentence boost:")
    for i, (h, boost) in enumerate(boosts[:3], 1):
        print(f"[{i}] Boost: {boost:.4f} | Sentence Score: {h.get('_sent_max', 0.0):.4f}")
        print(f"     Text: {h.get('chunk_text', '')[:150]}...")
        print()
    
    return hits_with_sent


def test_different_sentence_configs(store, query_text: str):
    """Test different sentence max-pooling configurations"""
    print("\n" + "="*60)
    print("SENTENCE CONFIGURATION COMPARISON")
    print("="*60)
    
    configs = [
        {"sent_bonus": 0.0, "sent_max_sents": 5, "name": "No sentence boost"},
        {"sent_bonus": 0.25, "sent_max_sents": 5, "name": "Low sentence boost"},
        {"sent_bonus": 0.5, "sent_max_sents": 5, "name": "Medium sentence boost"},
        {"sent_bonus": 1.0, "sent_max_sents": 5, "name": "High sentence boost"},
        {"sent_bonus": 0.5, "sent_max_sents": 3, "name": "Few sentences (3)"},
        {"sent_bonus": 0.5, "sent_max_sents": 8, "name": "Many sentences (8)"},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        cfg = RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            sent_maxpool=True,
            sent_bonus=config['sent_bonus'],
            sent_max_sents=config['sent_max_sents'],
            sent_min_len=15,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=5
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        hits_with_sent = sentence_maxpool_boost(store, qv, hits, cfg)
        
        # Show top result
        if hits_with_sent:
            top_hit = hits_with_sent[0]
            original_score = top_hit.get('rrf', 0.0)
            sent_score = top_hit.get('_sent_max', 0.0)
            final_score = top_hit.get('_score', original_score)
            boost = final_score - original_score
            
            print(f"Top result: {final_score:.4f} (orig: {original_score:.4f}, sent: {sent_score:.4f}, boost: {boost:.4f})")
            print(f"Text: {top_hit.get('chunk_text', '')[:100]}...")


def test_long_vs_short_documents(store, query_text: str):
    """Test how sentence max-pooling affects long vs short documents"""
    print("\n" + "="*60)
    print("LONG VS SHORT DOCUMENTS TEST")
    print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        sent_maxpool=True,
        sent_bonus=0.5,
        sent_max_sents=8,
        sent_min_len=10,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    hits_with_sent = sentence_maxpool_boost(store, qv, hits, cfg)
    
    # Analyze by document length
    short_docs = [h for h in hits_with_sent if len(h.get('chunk_text', '')) < 500]
    long_docs = [h for h in hits_with_sent if len(h.get('chunk_text', '')) >= 500]
    
    print(f"Query: '{query_text}'")
    print(f"Short documents (<500 chars): {len(short_docs)}")
    print(f"Long documents (>=500 chars): {len(long_docs)}")
    
    if short_docs:
        print("\nTop short document:")
        h = short_docs[0]
        print(f"Length: {len(h.get('chunk_text', ''))} chars")
        print(f"Score: {h.get('_score', 0.0):.4f} (sent: {h.get('_sent_max', 0.0):.4f})")
        print(f"Text: {h.get('chunk_text', '')[:200]}...")
    
    if long_docs:
        print("\nTop long document:")
        h = long_docs[0]
        print(f"Length: {len(h.get('chunk_text', ''))} chars")
        print(f"Score: {h.get('_score', 0.0):.4f} (sent: {h.get('_sent_max', 0.0):.4f})")
        print(f"Text: {h.get('chunk_text', '')[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Test Lost in the Middle mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=[
        "basic", "effectiveness", "configs", "length", "all"
    ], default="basic", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    # Load store
    try:
        store = load_store(args.store)
        print(f"Loaded store from {args.store}")
        print(f"Store contains {len(store.chunks)} chunks")
        print(f"Embedding model: {store.meta.get('embedding_model', 'unknown')}")
    except Exception as e:
        print(f"Failed to load store: {e}")
        return
    
    print(f"\nTesting Lost in the Middle with query: '{args.query}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_lost_in_middle(store, args.query, verbose=args.verbose)
    
    if args.test in ["effectiveness", "all"]:
        test_sentence_scoring_effectiveness(store, args.query)
    
    if args.test in ["configs", "all"]:
        test_different_sentence_configs(store, args.query)
    
    if args.test in ["length", "all"]:
        test_long_vs_short_documents(store, args.query)
    
    print("\n" + "="*60)
    print("LOST IN THE MIDDLE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()


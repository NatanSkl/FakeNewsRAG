#!/usr/bin/env python3
"""
Focused test for xQuAD (eXtended Query Aspect Diversification) mechanism.

This test verifies that xQuAD correctly diversifies results using multiple query aspects.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, xquad_diversify, 
    make_mqe_variants, encode
)


def test_xquad_basic(store, query_text: str, verbose: bool = True):
    """Test xQuAD with basic configuration"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: xQuAD Diversification")
        print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_xquad=True,
        xquad_k=12,
        xquad_lambda=0.6,
        xquad_aspects=3,
        use_cross_encoder=False,
        domain_cap=0,
        topn=15
    )
    
    # Get initial results without xQuAD
    hits_no_xquad, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    
    # Apply xQuAD
    variants = make_mqe_variants(query_text, None, store.emb)
    aspects_text = variants[:cfg.xquad_aspects]
    aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
    hits_xquad = xquad_diversify(store, qv, hits_no_xquad, list(aspects_vecs), cfg)
    
    if verbose:
        print(f"Query: '{query_text}'")
        print(f"Results without xQuAD: {len(hits_no_xquad)}")
        print(f"Results with xQuAD: {len(hits_xquad)}")
        
        # Show domain diversity
        print("\nDomain diversity comparison:")
        domains_no_xquad = [h.get('id', 'unknown') for h in hits_no_xquad[:10]]
        domains_xquad = [h.get('id', 'unknown') for h in hits_xquad[:10]]
        
        print("Without xQuAD:", set(domains_no_xquad))
        print("With xQuAD:", set(domains_xquad))
        
        print(f"\nUnique domains - No xQuAD: {len(set(domains_no_xquad))}")
        print(f"Unique domains - With xQuAD: {len(set(domains_xquad))}")
    
    return hits_xquad


def test_xquad_aspect_analysis(store, query_text: str):
    """Analyze how xQuAD uses different query aspects"""
    print("\n" + "="*60)
    print("xQuAD ASPECT ANALYSIS")
    print("="*60)
    
    # Generate query variants (aspects)
    variants = make_mqe_variants(query_text, None, store.emb)
    print(f"Query: '{query_text}'")
    print(f"Generated {len(variants)} query variants:")
    
    for i, variant in enumerate(variants, 1):
        print(f"[{i}] {variant[:150]}...")
    
    # Test with different numbers of aspects
    aspect_counts = [1, 2, 3]
    
    for aspect_count in aspect_counts:
        print(f"\n--- Using {aspect_count} aspects ---")
        
        cfg = RetrievalConfig(
            k_dense=30,
            k_bm25=30,
            use_xquad=True,
            xquad_k=15,
            xquad_lambda=0.6,
            xquad_aspects=aspect_count,
            use_cross_encoder=False,
            domain_cap=0,
            topn=20
        )
        
        hits_no_xquad, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        # Apply xQuAD with specified aspect count
        aspects_text = variants[:aspect_count]
        aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
        hits_xquad = xquad_diversify(store, qv, hits_no_xquad, list(aspects_vecs), cfg)
        
        domains_no_xquad = [h.get('id', 'unknown') for h in hits_no_xquad]
        domains_xquad = [h.get('id', 'unknown') for h in hits_xquad]
        
        diversity_no_xquad = len(set(domains_no_xquad)) / len(domains_no_xquad) if domains_no_xquad else 0
        diversity_xquad = len(set(domains_xquad)) / len(domains_xquad) if domains_xquad else 0
        
        print(f"Results: {len(hits_no_xquad)} → {len(hits_xquad)}")
        print(f"Diversity: {diversity_no_xquad:.2f} → {diversity_xquad:.2f}")
        print(f"Improvement: {diversity_xquad - diversity_no_xquad:.2f}")


def test_xquad_lambda_sensitivity(store, query_text: str):
    """Test sensitivity to different xQuAD lambda values"""
    print("\n" + "="*60)
    print("xQuAD LAMBDA SENSITIVITY TEST")
    print("="*60)
    
    # Test different lambda values
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    cfg_base = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_xquad=True,
        xquad_k=12,
        xquad_aspects=3,
        use_cross_encoder=False,
        domain_cap=0,
        topn=15
    )
    
    # Get base results
    hits_base, qv = hybrid_once(store, query_text, cfg_base, label_filter=None)
    
    # Generate aspects
    variants = make_mqe_variants(query_text, None, store.emb)
    aspects_text = variants[:cfg_base.xquad_aspects]
    aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
    
    results = []
    
    for lam in lambdas:
        cfg = RetrievalConfig(**cfg_base.__dict__, xquad_lambda=lam)
        hits_xquad = xquad_diversify(store, qv, hits_base, list(aspects_vecs), cfg)
        
        # Calculate diversity metrics
        domains = [h.get('id', 'unknown') for h in hits_xquad]
        unique_domains = len(set(domains))
        total_docs = len(domains)
        diversity_ratio = unique_domains / total_docs if total_docs > 0 else 0
        
        # Calculate average score
        avg_score = np.mean([h.get('rrf', 0.0) for h in hits_xquad]) if hits_xquad else 0.0
        
        results.append({
            'lambda': lam,
            'diversity_ratio': diversity_ratio,
            'unique_domains': unique_domains,
            'avg_score': avg_score
        })
        
        print(f"Lambda {lam:.1f}: Diversity {diversity_ratio:.2f} ({unique_domains}/{total_docs}), Avg Score {avg_score:.4f}")
    
    # Find optimal lambda
    best_lambda = max(results, key=lambda x: x['diversity_ratio'] + x['avg_score'])
    print(f"\nBest lambda: {best_lambda['lambda']:.1f} (Diversity: {best_lambda['diversity_ratio']:.2f}, Score: {best_lambda['avg_score']:.4f})")
    
    return results


def test_xquad_vs_no_xquad(store, query_text: str):
    """Compare xQuAD vs no xQuAD results"""
    print("\n" + "="*60)
    print("xQuAD vs NO-xQuAD COMPARISON")
    print("="*60)
    
    cfg_base = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_cross_encoder=False,
        domain_cap=0,
        topn=20
    )
    
    # No xQuAD
    cfg_no_xquad = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_xquad=False,
        use_cross_encoder=False,
        domain_cap=0,
        topn=20
    )
    hits_no_xquad, qv = hybrid_once(store, query_text, cfg_no_xquad, label_filter=None)
    
    # With xQuAD
    cfg_xquad = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_xquad=True,
        xquad_k=15,
        xquad_lambda=0.6,
        xquad_aspects=3,
        use_cross_encoder=False,
        domain_cap=0,
        topn=20
    )
    hits_base, qv = hybrid_once(store, query_text, cfg_xquad, label_filter=None)
    
    # Apply xQuAD
    variants = make_mqe_variants(query_text, None, store.emb)
    aspects_text = variants[:cfg_xquad.xquad_aspects]
    aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
    hits_xquad = xquad_diversify(store, qv, hits_base, list(aspects_vecs), cfg_xquad)
    
    print(f"Query: '{query_text}'")
    print(f"Results without xQuAD: {len(hits_no_xquad)}")
    print(f"Results with xQuAD: {len(hits_xquad)}")
    
    # Compare top results
    print("\nTop 5 results WITHOUT xQuAD:")
    for i, h in enumerate(hits_no_xquad[:5], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    print("Top 5 results WITH xQuAD:")
    for i, h in enumerate(hits_xquad[:5], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    # Compare diversity
    domains_no_xquad = [h.get('source_domain', 'unknown') for h in hits_no_xquad]
    domains_xquad = [h.get('source_domain', 'unknown') for h in hits_xquad]
    
    print(f"Domain diversity - No xQuAD: {len(set(domains_no_xquad))}/{len(domains_no_xquad)}")
    print(f"Domain diversity - With xQuAD: {len(set(domains_xquad))}/{len(domains_xquad)}")
    
    return hits_no_xquad, hits_xquad


def test_xquad_with_different_queries(store):
    """Test xQuAD with different types of queries"""
    print("\n" + "="*60)
    print("xQuAD WITH DIFFERENT QUERY TYPES")
    print("="*60)
    
    test_queries = [
        "artificial intelligence machine learning",
        "climate change global warming",
        "renewable energy solar power",
        "healthcare medical research",
        "economic inflation prices"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        cfg = RetrievalConfig(
            k_dense=25,
            k_bm25=25,
            use_xquad=True,
            xquad_k=12,
            xquad_lambda=0.6,
            xquad_aspects=3,
            use_cross_encoder=False,
            domain_cap=0,
            topn=15
        )
        
        hits_base, qv = hybrid_once(store, query, cfg, label_filter=None)
        
        # Apply xQuAD
        variants = make_mqe_variants(query, None, store.emb)
        aspects_text = variants[:cfg.xquad_aspects]
        aspects_vecs = store.emb.encode(aspects_text, normalize_embeddings=True).astype("float32")
        hits_xquad = xquad_diversify(store, qv, hits_base, list(aspects_vecs), cfg)
        
        domains_base = [h.get('id', 'unknown') for h in hits_base]
        domains_xquad = [h.get('id', 'unknown') for h in hits_xquad]
        
        diversity_base = len(set(domains_base)) / len(domains_base) if domains_base else 0
        diversity_xquad = len(set(domains_xquad)) / len(domains_xquad) if domains_xquad else 0
        
        print(f"Results: {len(hits_base)} → {len(hits_xquad)}")
        print(f"Diversity: {diversity_base:.2f} → {diversity_xquad:.2f}")
        print(f"Improvement: {diversity_xquad - diversity_base:.2f}")
        print(f"Top domains: {list(set(domains_xquad))[:3]}")


def test_xquad_aspect_quality(store, query_text: str):
    """Test the quality of different query aspects"""
    print("\n" + "="*60)
    print("xQuAD ASPECT QUALITY TEST")
    print("="*60)
    
    # Generate variants
    variants = make_mqe_variants(query_text, None, store.emb)
    print(f"Query: '{query_text}'")
    print(f"Generated {len(variants)} aspects:")
    
    for i, variant in enumerate(variants, 1):
        print(f"[{i}] {variant[:200]}...")
    
    # Test each aspect individually
    for i, aspect in enumerate(variants, 1):
        print(f"\n--- Testing Aspect {i} individually ---")
        
        cfg = RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            use_cross_encoder=False,
            domain_cap=0,
            topn=10
        )
        
        hits, qv = hybrid_once(store, aspect, cfg, label_filter=None)
        
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        total_docs = len(domains)
        diversity_ratio = unique_domains / total_docs if total_docs > 0 else 0
        
        print(f"Results: {len(hits)}, Diversity: {diversity_ratio:.2f} ({unique_domains}/{total_docs})")
        print(f"Top domains: {list(set(domains))[:3]}")


def main():
    parser = argparse.ArgumentParser(description="Test xQuAD diversity mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=[
        "basic", "aspects", "sensitivity", "comparison", "queries", "quality", "all"
    ], default="basic", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
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
    
    print(f"\nTesting xQuAD with query: '{args.query}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_xquad_basic(store, args.query, verbose=args.verbose)
    
    if args.test in ["aspects", "all"]:
        test_xquad_aspect_analysis(store, args.query)
    
    if args.test in ["sensitivity", "all"]:
        test_xquad_lambda_sensitivity(store, args.query)
    
    if args.test in ["comparison", "all"]:
        test_xquad_vs_no_xquad(store, args.query)
    
    if args.test in ["queries", "all"]:
        test_xquad_with_different_queries(store)
    
    if args.test in ["quality", "all"]:
        test_xquad_aspect_quality(store, args.query)
    
    print("\n" + "="*60)
    print("xQuAD TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

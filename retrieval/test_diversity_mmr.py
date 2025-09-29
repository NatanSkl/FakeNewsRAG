#!/usr/bin/env python3
"""
Focused test for MMR (Maximal Marginal Relevance) diversity mechanism.

This test verifies that MMR correctly balances relevance vs. diversity in retrieval results.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, mmr, encode
)


def test_mmr_basic(store, query_text: str, verbose: bool = True):
    """Test MMR with different lambda values"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: MMR (Maximal Marginal Relevance)")
        print("="*60)
    
    # Test different MMR lambda values
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for lam in lambdas:
        print(f"\n--- MMR Lambda = {lam} ---")
        
        cfg = RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            mmr_lambda=lam,
            mmr_k=15,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=10
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        print(f"Results: {len(hits)}")
        print("Top 5 domains:")
        domains = [h.get('id', 'unknown') for h in hits[:5]]
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count}")
        
        # Show diversity metrics
        unique_domains = len(set(domains))
        print(f"Unique domains in top 5: {unique_domains}")
    
    return hits


def test_mmr_diversity_analysis(store, query_text: str):
    """Analyze how MMR affects result diversity"""
    print("\n" + "="*60)
    print("MMR DIVERSITY ANALYSIS")
    print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    
    # Get results without MMR (lambda=1.0, pure relevance)
    cfg_no_mmr = RetrievalConfig(**cfg.__dict__, mmr_lambda=1.0, mmr_k=20)
    hits_no_mmr, qv = hybrid_once(store, query_text, cfg_no_mmr, label_filter=None)
    
    # Get results with MMR (lambda=0.5, balanced)
    cfg_mmr = RetrievalConfig(**cfg.__dict__, mmr_lambda=0.5, mmr_k=20)
    hits_mmr, qv = hybrid_once(store, query_text, cfg_mmr, label_filter=None)
    
    print(f"Query: '{query_text}'")
    print(f"Results without MMR: {len(hits_no_mmr)}")
    print(f"Results with MMR: {len(hits_mmr)}")
    
    # Analyze domain diversity
    def analyze_diversity(hits, name):
        domains = [h.get('id', 'unknown') for h in hits]
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        
        print(f"\n{name} domain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count}")
        
        unique_domains = len(set(domains))
        total_docs = len(domains)
        diversity_ratio = unique_domains / total_docs if total_docs > 0 else 0
        
        print(f"Unique domains: {unique_domains}/{total_docs} ({diversity_ratio:.2f})")
        return diversity_ratio
    
    diversity_no_mmr = analyze_diversity(hits_no_mmr, "Without MMR (pure relevance)")
    diversity_mmr = analyze_diversity(hits_mmr, "With MMR (balanced)")
    
    print(f"\nDiversity improvement: {diversity_mmr - diversity_no_mmr:.2f}")
    
    return hits_mmr


def test_mmr_lambda_sensitivity(store, query_text: str):
    """Test sensitivity to different MMR lambda values"""
    print("\n" + "="*60)
    print("MMR LAMBDA SENSITIVITY TEST")
    print("="*60)
    
    # Test a range of lambda values
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for lam in lambdas:
        cfg = RetrievalConfig(
            k_dense=25,
            k_bm25=25,
            mmr_lambda=lam,
            mmr_k=15,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=15
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        # Calculate diversity metrics
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        total_docs = len(domains)
        diversity_ratio = unique_domains / total_docs if total_docs > 0 else 0
        
        # Calculate average score
        avg_score = np.mean([h.get('rrf', 0.0) for h in hits]) if hits else 0.0
        
        results.append({
            'lambda': lam,
            'diversity_ratio': diversity_ratio,
            'unique_domains': unique_domains,
            'avg_score': avg_score
        })
        
        print(f"Lambda {lam:.1f}: Diversity {diversity_ratio:.2f} ({unique_domains}/{total_docs}), Avg Score {avg_score:.4f}")
    
    # Find optimal lambda (balance between diversity and relevance)
    best_lambda = max(results, key=lambda x: x['diversity_ratio'] + x['avg_score'])
    print(f"\nBest lambda: {best_lambda['lambda']:.1f} (Diversity: {best_lambda['diversity_ratio']:.2f}, Score: {best_lambda['avg_score']:.4f})")
    
    return results


def test_mmr_vs_no_mmr(store, query_text: str):
    """Compare MMR vs no MMR results"""
    print("\n" + "="*60)
    print("MMR vs NO-MMR COMPARISON")
    print("="*60)
    
    cfg_base = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    # No MMR (pure relevance)
    cfg_no_mmr = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        mmr_lambda=1.0,
        mmr_k=15,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    hits_no_mmr, qv = hybrid_once(store, query_text, cfg_no_mmr, label_filter=None)
    
    # With MMR
    cfg_mmr = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        mmr_lambda=0.5,
        mmr_k=15,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    hits_mmr, qv = hybrid_once(store, query_text, cfg_mmr, label_filter=None)
    
    print(f"Query: '{query_text}'")
    
    # Compare top results
    print("\nTop 5 results WITHOUT MMR (pure relevance):")
    for i, h in enumerate(hits_no_mmr[:5], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    print("Top 5 results WITH MMR (balanced):")
    for i, h in enumerate(hits_mmr[:5], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    # Compare diversity
    domains_no_mmr = [h.get('id', 'unknown') for h in hits_no_mmr]
    domains_mmr = [h.get('id', 'unknown') for h in hits_mmr]
    
    print(f"Domain diversity - No MMR: {len(set(domains_no_mmr))}/{len(domains_no_mmr)}")
    print(f"Domain diversity - With MMR: {len(set(domains_mmr))}/{len(domains_mmr)}")
    
    return hits_no_mmr, hits_mmr


def test_mmr_with_different_queries(store):
    """Test MMR with different types of queries"""
    print("\n" + "="*60)
    print("MMR WITH DIFFERENT QUERY TYPES")
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
            k_dense=20,
            k_bm25=20,
            mmr_lambda=0.5,
            mmr_k=10,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=8
        )
        
        hits, qv = hybrid_once(store, query, cfg, label_filter=None)
        
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        total_docs = len(domains)
        diversity_ratio = unique_domains / total_docs if total_docs > 0 else 0
        
        print(f"Results: {len(hits)}, Diversity: {diversity_ratio:.2f} ({unique_domains}/{total_docs})")
        print(f"Top domains: {list(set(domains))[:3]}")


def main():
    parser = argparse.ArgumentParser(description="Test MMR diversity mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=[
        "basic", "diversity", "sensitivity", "comparison", "queries", "all"
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
    
    print(f"\nTesting MMR with query: '{args.query}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_mmr_basic(store, args.query, verbose=args.verbose)
    
    if args.test in ["diversity", "all"]:
        test_mmr_diversity_analysis(store, args.query)
    
    if args.test in ["sensitivity", "all"]:
        test_mmr_lambda_sensitivity(store, args.query)
    
    if args.test in ["comparison", "all"]:
        test_mmr_vs_no_mmr(store, args.query)
    
    if args.test in ["queries", "all"]:
        test_mmr_with_different_queries(store)
    
    print("\n" + "="*60)
    print("MMR TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

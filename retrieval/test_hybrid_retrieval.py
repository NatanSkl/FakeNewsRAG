#!/usr/bin/env python3
"""
Focused test for Hybrid Retrieval mechanism.

This test verifies that hybrid retrieval correctly combines dense and sparse
retrieval methods using Reciprocal Rank Fusion (RRF).
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, rrf
)


def test_hybrid_basic(store, query_text: str, verbose: bool = True):
    """Test hybrid retrieval with different weight combinations"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Hybrid Retrieval (Dense + Sparse)")
        print("="*60)
    
    # Test different weight combinations
    weight_configs = [
        (1.0, 0.0, "Dense only"),
        (0.0, 1.0, "Sparse only"), 
        (1.35, 1.0, "Balanced hybrid")
    ]
    
    for w_dense, w_lex, name in weight_configs:
        print(f"\n--- {name} (w_dense={w_dense}, w_lex={w_lex}) ---")
        
        cfg = RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            w_dense=w_dense,
            w_lex=w_lex,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=10
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        print(f"Results: {len(hits)}")
        print("Top 3 results:")
        for i, h in enumerate(hits[:3], 1):
            rrf_score = h.get('rrf', 0.0)
            print(f"  [{i}] RRF Score: {rrf_score:.4f}")
            print(f"      Text: {h.get('chunk_text', '')[:80]}...")
            print(f"      Domain: {h.get('source_domain', 'unknown')}")
    
    return hits


def test_hybrid_weight_sensitivity(store, query_text: str):
    """Test sensitivity to different weight combinations"""
    print("\n" + "="*60)
    print("HYBRID WEIGHT SENSITIVITY TEST")
    print("="*60)
    
    # Test different dense weights
    dense_weights = [0.0, 0.5, 1.0, 1.35, 2.0, 3.0]
    lex_weight = 1.0
    
    results = []
    
    for w_dense in dense_weights:
        cfg = RetrievalConfig(
            k_dense=25,
            k_bm25=25,
            w_dense=w_dense,
            w_lex=lex_weight,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=15
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        # Calculate metrics
        avg_score = np.mean([h.get('rrf', 0.0) for h in hits]) if hits else 0.0
        max_score = max([h.get('rrf', 0.0) for h in hits]) if hits else 0.0
        
        # Calculate domain diversity
        domains = [h.get('source_domain', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        diversity_ratio = unique_domains / len(domains) if domains else 0
        
        results.append({
            'w_dense': w_dense,
            'avg_score': avg_score,
            'max_score': max_score,
            'diversity_ratio': diversity_ratio,
            'unique_domains': unique_domains
        })
        
        print(f"Dense weight {w_dense:.1f}: Avg score {avg_score:.4f}, Max score {max_score:.4f}, Diversity {diversity_ratio:.2f}")
    
    # Find optimal weight
    best_weight = max(results, key=lambda x: x['avg_score'] + x['diversity_ratio'])
    print(f"\nBest dense weight: {best_weight['w_dense']:.1f} (Score: {best_weight['avg_score']:.4f}, Diversity: {best_weight['diversity_ratio']:.2f})")
    
    return results


def test_dense_vs_sparse_vs_hybrid(store, query_text: str):
    """Compare dense-only, sparse-only, and hybrid retrieval"""
    print("\n" + "="*60)
    print("DENSE vs SPARSE vs HYBRID COMPARISON")
    print("="*60)
    
    cfg_base = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    
    # Dense only
    cfg_dense = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        w_dense=1.0,
        w_lex=0.0,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    hits_dense, qv = hybrid_once(store, query_text, cfg_dense, label_filter=None)
    
    # Sparse only
    cfg_sparse = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        w_dense=0.0,
        w_lex=1.0,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    hits_sparse, qv = hybrid_once(store, query_text, cfg_sparse, label_filter=None)
    
    # Hybrid
    cfg_hybrid = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        w_dense=1.35,
        w_lex=1.0,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    hits_hybrid, qv = hybrid_once(store, query_text, cfg_hybrid, label_filter=None)
    
    print(f"Query: '{query_text}'")
    print(f"Dense only results: {len(hits_dense)}")
    print(f"Sparse only results: {len(hits_sparse)}")
    print(f"Hybrid results: {len(hits_hybrid)}")
    
    # Compare top results
    print("\nTop 3 DENSE ONLY results:")
    for i, h in enumerate(hits_dense[:3], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | Domain: {h.get('source_domain', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    print("Top 3 SPARSE ONLY results:")
    for i, h in enumerate(hits_sparse[:3], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | Domain: {h.get('source_domain', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    print("Top 3 HYBRID results:")
    for i, h in enumerate(hits_hybrid[:3], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | Domain: {h.get('source_domain', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    # Compare metrics
    def analyze_results(hits, name):
        scores = [h.get('rrf', 0.0) for h in hits]
        domains = [h.get('source_domain', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        
        print(f"{name} metrics:")
        print(f"  Avg score: {np.mean(scores):.4f}")
        print(f"  Max score: {np.max(scores):.4f}")
        print(f"  Unique domains: {unique_domains}/{len(domains)}")
        print(f"  Top domains: {list(set(domains))[:3]}")
        print()
    
    analyze_results(hits_dense, "Dense only")
    analyze_results(hits_sparse, "Sparse only")
    analyze_results(hits_hybrid, "Hybrid")
    
    return hits_dense, hits_sparse, hits_hybrid


def test_rrf_effectiveness(store, query_text: str):
    """Test the effectiveness of RRF fusion"""
    print("\n" + "="*60)
    print("RRF EFFECTIVENESS TEST")
    print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        w_dense=1.35,
        w_lex=1.0,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    
    print(f"Query: '{query_text}'")
    print(f"Analyzing {len(hits)} hybrid results...")
    
    # Analyze score distribution
    scores = [h.get('rrf', 0.0) for h in hits]
    print(f"\nScore distribution:")
    print(f"  Min score: {np.min(scores):.4f}")
    print(f"  Max score: {np.max(scores):.4f}")
    print(f"  Avg score: {np.mean(scores):.4f}")
    print(f"  Std dev: {np.std(scores):.4f}")
    
    # Show top results with detailed scores
    print(f"\nTop 5 results with RRF scores:")
    for i, h in enumerate(hits[:5], 1):
        rrf_score = h.get('rrf', 0.0)
        print(f"[{i}] RRF Score: {rrf_score:.4f}")
        print(f"     Text: {h.get('chunk_text', '')[:150]}...")
        print(f"     Domain: {h.get('source_domain', 'unknown')}")
        print()
    
    # Analyze domain distribution
    domains = [h.get('source_domain', 'unknown') for h in hits]
    domain_counts = {}
    for d in domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"Domain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count}")
    
    return hits


def test_hybrid_with_different_queries(store):
    """Test hybrid retrieval with different types of queries"""
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL WITH DIFFERENT QUERY TYPES")
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
            w_dense=1.35,
            w_lex=1.0,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=12
        )
        
        hits, qv = hybrid_once(store, query, cfg, label_filter=None)
        
        # Calculate metrics
        scores = [h.get('rrf', 0.0) for h in hits]
        domains = [h.get('source_domain', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        
        print(f"Results: {len(hits)}")
        print(f"Avg score: {np.mean(scores):.4f}")
        print(f"Unique domains: {unique_domains}/{len(domains)}")
        print(f"Top domains: {list(set(domains))[:3]}")


def test_hybrid_candidate_sets(store, query_text: str):
    """Test how different candidate set sizes affect hybrid retrieval"""
    print("\n" + "="*60)
    print("HYBRID CANDIDATE SET SIZE TEST")
    print("="*60)
    
    candidate_sizes = [10, 20, 30, 50, 100]
    
    results = []
    
    for k in candidate_sizes:
        cfg = RetrievalConfig(
            k_dense=k,
            k_bm25=k,
            w_dense=1.35,
            w_lex=1.0,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=15
        )
        
        hits, qv = hybrid_once(store, query_text, cfg, label_filter=None)
        
        # Calculate metrics
        scores = [h.get('rrf', 0.0) for h in hits]
        domains = [h.get('source_domain', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        
        results.append({
            'k': k,
            'results': len(hits),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'unique_domains': unique_domains
        })
        
        print(f"Candidate size {k:3d}: {len(hits):2d} results, avg score {np.mean(scores):.4f}, {unique_domains:2d} domains")
    
    # Find optimal candidate size
    best_size = max(results, key=lambda x: x['avg_score'] + x['unique_domains'] / 10)
    print(f"\nBest candidate size: {best_size['k']} (Avg score: {best_size['avg_score']:.4f}, Domains: {best_size['unique_domains']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Retrieval mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=[
        "basic", "weights", "comparison", "rrf", "queries", "candidates", "all"
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
    
    print(f"\nTesting Hybrid Retrieval with query: '{args.query}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_hybrid_basic(store, args.query, verbose=args.verbose)
    
    if args.test in ["weights", "all"]:
        test_hybrid_weight_sensitivity(store, args.query)
    
    if args.test in ["comparison", "all"]:
        test_dense_vs_sparse_vs_hybrid(store, args.query)
    
    if args.test in ["rrf", "all"]:
        test_rrf_effectiveness(store, args.query)
    
    if args.test in ["queries", "all"]:
        test_hybrid_with_different_queries(store)
    
    if args.test in ["candidates", "all"]:
        test_hybrid_candidate_sets(store, args.query)
    
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()


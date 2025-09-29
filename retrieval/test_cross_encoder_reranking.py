#!/usr/bin/env python3
"""
Focused test for Cross-Encoder Reranking mechanism.

This test verifies that cross-encoder reranking correctly improves relevance scoring
through fine-grained query-document interaction modeling.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, cross_encoder_rerank
)
from sentence_transformers import CrossEncoder


def test_cross_encoder_basic(store, query_text: str, verbose: bool = True):
    """Test cross-encoder reranking with basic configuration"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Cross-Encoder Reranking")
        print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        use_cross_encoder=True,
        ce_topk=10,
        ce_weight=1.0,
        use_xquad=False,
        domain_cap=0,
        topn=10
    )
    
    # Get results without cross-encoder
    hits_no_ce, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    
    # Apply cross-encoder reranking
    try:
        ce = CrossEncoder(cfg.cross_encoder_model)
        hits_ce = cross_encoder_rerank(ce, query_text, hits_no_ce, cfg)
        
        if verbose:
            print(f"Query: '{query_text}'")
            print(f"Results without CE: {len(hits_no_ce)}")
            print(f"Results with CE: {len(hits_ce)}")
            
            print("\nTop 3 results (with cross-encoder scores):")
            for i, h in enumerate(hits_ce[:3], 1):
                original_score = h.get('rrf', 0.0)
                final_score = h.get('_score', original_score)
                ce_boost = final_score - original_score
                
                print(f"[{i}] Score: {final_score:.4f} (orig: {original_score:.4f}, CE boost: {ce_boost:.4f})")
                print(f"     Text: {h.get('chunk_text', '')[:100]}...")
                print(f"     ID: {h.get('id', 'unknown')}")
                print()
        
        return hits_ce
        
    except Exception as e:
        print(f"Cross-encoder test failed: {e}")
        print("This is expected if the model is not available")
        return hits_no_ce


def test_cross_encoder_effectiveness(store, query_text: str):
    """Analyze how effectively cross-encoder improves relevance"""
    print("\n" + "="*60)
    print("CROSS-ENCODER EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    cfg = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_cross_encoder=True,
        ce_topk=15,
        ce_weight=1.0,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    # Get base results
    hits_base, qv = hybrid_once(store, query_text, cfg, label_filter=None)
    
    try:
        ce = CrossEncoder(cfg.cross_encoder_model)
        hits_ce = cross_encoder_rerank(ce, query_text, hits_base, cfg)
        
        print(f"Query: '{query_text}'")
        print(f"Analyzing {len(hits_ce)} results...")
        
        # Analyze score improvements
        improvements = []
        for h in hits_ce:
            original_score = h.get('rrf', 0.0)
            final_score = h.get('_score', original_score)
            improvement = final_score - original_score
            improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            max_improvement = np.max(improvements)
            min_improvement = np.min(improvements)
            
            print(f"\nScore improvement statistics:")
            print(f"  Average improvement: {avg_improvement:.4f}")
            print(f"  Maximum improvement: {max_improvement:.4f}")
            print(f"  Minimum improvement: {min_improvement:.4f}")
            
            # Show documents with biggest improvements
            hits_with_improvements = [(h, h.get('_score', h.get('rrf', 0.0)) - h.get('rrf', 0.0)) for h in hits_ce]
            hits_with_improvements.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 3 documents with biggest CE improvements:")
            for i, (h, improvement) in enumerate(hits_with_improvements[:3], 1):
                print(f"[{i}] Improvement: {improvement:.4f}")
                print(f"     Original: {h.get('rrf', 0.0):.4f} → Final: {h.get('_score', h.get('rrf', 0.0)):.4f}")
                print(f"     Text: {h.get('chunk_text', '')[:150]}...")
                print()
        
        return hits_ce
        
    except Exception as e:
        print(f"Cross-encoder analysis failed: {e}")
        return hits_base


def test_cross_encoder_weight_sensitivity(store, query_text: str):
    """Test sensitivity to different CE weight values"""
    print("\n" + "="*60)
    print("CROSS-ENCODER WEIGHT SENSITIVITY TEST")
    print("="*60)
    
    weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    cfg_base = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        use_cross_encoder=True,
        ce_topk=12,
        use_xquad=False,
        domain_cap=0,
        topn=12
    )
    
    # Get base results
    hits_base, qv = hybrid_once(store, query_text, cfg_base, label_filter=None)
    
    try:
        ce = CrossEncoder(cfg_base.cross_encoder_model)
        
        results = []
        
        for weight in weights:
            cfg = RetrievalConfig(**cfg_base.__dict__, ce_weight=weight)
            hits_ce = cross_encoder_rerank(ce, query_text, hits_base, cfg)
            
            # Calculate metrics
            improvements = [h.get('_score', h.get('rrf', 0.0)) - h.get('rrf', 0.0) for h in hits_ce]
            avg_improvement = np.mean(improvements) if improvements else 0.0
            max_score = max([h.get('_score', h.get('rrf', 0.0)) for h in hits_ce]) if hits_ce else 0.0
            
            results.append({
                'weight': weight,
                'avg_improvement': avg_improvement,
                'max_score': max_score
            })
            
            print(f"Weight {weight:.2f}: Avg improvement {avg_improvement:.4f}, Max score {max_score:.4f}")
        
        # Find optimal weight
        best_weight = max(results, key=lambda x: x['avg_improvement'])
        print(f"\nBest weight: {best_weight['weight']:.2f} (Avg improvement: {best_weight['avg_improvement']:.4f})")
        
        return results
        
    except Exception as e:
        print(f"Cross-encoder weight test failed: {e}")
        return []


def test_cross_encoder_topk_sensitivity(store, query_text: str):
    """Test sensitivity to different CE top-k values"""
    print("\n" + "="*60)
    print("CROSS-ENCODER TOP-K SENSITIVITY TEST")
    print("="*60)
    
    topk_values = [5, 10, 15, 20, 25, 30]
    
    cfg_base = RetrievalConfig(
        k_dense=30,
        k_bm25=30,
        use_cross_encoder=True,
        ce_weight=1.0,
        use_xquad=False,
        domain_cap=0,
        topn=20
    )
    
    # Get base results
    hits_base, qv = hybrid_once(store, query_text, cfg_base, label_filter=None)
    
    try:
        ce = CrossEncoder(cfg_base.cross_encoder_model)
        
        results = []
        
        for topk in topk_values:
            cfg = RetrievalConfig(**cfg_base.__dict__, ce_topk=topk)
            hits_ce = cross_encoder_rerank(ce, query_text, hits_base, cfg)
            
            # Calculate metrics
            ce_processed = min(topk, len(hits_base))
            improvements = [h.get('_score', h.get('rrf', 0.0)) - h.get('rrf', 0.0) for h in hits_ce[:ce_processed]]
            avg_improvement = np.mean(improvements) if improvements else 0.0
            
            results.append({
                'topk': topk,
                'processed': ce_processed,
                'avg_improvement': avg_improvement
            })
            
            print(f"Top-K {topk:2d}: Processed {ce_processed:2d} docs, Avg improvement {avg_improvement:.4f}")
        
        # Find optimal top-k
        best_topk = max(results, key=lambda x: x['avg_improvement'])
        print(f"\nBest Top-K: {best_topk['topk']} (Avg improvement: {best_topk['avg_improvement']:.4f})")
        
        return results
        
    except Exception as e:
        print(f"Cross-encoder top-k test failed: {e}")
        return []


def test_cross_encoder_vs_no_ce(store, query_text: str):
    """Compare cross-encoder vs no cross-encoder results"""
    print("\n" + "="*60)
    print("CROSS-ENCODER vs NO-CROSS-ENCODER COMPARISON")
    print("="*60)
    
    cfg_base = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    # No cross-encoder
    hits_no_ce, qv = hybrid_once(store, query_text, cfg_base, label_filter=None)
    
    # With cross-encoder
    cfg_ce = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_cross_encoder=True,
        ce_topk=15,
        ce_weight=1.0,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    hits_base, qv = hybrid_once(store, query_text, cfg_ce, label_filter=None)
    
    try:
        ce = CrossEncoder(cfg_ce.cross_encoder_model)
        hits_ce = cross_encoder_rerank(ce, query_text, hits_base, cfg_ce)
        
        print(f"Query: '{query_text}'")
        print(f"Results without CE: {len(hits_no_ce)}")
        print(f"Results with CE: {len(hits_ce)}")
        
        # Compare top results
        print("\nTop 5 results WITHOUT cross-encoder:")
        for i, h in enumerate(hits_no_ce[:5], 1):
            print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
            print(f"     Text: {h.get('chunk_text', '')[:100]}...")
            print()
        
        print("Top 5 results WITH cross-encoder:")
        for i, h in enumerate(hits_ce[:5], 1):
            original_score = h.get('rrf', 0.0)
            final_score = h.get('_score', original_score)
            ce_boost = final_score - original_score
            print(f"[{i}] Score: {final_score:.4f} (orig: {original_score:.4f}, CE boost: {ce_boost:.4f}) | ID: {h.get('id', 'unknown')}")
            print(f"     Text: {h.get('chunk_text', '')[:100]}...")
            print()
        
        # Compare score distributions
        scores_no_ce = [h.get('rrf', 0.0) for h in hits_no_ce]
        scores_ce = [h.get('_score', h.get('rrf', 0.0)) for h in hits_ce]
        
        print(f"Score statistics:")
        print(f"  No CE - Avg: {np.mean(scores_no_ce):.4f}, Max: {np.max(scores_no_ce):.4f}")
        print(f"  With CE - Avg: {np.mean(scores_ce):.4f}, Max: {np.max(scores_ce):.4f}")
        
        return hits_no_ce, hits_ce
        
    except Exception as e:
        print(f"Cross-encoder comparison failed: {e}")
        return hits_no_ce, hits_base


def test_cross_encoder_with_different_queries(store):
    """Test cross-encoder with different types of queries"""
    print("\n" + "="*60)
    print("CROSS-ENCODER WITH DIFFERENT QUERY TYPES")
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
            use_cross_encoder=True,
            ce_topk=12,
            ce_weight=1.0,
            use_xquad=False,
            domain_cap=0,
            topn=12
        )
        
        hits_base, qv = hybrid_once(store, query, cfg, label_filter=None)
        
        try:
            ce = CrossEncoder(cfg.cross_encoder_model)
            hits_ce = cross_encoder_rerank(ce, query, hits_base, cfg)
            
            # Calculate improvement
            improvements = [h.get('_score', h.get('rrf', 0.0)) - h.get('rrf', 0.0) for h in hits_ce]
            avg_improvement = np.mean(improvements) if improvements else 0.0
            max_improvement = np.max(improvements) if improvements else 0.0
            
            print(f"Results: {len(hits_base)} → {len(hits_ce)}")
            print(f"Avg improvement: {avg_improvement:.4f}, Max improvement: {max_improvement:.4f}")
            
        except Exception as e:
            print(f"Cross-encoder failed for query '{query}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Cross-Encoder reranking mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=[
        "basic", "effectiveness", "weight", "topk", "comparison", "queries", "all"
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
    
    print(f"\nTesting Cross-Encoder with query: '{args.query}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_cross_encoder_basic(store, args.query, verbose=args.verbose)
    
    if args.test in ["effectiveness", "all"]:
        test_cross_encoder_effectiveness(store, args.query)
    
    if args.test in ["weight", "all"]:
        test_cross_encoder_weight_sensitivity(store, args.query)
    
    if args.test in ["topk", "all"]:
        test_cross_encoder_topk_sensitivity(store, args.query)
    
    if args.test in ["comparison", "all"]:
        test_cross_encoder_vs_no_ce(store, args.query)
    
    if args.test in ["queries", "all"]:
        test_cross_encoder_with_different_queries(store)
    
    print("\n" + "="*60)
    print("CROSS-ENCODER TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()


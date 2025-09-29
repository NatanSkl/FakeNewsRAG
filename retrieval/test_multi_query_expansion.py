#!/usr/bin/env python3
"""
Focused test for Multi-Query Expansion (MQE) mechanism.

This test verifies that MQE correctly generates multiple query variants
and improves retrieval coverage through query expansion.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from retrieval import (
    load_store, RetrievalConfig, hybrid_once, make_mqe_variants, claimify
)


def test_mqe_basic(store, article_text: str, title_hint: str = None, verbose: bool = True):
    """Test MQE with basic configuration"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Multi-Query Expansion (MQE)")
        print("="*60)
    
    # Generate query variants
    variants = make_mqe_variants(article_text, title_hint, store.emb)
    
    if verbose:
        print(f"Original text: {article_text[:200]}...")
        if title_hint:
            print(f"Title hint: {title_hint}")
        print(f"\nGenerated {len(variants)} query variants:")
        
        for i, variant in enumerate(variants, 1):
            print(f"[{i}] {variant[:150]}...")
    
    # Test retrieval with each variant
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=10
    )
    
    all_results = []
    
    for i, variant in enumerate(variants, 1):
        if verbose:
            print(f"\n--- Testing Variant {i} ---")
        
        hits, qv = hybrid_once(store, variant, cfg, label_filter=None)
        all_results.extend(hits[:5])  # Top 5 from each variant
        
        if verbose:
            print(f"Results: {len(hits)}")
            print(f"Top 3 IDs: {list(set([h.get('id', 'unknown') for h in hits[:3]]))}")
    
    if verbose:
        print(f"\nTotal unique results across all variants: {len(set([(h.get('id'), h.get('chunk_id')) for h in all_results]))}")
    
    return variants, all_results


def test_claimify_effectiveness(store, article_text: str):
    """Test the claimify function effectiveness"""
    print("\n" + "="*60)
    print("CLAIMIFY EFFECTIVENESS TEST")
    print("="*60)
    
    print(f"Original text: {article_text[:300]}...")
    
    # Test different claimify configurations
    max_sents_configs = [3, 6, 9, 12]
    
    for max_sents in max_sents_configs:
        print(f"\n--- Claimify with max_sents={max_sents} ---")
        
        claimified = claimify(article_text, store.emb, max_sents=max_sents)
        print(f"Claimified text: {claimified[:200]}...")
        print(f"Length: {len(claimified)} characters")
        
        # Test retrieval with claimified text
        cfg = RetrievalConfig(
            k_dense=15,
            k_bm25=15,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=8
        )
        
        hits, qv = hybrid_once(store, claimified, cfg, label_filter=None)
        
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        
        print(f"Retrieval results: {len(hits)}, Unique domains: {unique_domains}")
        print(f"Top domains: {list(set(domains))[:3]}")


def test_mqe_variant_quality(store, article_text: str, title_hint: str = None):
    """Analyze the quality of different MQE variants"""
    print("\n" + "="*60)
    print("MQE VARIANT QUALITY ANALYSIS")
    print("="*60)
    
    # Generate variants
    variants = make_mqe_variants(article_text, title_hint, store.emb)
    
    print(f"Original text: {article_text[:200]}...")
    if title_hint:
        print(f"Title hint: {title_hint}")
    
    print(f"\nGenerated {len(variants)} variants:")
    for i, variant in enumerate(variants, 1):
        print(f"[{i}] {variant[:200]}...")
        print(f"    Length: {len(variant)} characters")
        print()
    
    # Test each variant individually
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=10
    )
    
    variant_results = []
    
    for i, variant in enumerate(variants, 1):
        print(f"--- Testing Variant {i} individually ---")
        
        hits, qv = hybrid_once(store, variant, cfg, label_filter=None)
        
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        avg_score = np.mean([h.get('rrf', 0.0) for h in hits]) if hits else 0.0
        
        variant_results.append({
            'variant': i,
            'text': variant,
            'results': len(hits),
            'unique_domains': unique_domains,
            'avg_score': avg_score
        })
        
        print(f"Results: {len(hits)}, Unique domains: {unique_domains}, Avg score: {avg_score:.4f}")
        print(f"Top domains: {list(set(domains))[:3]}")
        print()
    
    # Compare variants
    print("Variant comparison:")
    for result in variant_results:
        print(f"Variant {result['variant']}: {result['results']} results, {result['unique_domains']} domains, score {result['avg_score']:.4f}")
    
    return variant_results


def test_mqe_vs_single_query(store, article_text: str, title_hint: str = None):
    """Compare MQE vs single query retrieval"""
    print("\n" + "="*60)
    print("MQE vs SINGLE QUERY COMPARISON")
    print("="*60)
    
    # Single query retrieval
    cfg = RetrievalConfig(
        k_dense=25,
        k_bm25=25,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=15
    )
    
    hits_single, qv = hybrid_once(store, article_text, cfg, label_filter=None)
    
    # MQE retrieval
    variants = make_mqe_variants(article_text, title_hint, store.emb)
    
    all_mqe_results = []
    for variant in variants:
        hits, qv = hybrid_once(store, variant, cfg, label_filter=None)
        all_mqe_results.extend(hits[:5])  # Top 5 from each variant
    
    # Remove duplicates based on doc_id and chunk_id
    seen = set()
    unique_mqe_results = []
    for h in all_mqe_results:
        key = (h.get('id'), h.get('chunk_id'))
        if key not in seen:
            seen.add(key)
            unique_mqe_results.append(h)
    
    print(f"Original text: {article_text[:200]}...")
    if title_hint:
        print(f"Title hint: {title_hint}")
    
    print(f"\nSingle query results: {len(hits_single)}")
    print(f"MQE results: {len(unique_mqe_results)}")
    
    # Compare domain diversity
    domains_single = [h.get('id', 'unknown') for h in hits_single]
    domains_mqe = [h.get('id', 'unknown') for h in unique_mqe_results]
    
    diversity_single = len(set(domains_single)) / len(domains_single) if domains_single else 0
    diversity_mqe = len(set(domains_mqe)) / len(domains_mqe) if domains_mqe else 0
    
    print(f"\nDomain diversity:")
    print(f"  Single query: {diversity_single:.2f} ({len(set(domains_single))}/{len(domains_single)})")
    print(f"  MQE: {diversity_mqe:.2f} ({len(set(domains_mqe))}/{len(domains_mqe)})")
    print(f"  Improvement: {diversity_mqe - diversity_single:.2f}")
    
    # Show top results
    print(f"\nTop 3 single query results:")
    for i, h in enumerate(hits_single[:3], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    print(f"Top 3 MQE results:")
    for i, h in enumerate(unique_mqe_results[:3], 1):
        print(f"[{i}] Score: {h.get('rrf', 0.0):.4f} | ID: {h.get('id', 'unknown')}")
        print(f"     Text: {h.get('chunk_text', '')[:100]}...")
        print()
    
    return hits_single, unique_mqe_results


def test_mqe_with_different_articles(store):
    """Test MQE with different types of articles"""
    print("\n" + "="*60)
    print("MQE WITH DIFFERENT ARTICLE TYPES")
    print("="*60)
    
    test_articles = [
        {
            "text": "The COVID-19 vaccines have been extensively tested for safety and efficacy. Multiple studies show they are safe and effective at preventing severe disease. The vaccines were developed using established scientific methods and have undergone rigorous clinical trials.",
            "title": "COVID-19 Vaccine Safety Study"
        },
        {
            "text": "Artificial intelligence and machine learning are transforming industries. Deep learning algorithms can process vast amounts of data to identify patterns and make predictions. These technologies are being applied in healthcare, finance, and transportation.",
            "title": "AI and Machine Learning Revolution"
        },
        {
            "text": "Climate change is causing global warming through greenhouse gas emissions. Rising temperatures affect weather patterns and sea levels worldwide. Renewable energy sources like solar and wind power are becoming more cost-effective alternatives.",
            "title": "Climate Change Impact Analysis"
        }
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n--- Article {i}: {article['title']} ---")
        
        variants = make_mqe_variants(article['text'], article['title'], store.emb)
        
        print(f"Generated {len(variants)} variants:")
        for j, variant in enumerate(variants, 1):
            print(f"  [{j}] {variant[:100]}...")
        
        # Test retrieval with first variant
        cfg = RetrievalConfig(
            k_dense=15,
            k_bm25=15,
            use_cross_encoder=False,
            use_xquad=False,
            domain_cap=0,
            topn=10
        )
        
        hits, qv = hybrid_once(store, variants[0], cfg, label_filter=None)
        
        domains = [h.get('id', 'unknown') for h in hits]
        unique_domains = len(set(domains))
        
        print(f"Retrieval results: {len(hits)}, Unique domains: {unique_domains}")
        print(f"Top domains: {list(set(domains))[:3]}")


def test_mqe_variant_overlap(store, article_text: str, title_hint: str = None):
    """Analyze overlap between different MQE variants"""
    print("\n" + "="*60)
    print("MQE VARIANT OVERLAP ANALYSIS")
    print("="*60)
    
    variants = make_mqe_variants(article_text, title_hint, store.emb)
    
    print(f"Original text: {article_text[:200]}...")
    if title_hint:
        print(f"Title hint: {title_hint}")
    
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        use_cross_encoder=False,
        use_xquad=False,
        domain_cap=0,
        topn=10
    )
    
    # Get results for each variant
    variant_results = []
    for i, variant in enumerate(variants, 1):
        hits, qv = hybrid_once(store, variant, cfg, label_filter=None)
        doc_ids = set([h.get('id') for h in hits])
        variant_results.append({
            'variant': i,
            'text': variant,
            'doc_ids': doc_ids,
            'count': len(hits)
        })
        print(f"Variant {i}: {len(hits)} results, {len(doc_ids)} unique documents")
    
    # Calculate overlaps
    print(f"\nOverlap analysis:")
    for i in range(len(variant_results)):
        for j in range(i + 1, len(variant_results)):
            overlap = variant_results[i]['doc_ids'] & variant_results[j]['doc_ids']
            overlap_ratio = len(overlap) / len(variant_results[i]['doc_ids'] | variant_results[j]['doc_ids']) if (variant_results[i]['doc_ids'] | variant_results[j]['doc_ids']) else 0
            print(f"Variants {i+1} & {j+1}: {len(overlap)} overlapping docs, ratio: {overlap_ratio:.2f}")
    
    # Overall coverage
    all_docs = set()
    for result in variant_results:
        all_docs.update(result['doc_ids'])
    
    print(f"\nTotal unique documents across all variants: {len(all_docs)}")
    
    return variant_results


def main():
    parser = argparse.ArgumentParser(description="Test Multi-Query Expansion mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--article", default="Artificial intelligence and machine learning are transforming industries. Deep learning algorithms can process vast amounts of data to identify patterns and make predictions.", help="Test article text")
    parser.add_argument("--title", default="AI and Machine Learning", help="Test article title")
    parser.add_argument("--test", choices=[
        "basic", "claimify", "quality", "comparison", "articles", "overlap", "all"
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
    
    print(f"\nTesting MQE with article: '{args.article[:100]}...'")
    print(f"Title: '{args.title}'")
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_mqe_basic(store, args.article, args.title, verbose=args.verbose)
    
    if args.test in ["claimify", "all"]:
        test_claimify_effectiveness(store, args.article)
    
    if args.test in ["quality", "all"]:
        test_mqe_variant_quality(store, args.article, args.title)
    
    if args.test in ["comparison", "all"]:
        test_mqe_vs_single_query(store, args.article, args.title)
    
    if args.test in ["articles", "all"]:
        test_mqe_with_different_articles(store)
    
    if args.test in ["overlap", "all"]:
        test_mqe_variant_overlap(store, args.article, args.title)
    
    print("\n" + "="*60)
    print("MQE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()


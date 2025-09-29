#!/usr/bin/env python3
"""
Test Domain Diversity mechanism.

This script tests the domain diversity mechanism that limits the number of results
from a single source domain to enhance diversity across different sources.
"""

import argparse
import sys
import os
from typing import List, Dict, Any, Tuple
from collections import Counter
from dataclasses import replace

# Add the parent directory to the path to import from retrieval module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import (
    load_store, hybrid_once, RetrievalConfig, Store,
    apply_domain_cap, filter_by_metadata
)


def test_domain_cap_basic(store: Store, query: str, verbose: bool = False) -> None:
    """Test basic domain capping functionality."""
    print("=" * 60)
    print("TEST: Domain Cap Basic Functionality")
    print("=" * 60)
    
    # Test with domain cap enabled
    cfg_with_cap = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        topn=10,
        domain_cap=3  # Limit to 3 results per domain
    )
    
    # Test without domain cap (set to 0 to disable)
    cfg_without_cap = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        topn=10,
        domain_cap=0  # Disable domain cap
    )
    
    print(f"Query: '{query}'")
    print(f"Domain cap: {cfg_with_cap.domain_cap}")
    print()
    
    # Get results with domain cap
    results_with_cap, _ = hybrid_once(store, query, cfg_with_cap, label_filter=None)
    results_with_cap = apply_domain_cap(results_with_cap, cfg_with_cap.domain_cap)
    
    # Get results without domain cap
    results_without_cap, _ = hybrid_once(store, query, cfg_without_cap, label_filter=None)
    results_without_cap = apply_domain_cap(results_without_cap, cfg_without_cap.domain_cap)
    
    print(f"Results with domain cap: {len(results_with_cap)}")
    print(f"Results without domain cap: {len(results_without_cap)}")
    print()
    
    # Analyze ID distribution (using ID as proxy for diversity)
    domains_with_cap = [r.get('id', 'unknown') for r in results_with_cap]
    domains_without_cap = [r.get('id', 'unknown') for r in results_without_cap]
    
    domain_counts_with_cap = Counter(domains_with_cap)
    domain_counts_without_cap = Counter(domains_without_cap)
    
    print("Domain distribution WITH domain cap:")
    for domain, count in domain_counts_with_cap.most_common():
        print(f"  {domain}: {count} results")
    print()
    
    print("Domain distribution WITHOUT domain cap:")
    for domain, count in domain_counts_without_cap.most_common():
        print(f"  {domain}: {count} results")
    print()
    
    # Check if domain cap is working
    max_domain_count = max(domain_counts_with_cap.values()) if domain_counts_with_cap else 0
    print(f"Maximum results from single domain (with cap): {max_domain_count}")
    print(f"Domain cap limit: {cfg_with_cap.domain_cap}")
    
    if max_domain_count <= cfg_with_cap.domain_cap:
        print("Domain cap is working correctly")
    else:
        print("Domain cap is not working - exceeded limit")
    
    if verbose:
        print("\nDetailed results with domain cap:")
        for i, result in enumerate(results_with_cap[:5]):
            print(f"  {i+1}. {result.get('title', 'No title')[:50]}...")
            print(f"     ID: {result.get('id', 'unknown')}")
            print(f"     Score: {result.get('score', 0):.4f}")
            print()


def test_domain_cap_effectiveness(store: Store, query: str) -> None:
    """Test effectiveness of domain capping."""
    print("=" * 60)
    print("TEST: Domain Cap Effectiveness")
    print("=" * 60)
    
    # Test different domain cap values
    cap_values = [1, 2, 3, 5, 10]
    
    print(f"Query: '{query}'")
    print()
    
    for cap in cap_values:
        cfg = RetrievalConfig(
            k_dense=30,
            k_bm25=30,
            topn=15,
            domain_cap=cap
        )
        
        results, _ = hybrid_once(store, query, cfg, label_filter=None)
        results = apply_domain_cap(results, cfg.domain_cap)
        domains = [r.get('id', 'unknown') for r in results]
        domain_counts = Counter(domains)
        
        unique_domains = len(domain_counts)
        max_from_single_domain = max(domain_counts.values()) if domain_counts else 0
        total_results = len(results)
        
        print(f"Domain cap {cap:2d}: {total_results:2d} results, {unique_domains:2d} unique domains, max {max_from_single_domain:2d} from single domain")
    
    print()


def test_domain_cap_vs_diversity(store: Store, query: str) -> None:
    """Compare domain capping with other diversity mechanisms."""
    print("=" * 60)
    print("TEST: Domain Cap vs Other Diversity Mechanisms")
    print("=" * 60)
    
    # Test configurations
    configs = {
        "No diversity": RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            topn=10,
            domain_cap=0,
            use_xquad=False
        ),
        "Domain cap only": RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            topn=10,
            domain_cap=2,
            use_xquad=False
        ),
        "MMR only": RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            topn=10,
            domain_cap=0,
            mmr_lambda=0.7,
            use_xquad=False
        ),
        "Domain cap + MMR": RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            topn=10,
            domain_cap=2,
            mmr_lambda=0.7,
            use_xquad=False
        )
    }
    
    print(f"Query: '{query}'")
    print()
    
    for name, cfg in configs.items():
        results, _ = hybrid_once(store, query, cfg, label_filter=None)
        results = apply_domain_cap(results, cfg.domain_cap)
        domains = [r.get('id', 'unknown') for r in results]
        domain_counts = Counter(domains)
        
        unique_domains = len(domain_counts)
        max_from_single_domain = max(domain_counts.values()) if domain_counts else 0
        total_results = len(results)
        
        print(f"{name:20s}: {total_results:2d} results, {unique_domains:2d} unique domains, max {max_from_single_domain:2d} from single domain")
    
    print()


def test_domain_cap_sensitivity(store: Store, query: str) -> None:
    """Test sensitivity to domain cap parameter."""
    print("=" * 60)
    print("TEST: Domain Cap Sensitivity")
    print("=" * 60)
    
    # Test different domain cap values
    cap_values = [1, 2, 3, 4, 5]
    
    print(f"Query: '{query}'")
    print()
    
    results_by_cap = {}
    
    for cap in cap_values:
        cfg = RetrievalConfig(
            k_dense=25,
            k_bm25=25,
            topn=15,
            domain_cap=cap
        )
        
        results, _ = hybrid_once(store, query, cfg, label_filter=None)
        results_by_cap[cap] = results
        
        domains = [r.get('id', 'unknown') for r in results]
        domain_counts = Counter(domains)
        
        unique_domains = len(domain_counts)
        max_from_single_domain = max(domain_counts.values()) if domain_counts else 0
        total_results = len(results)
        
        print(f"Cap {cap}: {total_results:2d} results, {unique_domains:2d} unique domains, max {max_from_single_domain:2d} from single domain")
    
    print()
    
    # Analyze overlap between different cap values
    print("Overlap analysis:")
    for i, cap1 in enumerate(cap_values):
        for cap2 in cap_values[i+1:]:
            results1 = results_by_cap[cap1]
            results2 = results_by_cap[cap2]
            
            # Get document IDs for comparison
            ids1 = set(r.get('id', '') for r in results1)
            ids2 = set(r.get('id', '') for r in results2)
            
            overlap = len(ids1.intersection(ids2))
            total_unique = len(ids1.union(ids2))
            overlap_pct = (overlap / total_unique * 100) if total_unique > 0 else 0
            
            print(f"  Cap {cap1} vs Cap {cap2}: {overlap}/{total_unique} overlap ({overlap_pct:.1f}%)")
    
    print()


def test_domain_cap_with_metadata(store: Store, query: str) -> None:
    """Test domain capping with metadata filtering."""
    print("=" * 60)
    print("TEST: Domain Cap with Metadata Filtering")
    print("=" * 60)
    
    # Test with different metadata filters
    test_cases = [
        ("No filter", None),
        ("English only", "en"),
        ("Recent articles", "2023"),
        ("Specific label", "fake")
    ]
    
    print(f"Query: '{query}'")
    print()
    
    for filter_name, filter_value in test_cases:
        print(f"Filter: {filter_name}")
        
        # Create base config
        cfg = RetrievalConfig(
            k_dense=20,
            k_bm25=20,
            topn=10,
            domain_cap=2
        )
        
        # Apply metadata filter if specified
        if filter_value == "en":
            cfg = replace(cfg, label_filter="en")
        elif filter_value == "2023":
            cfg = replace(cfg, date_from="2023-01-01", date_to="2023-12-31")
        elif filter_value == "fake":
            cfg = replace(cfg, label_filter="fake")
        
        results, _ = hybrid_once(store, query, cfg, label_filter=None)
        results = apply_domain_cap(results, cfg.domain_cap)
        domains = [r.get('id', 'unknown') for r in results]
        domain_counts = Counter(domains)
        
        unique_domains = len(domain_counts)
        max_from_single_domain = max(domain_counts.values()) if domain_counts else 0
        total_results = len(results)
        
        print(f"  Results: {total_results}, Unique domains: {unique_domains}, Max from single domain: {max_from_single_domain}")
        print(f"  Domain distribution: {dict(domain_counts.most_common(3))}")
        print()
    
    print()


def main():
    """Main function to run domain diversity tests."""
    parser = argparse.ArgumentParser(description="Test Domain Diversity mechanism")
    parser.add_argument("--store", default="index_tmp/store", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence", help="Test query")
    parser.add_argument("--test", choices=["basic", "effectiveness", "vs_diversity", "sensitivity", "metadata", "all"], 
                       default="all", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output for basic test")
    args = parser.parse_args()
    
    print("Domain Diversity Mechanism Test")
    print("=" * 60)
    print(f"Store: {args.store}")
    print(f"Query: {args.query}")
    print(f"Test: {args.test}")
    print()
    
    # Load the store
    try:
        store = load_store(args.store)
        print(f"Loaded store with {len(store.chunks)} chunks")
        print()
    except Exception as e:
        print(f"Failed to load store: {e}")
        return 1
    
    # Run tests
    try:
        if args.test in ["basic", "all"]:
            test_domain_cap_basic(store, args.query, args.verbose)
        
        if args.test in ["effectiveness", "all"]:
            test_domain_cap_effectiveness(store, args.query)
        
        if args.test in ["vs_diversity", "all"]:
            test_domain_cap_vs_diversity(store, args.query)
        
        if args.test in ["sensitivity", "all"]:
            test_domain_cap_sensitivity(store, args.query)
        
        if args.test in ["metadata", "all"]:
            test_domain_cap_with_metadata(store, args.query)
        
        print("=" * 60)
        print("All domain diversity tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

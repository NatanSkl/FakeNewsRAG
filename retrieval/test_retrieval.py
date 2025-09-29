#!/usr/bin/env python3
"""
Test script for the retrieve_evidence function from retrieval.py

This script tests the main retrieval function with our new index to see how it performs.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the path to import from retrieval module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import load_store, retrieve_evidence, RetrievalConfig


def test_retrieve_evidence(store_path: str, query: str, verbose: bool = True):
    """Test the retrieve_evidence function with a given query"""
    print("=" * 60)
    print("TEST: retrieve_evidence function")
    print("=" * 60)
    
    # Load the store
    try:
        store = load_store(store_path)
        print(f"Loaded store from {store_path}")
        print(f"Store contains {len(store.chunks)} chunks")
        print(f"Embedding model: {store.meta.get('embedding_model', 'unknown')}")
        print()
    except Exception as e:
        print(f"Failed to load store: {e}")
        return
    
    # Test with default configuration
    print(f"Query: '{query}'")
    print("Using default RetrievalConfig...")
    
    try:
        # Use the retrieve_evidence function
        results = retrieve_evidence(store, query, None, label_name="fake", cfg=RetrievalConfig(), verbose=verbose)
        
        print(f"Retrieved {len(results)} results")
        print()
        
        if verbose and results:
            print("Top 5 results:")
            for i, result in enumerate(results[:5], 1):
                print(f"[{i}] Score: {result.get('rrf', 0.0):.4f}")
                print(f"     Title: {result.get('title', 'No title')[:80]}...")
                print(f"     Text: {result.get('chunk_text', '')[:150]}...")
                print(f"     Label: {result.get('label', 'unknown')}")
                print(f"     ID: {result.get('id', 'unknown')}")
                print()
        
        return results
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_different_configs(store_path: str, query: str):
    """Test retrieve_evidence with different configurations"""
    print("=" * 60)
    print("TEST: Different RetrievalConfigs")
    print("=" * 60)
    
    try:
        store = load_store(store_path)
        print(f"Query: '{query}'")
        print()
        
        # Test different configurations
        configs = [
            ("Default", RetrievalConfig()),
            ("High k_dense", RetrievalConfig(k_dense=100, k_bm25=100)),
            ("Low k_dense", RetrievalConfig(k_dense=10, k_bm25=10)),
            ("Dense only", RetrievalConfig(k_dense=50, k_bm25=0, w_dense=1.0, w_lex=0.0)),
            ("Sparse only", RetrievalConfig(k_dense=0, k_bm25=50, w_dense=0.0, w_lex=1.0)),
        ]
        
        for name, config in configs:
            print(f"--- {name} ---")
            try:
                # Create a custom retrieve_evidence function that accepts config
                from retrieval import hybrid_once, sentence_maxpool_boost, cross_encoder_rerank
                from sentence_transformers import CrossEncoder
                
                # Get query vector
                qv = store.emb.encode([query], normalize_embeddings=True).astype("float32")[0]
                
                # Get hybrid results
                hits, _ = hybrid_once(store, query, config, label_filter=None)
                
                # Apply sentence max-pooling if enabled
                if config.sent_maxpool:
                    hits = sentence_maxpool_boost(store, qv, hits, config)
                
                # Apply cross-encoder reranking if enabled
                if config.use_cross_encoder:
                    try:
                        ce = CrossEncoder(config.cross_encoder_model)
                        hits = cross_encoder_rerank(ce, query, hits, config)
                    except Exception as e:
                        print(f"Cross-encoder failed: {e}")
                
                print(f"Results: {len(hits)}")
                if hits:
                    avg_score = sum(h.get('rrf', 0.0) for h in hits) / len(hits)
                    print(f"Average score: {avg_score:.4f}")
                    print(f"Top result: {hits[0].get('chunk_text', '')[:100]}...")
                print()
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                print()
    
    except Exception as e:
        print(f"Error loading store: {e}")


def test_with_different_queries(store_path: str):
    """Test retrieve_evidence with different types of queries"""
    print("=" * 60)
    print("TEST: Different Query Types")
    print("=" * 60)
    
    try:
        store = load_store(store_path)
        
        test_queries = [
            "artificial intelligence machine learning",
            "climate change global warming",
            "fake news misinformation",
            "renewable energy solar power",
            "healthcare medical research"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            try:
                results = retrieve_evidence(store, query, None, label_name="fake", cfg=RetrievalConfig(), verbose=False)
                print(f"Results: {len(results)}")
                if results:
                    print(f"Top result: {results[0].get('chunk_text', '')[:100]}...")
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    except Exception as e:
        print(f"Error loading store: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test retrieve_evidence function")
    parser.add_argument("--store", default="../index/store_slice", help="Path to index store")
    parser.add_argument("--query", default="artificial intelligence machine learning", help="Test query")
    parser.add_argument("--test", choices=["basic", "configs", "queries", "all"], 
                       default="basic", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    
    args = parser.parse_args()
    
    print("Retrieve Evidence Test")
    print("=" * 60)
    print(f"Store: {args.store}")
    print(f"Query: {args.query}")
    print(f"Test: {args.test}")
    print()
    
    # Run tests
    if args.test in ["basic", "all"]:
        test_retrieve_evidence(args.store, args.query, args.verbose)
    
    if args.test in ["configs", "all"]:
        test_different_configs(args.store, args.query)
    
    if args.test in ["queries", "all"]:
        test_with_different_queries(args.store)
    
    print("=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()

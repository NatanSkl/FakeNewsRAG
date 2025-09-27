#!/usr/bin/env python3
"""
Simple test that bypasses the complex retrieve_evidence function
"""

import argparse
import sys
from pathlib import Path
sys.path.append('../..')
from retrieval import load_store, hybrid_once, RetrievalConfig

def test_simple_retrieval(store_dir: str):
    """Test simple hybrid retrieval"""
    
    print(f"Loading store from: {store_dir}")
    store = load_store(store_dir)
    print(f"✓ Store loaded successfully")
    
    test_query = "Donald Trump"
    print(f"\nTesting with query: '{test_query}'")
    
    # Test hybrid_once directly
    cfg = RetrievalConfig(
        k_dense=20,
        k_bm25=20,
        topn=10,
    )
    
    try:
        hits, qv = hybrid_once(store, test_query, cfg, label_filter="fake")
        print(f"✓ Found {len(hits)} fake hits")
        
        for i, hit in enumerate(hits[:5], 1):
            print(f"  {i}. {hit.get('source_domain', 'unknown')} | {hit.get('title', 'No title')[:50]}...")
            print(f"     Score: {hit.get('rrf', 0.0):.4f}")
            print(f"     Text: {hit.get('chunk_text', '')[:100]}...")
            print()
    
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test without label filter
    try:
        hits, qv = hybrid_once(store, test_query, cfg, label_filter=None)
        print(f"\n✓ Found {len(hits)} total hits (no label filter)")
        
        for i, hit in enumerate(hits[:5], 1):
            print(f"  {i}. Label: {hit.get('label', 'unknown')} | {hit.get('title', 'No title')[:50]}...")
            print(f"     Score: {hit.get('rrf', 0.0):.4f}")
            print(f"     Text: {hit.get('chunk_text', '')[:100]}...")
            print()
    
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test simple hybrid retrieval")
    ap.add_argument("--store", default="../store", help="Path to index store")
    
    args = ap.parse_args()
    test_simple_retrieval(args.store)

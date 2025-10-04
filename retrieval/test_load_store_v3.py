#!/usr/bin/env python3
"""
Test script for the load_store function in retrieval_v3.py
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from retrieval_v3
sys.path.append(str(Path(__file__).parent))

from retrieval_v3 import load_store


def test_load_store():
    """Test loading a store from /StudentData/slice"""
    try:
        print("Testing load_store function...")
        
        # Load the store
        store = load_store("/StudentData/slice_backup02_10", verbose=True)
        
        print(f"✓ Successfully loaded store")
        print(f"  - Embedding model: {store.emb}")
        print(f"  - Index type: {type(store.index)}")
        print(f"  - Index dimension: {store.index.d}")
        print(f"  - Number of vectors: {store.index.ntotal}")
        print(f"  - BM25 loaded: {store.bm25 is not None}")
        print(f"  - v2d mapping size: {len(store.v2d)}")
        print(f"  - Original data shape: {store.original.shape}")
        
        # Test v2d mapping
        if store.v2d:
            sample_vector_id = list(store.v2d.keys())[0]
            sample_db_id = store.v2d[sample_vector_id]
            print(f"  - Sample mapping: vector_id {sample_vector_id} -> db_id {sample_db_id}")
        
        # Test original data
        if not store.original.empty:
            print(f"  - Original data columns: {list(store.original.columns)}")
            print(f"  - Sample original row: {store.original.iloc[0].to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading store: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_load_store()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)

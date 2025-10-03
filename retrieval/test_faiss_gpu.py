"""
Test file for FAISS GPU retrieval function.

This module tests the faiss_gpu_once function to ensure it works correctly
with GPU-accelerated FAISS indices.
"""

import numpy as np
import faiss
import torch
import tempfile
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from retrieval import Store, RetrievalConfig, faiss_gpu_once, load_store, load_index_v3_to_gpu




def test_faiss_gpu_basic():
    """Test basic FAISS GPU retrieval functionality."""
    print("Testing basic FAISS GPU retrieval...")
    
    # Load store and move to GPU
    store = load_index_v3_to_gpu("/StudentData/data")
    
    # Create retrieval config
    cfg = RetrievalConfig(
        k_dense=10,
        mmr_k=5,
        mmr_lambda=0.5
    )
    
    # Test query
    query = "technology and innovation in healthcare"
    hits, qv = faiss_gpu_once(store, query, cfg)
    
    print(f"Query: {query}")
    print(f"Number of hits: {len(hits)}")
    print(f"Query vector shape: {qv.shape}")
    
    # Verify results
    assert len(hits) <= cfg.mmr_k, f"Expected at most {cfg.mmr_k} hits, got {len(hits)}"
    assert qv.shape[1] == store.meta["embedding_dim"], "Query vector dimension mismatch"
    
    # Print results
    for i, hit in enumerate(hits):
        print(f"Hit {i+1}: {hit['chunk_text'][:80]}... (label: {hit['label']})")
    
    print("✓ Basic FAISS GPU test passed!")
    return True


def test_faiss_gpu_with_label_filter():
    """Test FAISS GPU retrieval with label filtering."""
    print("\nTesting FAISS GPU retrieval with label filtering...")
    
    store = load_store_to_gpu("/StudentData/data")
    cfg = RetrievalConfig(k_dense=20, mmr_k=10, mmr_lambda=0.5)
    
    query = "news and information"
    
    # Test with REAL label filter
    hits_real, _ = faiss_gpu_once(store, query, cfg, label_filter="REAL")
    print(f"REAL label hits: {len(hits_real)}")
    for hit in hits_real:
        assert hit["label"] == "REAL", f"Expected REAL label, got {hit['label']}"
        print(f"  - {hit['chunk_text'][:80]}...")
    
    # Test with FAKE label filter
    hits_fake, _ = faiss_gpu_once(store, query, cfg, label_filter="FAKE")
    print(f"FAKE label hits: {len(hits_fake)}")
    for hit in hits_fake:
        assert hit["label"] == "FAKE", f"Expected FAKE label, got {hit['label']}"
        print(f"  - {hit['chunk_text'][:80]}...")
    
    print("✓ Label filtering test passed!")
    return True


def test_faiss_gpu_vs_cpu():
    """Compare FAISS GPU vs CPU results."""
    print("\nComparing FAISS GPU vs CPU results...")
    
    # Load store for CPU
    store_cpu = load_index_v3_to_gpu("/StudentData/data")
    # Force CPU by moving back to CPU
    if hasattr(store_cpu.index, 'is_cpu') and not store_cpu.index.is_cpu:
        store_cpu.index = faiss.index_gpu_to_cpu(store_cpu.index)
    
    # Load store and move to GPU
    store_gpu = load_index_v3_to_gpu("/StudentData/data")
    
    cfg = RetrievalConfig(k_dense=10, mmr_k=5, mmr_lambda=0.5)
    query = "artificial intelligence and technology"
    
    # Get results from both
    hits_gpu, qv_gpu = faiss_gpu_once(store_gpu, query, cfg)
    hits_cpu, qv_cpu = faiss_gpu_once(store_cpu, query, cfg)
    
    print(f"GPU hits: {len(hits_gpu)}")
    print(f"CPU hits: {len(hits_cpu)}")
    
    # Results should be similar (exact match depends on MMR randomness)
    assert len(hits_gpu) == len(hits_cpu), "Different number of hits between GPU and CPU"
    assert qv_gpu.shape == qv_cpu.shape, "Different query vector shapes"
    
    print("✓ GPU vs CPU comparison passed!")
    return True


def test_faiss_gpu_performance():
    """Test FAISS GPU performance with timing."""
    print("\nTesting FAISS GPU performance...")
    
    import time
    
    store = load_store_to_gpu("/StudentData/data")
    cfg = RetrievalConfig(k_dense=50, mmr_k=20, mmr_lambda=0.5)
    
    queries = [
        "technology and innovation",
        "climate change and environment", 
        "artificial intelligence healthcare",
        "fake news social media",
        "moon landing conspiracy",
        "politics and government",
        "science and research",
        "economy and business"
    ]
    
    # Time multiple queries
    start_time = time.time()
    for query in queries:
        hits, qv = faiss_gpu_once(store, query, cfg)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(queries)
    
    print(f"Total time for {len(queries)} queries: {total_time:.3f}s")
    print(f"Average time per query: {avg_time:.3f}s")
    
    # Performance should be reasonable (adjust threshold as needed)
    assert avg_time < 2.0, f"Average query time too slow: {avg_time:.3f}s"
    
    print("✓ Performance test passed!")
    return True


def test_faiss_gpu_edge_cases():
    """Test FAISS GPU with edge cases."""
    print("\nTesting FAISS GPU edge cases...")
    
    store = load_store_to_gpu("/StudentData/data")
    cfg = RetrievalConfig(k_dense=5, mmr_k=3, mmr_lambda=0.5)
    
    # Test empty query
    hits, qv = faiss_gpu_once(store, "", cfg)
    print(f"Empty query hits: {len(hits)}")
    
    # Test very specific query that might not match anything well
    hits, qv = faiss_gpu_once(store, "xyzabc123nonexistent", cfg)
    print(f"Non-matching query hits: {len(hits)}")
    
    # Test with very small k_dense
    cfg_small = RetrievalConfig(k_dense=1, mmr_k=1, mmr_lambda=0.5)
    hits, qv = faiss_gpu_once(store, "technology", cfg_small)
    print(f"Small k_dense hits: {len(hits)}")
    assert len(hits) <= 1, "Expected at most 1 hit with k_dense=1"
    
    print("✓ Edge cases test passed!")
    return True


def main():
    """Run all tests."""
    print("Starting FAISS GPU retrieval tests...")
    print("=" * 50)
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, tests will run on CPU")
        
        test_faiss_gpu_basic()
        test_faiss_gpu_with_label_filter()
        test_faiss_gpu_vs_cpu()
        test_faiss_gpu_performance()
        test_faiss_gpu_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 All FAISS GPU tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()

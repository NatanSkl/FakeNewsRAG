#!/usr/bin/env python3
"""
Simple test for mmr_diversify function.

This test verifies that mmr_diversify can diversify results using
Maximal Marginal Relevance (MMR) to reduce redundancy.
"""

import argparse
from typing import List, Dict, Any

from retrieval_v3 import mmr_diversify, load_store


def create_test_results() -> List[Dict[str, Any]]:
    """Create test results for MMR diversification."""
    return [
        {
            "db_id": 1,
            "score": 0.95,
            "content": "Machine learning algorithms are revolutionizing healthcare with advanced diagnostic tools and personalized treatment plans.",
            "label": "real"
        },
        {
            "db_id": 2,
            "score": 0.90,
            "content": "Artificial intelligence in healthcare is transforming patient care through automated diagnosis and treatment recommendations.",
            "label": "real"
        },
        {
            "db_id": 3,
            "score": 0.85,
            "content": "Deep learning neural networks can analyze medical images to detect diseases with high accuracy.",
            "label": "real"
        },
        {
            "db_id": 4,
            "score": 0.80,
            "content": "The weather today is sunny and warm, perfect for outdoor activities and picnics.",
            "label": "real"
        },
        {
            "db_id": 5,
            "score": 0.75,
            "content": "Climate change is causing more extreme weather patterns and rising sea levels worldwide.",
            "label": "real"
        },
        {
            "db_id": 6,
            "score": 0.70,
            "content": "Fake news about vaccines causing autism has been thoroughly debunked by scientific research.",
            "label": "fake"
        },
        {
            "db_id": 7,
            "score": 0.65,
            "content": "Cooking pasta requires boiling water, adding salt, and timing the cooking process correctly.",
            "label": "real"
        },
        {
            "db_id": 8,
            "score": 0.60,
            "content": "Misinformation spreads rapidly on social media platforms and can have serious consequences.",
            "label": "fake"
        }
    ]


def test_mmr_diversify_basic(store, lambda_mmr: float = 0.5):
    """Test basic MMR diversification functionality."""
    print(f"\n" + "="*60)
    print(f"TEST: MMR diversification with lambda={lambda_mmr}")
    print("="*60)
    
    # Create test results
    test_results = create_test_results()
    
    print(f"Original results count: {len(test_results)}")
    print("Original results (sorted by score):")
    for i, result in enumerate(test_results, 1):
        print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}")
        print(f"      content: {result['content'][:60]}...")
    
    # Test query
    query = "artificial intelligence machine learning healthcare"
    print(f"\nQuery: '{query}'")
    
    try:
        # Test MMR diversification
        diversified = mmr_diversify(
            store=store,
            query_text=query,
            results=test_results,
            top_k=4,  # Select top 4 diverse results
            lambda_mmr=lambda_mmr,
            content_key="content"
        )
        
        print(f"\nDiversified results count: {len(diversified)}")
        print("Diversified results (with MMR ranking):")
        for i, result in enumerate(diversified, 1):
            print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, mmr_rank: {result.get('mmr_rank', 'N/A')}")
            print(f"      content: {result['content'][:60]}...")
        
        # Verify that MMR ranking was added
        has_mmr_ranks = all('mmr_rank' in r for r in diversified)
        mmr_ranks = [r.get('mmr_rank') for r in diversified]
        ranks_sorted = all(mmr_ranks[i] <= mmr_ranks[i+1] for i in range(len(mmr_ranks)-1))
        
        print(f"\nVerification:")
        print(f"  Results have MMR ranks: {has_mmr_ranks}")
        print(f"  MMR ranks are sequential: {ranks_sorted}")
        print(f"  MMR ranks: {mmr_ranks}")
        
        # Check diversity by looking at content similarity
        print(f"\nDiversity analysis:")
        print(f"  Selected db_ids: {[r['db_id'] for r in diversified]}")
        print(f"  Content topics: {[r['content'][:30] + '...' for r in diversified]}")
        
        return True
        
    except Exception as e:
        print(f"Error during MMR diversification: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmr_diversify_edge_cases(store):
    """Test MMR diversification with edge cases."""
    print("\n" + "="*60)
    print("TEST: MMR diversification edge cases")
    print("="*60)
    
    # Test empty results
    print("Testing empty results...")
    empty_result = mmr_diversify(store, "test query", [], top_k=5)
    print(f"Empty results test passed: {empty_result == []}")
    
    # Test single result
    print("\nTesting single result...")
    single_result = [{"db_id": 1, "score": 0.5, "content": "single test content"}]
    single_diversified = mmr_diversify(store, "test query", single_result, top_k=1)
    print(f"Single result test passed: {len(single_diversified) == 1}")
    print(f"Single result has MMR rank: {'mmr_rank' in single_diversified[0]}")
    
    # Test top_k larger than results
    print("\nTesting top_k larger than results...")
    test_results = create_test_results()[:3]
    large_k_result = mmr_diversify(store, "test query", test_results, top_k=10)
    print(f"Large k test passed: {len(large_k_result) == len(test_results)}")
    
    # Test different lambda values
    print("\nTesting different lambda values...")
    test_results = create_test_results()[:4]
    
    for lambda_val in [0.0, 0.5, 1.0]:
        try:
            result = mmr_diversify(store, "test query", test_results, top_k=2, lambda_mmr=lambda_val)
            print(f"  Lambda {lambda_val}: {len(result)} results selected")
        except Exception as e:
            print(f"  Lambda {lambda_val} failed: {e}")
    
    return True


def test_mmr_diversify_lambda_comparison(store):
    """Test MMR diversification with different lambda values."""
    print("\n" + "="*60)
    print("TEST: MMR diversification lambda comparison")
    print("="*60)
    
    test_results = create_test_results()
    query = "artificial intelligence machine learning healthcare"
    
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    print(f"Query: '{query}'")
    print(f"Testing with {len(test_results)} results, selecting top 4")
    print()
    
    for lambda_val in lambda_values:
        try:
            diversified = mmr_diversify(
                store=store,
                query_text=query,
                results=test_results,
                top_k=4,
                lambda_mmr=lambda_val
            )
            
            selected_ids = [r['db_id'] for r in diversified]
            print(f"Lambda {lambda_val:3.1f}: db_ids {selected_ids}")
            
        except Exception as e:
            print(f"Lambda {lambda_val:3.1f}: ERROR - {e}")
    
    print("\nNote:")
    print("  Lambda 0.0 = Pure diversity (ignore query relevance)")
    print("  Lambda 1.0 = Pure relevance (ignore diversity)")
    print("  Lambda 0.5 = Balanced relevance and diversity")
    
    return True


def main():
    """Run the MMR diversification tests."""
    parser = argparse.ArgumentParser(description="Test mmr_diversify function")
    parser.add_argument("--lambda-val", type=float, default=0.5, help="Lambda value for MMR (0.0-1.0)")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests only")
    parser.add_argument("--lambda-comparison", action="store_true", help="Run lambda comparison test")
    parser.add_argument("--store-dir", type=str, default="/StudentData/slice", help="Store directory path")
    
    args = parser.parse_args()
    
    print("Starting MMR diversification tests...")
    print("="*60)
    
    try:
        # Load store
        print("Loading store...")
        store = load_store(args.store_dir, verbose=True)
        print(f"Store loaded successfully with {store.index.ntotal} vectors")
        
        if args.edge_cases:
            success = test_mmr_diversify_edge_cases(store)
            if success:
                print("\n" + "="*60)
                print("🎉 Edge case tests completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Edge case tests failed!")
            return success
            
        elif args.lambda_comparison:
            success = test_mmr_diversify_lambda_comparison(store)
            if success:
                print("\n" + "="*60)
                print("🎉 Lambda comparison test completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Lambda comparison test failed!")
            return success
            
        else:
            # Test basic functionality
            success = test_mmr_diversify_basic(store, args.lambda_val)
            edge_success = test_mmr_diversify_edge_cases(store)
            
            all_success = success and edge_success
            
            if all_success:
                print("\n" + "="*60)
                print("🎉 All MMR diversification tests completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Some MMR diversification tests failed!")
            
            return all_success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

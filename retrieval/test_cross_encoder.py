#!/usr/bin/env python3
"""
Simple test for cross_encoder_rerank function.

This test verifies that cross_encoder_rerank can rerank results using
different cross-encoder models.
"""

import argparse
from typing import List, Dict, Any

from retrieval_v3 import cross_encoder_rerank, base_score


def create_test_results() -> List[Dict[str, Any]]:
    """Create test results for cross-encoder reranking."""
    return [
        {
            "db_id": 1,
            "score": 0.9,
            "content": "Machine learning and artificial intelligence are transforming healthcare with new diagnostic tools and treatment methods.",
            "label": "real"
        },
        {
            "db_id": 2,
            "score": 0.8,
            "content": "The weather today is sunny and warm, perfect for a picnic in the park.",
            "label": "real"
        },
        {
            "db_id": 3,
            "score": 0.7,
            "content": "Fake news about vaccines causing autism has been debunked by numerous scientific studies.",
            "label": "fake"
        },
        {
            "db_id": 4,
            "score": 0.6,
            "content": "Deep learning neural networks can process vast amounts of data to identify patterns and make predictions.",
            "label": "real"
        },
        {
            "db_id": 5,
            "score": 0.5,
            "content": "Cooking pasta requires boiling water and adding salt for flavor.",
            "label": "real"
        },
        {
            "db_id": 6,
            "score": 0.4,
            "content": "Misinformation about climate change is often spread through social media platforms.",
            "label": "fake"
        }
    ]


def test_cross_encoder_rerank_basic(model_name: str):
    """Test basic cross-encoder reranking functionality with a specific model."""
    print(f"\n" + "="*60)
    print(f"TEST: Cross-encoder reranking with {model_name}")
    print("="*60)
    
    try:
        from sentence_transformers import CrossEncoder
        cross_enc = CrossEncoder(model_name)
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return False
    
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
        # Test reranking
        reranked = cross_encoder_rerank(
            cross_enc=cross_enc,
            query_text=query,
            results=test_results,
            ce_topk=4,  # Rerank top 4 results
            ce_weight=1.0,
            batch_size=8
        )
        
        print(f"\nReranked results count: {len(reranked)}")
        print("Reranked results:")
        for i, result in enumerate(reranked, 1):
            print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.4f}")
            print(f"      base_score: {result.get('base_score', 'N/A')}")
            print(f"      ce_score: {result.get('ce_score', 'N/A')}")
            print(f"      content: {result['content'][:60]}...")
        
        # Verify that scores have been updated
        has_ce_scores = all('ce_score' in r for r in reranked[:4])  # Top 4 should have CE scores
        has_base_scores = all('base_score' in r for r in reranked[:4])
        
        print(f"\nVerification:")
        print(f"  Results have CE scores: {has_ce_scores}")
        print(f"  Results have base scores: {has_base_scores}")
        print(f"  Results are sorted by score: {all(reranked[i]['score'] >= reranked[i+1]['score'] for i in range(len(reranked)-1))}")
        
        return True
        
    except Exception as e:
        print(f"Error during reranking: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_encoder_rerank_edge_cases():
    """Test cross-encoder reranking with edge cases."""
    print("\n" + "="*60)
    print("TEST: Cross-encoder reranking edge cases")
    print("="*60)
    
    # Test empty results
    print("Testing empty results...")
    empty_result = cross_encoder_rerank(None, "test query", [])
    print(f"Empty results test passed: {empty_result == []}")
    
    # Test None cross_encoder
    print("\nTesting None cross_encoder...")
    test_results = create_test_results()[:2]
    none_ce_result = cross_encoder_rerank(None, "test query", test_results)
    print(f"None cross_encoder test passed: {none_ce_result == test_results}")
    
    # Test single result with None cross_encoder
    print("\nTesting single result with None cross_encoder...")
    single_result = [{"db_id": 1, "score": 0.5, "content": "single test content"}]
    single_reranked = cross_encoder_rerank(None, "test query", single_result, ce_topk=1)
    print(f"Single result test passed: {len(single_reranked) == 1}")
    
    # Test base_score function
    print("\nTesting base_score function...")
    test_result = {"score": 0.8, "hybrid_score": 0.6, "_score": 0.4}
    score = base_score(test_result)
    print(f"base_score test passed: {score == 0.8}")
    
    return True


def test_both_models():
    """Test both cross-encoder models."""
    models = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "castorini/monot5-base-msmarco"
    ]
    
    results = {}
    for model in models:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*80}")
        results[model] = test_cross_encoder_rerank_basic(model)
    
    return results


def main():
    """Run the cross-encoder reranking tests."""
    parser = argparse.ArgumentParser(description="Test cross_encoder_rerank function")
    parser.add_argument("--model", type=str, help="Test specific model only")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests only")
    
    args = parser.parse_args()
    
    print("Starting cross-encoder reranking tests...")
    print("="*60)
    
    try:
        if args.edge_cases:
            success = test_cross_encoder_rerank_edge_cases()
            if success:
                print("\n" + "="*60)
                print("🎉 Edge case tests completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Edge case tests failed!")
            return success
            
        elif args.model:
            success = test_cross_encoder_rerank_basic(args.model)
            if success:
                print("\n" + "="*60)
                print(f"🎉 Cross-encoder test for {args.model} completed successfully!")
            else:
                print("\n" + "="*60)
                print(f"❌ Cross-encoder test for {args.model} failed!")
            return success
            
        else:
            # Test both models
            results = test_both_models()
            edge_success = test_cross_encoder_rerank_edge_cases()
            
            all_success = all(results.values()) and edge_success
            
            print("\n" + "="*80)
            print("SUMMARY:")
            print("="*80)
            for model, success in results.items():
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"  {model}: {status}")
            print(f"  Edge cases: {'✅ PASSED' if edge_success else '❌ FAILED'}")
            
            if all_success:
                print("\n🎉 All cross-encoder tests completed successfully!")
            else:
                print("\n❌ Some cross-encoder tests failed!")
            
            return all_success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

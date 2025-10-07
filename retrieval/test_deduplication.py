#!/usr/bin/env python3
"""
Simple test for deduplicate function.

This test verifies that deduplicate can remove duplicate db_id entries
from results, keeping only the one with the highest score.
"""

import argparse
from typing import List, Dict, Any

from retrieval_v3 import deduplicate


def create_test_results() -> List[Dict[str, Any]]:
    """Create test results with duplicates for testing deduplication."""
    return [
        {
            "db_id": 1,
            "score": 0.9,
            "content": "First result for db_id 1",
            "label": "fake"
        },
        {
            "db_id": 2,
            "score": 0.8,
            "content": "First result for db_id 2",
            "label": "real"
        },
        {
            "db_id": 1,  # Duplicate db_id
            "score": 0.7,  # Lower score than first
            "content": "Second result for db_id 1",
            "label": "fake"
        },
        {
            "db_id": 3,
            "score": 0.85,
            "content": "First result for db_id 3",
            "label": "real"
        },
        {
            "db_id": 2,  # Duplicate db_id
            "score": 0.95,  # Higher score than first
            "content": "Second result for db_id 2",
            "label": "real"
        },
        {
            "db_id": 1,  # Another duplicate db_id
            "score": 0.6,  # Lowest score
            "content": "Third result for db_id 1",
            "label": "fake"
        }
    ]


def test_deduplicate_basic():
    """Test basic deduplication functionality."""
    print("\n" + "="*60)
    print("TEST: Basic deduplication functionality")
    print("="*60)
    
    # Create test results with duplicates
    test_results = create_test_results()
    
    print(f"Original results count: {len(test_results)}")
    print("Original results:")
    for i, result in enumerate(test_results, 1):
        print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, content: {result['content']}")
    
    # Test deduplication
    try:
        deduplicated = deduplicate(test_results)
        
        print(f"\nDeduplicated results count: {len(deduplicated)}")
        print("Deduplicated results:")
        for i, result in enumerate(deduplicated, 1):
            print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, content: {result['content']}")
        
        # Verify deduplication worked correctly
        db_ids = [result['db_id'] for result in deduplicated]
        unique_db_ids = set(db_ids)
        
        print(f"\nVerification:")
        print(f"  Unique db_ids: {len(unique_db_ids)}")
        print(f"  Total results: {len(deduplicated)}")
        print(f"  All unique: {len(unique_db_ids) == len(deduplicated)}")
        
        # Check that highest scores were kept
        db_id_scores = {}
        for result in test_results:
            db_id = result['db_id']
            score = result['score']
            if db_id not in db_id_scores or score > db_id_scores[db_id]:
                db_id_scores[db_id] = score
        
        print(f"  Expected highest scores per db_id: {db_id_scores}")
        
        for result in deduplicated:
            db_id = result['db_id']
            score = result['score']
            expected_score = db_id_scores[db_id]
            print(f"    db_id {db_id}: got {score:.2f}, expected {expected_score:.2f}, match: {abs(score - expected_score) < 0.001}")
        
        # Check that results are sorted by score (highest first)
        scores = [result['score'] for result in deduplicated]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        print(f"  Results sorted by score (highest first): {is_sorted}")
        
        return True
        
    except Exception as e:
        print(f"Error during deduplication: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deduplicate_edge_cases():
    """Test deduplication with edge cases."""
    print("\n" + "="*60)
    print("TEST: Deduplication edge cases")
    print("="*60)
    
    # Test empty list
    print("Testing empty list...")
    empty_result = deduplicate([])
    print(f"Empty list result: {empty_result}")
    print(f"Empty list test passed: {empty_result == []}")
    
    # Test single result
    print("\nTesting single result...")
    single_result = [{"db_id": 1, "score": 0.5, "content": "single"}]
    deduplicated_single = deduplicate(single_result)
    print(f"Single result: {deduplicated_single}")
    print(f"Single result test passed: {deduplicated_single == single_result}")
    
    # Test no duplicates
    print("\nTesting no duplicates...")
    no_duplicates = [
        {"db_id": 1, "score": 0.9, "content": "first"},
        {"db_id": 2, "score": 0.8, "content": "second"},
        {"db_id": 3, "score": 0.7, "content": "third"}
    ]
    deduplicated_no_dup = deduplicate(no_duplicates)
    print(f"No duplicates result count: {len(deduplicated_no_dup)}")
    print(f"No duplicates test passed: {len(deduplicated_no_dup) == len(no_duplicates)}")
    
    # Test custom score key
    print("\nTesting custom score key...")
    custom_score_results = [
        {"db_id": 1, "custom_score": 0.9, "content": "first"},
        {"db_id": 1, "custom_score": 0.7, "content": "second"},
        {"db_id": 2, "custom_score": 0.8, "content": "third"}
    ]
    deduplicated_custom = deduplicate(custom_score_results, score_key="custom_score")
    print(f"Custom score result count: {len(deduplicated_custom)}")
    print(f"Custom score results: {deduplicated_custom}")
    print(f"Custom score test passed: {len(deduplicated_custom) == 2}")
    
    return True


def main():
    """Run the deduplication tests."""
    parser = argparse.ArgumentParser(description="Test deduplicate function")
    
    print("Starting deduplication tests...")
    print("="*60)
    
    try:
        success1 = test_deduplicate_basic()
        success2 = test_deduplicate_edge_cases()
        
        if success1 and success2:
            print("\n" + "="*60)
            print("🎉 All deduplication tests completed successfully!")
            return True
        else:
            print("\n" + "="*60)
            print("❌ Some deduplication tests failed!")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()



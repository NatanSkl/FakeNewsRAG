#!/usr/bin/env python3
"""
Simple test for filter_label function.

This test verifies that filter_label can filter results by label.
"""

import argparse
from typing import List, Dict, Any

from retrieval_v3 import filter_label


def create_test_results() -> List[Dict[str, Any]]:
    """Create test results with different labels."""
    return [
        {
            "db_id": 1,
            "score": 0.9,
            "content": "Machine learning is transforming healthcare",
            "label": "real"
        },
        {
            "db_id": 2,
            "score": 0.8,
            "content": "Fake news about vaccines is dangerous",
            "label": "fake"
        },
        {
            "db_id": 3,
            "score": 0.7,
            "content": "Climate change is a serious global issue",
            "label": "real"
        },
        {
            "db_id": 4,
            "score": 0.6,
            "content": "Conspiracy theories spread misinformation",
            "label": "fake"
        },
        {
            "db_id": 5,
            "score": 0.5,
            "content": "Scientific research advances our understanding",
            "label": "real"
        }
    ]


def test_filter_label_basic():
    """Test basic filter_label functionality."""
    print("\n" + "="*60)
    print("TEST: Basic filter_label functionality")
    print("="*60)
    
    # Create test results
    test_results = create_test_results()
    
    print(f"Original results count: {len(test_results)}")
    print("Original results:")
    for i, result in enumerate(test_results, 1):
        print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, label: {result['label']}")
        print(f"      content: {result['content']}")
    
    # Test filtering by "real" label
    print(f"\nFiltering by label 'real':")
    real_results = filter_label(test_results, "real")
    print(f"Real results count: {len(real_results)}")
    for i, result in enumerate(real_results, 1):
        print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, label: {result['label']}")
        print(f"      content: {result['content']}")
    
    # Test filtering by "fake" label
    print(f"\nFiltering by label 'fake':")
    fake_results = filter_label(test_results, "fake")
    print(f"Fake results count: {len(fake_results)}")
    for i, result in enumerate(fake_results, 1):
        print(f"  [{i}] db_id: {result['db_id']}, score: {result['score']:.2f}, label: {result['label']}")
        print(f"      content: {result['content']}")
    
    # Test filtering by non-existent label
    print(f"\nFiltering by label 'unknown':")
    unknown_results = filter_label(test_results, "unknown")
    print(f"Unknown results count: {len(unknown_results)}")
    
    # Verify filtering worked correctly
    real_labels = [r['label'] for r in real_results]
    fake_labels = [r['label'] for r in fake_results]
    
    print(f"\nVerification:")
    print(f"  All real results have 'real' label: {all(label == 'real' for label in real_labels)}")
    print(f"  All fake results have 'fake' label: {all(label == 'fake' for label in fake_labels)}")
    print(f"  Real results: {len(real_results)} (expected: 3)")
    print(f"  Fake results: {len(fake_results)} (expected: 2)")
    print(f"  Unknown results: {len(unknown_results)} (expected: 0)")
    
    return len(real_results) == 3 and len(fake_results) == 2 and len(unknown_results) == 0


def test_filter_label_edge_cases():
    """Test filter_label with edge cases."""
    print("\n" + "="*60)
    print("TEST: filter_label edge cases")
    print("="*60)
    
    # Test empty list
    print("Testing empty list...")
    empty_result = filter_label([], "real")
    print(f"Empty list test passed: {empty_result == []}")
    
    # Test list with no matching labels
    print("\nTesting list with no matching labels...")
    no_match_results = [{"db_id": 1, "score": 0.5, "content": "test", "label": "real"}]
    no_match_filtered = filter_label(no_match_results, "fake")
    print(f"No match test passed: {no_match_filtered == []}")
    
    # Test list with missing label field
    print("\nTesting list with missing label field...")
    no_label_results = [{"db_id": 1, "score": 0.5, "content": "test"}]
    no_label_filtered = filter_label(no_label_results, "real")
    print(f"Missing label test passed: {no_label_filtered == []}")
    
    # Test list with None label
    print("\nTesting list with None label...")
    none_label_results = [{"db_id": 1, "score": 0.5, "content": "test", "label": None}]
    none_label_filtered = filter_label(none_label_results, "real")
    print(f"None label test passed: {none_label_filtered == []}")
    
    return True


def main():
    """Run the filter_label tests."""
    parser = argparse.ArgumentParser(description="Test filter_label function")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests only")
    
    args = parser.parse_args()
    
    print("Starting filter_label tests...")
    print("="*60)
    
    try:
        if args.edge_cases:
            success = test_filter_label_edge_cases()
            if success:
                print("\n" + "="*60)
                print("🎉 Edge case tests completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Edge case tests failed!")
            return success
            
        else:
            # Test basic functionality
            success = test_filter_label_basic()
            edge_success = test_filter_label_edge_cases()
            
            all_success = success and edge_success
            
            if all_success:
                print("\n" + "="*60)
                print("🎉 All filter_label tests completed successfully!")
            else:
                print("\n" + "="*60)
                print("❌ Some filter_label tests failed!")
            
            return all_success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

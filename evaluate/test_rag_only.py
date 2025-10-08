"""
Test RAG Retrieval System Only
Tests the retrieval component of our RAG without Llama to avoid crashes
Professional output without emojis
"""

import sys
import os
import time
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def test_rag_retrieval():
    """Test RAG retrieval component."""
    print("="*80)
    print("TESTING RAG RETRIEVAL SYSTEM")
    print("="*80)
    
    try:
        from retrieval import load_store, retrieve_evidence, RetrievalConfig
        
        # Try mini_index first since it was visible in project layout
        store_paths = [
            "mini_index/store",
            "/StudentData/slice_backup02_10",
            "index/store",
            "improved_index"
        ]
        
        store = None
        store_path_used = None
        
        for path in store_paths:
            if Path(path).exists():
                print(f"\nTrying to load store from: {path}")
                try:
                    store = load_store(path, verbose=False)
                    store_path_used = path
                    print(f"SUCCESS: Store loaded from {path}")
                    print(f"Store contains {store.index.ntotal} vectors")
                    break
                except Exception as e:
                    print(f"FAILED: Could not load from {path}: {e}")
                    continue
        
        if store is None:
            print("\nERROR: No valid store found in any location")
            print("\nSearched locations:")
            for path in store_paths:
                exists = "EXISTS" if Path(path).exists() else "NOT FOUND"
                print(f"  {path}: {exists}")
            return False
        
        print("\n" + "="*80)
        print("TESTING EVIDENCE RETRIEVAL")
        print("="*80)
        
        # Test articles
        test_cases = [
            {
                'title': 'Bitcoin cryptocurrency manipulation',
                'content': 'Government agencies manipulating Bitcoin markets secretly',
                'expected': 'Should find fake news evidence'
            },
            {
                'title': 'Medical research heart disease',
                'content': 'Peer-reviewed scientific study on cardiovascular health',
                'expected': 'Should find reliable news evidence'
            },
            {
                'title': 'Climate change global warming',
                'content': 'Scientific research on rising temperatures and climate impacts',
                'expected': 'Should find reliable news evidence'
            }
        ]
        
        config = RetrievalConfig(k=5, ce_model=None, diversity_type=None, verbose=False)
        
        total_tests = 0
        successful_retrieval = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {test['title']}")
            print(f"Expected: {test['expected']}")
            print("-"*80)
            
            start_time = time.time()
            
            try:
                # Retrieve fake evidence
                fake_hits = retrieve_evidence(
                    store=store,
                    article_text=test['content'],
                    title_hint=test['title'],
                    label_name="fake",
                    cfg=config
                )
                
                # Retrieve reliable evidence  
                reliable_hits = retrieve_evidence(
                    store=store,
                    article_text=test['content'],
                    title_hint=test['title'],
                    label_name="reliable",
                    cfg=config
                )
                
                elapsed = time.time() - start_time
                
                print(f"Retrieval completed in {elapsed:.2f}s")
                print(f"  Fake evidence chunks: {len(fake_hits)}")
                print(f"  Reliable evidence chunks: {len(reliable_hits)}")
                
                # Show top results
                if fake_hits:
                    print(f"\n  Top fake evidence:")
                    for j, hit in enumerate(fake_hits[:3], 1):
                        score = hit.get('score', 0.0)
                        text = hit.get('chunk_text', hit.get('content', 'No text'))[:100]
                        print(f"    {j}. Score: {score:.3f}")
                        print(f"       {text}...")
                
                if reliable_hits:
                    print(f"\n  Top reliable evidence:")
                    for j, hit in enumerate(reliable_hits[:3], 1):
                        score = hit.get('score', 0.0)
                        text = hit.get('chunk_text', hit.get('content', 'No text'))[:100]
                        print(f"    {j}. Score: {score:.3f}")
                        print(f"       {text}...")
                
                total_tests += 1
                if len(fake_hits) > 0 or len(reliable_hits) > 0:
                    successful_retrieval += 1
                    print(f"\n  Status: SUCCESS - Retrieved evidence")
                else:
                    print(f"\n  Status: WARNING - No evidence retrieved")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("RETRIEVAL TEST SUMMARY")
        print("="*80)
        print(f"Store used: {store_path_used}")
        print(f"Total vectors in store: {store.index.ntotal}")
        print(f"Tests run: {total_tests}")
        print(f"Successful retrievals: {successful_retrieval}/{total_tests}")
        print(f"Success rate: {successful_retrieval/total_tests*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_rag_components():
    """Check what RAG components are available."""
    print("\n" + "="*80)
    print("CHECKING RAG COMPONENTS")
    print("="*80)
    
    components = {
        'retrieval_module': False,
        'rag_pipeline': False,
        'embedding_model': False,
        'store': False
    }
    
    # Check retrieval module
    try:
        from retrieval import load_store, retrieve_evidence
        components['retrieval_module'] = True
        print("OK: retrieval module available")
    except Exception as e:
        print(f"MISSING: retrieval module - {e}")
    
    # Check RAG pipeline
    try:
        from pipeline.rag_pipeline import classify_article_rag
        components['rag_pipeline'] = True
        print("OK: RAG pipeline available")
    except Exception as e:
        print(f"INFO: RAG pipeline - {e}")
    
    # Check for store
    store_paths = ["mini_index/store", "index/store"]
    for path in store_paths:
        if Path(path).exists():
            components['store'] = True
            print(f"OK: Store found at {path}")
            
            # List files in store
            store_files = list(Path(path).glob("*"))
            print(f"    Store contains {len(store_files)} files:")
            for f in store_files:
                size = f.stat().st_size / (1024**2)
                print(f"      - {f.name} ({size:.2f} MB)")
            break
    
    if not components['store']:
        print("MISSING: No store found")
    
    return components


def main():
    """Main test function."""
    print("RAG SYSTEM COMPONENT TEST")
    print("Professional evaluation without emojis\n")
    
    # Check components
    components = check_rag_components()
    
    # Test retrieval if available
    if components['retrieval_module'] and components['store']:
        test_rag_retrieval()
    else:
        print("\nCannot test RAG - missing required components")
        if not components['retrieval_module']:
            print("  Missing: retrieval module")
        if not components['store']:
            print("  Missing: store/index")
            print("\nTo build index:")
            print("  cd index")
            print("  python build_index_v3.py --input /path/to/data --output store")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

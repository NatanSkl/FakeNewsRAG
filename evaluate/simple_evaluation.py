"""
Simple Evaluation Script
Tests what we have available and runs evaluation
No emojis, professional output
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def check_requirements():
    """Check what's available for evaluation."""
    print("="*80)
    print("CHECKING SYSTEM REQUIREMENTS")
    print("="*80)
    
    checks = {
        'llama_cpp': False,
        'llama_model': False,
        'store': False,
        'test_data': False
    }
    
    # Check llama-cpp-python
    try:
        import llama_cpp
        checks['llama_cpp'] = True
        print("OK: llama-cpp-python installed")
    except:
        print("MISSING: llama-cpp-python")
    
    # Check Llama model
    model_path = Path("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    if model_path.exists():
        checks['llama_model'] = True
        print(f"OK: Llama model found ({model_path.stat().st_size / (1024**3):.2f} GB)")
    else:
        print("MISSING: Llama model")
    
    # Check for store/index
    store_paths = [
        "/StudentData/slice_backup02_10",
        "index/store",
        "mini_index/store",
        "improved_index"
    ]
    
    for path in store_paths:
        if Path(path).exists():
            checks['store'] = True
            print(f"OK: Store found at {path}")
            break
    
    if not checks['store']:
        print("MISSING: No store/index found")
    
    # Check for test data
    if Path("data/test.csv").exists():
        checks['test_data'] = True
        print("OK: Test data found")
    else:
        print("INFO: No test.csv, will use store data")
    
    print("\n" + "="*80)
    print("SYSTEM STATUS")
    print("="*80)
    
    all_ready = all(checks.values())
    
    if checks['llama_cpp'] and checks['llama_model']:
        print("STATUS: Can run Llama baseline tests")
    
    if checks['store']:
        print("STATUS: Can run RAG retrieval tests")
    
    if all_ready:
        print("STATUS: READY FOR FULL EVALUATION")
    else:
        print("STATUS: PARTIAL EVALUATION POSSIBLE")
    
    return checks


def run_available_tests(checks: dict):
    """Run whatever tests are possible."""
    print("\n" + "="*80)
    print("RUNNING AVAILABLE TESTS")
    print("="*80)
    
    if checks['llama_cpp'] and checks['llama_model']:
        print("\nTest 1: Llama Classification")
        print("-"*80)
        run_llama_test()
    
    if checks['store']:
        print("\nTest 2: RAG Retrieval")
        print("-"*80)
        run_rag_retrieval_test()


def run_llama_test():
    """Test Llama classification."""
    try:
        from llama_cpp import Llama
        
        print("Loading Llama model...")
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False
        )
        
        test_articles = [
            ("Bitcoin manipulation conspiracy theory", "fake"),
            ("Scientific peer-reviewed research study", "reliable")
        ]
        
        correct = 0
        total = 0
        
        for content, true_label in test_articles:
            prompt = f"Classify as FAKE or RELIABLE: {content}\n\nClassification:"
            
            response = llm(prompt, max_tokens=10, temperature=0.1, stop=["\n"])
            response_text = response['choices'][0]['text'].strip().upper()
            
            if "FAKE" in response_text:
                pred = "fake"
            else:
                pred = "reliable"
            
            is_correct = pred == true_label
            correct += is_correct
            total += 1
            
            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"  Article: {content[:50]}...")
            print(f"  Predicted: {pred}, True: {true_label} - {status}")
        
        accuracy = correct / total
        print(f"\nLlama Baseline Accuracy: {accuracy:.1%} ({correct}/{total})")
        
        return True
        
    except Exception as e:
        print(f"Error in Llama test: {e}")
        return False


def run_rag_retrieval_test():
    """Test RAG retrieval."""
    try:
        from retrieval import load_store, retrieve_evidence, RetrievalConfig
        
        # Try to find store
        store_paths = [
            "/StudentData/slice_backup02_10",
            "index/store",
            "mini_index/store"
        ]
        
        store = None
        for path in store_paths:
            if Path(path).exists():
                print(f"Loading store from: {path}")
                try:
                    store = load_store(path, verbose=False)
                    print(f"Store loaded: {store.index.ntotal} vectors")
                    break
                except:
                    continue
        
        if store is None:
            print("No valid store found for RAG testing")
            return False
        
        # Test retrieval
        test_query = "Bitcoin cryptocurrency manipulation conspiracy"
        
        config = RetrievalConfig(k=5, ce_model=None, diversity_type=None, verbose=False)
        
        fake_hits = retrieve_evidence(
            store,
            test_query,
            "fake",
            None,
            None,
            config.k,
            False
        )
        
        reliable_hits = retrieve_evidence(
            store,
            test_query,
            "reliable",
            None,
            None,
            config.k,
            False
        )
        
        print(f"  Query: {test_query}")
        print(f"  Fake evidence: {len(fake_hits)} chunks")
        print(f"  Reliable evidence: {len(reliable_hits)} chunks")
        
        if fake_hits:
            print(f"  Sample fake: {fake_hits[0].get('content', '')[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"Error in RAG test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("FAKE NEWS RAG EVALUATION SYSTEM")
    print("Professional evaluation without emojis\n")
    
    # Check what's available
    checks = check_requirements()
    
    # Run available tests
    run_available_tests(checks)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if not all(checks.values()):
        print("\nTo run full evaluation:")
        if not checks['llama_model']:
            print("1. Download Llama: python evaluate/download_llama_model.py")
        if not checks['store']:
            print("2. Build index: cd index && python build_index_v3.py")
        print("3. Run: python evaluate/run_real_evaluation.py")


if __name__ == "__main__":
    main()

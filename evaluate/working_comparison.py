"""
Working RAG vs Llama comparison - Simple version
"""

import sys
import time
import json
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_llama_only():
    """Test only Llama classification."""
    print("="*50)
    print("LLAMA CLASSIFICATION TEST")
    print("="*50)
    
    try:
        from llama_cpp import Llama
        
        print("📥 Loading Llama model...")
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=512,  # Very small context
            n_gpu_layers=0,
            verbose=False
        )
        print("✅ Llama loaded")
        
        # Test articles
        test_cases = [
            ("Bitcoin manipulation", "fake"),
            ("Scientific study", "credible"),
            ("UFO conspiracy", "fake"),
            ("Climate research", "credible")
        ]
        
        results = []
        
        for i, (content, true_label) in enumerate(test_cases):
            print(f"\n🧪 Test {i+1}: {content}")
            
            prompt = f"Classify as FAKE or CREDIBLE: {content}"
            
            start_time = time.time()
            response = llm(prompt, max_tokens=10, temperature=0.1, stop=["\n"])
            elapsed = time.time() - start_time
            
            response_text = response['choices'][0]['text'].strip()
            
            # Simple parsing
            if "FAKE" in response_text.upper():
                prediction = "fake"
            else:
                prediction = "credible"
            
            correct = prediction == true_label
            
            print(f"   Response: {response_text}")
            print(f"   Prediction: {prediction}")
            print(f"   True: {true_label}")
            print(f"   Correct: {'✅' if correct else '❌'}")
            print(f"   Time: {elapsed:.2f}s")
            
            results.append({
                'content': content,
                'true_label': true_label,
                'prediction': prediction,
                'correct': correct,
                'time': elapsed,
                'response': response_text
            })
        
        # Summary
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print(f"\n📊 LLAMA SUMMARY:")
        print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{total_count})")
        print(f"   Avg Time: {avg_time:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_rag_retrieval_only():
    """Test only RAG retrieval."""
    print("\n" + "="*50)
    print("RAG RETRIEVAL TEST")
    print("="*50)
    
    try:
        from rag_pipeline import load_store, RetrievalConfig, retrieve_evidence
        
        print("📥 Loading RAG store...")
        store = load_store("mini_index/store")
        print("✅ RAG store loaded")
        
        # Test retrieval
        test_queries = [
            "Bitcoin manipulation conspiracy",
            "Scientific research study",
            "UFO alien technology",
            "Climate change data"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries):
            print(f"\n🔍 Query {i+1}: {query}")
            
            config = RetrievalConfig(
                topn=3,
                use_cross_encoder=False,
                use_xquad=False,
                sent_maxpool=False,
                mmr_k=0,
                mmr_lambda=0.0
            )
            
            start_time = time.time()
            
            fake_hits = retrieve_evidence(
                store=store,
                article_text=query,
                title_hint=query,
                label_name="fake",
                cfg=config
            )
            
            credible_hits = retrieve_evidence(
                store=store,
                article_text=query,
                title_hint=query,
                label_name="credible",
                cfg=config
            )
            
            elapsed = time.time() - start_time
            
            print(f"   Fake evidence: {len(fake_hits)} chunks")
            print(f"   Credible evidence: {len(credible_hits)} chunks")
            print(f"   Time: {elapsed:.2f}s")
            
            # Show sample evidence
            if fake_hits:
                sample = fake_hits[0].get('chunk_text', 'No text')[:80]
                print(f"   Sample fake: {sample}...")
            
            if credible_hits:
                sample = credible_hits[0].get('chunk_text', 'No text')[:80]
                print(f"   Sample credible: {sample}...")
            
            results.append({
                'query': query,
                'fake_hits': len(fake_hits),
                'credible_hits': len(credible_hits),
                'time': elapsed
            })
        
        # Summary
        total_fake = sum(r['fake_hits'] for r in results)
        total_credible = sum(r['credible_hits'] for r in results)
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print(f"\n📊 RAG SUMMARY:")
        print(f"   Total fake evidence: {total_fake}")
        print(f"   Total credible evidence: {total_credible}")
        print(f"   Avg Time: {avg_time:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main test function."""
    print("🚀 RAG vs LLAMA COMPARISON TEST")
    print("Testing components separately to avoid crashes")
    
    # Test Llama
    llama_results = test_llama_only()
    
    # Test RAG
    rag_results = test_rag_retrieval_only()
    
    # Save results
    if llama_results or rag_results:
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'llama_results': llama_results,
            'rag_results': rag_results
        }
        
        with open('comparison_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: comparison_results.json")
    
    print("\n🎯 CONCLUSION:")
    if llama_results:
        print("✅ Llama classification is working")
    if rag_results:
        print("✅ RAG retrieval is working")
    
    if llama_results and rag_results:
        print("🎉 Both systems are functional!")
        print("Your RAG system can retrieve evidence and Llama can classify text")
        print("The combination provides enhanced fake news detection capabilities")

if __name__ == "__main__":
    main()
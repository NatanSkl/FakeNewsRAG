"""
Simple RAG vs Llama comparison test
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_llama_simple():
    """Test Llama with simple classification."""
    print("="*50)
    print("SIMPLE LLAMA TEST")
    print("="*50)
    
    try:
        from llama_cpp import Llama
        
        print("📥 Loading Llama model...")
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=1024,  # Smaller context
            n_gpu_layers=0,
            verbose=False
        )
        print("✅ Llama loaded")
        
        # Simple test
        test_prompt = "Classify this as FAKE or CREDIBLE: Bitcoin prices are skyrocketing due to secret government manipulation!"
        
        print("🧪 Testing classification...")
        start_time = time.time()
        
        response = llm(
            test_prompt,
            max_tokens=50,
            temperature=0.1,
            stop=["\n\n"]
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ Response: {response['choices'][0]['text']}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_rag_simple():
    """Test RAG with simple retrieval."""
    print("\n" + "="*50)
    print("SIMPLE RAG TEST")
    print("="*50)
    
    try:
        from rag_pipeline import load_store, RetrievalConfig, retrieve_evidence
        
        print("📥 Loading RAG store...")
        store = load_store("mini_index/store")
        print("✅ RAG store loaded")
        
        # Simple retrieval test
        test_text = "Bitcoin prices are skyrocketing due to secret government manipulation!"
        
        print("🧪 Testing retrieval...")
        start_time = time.time()
        
        config = RetrievalConfig(
            topn=3,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=0,
            mmr_lambda=0.0
        )
        
        fake_hits = retrieve_evidence(
            store=store,
            article_text=test_text,
            title_hint="Bitcoin manipulation",
            label_name="fake",
            cfg=config
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ Retrieved {len(fake_hits)} fake evidence chunks")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        if fake_hits:
            print("📄 Sample evidence:")
            for i, hit in enumerate(fake_hits[:2]):
                chunk_text = hit.get('chunk_text', 'No text')[:100]
                print(f"   {i+1}. {chunk_text}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Starting Simple RAG vs Llama Test")
    
    # Test Llama
    llama_success = test_llama_simple()
    
    # Test RAG
    rag_success = test_rag_simple()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if llama_success:
        print("✅ Llama: WORKING")
    else:
        print("❌ Llama: FAILED")
    
    if rag_success:
        print("✅ RAG: WORKING")
    else:
        print("❌ RAG: FAILED")
    
    if llama_success and rag_success:
        print("\n🎉 Both systems are working!")
        print("You can now run full comparison tests")
    else:
        print("\n⚠️  Some systems failed")
        print("Check the errors above")

if __name__ == "__main__":
    main()
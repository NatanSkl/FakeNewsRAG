"""
Test script to verify our RAG retrieval components work
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import load_store, RetrievalConfig, retrieve_evidence

def test_rag_retrieval():
    """Test our RAG retrieval components."""
    
    print("="*60)
    print("TESTING OUR RAG RETRIEVAL COMPONENTS")
    print("="*60)
    
    # Load store
    print("Loading store...")
    store = load_store("mini_index/store")
    print(f"✅ Store loaded: {len(store.chunks)} chunks")
    print(f"  - Embedding model: {store.meta.get('embedding_model', 'Unknown')}")
    print(f"  - Index type: {type(store.index).__name__}")
    
    # Show label distribution
    label_counts = {}
    for chunk in store.chunks:
        label = chunk.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  - Label distribution: {label_counts}")
    
    # Test article (fake news)
    test_article = {
        'title': "Bitcoin & Blockchain Searches Exceed Trump! Blockchain Stocks Are Next!",
        'content': """Scientists have discovered fossil remains of a new carnivorous mammal in Turkey, 
        one of the biggest marsupial relatives ever discovered in the northern hemisphere. 
        The findings, by Dr Robin Beck from the University of Salford in the UK and Dr Murat Maga, 
        of the University of Washington who discovered the fossil, are published today in the journal PLoS ONE. 
        The new fossil is a 43 million year old cat-sized mammal that had powerful teeth and jaws for crushing hard food, 
        like the modern Tasmanian Devil. It is related to the pouched mammals, or marsupials, of Australia and South America, 
        and it shows that marsupial relatives, or metatherians, were far more diverse in the northern hemisphere than previously believed."""
    }
    
    print(f"\nTesting article: {test_article['title'][:50]}...")
    
    # Test different retrieval configurations
    configs = {
        'basic': RetrievalConfig(
            topn=5,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=0,
            mmr_lambda=0.0
        ),
        'with_mmr': RetrievalConfig(
            topn=5,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=20,
            mmr_lambda=0.4
        ),
        'with_reranking': RetrievalConfig(
            topn=5,
            use_cross_encoder=True,
            use_xquad=False,
            sent_maxpool=True,
            mmr_k=20,
            mmr_lambda=0.4
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*20} Testing {config_name.upper()} {'='*20}")
        
        try:
            start_time = time.time()
            
            # Test fake evidence retrieval
            fake_hits = retrieve_evidence(
                store=store,
                article_text=test_article['content'],
                title_hint=test_article['title'],
                label_name="fake",
                cfg=config
            )
            
            # Test credible evidence retrieval
            credible_hits = retrieve_evidence(
                store=store,
                article_text=test_article['content'],
                title_hint=test_article['title'],
                label_name="credible",
                cfg=config
            )
            
            processing_time = time.time() - start_time
            
            print(f"✅ {config_name} retrieval completed!")
            print(f"  - Fake evidence: {len(fake_hits)} chunks")
            print(f"  - Credible evidence: {len(credible_hits)} chunks")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            # Show top fake evidence
            if fake_hits:
                print(f"\n  Top fake evidence:")
                for i, hit in enumerate(fake_hits[:3]):
                    score = hit.get('score', 0.0)
                    chunk_text = hit.get('chunk_text', 'No text')[:80]
                    print(f"    {i+1}. Score: {score:.3f} - {chunk_text}...")
            
            # Show top credible evidence
            if credible_hits:
                print(f"\n  Top credible evidence:")
                for i, hit in enumerate(credible_hits[:3]):
                    score = hit.get('score', 0.0)
                    chunk_text = hit.get('chunk_text', 'No text')[:80]
                    print(f"    {i+1}. Score: {score:.3f} - {chunk_text}...")
            
            results[config_name] = {
                'fake_hits': len(fake_hits),
                'credible_hits': len(credible_hits),
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error in {config_name} retrieval: {e}")
            import traceback
            traceback.print_exc()
            results[config_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*60)
    print("RETRIEVAL TEST SUMMARY")
    print("="*60)
    
    for config_name, result in results.items():
        if result['success']:
            print(f"✅ {config_name}: {result['fake_hits']} fake + {result['credible_hits']} credible hits ({result['processing_time']:.2f}s)")
        else:
            print(f"❌ {config_name}: FAILED - {result['error']}")
    
    # Check if our RAG retrieval is working
    working_configs = [name for name, result in results.items() if result['success']]
    
    if working_configs:
        print(f"\n🎉 RAG retrieval is working! Working configurations: {', '.join(working_configs)}")
        print("✅ Our RAG pipeline retrieval components are functional")
        
        if len(working_configs) > 1:
            print("✅ Multiple retrieval configurations are working")
        
        return True
    else:
        print("\n❌ RAG retrieval is not working")
        print("❌ Our RAG pipeline has issues")
        return False

def test_embedding_quality():
    """Test embedding quality and similarity."""
    
    print("\n" + "="*60)
    print("TESTING EMBEDDING QUALITY")
    print("="*60)
    
    try:
        store = load_store("mini_index/store")
        
        # Get some fake and credible chunks
        fake_chunks = [chunk for chunk in store.chunks if chunk.get("label") == "fake"][:5]
        credible_chunks = [chunk for chunk in store.chunks if chunk.get("label") == "credible"][:5]
        
        print(f"Testing with {len(fake_chunks)} fake chunks and {len(credible_chunks)} credible chunks")
        
        # Test embedding encoding
        test_text = "This is a test article about fake news detection"
        
        start_time = time.time()
        embedding = store.emb.encode([test_text], normalize_embeddings=True)
        encoding_time = time.time() - start_time
        
        print(f"✅ Embedding encoding works!")
        print(f"  - Embedding dimension: {embedding.shape[1]}")
        print(f"  - Encoding time: {encoding_time:.3f}s")
        
        # Test similarity search
        if fake_chunks:
            fake_text = fake_chunks[0].get("chunk_text", "")
            if fake_text:
                start_time = time.time()
                fake_embedding = store.emb.encode([fake_text], normalize_embeddings=True)
                similarity = store.index.search(fake_embedding, k=5)
                search_time = time.time() - start_time
                
                print(f"✅ Similarity search works!")
                print(f"  - Search time: {search_time:.3f}s")
                print(f"  - Top similarity scores: {similarity[0][0][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in embedding test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing our RAG retrieval components...")
    
    # Test retrieval
    retrieval_success = test_rag_retrieval()
    
    # Test embeddings
    embedding_success = test_embedding_quality()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if retrieval_success:
        print("✅ RAG Retrieval: WORKING")
    else:
        print("❌ RAG Retrieval: FAILED")
    
    if embedding_success:
        print("✅ Embedding System: WORKING")
    else:
        print("❌ Embedding System: FAILED")
    
    if retrieval_success and embedding_success:
        print("\n🎉 Our RAG retrieval system is working!")
        print("✅ The retrieval components of our RAG pipeline are functional")
        print("⚠️  Only LLM classification is missing (needs LLM server)")
        print("\nTo test complete RAG pipeline:")
        print("1. Start LLM server: python -m llama_cpp.server --model <path> --port 8010")
        print("2. Run: python test_real_rag.py")
    else:
        print("\n❌ Our RAG system has issues that need to be fixed")
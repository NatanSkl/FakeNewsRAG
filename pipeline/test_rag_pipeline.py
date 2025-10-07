#!/usr/bin/env python3
"""
Test script for the full RAG pipeline.

This script tests the complete RAG pipeline including:
1. Retrieval of fake and credible evidence
2. Generation of contrastive summaries using LLMs
3. Article classification
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import classify_article_rag, RAGOutput
from common.llm_client import Llama, Mistral, LocalLLM
from retrieval import RetrievalConfig


def test_rag_pipeline(
    article_title: str,
    article_content: str,
    store_path: str,
    llm_url: str = "http://127.0.0.1:8010",
    llm_type: str = "llama",
    verbose: bool = False
) -> RAGOutput:
    """
    Test the complete RAG pipeline.
    
    Args:
        article_title: Title of the article to classify
        article_content: Content of the article to classify
        store_path: Path to the index store
        llm_url: URL of the LLM server
        llm_type: Type of LLM ("llama" or "mistral")
        verbose: Whether to print verbose output
    
    Returns:
        RAGOutput with classification results and evidence
    """
    print("=" * 80)
    print("RAG PIPELINE TEST")
    print("=" * 80)
    print(f"Article Title: {article_title}")
    print(f"Article Content: {article_content[:100]}...")
    print(f"Store Path: {store_path}")
    print(f"LLM URL: {llm_url}")
    print(f"LLM Type: {llm_type}")
    print()
    
    # Initialize LLM
    print(f"[{datetime.now().strftime('%H:%M')}] Initializing LLM...")
    try:
        if llm_type == "llama":
            llm = Llama(llm_url)
        elif llm_type == "mistral":
            llm = Mistral(llm_url)
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
        print(f"[{datetime.now().strftime('%H:%M')}] LLM initialized successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M')}] Error initializing LLM: {e}")
        return None
    
    # Configure retrieval
    retrieval_config = RetrievalConfig(
        k_dense=100,  # Smaller for faster testing
        k_bm25=100,
        topn=10,
        domain_cap=3,
        use_xquad=True,
        use_cross_encoder=True
    )
    
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M')}] Retrieval config: {retrieval_config}")
    
    # Run RAG pipeline
    print(f"[{datetime.now().strftime('%H:%M')}] Starting RAG pipeline...")
    start_time = time.time()
    
    try:
        result = classify_article_rag(
            article_title=article_title,
            article_content=article_content,
            store_dir=store_path,
            llm=llm,
            retrieval_config=retrieval_config,
            verbose=verbose
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[{datetime.now().strftime('%H:%M')}] RAG pipeline completed in {duration:.2f} seconds")
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        print(f"\n🎯 CLASSIFICATION:")
        print(f"   Prediction: {result.classification.prediction}")
        print(f"   Confidence: {result.classification.confidence:.3f}")
        print(f"   Reasoning: {result.classification.reasoning}")
        
        print(f"\n📊 EVIDENCE SUMMARY:")
        print(f"   Fake evidence: {len(result.fake_evidence)} chunks")
        print(f"   Credible evidence: {len(result.reliable_evidence)} chunks")
        
        if result.fake_evidence:
            print(f"\n🔴 FAKE EVIDENCE (Top 3):")
            for i, evidence in enumerate(result.fake_evidence[:3], 1):
                print(f"   [{i}] ID: {evidence.id}")
                print(f"       Title: {evidence.title[:50]}...")
                print(f"       Content: {evidence.text[:100]}...")
                print()
        
        if result.reliable_evidence:
            print(f"\n🟢 CREDIBLE EVIDENCE (Top 3):")
            for i, evidence in enumerate(result.reliable_evidence[:3], 1):
                print(f"   [{i}] ID: {evidence.id}")
                print(f"       Title: {evidence.title[:50]}...")
                print(f"       Content: {evidence.text[:100]}...")
                print()
        
        print(f"\n📝 FAKE SUMMARY:")
        print(f"   {result.fake_summary}")
        
        print(f"\n📝 CREDIBLE SUMMARY:")
        print(f"   {result.reliable_summary}")
        
        return result
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M')}] Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_articles():
    """Test the RAG pipeline with multiple sample articles."""
    
    test_articles = [
        {
            "title": "Scientists Discover New Planet with Alien Life",
            "content": "A team of researchers at NASA has announced the discovery of a new exoplanet that shows signs of alien life. The planet, located 50 light-years away, has an atmosphere containing oxygen and methane, which are strong indicators of biological activity. The discovery was made using the James Webb Space Telescope and represents a breakthrough in the search for extraterrestrial life."
        },
        {
            "title": "COVID-19 Vaccine Causes Autism in Children",
            "content": "A new study published in a medical journal reveals that the COVID-19 vaccine causes autism in children. The research, conducted over 5 years with 10,000 participants, shows a direct correlation between vaccination and autism spectrum disorders. Parents are being warned to avoid vaccinating their children."
        },
        {
            "title": "Climate Change Accelerating Faster Than Predicted",
            "content": "New research from leading climate scientists shows that global warming is accelerating at a rate 20% faster than previously predicted. The study, published in Nature Climate Change, analyzed temperature data from the past decade and found that sea levels are rising more rapidly than expected, with significant implications for coastal cities worldwide."
        }
    ]
    
    print("=" * 80)
    print("TESTING MULTIPLE ARTICLES")
    print("=" * 80)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n{'='*20} ARTICLE {i} {'='*20}")
        result = test_rag_pipeline(
            article_title=article["title"],
            article_content=article["content"],
            store_path="/StudentData/slice",
            llm_url="http://127.0.0.1:8010",
            llm_type="llama",
            verbose=True
        )
        
        if result:
            print(f"✅ Article {i} processed successfully")
        else:
            print(f"❌ Article {i} failed to process")
        
        print("\n" + "-" * 80)


def main():
    """Main function to run RAG pipeline tests."""
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    parser.add_argument("--title", help="Article title to test")
    parser.add_argument("--content", help="Article content to test")
    parser.add_argument("--store", default="../index/store_slice", help="Path to index store")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8010", help="LLM server URL")
    parser.add_argument("--llm-type", choices=["llama", "mistral"], default="llama", help="LLM type")
    parser.add_argument("--test", choices=["single", "multiple"], default="single", help="Test type")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.test == "multiple":
        test_multiple_articles()
    else:
        # Use default article if none provided
        if not args.title or not args.content:
            args.title = "Artificial Intelligence Breakthrough in Medical Diagnosis"
            args.content = "Researchers at Stanford University have developed a new AI system that can diagnose diseases with 95% accuracy. The system uses machine learning algorithms to analyze medical images and patient data, potentially revolutionizing healthcare. The breakthrough could lead to earlier detection of cancer and other serious conditions."
        
        result = test_rag_pipeline(
            article_title=args.title,
            article_content=args.content,
            store_path=args.store,
            llm_url=args.llm_url,
            llm_type=args.llm_type,
            verbose=args.verbose
        )
        
        if result:
            print("\n✅ RAG pipeline test completed successfully!")
        else:
            print("\n❌ RAG pipeline test failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()

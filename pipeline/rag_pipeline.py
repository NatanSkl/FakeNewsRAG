"""
Full RAG pipeline for article classification.

This module orchestrates the complete RAG system:
1. Retrieves fake and credible evidence
2. Generates contrastive summaries using LLMs
3. Classifies articles as fake/credible
"""

from __future__ import annotations
from generate.summary import Article, EvidenceChunk, contrastive_summaries
from classify.classifier import classify_article, ClassificationResult
from common.llm_client import Llama, Mistral, LocalLLM
from retrieval import load_store, retrieve_evidence, RetrievalConfig

import numpy as np
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RAGOutput:
    classification: ClassificationResult
    fake_summary: str
    reliable_summary: str
    fake_evidence: List[EvidenceChunk]
    reliable_evidence: List[EvidenceChunk]
    retrieval_config: RetrievalConfig


def _get_timestamp() -> str:
    """Get current time in HH:MM format."""
    return dt.datetime.now().strftime("%H:%M")


def classify_article_rag(
    article_title: str,
    article_content: str,
    *,
    store_dir: str = "mini_index/store",
    llm: LocalLLM,
    retrieval_config: RetrievalConfig | None = None,
    verbose: bool = False
) -> RAGOutput:
    """
    Full RAG pipeline for article classification.
    
    Args:
        article_title: Title of the article to classify
        article_content: Content of the article to classify
        store_dir: Path to the index store
        llm: Language model for summarization and classification
        retrieval_config: Configuration for retrieval (uses defaults if None)
        verbose: Whether to print verbose output with timestamps
    
    Returns:
        RAGOutput with classification results and evidence
    """
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Starting classification for: '{article_title[:50]}...'")
        print(f"[{_get_timestamp()}] [RAG Pipeline] Store directory: {store_dir}")
    
    # Load the index store
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Loading index store...")
    store = load_store(store_dir)
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Index store loaded successfully")
    
    # Use default config if none provided
    if retrieval_config is None:
        retrieval_config = RetrievalConfig()
        if verbose:
            print(f"[{_get_timestamp()}] [RAG Pipeline] Using default retrieval config")
    
    # Retrieve evidence for both labels
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Retrieving fake evidence...")
    fake_hits = retrieve_evidence(
        store, 
        article_content, 
        title_hint=article_title, 
        label_name="fake", 
        cfg=retrieval_config,
        verbose=verbose
    )
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Retrieved {len(fake_hits)} fake evidence chunks")
    
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Retrieving credible evidence...")
    credible_hits = retrieve_evidence(
        store, 
        article_content, 
        title_hint=article_title, 
        label_name="credible", 
        cfg=retrieval_config,
        verbose=verbose
    )
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Retrieved {len(credible_hits)} credible evidence chunks")
    
    # Convert hits to EvidenceChunk objects
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Converting evidence to chunks...")
    fake_evidence = [
        EvidenceChunk(
            id=h.get("id", "unknown"),
            title=h.get("title", ""),
            text=h["chunk_text"],
            label="fake"
        )
        for h in fake_hits
    ]
    
    credible_evidence = [
        EvidenceChunk(
            id=h.get("id", "unknown"),
            title=h.get("title", ""),
            text=h["chunk_text"],
            label="reliable"
        )
        for h in credible_hits
    ]
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Converted to {len(fake_evidence)} fake and {len(credible_evidence)} credible evidence chunks")
    
    # Create Article object
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Creating article object...")
    article = Article(
        id="test_article",
        title=article_title,
        text=article_content
    )
    
    # Generate contrastive summaries
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Generating contrastive summaries...")
    summaries = contrastive_summaries(
        llm, article, fake_evidence, credible_evidence
    )
    fake_summary = summaries["fake_summary"]
    reliable_summary = summaries["reliable_summary"]
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Summaries generated successfully")
    
    # Classify the article
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Classifying article...")
    classification = classify_article(
        llm,
        article_title, 
        article_content, 
        fake_summary, 
        reliable_summary
    )
    if verbose:
        print(f"[{_get_timestamp()}] [RAG Pipeline] Classification completed: {classification.prediction} (confidence: {classification.confidence:.3f})")
        print(f"[{_get_timestamp()}] [RAG Pipeline] RAG pipeline completed successfully!")
    
    return RAGOutput(
        classification=classification,
        fake_summary=fake_summary,
        reliable_summary=reliable_summary,
        fake_evidence=fake_evidence,
        reliable_evidence=credible_evidence,
        retrieval_config=retrieval_config
    )


def main():
    """Example usage of the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full RAG pipeline on an article")
    parser.add_argument("--title", required=True, help="Article title")
    parser.add_argument("--content", required=True, help="Article content")
    parser.add_argument("--store", default="mini_index/store", help="Path to index store")
    parser.add_argument("--llm-type", choices=["llama", "mistral"], default="llama", help="LLM type")
    parser.add_argument("--llm-url", default="http://localhost:8000", help="LLM server URL")
    args = parser.parse_args()
    
    # Initialize LLM
    if args.llm_type == "llama":
        llm = Llama(args.llm_url)
    else:
        llm = Mistral(args.llm_url)
    
    # Run RAG pipeline
    result = classify_article_rag(
        article_title=args.title,
        article_content=args.content,
        store_dir=args.store,
        llm=llm
    )
    
    # Print results
    print(f"\n=== CLASSIFICATION ===")
    print(f"Prediction: {result.classification.prediction}")
    print(f"Confidence: {result.classification.confidence:.3f}")
    print(f"Reasoning: {result.classification.reasoning}")
    
    print(f"\n=== FAKE SUMMARY ===")
    print(result.fake_summary)
    
    print(f"\n=== CREDIBLE SUMMARY ===")
    print(result.reliable_summary)
    
    print(f"\n=== EVIDENCE COUNTS ===")
    print(f"Fake evidence: {len(result.fake_evidence)} chunks")
    print(f"Credible evidence: {len(result.reliable_evidence)} chunks")


if __name__ == "__main__":
    main()
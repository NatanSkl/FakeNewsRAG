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


def classify_article_rag(
    article_title: str,
    article_content: str,
    *,
    store_dir: str = "mini_index/store",
    llm: LocalLLM,
    retrieval_config: RetrievalConfig | None = None
) -> RAGOutput:
    """
    Full RAG pipeline for article classification.
    
    Args:
        article_title: Title of the article to classify
        article_content: Content of the article to classify
        store_dir: Path to the index store
        llm: Language model for summarization and classification
        retrieval_config: Configuration for retrieval (uses defaults if None)
    
    Returns:
        RAGOutput with classification results and evidence
    """
    # Load the index store
    store = load_store(store_dir)
    
    # Use default config if none provided
    if retrieval_config is None:
        retrieval_config = RetrievalConfig()
    
    # Retrieve evidence for both labels
    print("Retrieving fake evidence...")
    fake_hits = retrieve_evidence(
        store, 
        article_content, 
        title_hint=article_title, 
        label_name="fake", 
        cfg=retrieval_config
    )
    
    print("Retrieving credible evidence...")
    credible_hits = retrieve_evidence(
        store, 
        article_content, 
        title_hint=article_title, 
        label_name="credible", 
        cfg=retrieval_config
    )
    
    # Convert hits to EvidenceChunk objects
    fake_evidence = [
        EvidenceChunk(
            content=h["chunk_text"],
            source=h.get("source_domain", ""),
            title=h.get("title", ""),
            url=h.get("url", ""),
            published_at=h.get("published_at", ""),
            score=h.get("_score", h.get("rrf", 0.0))
        )
        for h in fake_hits
    ]
    
    credible_evidence = [
        EvidenceChunk(
            content=h["chunk_text"],
            source=h.get("source_domain", ""),
            title=h.get("title", ""),
            url=h.get("url", ""),
            published_at=h.get("published_at", ""),
            score=h.get("_score", h.get("rrf", 0.0))
        )
        for h in credible_hits
    ]
    
    # Create Article object
    article = Article(
        title=article_title,
        content=article_content
    )
    
    # Generate contrastive summaries
    print("Generating summaries...")
    fake_summary, reliable_summary = contrastive_summaries(
        article, fake_evidence, credible_evidence, llm
    )
    
    # Classify the article
    print("Classifying article...")
    classification = classify_article(
        article_title, 
        article_content, 
        fake_summary, 
        reliable_summary, 
        llm
    )
    
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
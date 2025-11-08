"""
Full RAG pipeline for article classification.

This module orchestrates the complete RAG system:
1. Retrieves fake and reliable evidence
2. Generates contrastive summaries using LLMs
3. Classifies articles as fake/reliable
"""

from __future__ import annotations
from generate.summary import Article, EvidenceChunk, contrastive_summaries
from classify.classifier import classify_article, ClassificationResult
from common.llm_client import Llama, Mistral, LocalLLM
from retrieval import load_store, retrieve_evidence, RetrievalConfig

import numpy as np
import datetime as dt
import time
from dataclasses import dataclass
from typing import List, Dict, Any

# Import logging utilities
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from custom_logging.logger import setup_logging, get_logger

# Setup logging
setup_logging('pipeline.log', log_level=logging.DEBUG, include_console=True)

# Initialize logger
logger = get_logger(__name__)


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


def _update_progress(callback, stage: str, progress: float, sleep_time: float = 0.1):
    """Helper to update progress and add a small delay for visibility."""
    if callback:
        callback(stage, progress)
        time.sleep(sleep_time)


def classify_article_rag(
    article_title: str,
    article_content: str,
    stores: Dict[str, Any],
    *,
    ce_model = None,
    diversify_type: str = None,
    llm: LocalLLM,
    retrieval_config: RetrievalConfig | None = None,
    verbose: bool = False,
    prompt_type: int = 0,
    naming_convention: str = "fake_reliable",
    debug_mode: bool = False,
    progress_callback = None
) -> RAGOutput:
    """
    Full RAG pipeline for article classification.
    
    Args:
        article_title: Title of the article to classify
        article_content: Content of the article to classify
        stores: Dictionary of pre-loaded Store objects with keys "fake" and "reliable"
        ce_model: Cross-encoder model for reranking (None to skip)
        diversify_type: Diversity method ("mmr" or None to skip)
        llm: Language model for summarization and classification
        retrieval_config: Configuration for retrieval (uses defaults if None)
        verbose: Whether to print verbose output with timestamps
        prompt_type: Type of prompt to use for summarization and classification
        naming_convention: Naming convention for classification labels
        debug_mode: If True, returns debug information including prompts
        progress_callback: Optional callback function(stage: str, progress: float) called at each stage
    
    Returns:
        RAGOutput with classification results and evidence
    """
    logger.info(f"[RAG Pipeline] Starting classification for: '{article_title[:50]}...'")
    logger.info("[RAG Pipeline] Using pre-loaded stores")
    
    _update_progress(progress_callback, "Initializing retrieval configuration", 0.10)
    
    # Use default config if none provided
    if retrieval_config is None:
        retrieval_config = RetrievalConfig(
            k=10, 
            ce_model=ce_model,
            diversity_type=diversify_type,
            verbose=verbose
        )
        logger.info("[RAG Pipeline] Using default retrieval config")
    
    # Retrieve evidence for both labels
    _update_progress(progress_callback, "Retrieving fake evidence", 0.20)
    logger.info("[RAG Pipeline] Retrieving fake evidence...")
    fake_hits = retrieve_evidence(
        stores, 
        article_content, 
        "fake", 
        retrieval_config
    )
    logger.info(f"[RAG Pipeline] Retrieved {len(fake_hits)} fake evidence chunks")
    _update_progress(progress_callback, "Retrieved fake evidence", 0.35)
    
    _update_progress(progress_callback, "Retrieving reliable evidence", 0.40)
    logger.info("[RAG Pipeline] Retrieving reliable evidence...")
    credible_hits = retrieve_evidence(
        stores, 
        article_content, 
        "reliable", 
        retrieval_config
    )
    logger.info(f"[RAG Pipeline] Retrieved {len(credible_hits)} reliable evidence chunks")
    _update_progress(progress_callback, "Retrieved reliable evidence", 0.50)
    
    # Convert hits to EvidenceChunk objects
    logger.info("[RAG Pipeline] Converting evidence to chunks...")
    fake_evidence = [
        EvidenceChunk(
            id=h.get("db_id", "unknown"),
            title=h.get("title", ""),
            text=h["full_content"],  # Use full_content if available, fallback to content
            label="fake"
        )
        for h in fake_hits
    ]
    
    credible_evidence = [
        EvidenceChunk(
            id=h.get("db_id", "unknown"),
            title=h.get("title", ""),
            text=h["full_content"],  # Use full_content if available, fallback to content
            label="reliable"
        )
        for h in credible_hits
    ]
    logger.info(f"[RAG Pipeline] Converted to {len(fake_evidence)} fake and {len(credible_evidence)} reliable evidence chunks")
    
    # Create Article object
    logger.info("[RAG Pipeline] Creating article object...")
    article = Article(
        id="test_article",
        title=article_title,
        text=article_content
    )
    
    # Generate contrastive summaries
    _update_progress(progress_callback, "Generating contrastive summaries (calling LLM)", 0.60)
    logger.info("[RAG Pipeline] Generating contrastive summaries...")
    summaries = contrastive_summaries(
        llm, article, fake_evidence, credible_evidence, promt_type=prompt_type, return_debug=debug_mode
    )
    fake_summary = summaries["fake_summary"]
    reliable_summary = summaries["reliable_summary"]
    logger.info("[RAG Pipeline] Summaries generated successfully")
    _update_progress(progress_callback, "Summaries generated", 0.75)
    
    # Classify the article
    _update_progress(progress_callback, "Classifying article (calling LLM)", 0.85)
    logger.info("[RAG Pipeline] Classifying article...")
    classification = classify_article(
        llm,
        article_title, 
        article_content, 
        fake_summary, 
        reliable_summary,
        naming_convention=naming_convention,
        promt_type=prompt_type,
        return_debug=debug_mode
    )
    logger.info(f"[RAG Pipeline] Classification completed: {classification.prediction} (confidence: {classification.confidence:.3f})")
    _update_progress(progress_callback, "Classification complete", 1.00)
    logger.info("[RAG Pipeline] RAG pipeline completed successfully!")
    
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
    parser.add_argument("--store", default="/StudentData/slice", help="Path to index store")
    parser.add_argument("--llm-type", choices=["llama", "mistral"], default="llama", help="LLM type")
    parser.add_argument("--llm-url", default="http://localhost:8000", help="LLM server URL")
    args = parser.parse_args()
    
    # Initialize LLM
    if args.llm_type == "llama":
        llm = Llama(args.llm_url)
    else:
        llm = Mistral(args.llm_url)
    
    # Load stores
    logger.info("Loading stores...")
    fake_store = load_store(args.store + "_fake", verbose=True)
    reliable_store = load_store(args.store + "_reliable", verbose=True)
    stores = {"fake": fake_store, "reliable": reliable_store}
    
    # Run RAG pipeline
    result = classify_article_rag(
        article_title=args.title,
        article_content=args.content,
        stores=stores,
        llm=llm
    )
    
    # Print results
    logger.info(f"\n=== CLASSIFICATION ===")
    logger.info(f"Prediction: {result.classification.prediction}")
    logger.info(f"Confidence: {result.classification.confidence:.3f}")
    logger.info(f"Reasoning: {result.classification.reasoning}")
    
    logger.info(f"\n=== FAKE SUMMARY ===")
    logger.info(result.fake_summary)
    
    logger.info(f"\n=== CREDIBLE SUMMARY ===")
    logger.info(result.reliable_summary)
    
    logger.info(f"\n=== EVIDENCE COUNTS ===")
    logger.info(f"Fake evidence: {len(result.fake_evidence)} chunks")
    logger.info(f"Credible evidence: {len(result.reliable_evidence)} chunks")


if __name__ == "__main__":
    main()
"""
Pipeline module for the FakeNewsRAG system.

This module contains the full RAG pipeline for article classification.
"""

from .rag_pipeline import (
    classify_article_rag,
    RAGOutput
)

__all__ = [
    'classify_article_rag',
    'RAGOutput'
]

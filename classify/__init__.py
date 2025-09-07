"""
Classification module for Fake News RAG project.

This module provides functionality to classify articles as fake or reliable
using contrastive summaries generated from retrieved evidence.

Main components:
- classify_article: Full classification with confidence and reasoning
- classify_article_simple: Simple classification returning only prediction
- ClassificationResult: Data class for classification results
"""

from .classifier import (
    classify_article,
    classify_article_simple,
    ClassificationResult
)

__all__ = [
    "classify_article",
    "classify_article_simple", 
    "ClassificationResult"
]

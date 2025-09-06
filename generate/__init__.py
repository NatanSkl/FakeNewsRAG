"""
Generation module for Fake News RAG project.

This module provides functionality to generate contrastive summaries using local LLMs.
The main functionality includes:

- LocalLLM: Client for communicating with llama.cpp server
- contrastive_summaries: Generate two summaries (fake vs credible) based on retrieved evidence
- Article and EvidenceChunk: Data models for representing articles and evidence
"""

from common.llm_client import LocalLLM, ChatResponse
from .summary import Article, EvidenceChunk, contrastive_summaries

__all__ = [
    "LocalLLM",
    "ChatResponse", 
    "Article",
    "EvidenceChunk",
    "contrastive_summaries"
]

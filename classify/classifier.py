"""
Classification module for Fake News RAG project.

This module provides functionality to classify articles as fake or reliable
using contrastive summaries generated from retrieved evidence.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional

import tiktoken

from common.llm_client import LocalLLM


def _trim_tokens(s: str, max_tokens: int) -> str:
    """Trim text to approximately max_tokens using tiktoken encoding."""
    s = (s or "").strip()
    if not s:
        return s
    
    # Use cl100k_base encoding (GPT-4 tokenizer)
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(s)
    
    if len(tokens) <= max_tokens:
        return s
    
    # Truncate to max_tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)


@dataclass
class ClassificationResult:
    """Result of article classification."""
    prediction: str  # "fake" or "reliable"
    confidence: float  # confidence score between 0 and 1
    reasoning: str  # explanation for the classification
    raw_response: str  # raw LLM response


# TODO test cases of different personalities / instructions / level of detail
# Classification prompt templates
CLASSIFICATION_SYSTEM = (
    "You are an expert fact-checker and news analyst. Your task is to determine whether "
    "a given article is FAKE NEWS or RELIABLE based on evidence summaries. "
    "Analyze the article content against both the fake news evidence and reliable evidence summaries. "
    "Consider which evidence better aligns with the facts presented in the article. "
    "Provide a clear classification with reasoning."
)

# TODO add limits to these templates

CLASSIFICATION_USER_TEMPLATE = (
    """
ARTICLE TO CLASSIFY
-------------------
Title: {article_title}
Content: {article_content}

FAKE NEWS EVIDENCE SUMMARY
--------------------------
{fake_summary}

RELIABLE NEWS EVIDENCE SUMMARY
------------------------------
{reliable_summary}

TASK
----
Based on the evidence summaries above, determine if the article is FAKE NEWS or RELIABLE.

Consider:
1. Which evidence summary better aligns with the facts in the article?
2. Are there contradictions between the article and reliable evidence?
3. Does the article contain claims that are supported by fake news evidence?

Respond with:
- Classification: [FAKE/RELIABLE]
- Confidence: [0.0-1.0]   This should be a number between 0 and 1 that represents the confidence in your classification.
- Reasoning: [Brief explanation of your decision]

Format your response exactly as:
Classification: [FAKE/RELIABLE]
Confidence: [0.0-1.0]
Reasoning: [Your explanation here]
"""
)


def classify_article(
    llm: LocalLLM,
    article_title: str,
    article_content: str,
    fake_summary: str,
    reliable_summary: str,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> ClassificationResult:
    """
    Classify an article as fake or reliable based on contrastive summaries.
    
    Args:
        llm: LocalLLM instance for classification
        article_title: Title of the article to classify
        article_content: Content/text of the article to classify
        fake_summary: Summary based on fake news evidence
        reliable_summary: Summary based on reliable news evidence
        temperature: LLM temperature for generation (lower = more deterministic)
        max_tokens: Maximum tokens for the response
        
    Returns:
        ClassificationResult with prediction, confidence, reasoning, and raw response
    """
    
    # Apply token limits
    article_content_trimmed = _trim_tokens(article_content, 600)
    fake_summary_trimmed = _trim_tokens(fake_summary, 300)
    reliable_summary_trimmed = _trim_tokens(reliable_summary, 300)
    
    # Format the prompt
    prompt = CLASSIFICATION_USER_TEMPLATE.format(
        article_title=article_title,
        article_content=article_content_trimmed,
        fake_summary=fake_summary_trimmed,
        reliable_summary=reliable_summary_trimmed
    )
    
    # Create messages for the LLM
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    # Get response from LLM
    response = llm.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Parse the response
    return _parse_classification_response(response.text)


def _parse_classification_response(response_text: str) -> ClassificationResult:
    """
    Parse the LLM response to extract classification, confidence, and reasoning.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        ClassificationResult with parsed information
    """
    lines = response_text.strip().split('\n')
    
    prediction = "reliable"  # default
    confidence = 0.5  # default
    reasoning = "Unable to parse response"
    
    for line in lines:
        line = line.strip()
        if line.startswith("Classification:"):
            classification_text = line.split(":", 1)[1].strip().upper()
            if "FAKE" in classification_text:
                prediction = "fake"
            elif "RELIABLE" in classification_text:
                prediction = "reliable"
        elif line.startswith("Confidence:"):
            try:
                confidence_text = line.split(":", 1)[1].strip()
                confidence = float(confidence_text)
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                confidence = 0.5
        elif line.startswith("Reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
    
    return ClassificationResult(
        prediction=prediction,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=response_text
    )


def classify_article_simple(
    llm: LocalLLM,
    article_title: str,
    article_content: str,
    fake_summary: str,
    reliable_summary: str,
    **kwargs
) -> str:
    """
    Simple classification that returns only the prediction (fake/reliable).
    
    Args:
        llm: LocalLLM instance for classification
        article_title: Title of the article to classify
        article_content: Content/text of the article to classify
        fake_summary: Summary based on fake news evidence
        reliable_summary: Summary based on reliable news evidence
        **kwargs: Additional arguments passed to classify_article
        
    Returns:
        String: "fake" or "reliable"
    """
    result = classify_article(
        llm=llm,
        article_title=article_title,
        article_content=article_content,
        fake_summary=fake_summary,
        reliable_summary=reliable_summary,
        **kwargs
    )
    return result.prediction

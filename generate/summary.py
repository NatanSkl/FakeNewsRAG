"""
Generate class-conditional, query-conditioned (contrastive) summaries.

Core API:
    contrastive_summaries(query, fake_evidence, credible_evidence, llm, ...)

- "contrastive" = produce two summaries that relate retrieved evidence to the
  query article: one using FAKE-labeled evidence, one using CREDIBLE-labeled.
- This module is retrieval-agnostic: it takes already-matched evidence.
- A small CLI remains for quick manual runs on a CSV sample.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tiktoken

from common.llm_client import LocalLLM

# Import logging utilities
sys.path.append(str(Path(__file__).parent.parent))
from custom_logging.logger import setup_logging, get_logger

# Setup logging
setup_logging('generate.log', log_level=logging.DEBUG, include_console=True)

# Initialize logger
logger = get_logger(__name__)


# ----------------------------- data models -----------------------------
@dataclass
class Article:
    id: str
    title: str
    text: str


@dataclass
class EvidenceChunk:
    id: str
    title: str
    text: str
    label: str  # "fake" | "reliable"


def _first_present(cols: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _trim(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s[:max_chars]


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


# ----------------------------- prompting -----------------------------
# TODO test cases of [General summary, Fake news detection summary, etc]
SUMMARY_SYSTEMS = [
    (
        "You are a careful fact-focused assistant. Summarize ARTICLES."
    ),
    (
        "You are a careful fact-focused assistant. Summarize ARTICLES."
    ),
    (
        """You are an expert journalist and fact-checker. Your task is to write a comprehensive, 
        well-structured news article based on the provided ARTICLES. Create an article that 
        accurately represents the facts from the ARTICLES while maintaining journalistic 
        standards and objectivity."""
    )
]

SUMMARY_USER_TEMPLATES = [
    (
        """
ARTICLES
-----------------------
{evidence_bullets}

TASK
----
Write a concise summary focusing on the facts of the ARTICLES. 
Limit to 300 tokens.
"""
    ),
    (
        """
ARTICLES
-----------------------
{evidence_bullets}

TASK
----
Write a concise summary focusing on the facts of the ARTICLES. 
The summary will be used to classify new articles as fake or reliable based on their closeness to the summary. 
Avoid direct mentions of which group of articles you are summarizing (e.g. fake or reliable).
Limit to 300 tokens.
"""
    ),
    (
        """
ARTICLES
-----------------
{evidence_bullets}

TASK
----
Write a new article based on the facts of the ARTICLES. 
Avoid direct mentions of the articles you are using.
Limit to 300 tokens.
"""
    )
]


def _format_evidence_bullets(chunks: List[EvidenceChunk], max_tokens: int = 300) -> str:
    """Format evidence chunks into a long string with article headers."""
    logger.info(f"Formatting {len(chunks)} evidence chunks into article format (max {max_tokens} tokens each)")
    
    articles = []
    for i, ch in enumerate(chunks, 1):
        article_header = f"Article {i} - {_trim(ch.title, 120)}"
        article_content = _trim_tokens(ch.text, max_tokens)
        article_text = f"{article_header}\n{article_content}"
        articles.append(article_text)
        logger.debug(f"Evidence chunk {i}/{len(chunks)}: {ch.id} - {ch.label} - {len(ch.text)} chars")
    
    result = "\n\n".join(articles) if articles else "(no evidence)"
    logger.info(f"Formatted evidence articles: {len(result)} total characters")
    
    return result


# ----------------------------- public API -----------------------------

def contrastive_summaries(
    llm: LocalLLM,
    query: Article,
    fake_evidence: List[EvidenceChunk],
    reliable_evidence: List[EvidenceChunk],
    temperature: float = 0.2,
    max_tokens: int = 600,
    promt_type: int = 0,
) -> Dict[str, str]:
    """Produce two summaries conditioned on the same query with different evidence sets.

    Args:
        llm: LocalLLM instance for generating summaries
        query: Article to summarize in relation to
        fake_evidence: List of EvidenceChunk objects labeled as "fake"
        reliable_evidence: List of EvidenceChunk objects labeled as "reliable"
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens for each summary
        promt_type: Prompt type (0 = general persona, 1 = persona with background information, 2 = new article)

    Returns:
        Dict with keys "fake_summary" and "reliable_summary" containing the generated summaries
    """
    summary_system = SUMMARY_SYSTEMS[promt_type]
    summary_user_template = SUMMARY_USER_TEMPLATES[promt_type]

    logger.info(f"Starting contrastive summaries generation for query: '{query.title[:50]}...'")
    logger.info(f"Fake evidence: {len(fake_evidence)} chunks, Reliable evidence: {len(reliable_evidence)} chunks")

    q_title = _trim(query.title or "(untitled)", 160)
    q_body = _trim(query.text, 2000)

    logger.info("Generating fake evidence prompt...")
    prompt_fake = summary_user_template.format(
        #q_title=q_title,
        #q_body=q_body,
        evidence_bullets=_format_evidence_bullets(fake_evidence),
    )

    logger.info("Generating reliable evidence prompt...")
    prompt_reliable = summary_user_template.format(
        #q_title=q_title,
        #q_body=q_body,
        evidence_bullets=_format_evidence_bullets(reliable_evidence),
    )

    # Log the prompts
    logger.info("=== FAKE EVIDENCE PROMPT ===")
    logger.info(f"System: {summary_system}")
    logger.info(f"User: {prompt_fake}")

    logger.info("=== RELIABLE EVIDENCE PROMPT ===")
    logger.info(f"System: {summary_system}")
    logger.info(f"User: {prompt_reliable}")

    msgs_fake = [{"role": "system", "content": summary_system}, {"role": "user", "content": prompt_fake}]
    msgs_reliable = [{"role": "system", "content": summary_system}, {"role": "user", "content": prompt_reliable}]

    logger.info(f"Calling LLM for fake summary (temperature={temperature}, max_tokens={max_tokens})...")
    resp_fake = llm.chat(msgs_fake, temperature=temperature, max_tokens=max_tokens)
    logger.info(f"Fake summary generated: {len(resp_fake.text)} characters")

    logger.info(f"Calling LLM for reliable summary (temperature={temperature}, max_tokens={max_tokens})...")
    resp_reliable = llm.chat(msgs_reliable, temperature=temperature, max_tokens=max_tokens)
    logger.info(f"Reliable summary generated: {len(resp_reliable.text)} characters")

    logger.info("Contrastive summaries generation completed successfully!")
    return {"fake_summary": resp_fake.text.strip(), "reliable_summary": resp_reliable.text.strip()}

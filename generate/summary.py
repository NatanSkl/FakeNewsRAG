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

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from common.llm_client import LocalLLM


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


# ----------------------------- prompting -----------------------------
# TODO test cases of [General summary, Fake news detection summary, etc]
SUMMARY_SYSTEM = (
    "You are a careful fact-focused assistant. Summarize EVIDENCE in relation to the QUERY. "
    "Quote short spans from evidence with (chunk_id) markers. If evidence is insufficient, say so."
)

SUMMARY_USER_TEMPLATE = (
    """
QUERY ARTICLE
-------------
Title: {q_title}
Body (truncated):\n{q_body}

EVIDENCE ({label_upper})
-----------------------
{evidence_bullets}

TASK
----
Write a concise summary (~6-10 sentences) explaining how the EVIDENCE {stance_word} the QUERY. 
Use inline citations like ({{chunk_id}}). End with a one-line verdict about {label_lower} sources.
"""
)


def _format_evidence_bullets(chunks: List[EvidenceChunk], max_each: int = 600) -> str:
    bullets = []
    for ch in chunks:
        bullets.append(f"- [{ch.id}] { _trim(ch.title, 120) } — {_trim(ch.text, max_each)}")
    return "\n".join(bullets) if bullets else "(no evidence)"


# ----------------------------- public API -----------------------------

def contrastive_summaries(
    llm: LocalLLM,
    query: Article,
    fake_evidence: List[EvidenceChunk],
    reliable_evidence: List[EvidenceChunk],
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> Dict[str, str]:
    """Produce two summaries conditioned on the same query with different evidence sets.

    Args:
        llm: LocalLLM instance for generating summaries
        query: Article to summarize in relation to
        fake_evidence: List of EvidenceChunk objects labeled as "fake"
        reliable_evidence: List of EvidenceChunk objects labeled as "reliable"
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens for each summary

    Returns:
        Dict with keys "fake_summary" and "reliable_summary" containing the generated summaries
    """
    q_title = _trim(query.title or "(untitled)", 160)
    q_body = _trim(query.text, 2000)

    prompt_fake = SUMMARY_USER_TEMPLATE.format(
        q_title=q_title,
        q_body=q_body,
        evidence_bullets=_format_evidence_bullets(fake_evidence),
        label_upper="FAKE",
        label_lower="fake",
        stance_word="supports or aligns with",
    )
    prompt_reliable = SUMMARY_USER_TEMPLATE.format(
        q_title=q_title,
        q_body=q_body,
        evidence_bullets=_format_evidence_bullets(reliable_evidence),
        label_upper="RELIABLE",
        label_lower="reliable",
        stance_word="supports or aligns with",
    )

    msgs_fake = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt_fake}]
    msgs_reliable = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt_reliable}]

    resp_fake = llm.chat(msgs_fake, temperature=temperature, max_tokens=max_tokens)
    resp_reliable = llm.chat(msgs_reliable, temperature=temperature, max_tokens=max_tokens)

    return {"fake_summary": resp_fake.text.strip(), "reliable_summary": resp_reliable.text.strip()}
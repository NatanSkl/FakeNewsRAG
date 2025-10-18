"""
Test module for the summary generation functionality.

This module simulates a simple retrieval mechanism and tests that the summarization
works well with the local LLM. It doesn't focus on the retrieval quality but rather
on ensuring the generation module functions correctly.
"""
from __future__ import annotations

import os
import re
import math
import random
import json
from collections import Counter
from typing import Tuple, Optional

import pandas as pd
import pytest

from .summary import Article, EvidenceChunk, contrastive_summaries, _first_present
from common.llm_client import Llama, Mistral

# Global LLM instance - easily switch between models
# Change this to Mistral() to use Mistral instead of Llama
llm = Llama()

# TOOD verify that summary module behaves correctly


COMMON_TEXT_COLS  = ["text", "body", "content", "article", "full_text"]
COMMON_TITLE_COLS = ["title", "headline"]
COMMON_LABEL_COLS = ["label", "class", "tag", "type", "labels"]
COMMON_ID_COLS    = ["id", "article_id", "doc_id"]


def _coerce_schema(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    text_col = _first_present(COMMON_TEXT_COLS, df)
    label_col = _first_present(COMMON_LABEL_COLS, df)
    title_col = _first_present(COMMON_TITLE_COLS, df) or text_col
    id_col = _first_present(COMMON_ID_COLS, df)
    if not text_col or not label_col:
        raise ValueError(f"CSV must contain at least text and label columns. Found: {list(df.columns)}")
    return text_col, label_col, title_col, id_col


# ----------------------- simple similarity helpers (no sklearn) -----------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","by","at","from","as",
    "is","are","was","were","be","been","that","this","it","its","into","about",
}

def tokens(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "") if t.lower() not in _STOP]

def bow(s: str) -> Counter:
    return Counter(tokens(s))

def cosine(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    # dot product
    dot = sum(c1[t]*c2.get(t,0) for t in c1)
    if dot == 0:
        return 0.0
    n1 = math.sqrt(sum(v*v for v in c1.values()))
    n2 = math.sqrt(sum(v*v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot/(n1*n2)


def top_k_similar(df, query_idx, title_col, label_col, k=4, type_filter=None):
    """Simple retrieval simulation using title-based cosine similarity.
    
    This is a basic implementation for testing purposes - not meant for production.
    
    Args:
        df: DataFrame containing articles
        query_idx: Index of the query article
        title_col: Column name for titles
        label_col: Column name for labels
        k: Number of similar articles to return
        type_filter: Optional filter for specific label type ("fake" or "reliable")
    
    Returns:
        List of indices of similar articles, or dict with per-label results if type_filter is None
    """
    q = df.iloc[query_idx]
    q_title = str(q.get(title_col, ""))
    q_vec = bow(q_title)

    # compute similarity to every other row
    sims = []
    for i, r in df.iterrows():
        if i == query_idx:
            continue
        r_title = str(r.get(title_col, ""))
        sims.append((i, cosine(q_vec, bow(r_title))))

    # sort by similarity desc
    sims.sort(key=lambda x: x[1], reverse=True)

    # normalize label function
    def norm_label(x: str) -> str:
        x = str(x).lower()
        if x in ("1","true","reliable"): return "reliable"
        if x in ("0","false","fake"): return "fake"
        return x

    # If type_filter is specified, return only articles of that type
    if type_filter is not None:
        type_filter = type_filter.lower()
        result = []
        for i, sim in sims:
            lab = norm_label(df.iloc[i][label_col])
            if lab == type_filter and len(result) < k:
                result.append(i)
        return result
    
    # Otherwise, return per-label results (backward compatibility)
    per_label = {"fake": [], "reliable": []}
    for i, sim in sims:
        lab = norm_label(df.iloc[i][label_col])
        if lab in per_label and len(per_label[lab]) < k:
            per_label[lab].append(i)
        if all(len(v) >= k for v in per_label.values()):
            break

    return per_label


@pytest.mark.integration
def test_contrastive_summaries_real_llm_title_similarity():
    """
    Test the contrastive summarization with a real local LLM using simple retrieval simulation.
    DEBUG to see how the test runs
    """
    csv_path = "news_sample.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"Sample CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    text_col, label_col, title_col, id_col = _coerce_schema(df)

    # Use a stable but randomish query to vary across runs slightly
    random.seed(42)
    query_idx = random.randrange(0, len(df))
    row = df.iloc[query_idx]
    query = Article(
        id=str(row.get(id_col, query_idx)),
        title=str(row.get(title_col, "")),
        text=str(row.get(text_col, "")),
    )

    # quick health check for the local llama server; skip if unavailable
    try:
        ping = llm.simple("Say OK.", system="You only say OK.", max_tokens=3)
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")

    # simple retrieval: pick top-k by title cosine similarity per label
    fake_idxs = top_k_similar(df, query_idx, title_col, label_col, k=3, type_filter="fake")
    reliable_idxs = top_k_similar(df, query_idx, title_col, label_col, k=3, type_filter="reliable")

    def rows_to_chunks(idxs, label_value):
        chunks = []
        for i in idxs:
            r = df.iloc[i]
            chunks.append(EvidenceChunk(
                id=str(r.get(id_col, i)),
                title=str(r.get(title_col, "")),
                text=str(r.get(text_col, ""))[:1500],
                label=label_value,
            ))
        return chunks

    ev_fake = rows_to_chunks(fake_idxs, "fake")
    ev_reliable = rows_to_chunks(reliable_idxs, "reliable")

    # if either side is empty, backfill with random examples of that label
    def backfill(label_name, dest):
        if dest:
            return dest
        mask = df[label_col].astype(str).str.lower().isin([
            label_name, "1" if label_name == "reliable" else "0", "true" if label_name == "reliable" else "false"
        ])
        sample = df[mask].head(2)
        return [
            EvidenceChunk(id=str(i), title=str(r.get(title_col, "")), text=str(r.get(text_col, ""))[:1500], label=label_name)
            for i, r in sample.iterrows()
        ]

    ev_fake = backfill("fake", ev_fake)
    ev_reliable = backfill("reliable", ev_reliable)

    out = contrastive_summaries(llm, query, ev_fake, ev_reliable, temperature=0.2, max_tokens=400)

    # minimal assertions: non-empty, somewhat long, and different strings
    assert isinstance(out.get("fake_summary"), str) and len(out["fake_summary"]) > 40
    assert isinstance(out.get("reliable_summary"), str) and len(out["reliable_summary"]) > 40
    # optional: ensure the two aren't identical
    assert out["fake_summary"] != out["reliable_summary"]

    # drop a debug artifact for quick inspection
    artifact = "artifacts/test_real_llm_contrastive_summaries.json"
    with open(artifact, "w", encoding="utf-8") as f:
        json.dump({
            "query": query.__dict__,
            "ev_fake": [c.__dict__ for c in ev_fake],
            "ev_reliable": [c.__dict__ for c in ev_reliable],
            **out,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved artifact to {artifact}")


def test_llm_connection():
    """Test that the local LLM server is reachable and responding."""
    try:
        response = llm.simple("Hello, respond with just 'OK'", max_tokens=5)
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"LLM response: {response}")
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")


def test_summary_with_mock_data():
    """Test the summary generation with mock data (no LLM required)."""
    # Create mock LLM that returns predictable responses
    class MockLLM:
        def chat(self, messages, **kwargs):
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            return MockResponse("This is a mock summary for testing purposes.")
    
    mock_llm = MockLLM()
    
    # Create test data
    query = Article(
        id="test_query",
        title="Test Article Title",
        text="This is a test article about some topic."
    )
    
    fake_evidence = [
        EvidenceChunk(
            id="fake1",
            title="Fake Evidence 1",
            text="This is fake evidence that contradicts the query.",
            label="fake"
        )
    ]
    
    reliable_evidence = [
        EvidenceChunk(
            id="reliable1",
            title="Reliable Evidence 1", 
            text="This is reliable evidence that supports the query.",
            label="reliable"
        )
    ]
    
    # Test the function
    result = contrastive_summaries(mock_llm, query, fake_evidence, reliable_evidence)
    
    # Assertions
    assert "fake_summary" in result
    assert "reliable_summary" in result
    assert isinstance(result["fake_summary"], str)
    assert isinstance(result["reliable_summary"], str)
    assert len(result["fake_summary"]) > 0
    assert len(result["reliable_summary"]) > 0


@pytest.mark.integration
def test_static_data_summary():
    """Test the contrastive summarization with static data using a real local LLM."""
    
    # Test LLM connection first
    try:
        test_response = llm.simple("Hello, respond with just 'OK'", max_tokens=5)
        assert isinstance(test_response, str)
        assert len(test_response.strip()) > 0
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")
    
    # Create a sample query article
    query = Article(
        id="query_001",
        title="New Study Shows Climate Change Impact",
        text="A recent study published in Nature Climate Change reveals that global temperatures have risen by 1.5 degrees Celsius over the past century. The research, conducted by an international team of scientists, analyzed temperature data from over 100 weather stations worldwide. The findings suggest that human activities, particularly greenhouse gas emissions, are the primary driver of this warming trend."
    )
    
    # Create sample fake evidence
    fake_evidence = [
        EvidenceChunk(
            id="fake_001",
            title="Climate Change is a Hoax",
            text="Climate change is a manufactured crisis created by global elites to control populations. The so-called 'scientific consensus' is actually a result of funding bias and political pressure. Real scientists know that climate variations are natural and not caused by human activities.",
            label="fake"
        ),
        EvidenceChunk(
            id="fake_002", 
            title="Temperature Data Manipulated",
            text="The temperature records used in climate studies have been systematically manipulated to show warming trends. Weather stations have been moved to urban heat islands, and historical data has been adjusted to fit the global warming narrative.",
            label="fake"
        )
    ]
    
    # Create sample reliable evidence
    reliable_evidence = [
        EvidenceChunk(
            id="reliable_001",
            title="NASA Confirms Global Warming Trend",
            text="NASA's Goddard Institute for Space Studies has confirmed that Earth's average surface temperature has increased by about 1.1 degrees Celsius since the late 19th century. This warming is primarily driven by increased carbon dioxide and other greenhouse gas emissions from human activities.",
            label="reliable"
        ),
        EvidenceChunk(
            id="reliable_002",
            title="IPCC Report on Climate Change",
            text="The Intergovernmental Panel on Climate Change (IPCC) has released its latest assessment report, confirming that human influence has warmed the climate at a rate that is unprecedented in at least the last 2000 years. The report is based on thousands of peer-reviewed scientific studies.",
            label="reliable"
        )
    ]
    
    # Generate contrastive summaries
    summaries = contrastive_summaries(
        llm=llm,
        query=query,
        fake_evidence=fake_evidence,
        reliable_evidence=reliable_evidence,
        temperature=0.2,
        max_tokens=400
    )
    
    # Assertions
    assert "fake_summary" in summaries
    assert "reliable_summary" in summaries
    assert isinstance(summaries["fake_summary"], str)
    assert isinstance(summaries["reliable_summary"], str)
    assert len(summaries["fake_summary"]) > 40
    assert len(summaries["reliable_summary"]) > 40
    assert summaries["fake_summary"] != summaries["reliable_summary"]
    
    # Save debug artifact
    artifact = "artifacts/test_static_data_summary.json"
    with open(artifact, "w", encoding="utf-8") as f:
        json.dump({
            "query": query.__dict__,
            "fake_evidence": [c.__dict__ for c in fake_evidence],
            "reliable_evidence": [c.__dict__ for c in reliable_evidence],
            **summaries,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved artifact to {artifact}")


if __name__ == "__main__":
    # Run a simple test if called directly
    print("Testing LLM connection...")
    test_llm_connection()
    print("Testing with mock data...")
    test_summary_with_mock_data()
    print("All tests passed!")

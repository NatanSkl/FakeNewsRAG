# Generation Module

This module provides functionality to generate contrastive summaries using local LLMs for the Fake News RAG project.

## Overview

The generation module creates two types of summaries based on retrieved evidence:
- **Fake Summary**: Generated from articles labeled as "fake news"
- **Credible Summary**: Generated from articles labeled as "credible"

These summaries are created in relation to a query article and help determine whether the query article is likely fake or credible.

## Components

### 1. LLM Client (`llm_client.py`)
- `LocalLLM`: Client for communicating with llama.cpp server
- `ChatResponse`: Response wrapper for LLM outputs
- Supports OpenAI-compatible API endpoints

### 2. Summary Generation (`summary.py`)
- `contrastive_summaries()`: Main function for generating contrastive summaries
- `Article`: Data model for query articles
- `EvidenceChunk`: Data model for retrieved evidence
- Schema coercion utilities for CSV data

### 3. Testing (`test_summary.py`)
- Integration tests with real LLM
- Mock tests for unit testing
- Simple retrieval simulation for testing

## Usage

### Basic Usage

```python
from generate import LocalLLM, Article, EvidenceChunk, contrastive_summaries

# Initialize LLM client
llm = LocalLLM(base_url="http://127.0.0.1:8010/v1", model="local-llama")

# Create query article
query = Article(
    id="query_001",
    title="Your Article Title",
    text="Your article content..."
)

# Create evidence chunks
fake_evidence = [
    EvidenceChunk(id="fake1", title="Fake Article", text="...", label="fake")
]
credible_evidence = [
    EvidenceChunk(id="cred1", title="Credible Article", text="...", label="credible")
]

# Generate summaries
summaries = contrastive_summaries(
    llm=llm,
    query=query,
    fake_evidence=fake_evidence,
    credible_evidence=credible_evidence
)

print("Fake summary:", summaries["fake_summary"])
print("Credible summary:", summaries["credible_summary"])
```

### Running Tests

```bash
# Run all tests
python -m pytest generate/test_summary.py -v

# Run specific test
python -m pytest generate/test_summary.py::test_llm_connection -v

# Run example
python generate/example_usage.py
```

### CLI Usage

```bash
# Generate summaries from CSV
python -m generate.summary --csv news_sample.csv --row-index 0 --k 5
```

## Requirements

- Local LLM server running (llama.cpp)
- Python packages: requests, pandas, pytest
- CSV data with text, title, and label columns

## Server Setup

Make sure your llama.cpp server is running:

```bash
python -m llama_cpp.server \
  --model ./models/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --n_gpu_layers -1 --n_ctx 2048 --n_batch 256 --offload_kqv true \
  --host 127.0.0.1 --port 8010
```

## Configuration

Set environment variables for LLM connection:
- `LLM_BASE_URL`: Base URL for LLM server (default: http://127.0.0.1:8010/v1)
- `LLM_MODEL`: Model name (default: local-llama)
- `LLM_API_KEY`: API key (default: unused)

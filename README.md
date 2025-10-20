# 📰 FakeNewsRAG  
**Retrieval-Augmented Generation Pipeline for Fake News Detection**

---

### 🚀 Overview

FakeNewsRAG is an **fake-news detection system** that combines  
**Retrieval-Augmented Generation (RAG)** with **Large Language Models (LLMs)**.  
Instead of treating fake-news classification as a black-box task, the system  
**mimics a human fact-checker**:  
1. It **retrieves** relevant evidence from a labeled news corpus (fake/reliable).  
2. It **summarizes** both sides using contrastive prompts.  
3. It **classifies** the new article based on which evidence aligns more closely.

This approach makes the classification **transparent and verifiable**,  
allowing users to inspect retrieved evidence, summaries, and reasoning.

---

### 💡 Key Innovations

| Feature | Description |
|----------|-------------|
| **RAG-based Fact-Checking** | Integrates semantic retrieval (FAISS + BM25) with LLM-based reasoning. |
| **Contrastive Summaries** | Generates two opposing factual summaries — one from *fake* and one from *reliable* evidence — for side-by-side evaluation. |
| **Explainable Classification** | The model’s verdict (“Fake” or “Reliable”) is backed by textual evidence and LLM reasoning. |
| **Dynamic Cross-Encoder Re-Ranking** | Enhances retrieval accuracy using a cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`). |
| **Diversity-Aware Retrieval** | Optional Maximal Marginal Relevance (MMR) and xQuAD diversification ensure balanced evidence. |
| **Transparent UI** | Built with **Streamlit**, showing pipeline stages, live progress, and full console output. |

---

### 🧠 Pipeline Architecture
       ┌──────────────────────────────────────────────┐
       │                 Input Article                │
       │   title + content                            │
       └──────────────────────────────────────────────┘
                           │
                           ▼
                 1️⃣ Evidence Retrieval
       ┌──────────────────────────────────────────────┐
       │ Retrieve top-k fake & reliable evidence       │
       │  • FAISS dense retrieval                     │
       │  • BM25 lexical retrieval (optional)          │
       │  • Cross-encoder reranking                    │
       │  • MMR/xQuAD diversity                        │
       └──────────────────────────────────────────────┘
                           │
                           ▼
              2️⃣ Contrastive Summarization
       ┌──────────────────────────────────────────────┐
       │ Summarize each evidence set using the LLM     │
       │  → Fake summary                               │
       │  → Reliable summary                           │
       └──────────────────────────────────────────────┘
                           │
                           ▼
                 3️⃣ Classification
       ┌──────────────────────────────────────────────┐
       │ LLM compares article vs. both summaries       │
       │  → Prediction (Fake / Reliable)               │
       │  → Confidence                                │
       │  → Reasoning                                 │
       └──────────────────────────────────────────────┘
                           │
                           ▼
                 4️⃣ Streamlit Visualization
           Live stages • Console output • Verdict

         
---

### ⚙️ Setup & Installation

#### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/FakeNewsRAG.git
cd FakeNewsRAG
```

#### 2️⃣ Create and activate a virtual environment

**Windows (PowerShell):**

python -m venv .venv

.venv\Scripts\activate

**macOS / Linux:**

python3 -m venv .venv

source .venv/bin/activate

#### 3️⃣ Install dependencies
pip install -r requirements.txt

#### 🧭Running the LLM Backend

The pipeline uses a local Llama and Gemma server.

Before launching the app, make sure you have a running LLM server:
```
Llama:
python -m llama_cpp.server `
  --model ./models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf `
  --n_gpu_layers -1 --n_ctx 2048 --n_batch 256 --offload_kqv true `
  --host 127.0.0.1 --port 8011

Gemma:
python -m llama_cpp.server \
  --model /home/student/models/gemma-3-1b-it-Q4_K_M.gguf \
  --n_gpu_layers -1 \
  --n_ctx 4096 \
  --n_batch 192 \
  --offload_kqv true \
  --host 127.0.0.1 --port 8012
```

#### 🧭 Running the Streamlit App

From the project root:

```
python -m streamlit run app.py
```

#### 🧠 UI Workflow

Paste an article title and content

Click “⚖️ Analyze”

Watch dynamic stage updates:
Loading FAISS → Retrieving evidence → Generating summaries → Classifying article

View printed console output and final classification

✅ RELIABLE or 🚩 FAKE NEWS DETECTED

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
| **RAG-based Fact-Checking** | Integrates semantic retrieval (FAISS) with LLM-based reasoning. |
| **Contrastive Summaries** | Generates two opposing factual summaries — one from *fake* and one from *reliable* evidence — for side-by-side evaluation. |
| **Explainable Classification** | The model’s verdict (“Fake” or “Reliable”) is backed by textual evidence and LLM reasoning. |
| **Dynamic Cross-Encoder Re-Ranking** | Enhances retrieval accuracy using a cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`). |
| **Diversity-Aware Retrieval** | Optional Maximal Marginal Relevance (MMR) ensures balanced evidence. |
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
       │  • Cross-encoder reranking                    │
       │  • MMR diversity                              │
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

#### 2️⃣ Configure storage directory
Edit `params.env` to set `STORAGE_DIR` to your desired directory for all downloads and installations:
```bash
STORAGE_DIR=/path/to/your/storage
```
Note: The entire STORAGE_DIR takes about 70GB out of 100GB in /StudentData on our VM.
For a clean install, make sure to delete it beforehand.

#### 3️⃣ Create conda environments
```bash
bash reproduce/create_conda_envs.sh
```
This creates two conda environments:
- `$STORAGE_DIR/llama-cuda` - for running LLM servers
- `$STORAGE_DIR/rag2` - for running experiments and evaluations

#### 4️⃣ Download LLM models
Set your HuggingFace token and download models:
```bash
export HF_TOKEN=your_huggingface_token_here
bash reproduce/download_llms.sh
```

#### 5️⃣ Download data and build indices
```bash
bash reproduce/get_files_build_index.sh
```
This downloads the news dataset, preprocesses it, and builds FAISS indices for fake and reliable news.

---

### 🚀 Running Experiments

The evaluation pipeline requires two terminals running simultaneously.

#### Terminal A: LLM Server
```bash
conda activate $STORAGE_DIR/llama-cuda
bash reproduce/run_llm.sh
```

#### Terminal B: Run Experiments
```bash
conda activate $STORAGE_DIR/rag2
bash reproduce/run_experiments.sh
```

**After Terminal B finishes:**

1. **In Terminal A**: Kill the current script (Ctrl+C) and restart with 8B model:
   ```bash
   bash reproduce/run_llm.sh 8B
   ```

2. **In Terminal B**: Run reasoning support evaluation:
   ```bash
   python reproduce/run_reasoning_support_eval.py
   ```

3. **In Terminal B**: Run main evaluations:
   ```bash
   python reproduce/run_evaluations.py
   ```

4. **In Terminal B**: Generate visualizations:
   ```bash
   python reproduce/visualize_metrics.py
   ```

---

### 📊 Results

- **JSON results**: `$STORAGE_DIR/metrics_reason_support/`
- **Graphs/visualizations**: `$STORAGE_DIR/visualized/`

---

### 🧭 Running the Streamlit App

From the project root:

```bash
python -m streamlit run app.py --server.port 8888
```

#### 🧠 UI Workflow

1. Paste an article title and content
2. Click "⚖️ Analyze"
3. Watch dynamic stage updates:
   - Loading FAISS → Retrieving evidence → Generating summaries → Classifying article
4. View printed console output and final classification
5. ✅ RELIABLE or 🚩 FAKE NEWS DETECTED

---

### 📦 Dependencies

**Note:** All dependencies are automatically downloaded and installed when running `reproduce/create_conda_envs.sh` and `reproduce/download_llms.sh`. Manual installation is not required.

#### Conda Environments

The setup creates two conda environments:

**1. llama-cuda Environment** (`$STORAGE_DIR/llama-cuda`)
- Python 3.11
- **Conda packages:**
  - `cuda-nvcc=11.8.89` (NVCC compiler)
  - `cuda-toolkit=11.8.0` (CUDA toolkit)
  - `cmake`, `ninja`, `git` (build tools)
- **Python packages** (from `requirements/llama_cuda_requirements.txt`):
  - `llama-cpp-python==0.3.16` (built with CUDA support)
  - `fastapi==0.118.0`
  - `uvicorn==0.37.0`
  - `pydantic==2.11.10`
  - And other dependencies for LLM server

**2. rag2 Environment** (`$STORAGE_DIR/rag2`)
- Python 3.10
- **Conda packages:**
  - `cudatoolkit=11.8`
  - `faiss-gpu=1.7.2`
- **Python packages** (from `requirements/rag2_requirements.txt`):
  - `torch==2.1.2+cu121` (PyTorch with CUDA 12.1)
  - `torchvision==0.16.2+cu121`
  - `torchaudio==2.1.2+cu121`
  - `sentence-transformers==2.6.1`
  - `transformers==4.43.3`
  - `streamlit==1.51.0`
  - `scikit-learn==1.7.2`
  - `pandas==2.3.3`
  - `numpy<2`
  - And other dependencies for RAG pipeline

#### LLM Models

Downloaded via `reproduce/download_llms.sh` (requires `HF_TOKEN`):
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `Llama-3.2-8B-Instruct-Q4_K_M.gguf`

Location: `$STORAGE_DIR/models/`

#### Embedding & Reranking Models

These models are automatically downloaded on first use:

- **Sentence Transformer** (for embeddings):
  - `sentence-transformers/all-MiniLM-L6-v2`
  - Used for building FAISS indices and retrieval

- **Cross-Encoder** (for reranking):
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Used for reranking retrieved evidence

#### Tokenizer

- **tiktoken** with encoding `cl100k_base`
  - Used for text chunking and tokenization
  - Automatically downloaded via `tiktoken` package

---

### ⚠️ Disclaimer

This project is for **educational and research purposes only**. The fake news detection system provided here is not intended for production use or real-world fact-checking applications. Results should not be considered definitive or used as the sole basis for determining the veracity of news articles. Always verify information through multiple reliable sources and consult professional fact-checking organizations.

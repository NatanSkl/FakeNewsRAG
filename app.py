#python -m streamlit run app.py
import streamlit as st
import io
import sys
import contextlib
import logging
import time
from datetime import datetime
from pipeline.rag_pipeline import classify_article_rag
from retrieval import load_store
from common.llm_client import Llama, Mistral

# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="📰 FakeNews RAG", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center;'>📰 FakeNews RAG</h1>
    <p style='text-align:center; color:gray; font-size:16px'>
    Retrieval-Augmented Fake News Detection — step-by-step progress viewer.
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------- Defaults -----------------
STORE_DIR = "/StudentData/index"
#STORE_DIR = "./index"
LLM_URL = "http://127.0.0.1:8010/v1"
LLM_TYPE = "llama"

# ----------------- Inputs -----------------
st.subheader("1️⃣ Paste an article")
title = st.text_input("Title (optional)")
content = st.text_area("Article content", height=250, placeholder="Paste the article body here…")

# ----------------- Helpers -----------------
def disable_all_loggers():
    """Silence all loggers (so only prints appear)."""
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).disabled = True
    logging.getLogger().disabled = True


def run_pipeline_with_capture(title: str, content: str, progress_callback=None):
    """Run the RAG pipeline and capture printed output + exceptions."""
    output_capture = io.StringIO()
    result = None
    error_msg = None
    start_time = datetime.now()

    with contextlib.redirect_stdout(output_capture):
        try:
            print(f"[{start_time.strftime('%H:%M:%S')}] Starting RAG pipeline...")
            
            # Initialize LLM
            if progress_callback:
                progress_callback("Initializing LLM", 0.02)
                time.sleep(0.3)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing {LLM_TYPE.upper()} LLM at {LLM_URL}...")
            if LLM_TYPE == "llama":
                llm = Llama(LLM_URL)
            else:
                llm = Mistral(LLM_URL)
            
            # Load stores
            if progress_callback:
                progress_callback("Loading FAISS stores", 0.05)
                time.sleep(0.3)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading FAISS stores from {STORE_DIR}...")
            fake_store = load_store(f"{STORE_DIR}_fake", verbose=False)
            reliable_store = load_store(f"{STORE_DIR}_reliable", verbose=False)
            stores = {"fake": fake_store, "reliable": reliable_store}
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Stores loaded successfully!")
            
            # Run RAG pipeline with progress callback
            if progress_callback:
                progress_callback("Stores loaded, starting RAG pipeline", 0.08)
                time.sleep(0.3)
            result = classify_article_rag(
                article_title=title or "(untitled)",
                article_content=content,
                stores=stores,
                llm=llm,
                verbose=False,
                progress_callback=progress_callback
            )
        except Exception as e:
            error_msg = f"Pipeline execution error: {e}"
            import traceback
            traceback.print_exc(file=output_capture)

    return result, output_capture.getvalue(), error_msg


def update_stage(stage_text, color="blue"):
    """Display or update the current stage label dynamically."""
    emoji_map = {
        "blue": "🔵",
        "green": "🟢",
        "purple": "🟣",
        "orange": "🟠",
        "red": "🔴",
        "gray": "⚪",
    }
    stage_placeholder.markdown(
        f"<h4 style='text-align:center;color:{color};'>{emoji_map.get(color,'🔵')} Current Stage: {stage_text}</h4>",
        unsafe_allow_html=True,
    )


# ----------------- Run Button -----------------
if st.button("⚖️ Analyze"):
    if not content.strip():
        st.warning("Please paste article content first.")
        st.stop()

    disable_all_loggers()  # silence logger spam

    st.info("🧠 Running FakeNews RAG pipeline… it might take few minutes.")
    progress = st.progress(0)
    stage_placeholder = st.empty()
    
    # Define progress callback to update UI in real-time
    def progress_update(stage: str, progress_pct: float):
        """Update progress bar and stage text based on current pipeline stage."""
        # Determine color based on stage
        stage_lower = stage.lower()
        if "initializ" in stage_lower or "loading" in stage_lower or "loaded" in stage_lower:
            color = "blue"
        elif "retriev" in stage_lower:
            color = "green"
        elif "summar" in stage_lower:
            color = "purple"
        elif "classif" in stage_lower:
            color = "orange"
        elif "complete" in stage_lower:
            color = "green"
        else:
            color = "blue"
        
        # Update UI
        progress.progress(progress_pct)
        update_stage(stage, color=color)

    # ---- Run pipeline with real-time progress ----
    result, printed_output, error_msg = run_pipeline_with_capture(title, content, progress_callback=progress_update)

    progress.progress(1.0)

    # ---- Outcome ----
    st.subheader("📜 Pipeline Output")
    st.text_area("Console output", printed_output, height=300)

    if error_msg or result is None:
        update_stage("Pipeline failed!", color="red")
        st.error("❌ Pipeline failed! See printed output above for details.")
    else:
        update_stage("Pipeline completed successfully!", color="green")
        st.success("✅ Pipeline completed successfully!")

        # ---- Show classification ----
        st.subheader("2️⃣ Classification Result")
        verdict = result.classification.prediction.upper()
        conf = result.classification.confidence

        if verdict == "FAKE":
            st.markdown("### 🚩 **FAKE NEWS DETECTED**")
        else:
            st.markdown("### ✅ **RELIABLE ARTICLE**")

        st.metric("Confidence", f"{conf:.2f}")
        st.write("**Reasoning:**", result.classification.reasoning)

        st.markdown("#### 🟥 Fake Summary")
        st.write(result.fake_summary)
        st.markdown("#### 🟩 Reliable Summary")
        st.write(result.reliable_summary)

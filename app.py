#python -m streamlit run app.py
import streamlit as st
import io
import sys
import contextlib
import logging
from datetime import datetime
from pipeline.test_rag_pipeline import test_rag_pipeline

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
LLM_URL = "http://127.0.0.1:8010"
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


def run_pipeline_with_capture(title: str, content: str):
    """Run the RAG pipeline and capture printed output + exceptions."""
    output_capture = io.StringIO()
    result = None
    error_msg = None
    start_time = datetime.now()

    with contextlib.redirect_stdout(output_capture):
        try:
            print(f"[{start_time.strftime('%H:%M:%S')}] Starting RAG pipeline test...")
            result = test_rag_pipeline(
                article_title=title or "(untitled)",
                article_content=content,
                store_path=STORE_DIR,
                llm_url=LLM_URL,
                llm_type=LLM_TYPE,
                verbose=False,
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

    # ---- Stage transitions ----
    update_stage("Initializing LLM and store", color="blue")
    progress.progress(10)
    st.write("Loading FAISS index and embedding model…")

    update_stage("Retrieving evidence", color="green")
    progress.progress(35)
    st.write("Searching for relevant fake and reliable evidence chunks…")

    update_stage("Generating contrastive summaries", color="purple")
    progress.progress(60)
    st.write("Calling LLM to summarize fake vs reliable evidence…")

    update_stage("Classifying article", color="orange")
    progress.progress(85)
    st.write("Comparing summaries and deciding if article is fake or reliable…")

    # ---- Run pipeline ----
    result, printed_output, error_msg = run_pipeline_with_capture(title, content)

    progress.progress(100)

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

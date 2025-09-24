import json
from pathlib import Path
import streamlit as st

from pipeline import classify_article_rag, RAGOutput
from common.llm_client import Llama, Mistral
#python -m streamlit run app.py

# TODO add query functions to index/query
# TODO test retrieval code

st.set_page_config(page_title="FakeNews RAG — Real vs Fake", layout="wide")
st.title("📰 FakeNews RAG")

with st.sidebar:
    st.header("Index & Model")
    store_dir = st.text_input("Store directory", "mini_index/store")
    model_choice = st.selectbox("LLM", ["Llama (default)", "Mistral"])
    topn = st.slider("Evidence per label", 4, 24, 12, 1)
    st.caption("Tip: if results feel off, rebuild your mini index with bge-large, and increase top-k in retrieval.")

st.subheader("1) Paste an article")
title = st.text_input("Title (optional, helps retrieval)")
text = st.text_area("Article body", height=260, placeholder="Paste the full article here…")

run = st.button("⚖️ Analyze authenticity")

def verdict_badge(pred: str) -> str:
    if pred.lower() == "fake":
        return "🚩 **FAKE**"
    elif pred.lower() == "reliable":
        return "✅ **RELIABLE**"
    return f"ℹ️ {pred}"

if run:
    if not text or len(text.strip()) < 40:
        st.warning("Please paste at least ~40 characters of article text.")
        st.stop()

    # choose LLM client
    llm = Llama() if model_choice.startswith("Llama") else Mistral()

    with st.spinner("Retrieving evidence, summarizing, and classifying…"):
        try:
            out: RAGOutput = classify_article_rag(
                article_title=title,
                article_content=text,
                store_dir=store_dir,
                llm=llm,
                title_hint=title,
                topn_per_label=topn,
            )
        except Exception as e:
            st.exception(e)
            st.stop()


    st.subheader("2) Verdict")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.markdown(verdict_badge(out.classification.prediction))
    with c2:
        st.metric("Confidence", f"{out.classification.confidence:.2f}")
    with c3:
        st.write("**Reasoning:**", out.classification.reasoning)


    tab1, tab2 = st.tabs(["✅ Reliable evidence", "🚩 Fake evidence"])

    def _evidence_table(items):
        return [{
            "chunk_id": c.id,
            "title": c.title[:120],
            "snippet": c.text[:300],
        } for c in items]

    with tab1:
        st.markdown("#### Summary (reliable)")
        st.write(out.reliable_summary or "(no summary)")
        st.markdown("#### Evidence used")
        st.dataframe(_evidence_table(out.reliable_evidence), use_container_width=True)
        pack = {
            "label": "reliable",
            "query_title": title, "query_text": text,
            "summary": out.reliable_summary,
            "evidence": [c.__dict__ for c in out.reliable_evidence],
        }
        st.download_button("⬇️ Download reliable pack (JSON)", data=json.dumps(pack, ensure_ascii=False, indent=2),
                           file_name="reliable_pack.json", mime="application/json")

    with tab2:
        st.markdown("#### Summary (fake)")
        st.write(out.fake_summary or "(no summary)")
        st.markdown("#### Evidence used")
        st.dataframe(_evidence_table(out.fake_evidence), use_container_width=True)
        pack = {
            "label": "fake",
            "query_title": title, "query_text": text,
            "summary": out.fake_summary,
            "evidence": [c.__dict__ for c in out.fake_evidence],
        }
        st.download_button("⬇️ Download fake pack (JSON)", data=json.dumps(pack, ensure_ascii=False, indent=2),
                           file_name="fake_pack.json", mime="application/json")

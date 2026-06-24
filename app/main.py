import html
import re
from pathlib import Path

import streamlit as st

from rag_engine import (
    analyze_risks,
    summarize_contract,
    compare_contracts
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown(
    """
<style>
.main {
    padding-top: 1rem;
}

.header-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #0f172a, #1e293b);
    color: white;
    margin-bottom: 25px;
}

.answer-box {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
    margin-top: 10px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.badge {
    background-color: #dcfce7;
    color: #166534;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 10px;
    font-weight: 600;
    border: 1px solid #bbf7d0;
    overflow-wrap: anywhere;
}

.step-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    font-weight: 600;
}

.soft-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True
)

# ==================================================
# HELPERS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

def count_contracts() -> int:
    try:
        return len(list((BASE_DIR / "data").rglob("*.pdf")))
    except Exception:
        return 0

def escape_html(text: str) -> str:
    return html.escape(text or "")

def extract_first(pattern: str, text: str, default: str = "—") -> str:
    match = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return default

def extract_categories(answer: str):
    pattern = r"Risk Categories:\s*(.*?)(?:\n\s*Overall Risk Score:|\Z)"
    match = re.search(pattern, answer or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    block = match.group(1)
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("•") or line.startswith("*"):
            cleaned = line.lstrip("-•* ").strip()
            if cleaned:
                lines.append(cleaned)
    return lines

def render_file_badges(files):
    files = [f for f in files if f]
    if not files:
        st.info("No source files found.")
        return

    cols = st.columns(min(4, max(1, len(files))))
    for i, file_name in enumerate(files):
        cols[i % len(cols)].markdown(
            f"""
            <div class="badge">
                📄 {escape_html(file_name)}
            </div>
            """,
            unsafe_allow_html=True
        )

def render_answer_block(title: str, answer: str):
    st.subheader(title)
    st.markdown(
        f"""
        <div class="answer-box">
            <pre style="margin:0; white-space:pre-wrap; font-family:inherit;">{escape_html(answer or "")}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_step_guide():
    st.markdown("### Project Flow")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="step-card">1. Ask a Question</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-card">2. Contract Summary</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-card">3. Compare Contracts</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step-card">4. Review Evidence</div>', unsafe_allow_html=True)

# ==================================================
# CONTRACT COUNT
# ==================================================
total_contracts = count_contracts()

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
<div class="header-box">
    <h1>📄 Contract Risk Analyzer</h1>
    <p>
        AI-Powered Legal Contract Search & Analysis using
        ChromaDB, HuggingFace Embeddings, RAG, and Llama 3.
    </p>
</div>
""",
    unsafe_allow_html=True
)

# ==================================================
# TOP METRICS
# ==================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Contracts Indexed", total_contracts)
with col2:
    st.metric("Vector Database", "Ready")
with col3:
    st.metric("AI Model", "Llama 3")

st.divider()
render_step_guide()
st.divider()

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3 = st.tabs(
    ["🔎 Ask a Question", "🧾 Contract Summary", "⚖️ Compare Contracts"]
)

# ==================================================
# TAB 1: QUESTION ANSWERING
# ==================================================
with tab1:
    st.markdown("### Ask a Contract Question")
    query = st.text_input(
        "Ask a Contract Question",
        placeholder="Which contracts contain arbitration clauses?",
        key="qa_query"
    )

    if st.button("🔍 Analyze Contracts", key="qa_btn"):
        if not query.strip():
            st.warning("Please enter a contract question.")
            st.stop()

        with st.spinner("Analyzing contracts and retrieving evidence..."):
            result = analyze_risks(query)

        st.success("Analysis Complete")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieved Files", result.get("retrieved_count", 0))
        with col2:
            st.metric("Question Length", len(query.split()))

        st.divider()

        render_answer_block("📌 Analysis Result", result.get("answer", ""))

        st.divider()
        st.subheader("📂 Retrieved Contract Files")
        render_file_badges(result.get("files", []))

# ==================================================
# TAB 2: CONTRACT SUMMARY
# ==================================================
with tab2:
    st.markdown("### Single Contract Summary + Risk Profile")
    contract_ref = st.text_input(
        "Enter contract file name or keyword",
        placeholder="emp_01.pdf or Employee Agreement",
        key="summary_query"
    )

    if st.button("🧾 Generate Summary", key="summary_btn"):
        if not contract_ref.strip():
            st.warning("Please enter a contract name or file.")
            st.stop()

        with st.spinner("Generating contract summary and risk profile..."):
            result = summarize_contract(contract_ref)

        st.success("Analysis Complete")

        answer = result.get("answer", "")
        score = extract_first(r"Overall Risk Score:\s*([0-9]+)", answer, "—")
        level = extract_first(
            r"Overall Risk Level:\s*(Low|Medium|High)",
            answer,
            "—"
        )
        categories = extract_categories(answer)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retrieved Files", result.get("retrieved_count", 0))
        with col2:
            st.metric("Risk Score", score)
        with col3:
            st.metric("Risk Level", level)

        st.divider()

        render_answer_block("🧠 Contract Summary", answer)

        if categories:
            st.subheader("🚨 Risk Categories")
            for category in categories:
                st.markdown(f"- {category}")

        st.divider()
        st.subheader("📂 Retrieved Contract Files")
        render_file_badges(result.get("files", []))

# ==================================================
# TAB 3: CONTRACT COMPARISON
# ==================================================
with tab3:
    st.markdown("### Compare Two Contracts")
    col1, col2 = st.columns(2)

    with col1:
        contract_a = st.text_input(
            "First contract",
            placeholder="emp_01.pdf or Employee Agreement",
            key="compare_a"
        )

    with col2:
        contract_b = st.text_input(
            "Second contract",
            placeholder="SA_3.pdf or Service Agreement",
            key="compare_b"
        )

    if st.button("⚖️ Compare Contracts", key="compare_btn"):
        if not contract_a.strip() or not contract_b.strip():
            st.warning("Please enter both contract names.")
            st.stop()

        with st.spinner("Comparing contracts..."):
            result = compare_contracts(contract_a, contract_b)

        st.success("Comparison Complete")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieved Files", result.get("retrieved_count", 0))
        with col2:
            st.metric("Input Contracts", 2)

        st.divider()

        render_answer_block("📌 Comparison Result", result.get("answer", ""))

        st.divider()
        st.subheader("📂 Retrieved Contract Files")
        render_file_badges(result.get("files", []))

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Powered by ChromaDB • HuggingFace Embeddings • Groq Llama 3 • RAG")
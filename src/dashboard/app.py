"""
Market Intelligence Streamlit Dashboard
A beautiful web interface for your AI analyst agent.

Run with: streamlit run src/dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from src.agent.market_agent import ask, search_news, get_sentiment, get_pipeline_stats

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Platform",
    page_icon="📈",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────
st.title("📈 Market Intelligence Platform")
st.markdown(
    "AI-powered financial analyst — "
    "powered by FinBERT + GPT-4o + RAG"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Sentiment Check")
    st.markdown("Test your FinBERT model directly:")

    headline = st.text_input(
        "Enter a headline:",
        placeholder="NVIDIA reports record earnings...",
    )

    if st.button("Analyse Sentiment"):
        if headline:
            with st.spinner("Analysing..."):
                result = get_sentiment.invoke(headline)
            st.success(result)
        else:
            st.warning("Please enter a headline")

    st.divider()

    st.header("Pipeline Stats")
    ticker = st.text_input(
        "Check ticker:",
        placeholder="NVDA",
        value="NVDA",
    )
    if st.button("Get Stats"):
        with st.spinner("Fetching..."):
            stats = get_pipeline_stats.invoke(ticker)
        st.info(stats)

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("- FinBERT (97.79% accuracy)")
    st.markdown("- GPT-4o")
    st.markdown("- Pinecone RAG")
    st.markdown("- Apache Kafka + Spark")
    st.markdown("- Delta Lake + dbt")

# ── Main area ─────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Ask the AI Analyst")

    # Quick question buttons
    st.markdown("**Quick questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)

    with q_col1:
        if st.button("NVIDIA outlook"):
            st.session_state.question = (
                "What is the latest news and sentiment around NVIDIA?"
            )
    with q_col2:
        if st.button("Market sentiment"):
            st.session_state.question = (
                "What is the overall market sentiment right now?"
            )
    with q_col3:
        if st.button("AI stocks"):
            st.session_state.question = (
                "What is happening with AI related stocks?"
            )

    # Question input
    question = st.text_area(
        "Or ask your own question:",
        value=st.session_state.get("question", ""),
        placeholder="What is the current sentiment around NVIDIA?",
        height=100,
    )

    if st.button("Ask Agent", type="primary"):
        if question:
            with st.spinner(
                "Agent is thinking... "
                "searching news, analysing sentiment..."
            ):
                answer = ask(question)

            st.success("Analysis complete!")
            st.markdown("### Agent Response")
            st.markdown(answer)

            # Save to history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "question": question,
                "answer": answer,
            })
        else:
            st.warning("Please enter a question")

with col2:
    st.header("Recent News Search")

    search_query = st.text_input(
        "Search articles:",
        placeholder="NVIDIA earnings",
    )

    if st.button("Search"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_news.invoke(search_query)
            st.markdown(results)
        else:
            st.warning("Enter a search term")

# ── History ───────────────────────────────────────────────
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.header("Question History")
    for i, item in enumerate(
        reversed(st.session_state.history[-5:])
    ):
        with st.expander(f"Q: {item['question'][:60]}..."):
            st.markdown(item["answer"])
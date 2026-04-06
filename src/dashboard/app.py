"""
Market Intelligence Dashboard - Cloud Version
Works on Streamlit Cloud without local servers.
Uses OpenAI and Pinecone directly.
"""

import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Platform",
    page_icon="📈",
    layout="wide",
)

# ── Initialize clients ────────────────────────────────────
@st.cache_resource
def get_clients():
    # On Streamlit Cloud secrets are in st.secrets
    # Locally they are in .env file
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai_key = os.getenv("OPENAI_API_KEY", "")

    try:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
    except Exception:
        pinecone_key = os.getenv("PINECONE_API_KEY", "")

    try:
        pinecone_index_name = st.secrets["PINECONE_INDEX"]
    except Exception:
        pinecone_index_name = os.getenv(
            "PINECONE_INDEX", "market-intel"
        )

    if not openai_key:
        st.error("OPENAI_API_KEY not found in secrets!")
        st.stop()

    if not pinecone_key:
        st.error("PINECONE_API_KEY not found in secrets!")
        st.stop()

    openai_client = OpenAI(api_key=openai_key)
    pc            = Pinecone(api_key=pinecone_key)
    index         = pc.Index(pinecone_index_name)

    return openai_client, index

def get_embedding(text: str, client: OpenAI) -> list:
    """Convert text to vector embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def search_articles(query: str, index, client: OpenAI) -> list:
    """Search Pinecone for relevant articles."""
    vector = get_embedding(query, client)
    results = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True,
    )
    return results.matches


def analyze_sentiment(text: str, client: OpenAI) -> dict:
    """
    Use GPT-4o to analyze sentiment when
    local FinBERT server is not available.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment analyzer. "
                    "Classify the sentiment of financial text as "
                    "POSITIVE, NEUTRAL, or NEGATIVE. "
                    "Reply with exactly: "
                    "LABEL|CONFIDENCE where confidence is 0-100."
                    "Example: POSITIVE|95"
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0,
        max_tokens=20,
    )
    result = response.choices[0].message.content.strip()
    parts  = result.split("|")
    label  = parts[0] if len(parts) > 0 else "NEUTRAL"
    conf   = parts[1] if len(parts) > 1 else "50"
    return {
        "label":      label,
        "confidence": f"{conf}%",
    }


def ask_agent(question: str, client: OpenAI, index) -> str:
    """
    Ask the AI agent a financial question.
    Searches Pinecone for context then asks GPT-4o.
    """
    # Search for relevant articles
    matches = search_articles(question, index, client)

    # Build context from retrieved articles
    context = ""
    if matches:
        context = "Relevant financial news:\n"
        for m in matches:
            title = m.metadata.get("title", "")
            source = m.metadata.get("source", "")
            score = round(m.score, 3)
            context += f"- [{score}] {title} ({source})\n"
    else:
        context = "No specific articles found in the database."

    # Ask GPT-4o with the context
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst AI. "
                    "Answer questions using the provided news context. "
                    "Be specific, structured, and insightful. "
                    "Always mention sentiment and key signals."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ── Header ────────────────────────────────────────────────
st.title("📈 Market Intelligence Platform")
st.markdown(
    "AI-powered financial analyst — "
    "GPT-4o + Pinecone RAG + FinBERT Sentiment"
)
st.divider()

# ── Load clients ──────────────────────────────────────────
try:
    openai_client, pinecone_index = get_clients()
except Exception as e:
    st.error(f"Connection error: {e}")
    st.info(
        "Make sure OPENAI_API_KEY and PINECONE_API_KEY "
        "are set in Streamlit secrets."
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Sentiment Check")
    st.markdown("Analyze any financial headline:")

    headline = st.text_input(
        "Enter a headline:",
        placeholder="NVIDIA reports record earnings...",
    )

    if st.button("Analyse Sentiment"):
        if headline:
            with st.spinner("Analysing..."):
                result = analyze_sentiment(
                    headline, openai_client
                )
            st.success(
                f"Sentiment: {result['label']} "
                f"({result['confidence']} confidence)"
            )
        else:
            st.warning("Please enter a headline")

    st.divider()
    st.header("Article Search")
    search_query = st.text_input(
        "Search news:",
        placeholder="NVIDIA earnings",
    )
    if st.button("Search"):
        if search_query:
            with st.spinner("Searching Pinecone..."):
                matches = search_articles(
                    search_query,
                    pinecone_index,
                    openai_client,
                )
            if matches:
                for m in matches:
                    title  = m.metadata.get("title", "No title")
                    source = m.metadata.get("source", "unknown")
                    score  = round(m.score, 3)
                    st.markdown(
                        f"**[{score}]** {title} *({source})*"
                    )
            else:
                st.info("No articles found")

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("- FinBERT (97.79% accuracy)")
    st.markdown("- GPT-4o")
    st.markdown("- Pinecone RAG")
    st.markdown("- Apache Kafka + Spark")
    st.markdown("- Delta Lake + dbt")
    st.markdown("- LangChain ReAct Agent")

# ── Main area ─────────────────────────────────────────────
st.header("Ask the AI Analyst")

# Quick question buttons
st.markdown("**Quick questions:**")
q1, q2, q3 = st.columns(3)

with q1:
    if st.button("NVIDIA outlook"):
        st.session_state.question = (
            "What is the latest news and "
            "sentiment around NVIDIA?"
        )
with q2:
    if st.button("Market sentiment"):
        st.session_state.question = (
            "What is the overall stock market "
            "sentiment right now?"
        )
with q3:
    if st.button("AI stocks"):
        st.session_state.question = (
            "What is happening with "
            "AI related stocks?"
        )

# Question input
question = st.text_area(
    "Or ask your own question:",
    value=st.session_state.get("question", ""),
    placeholder=(
        "What is the current sentiment around NVIDIA?"
    ),
    height=100,
)

if st.button("Ask Agent", type="primary"):
    if question:
        with st.spinner(
            "Searching news, analysing sentiment, "
            "generating response..."
        ):
            answer = ask_agent(
                question,
                openai_client,
                pinecone_index,
            )

        st.success("Analysis complete!")
        st.markdown("### Agent Response")
        st.markdown(answer)

        # Save to history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "question": question,
            "answer":   answer,
        })
    else:
        st.warning("Please enter a question")

# ── History ───────────────────────────────────────────────
if (
    "history" in st.session_state
    and st.session_state.history
):
    st.divider()
    st.header("Question History")
    for item in reversed(
        st.session_state.history[-5:]
    ):
        with st.expander(
            f"Q: {item['question'][:60]}..."
        ):
            st.markdown(item["answer"])
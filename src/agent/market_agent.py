"""
Market Intelligence LangChain Agent
A ReAct agent that uses 3 tools to answer
financial questions:
  1. search_news - finds relevant articles from Pinecone
  2. get_sentiment - scores sentiment via your FinBERT API
  3. get_pipeline_stats - gets article volume stats

Run with: python src/agent/market_agent.py
"""

import os
import json
import logging
import requests
import pandas as pd
import glob
from openai import OpenAI
from pinecone import Pinecone
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "market-intel")
SENTIMENT_URL    = os.getenv(
    "SENTIMENT_API_URL",
    "http://localhost:8000"
)
DELTA_PATH       = os.getenv(
    "DELTA_PATH",
    "../market-intel-data-pipeline/data/delta/articles/news"
)
EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Global clients ────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(INDEX_NAME)


# ── Tool 1: Search news ───────────────────────────────────
@tool
def search_news(query: str) -> str:
    """
    Search for relevant financial news articles
    related to the query. Use this to find recent
    news about specific stocks, companies, or
    market events. Input should be a search query
    like 'NVIDIA earnings' or 'Federal Reserve rates'.
    """
    try:
        # Convert query to embedding
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        query_vector = response.data[0].embedding

        # Search Pinecone for similar articles
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
        )

        if not results.matches:
            return "No relevant articles found."

        # Format results
        articles = []
        for match in results.matches:
            meta  = match.metadata
            score = round(match.score, 3)
            articles.append(
                f"- [{score}] {meta.get('title', 'No title')} "
                f"(Source: {meta.get('source', 'unknown')}, "
                f"Type: {meta.get('content_type', 'news')})"
            )

        return (
            f"Found {len(articles)} relevant articles:\n"
            + "\n".join(articles)
        )

    except Exception as e:
        return f"Search error: {str(e)}"


# ── Tool 2: Get sentiment ─────────────────────────────────
@tool
def get_sentiment(text: str) -> str:
    """
    Analyze the sentiment of a financial text or
    headline using the fine-tuned FinBERT model.
    Returns positive, neutral, or negative sentiment
    with a confidence score. Input should be a
    financial headline or short text to analyze.
    """
    try:
        response = requests.post(
            f"{SENTIMENT_URL}/predict",
            json={"texts": [text]},
            timeout=10,
        )
        response.raise_for_status()
        data   = response.json()
        result = data["results"][0]

        return (
            f"Sentiment: {result['label'].upper()} "
            f"(confidence: {result['confidence']:.1%})\n"
            f"Breakdown: "
            f"Positive={result['probabilities']['positive']:.1%}, "
            f"Neutral={result['probabilities']['neutral']:.1%}, "
            f"Negative={result['probabilities']['negative']:.1%}"
        )

    except requests.exceptions.ConnectionError:
        return (
            "Sentiment API not available. "
            "Start it with: uvicorn src.serving.local_server:app "
            "--port 8000 (in market-intel-mlops repo)"
        )
    except Exception as e:
        return f"Sentiment error: {str(e)}"


# ── Tool 3: Pipeline stats ────────────────────────────────
@tool
def get_pipeline_stats(ticker: str) -> str:
    """
    Get article volume statistics for a specific
    stock ticker from the data pipeline. Shows how
    many articles mention this ticker and recent
    activity. Input should be a ticker symbol
    like 'NVDA', 'AAPL', or 'TSLA'.
    """
    try:
        files = glob.glob(
            f"{DELTA_PATH}/*.snappy.parquet"
        )
        if not files:
            return "No pipeline data available."

        dfs = [pd.read_parquet(f) for f in files]
        df  = pd.concat(dfs, ignore_index=True)

        # Filter articles mentioning this ticker
        ticker_upper = ticker.upper()
        mask = df["tickers_mentioned"].apply(
            lambda x: ticker_upper in (x if x is not None else [])
            if hasattr(x, "__iter__")
            else False
        )
        ticker_df = df[mask]

        total     = len(df)
        mentioned = len(ticker_df)

        if mentioned == 0:
            return (
                f"No articles found mentioning {ticker_upper} "
                f"in the pipeline. "
                f"Total articles in pipeline: {total}"
            )

        return (
            f"Pipeline stats for {ticker_upper}:\n"
            f"- Total articles in pipeline: {total}\n"
            f"- Articles mentioning {ticker_upper}: {mentioned}\n"
            f"- Coverage: {mentioned/total:.1%} of all articles\n"
            f"- Most recent: "
            f"{ticker_df['ingested_at'].max()}"
        )

    except Exception as e:
        return f"Pipeline stats error: {str(e)}"


# ── Agent prompt ──────────────────────────────────────────
AGENT_PROMPT = PromptTemplate.from_template("""
You are an expert financial analyst AI assistant.
You have access to real-time financial news data
and a sentiment analysis model.

Answer the user's question by using your tools
to gather relevant information, then provide a
clear, structured analyst-style response.

Always:
1. Search for relevant news first
2. Analyze sentiment of key headlines
3. Check pipeline stats for volume signals
4. Synthesize into a clear answer

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your complete analyst response

Question: {input}
Thought: {agent_scratchpad}
""")


# ── Build agent ───────────────────────────────────────────
def build_agent() -> AgentExecutor:
    """Build the ReAct agent with all 3 tools."""
    llm   = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    tools = [search_news, get_sentiment, get_pipeline_stats]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=AGENT_PROMPT,
    )

    return AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    max_execution_time=60,
    handle_parsing_errors=True,
)


# ── Ask the agent ─────────────────────────────────────────
def ask(question: str) -> str:
    """Ask the agent a financial question."""
    log.info(f"Question: {question}")
    agent = build_agent()

    try:
        result = agent.invoke({"input": question})
        return result["output"]
    except Exception as e:
        log.error(f"Agent error: {e}")
        return f"Error: {str(e)}"


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nMarket Intelligence Agent")
    print("=" * 40)

    questions = [
        "What is the current market sentiment?",
        "What are the latest news about NVIDIA?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        answer = ask(question)
        print(f"\nAnswer: {answer}")
        print("=" * 40)
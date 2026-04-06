# market-intel-ai-agent# Market Intel — AI Agent

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![GPT4o](https://img.shields.io/badge/GPT--4o-OpenAI-black)
![Pinecone](https://img.shields.io/badge/Pinecone-RAG-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

LangChain ReAct agent with RAG pipeline, FinBERT
sentiment tools, and Streamlit dashboard.
Part 3 of 3 in the Market Intelligence Platform.

## Architecture
```
Question ──► LangChain ReAct Agent (GPT-4o)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    search_news  get_sentiment  get_pipeline_stats
          │           │           │
       Pinecone    FinBERT      Delta Lake
       (46 vectors) (97.79%)   (parquet files)
          │           │           │
          └───────────┴───────────┘
                      │
                      ▼
             Analyst Quality Answer
                      │
                      ▼
             Streamlit Dashboard
```

## What This Does

An AI analyst agent that answers natural language
financial questions by combining:
- Semantic search over real financial news (RAG)
- Fine-tuned FinBERT sentiment classification
- Real-time pipeline volume statistics

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | Agent framework and tool orchestration |
| GPT-4o | Large language model brain |
| Pinecone | Vector database for semantic search |
| OpenAI Embeddings | text-embedding-3-small |
| FastAPI | REST API backend |
| Streamlit | Interactive web dashboard |

## Quick Start

### 1. Prerequisites
Make sure Phase 1 and Phase 2 are running:
- Kafka + Spark pipeline (Phase 1)
- FinBERT inference server on port 8000 (Phase 2)

### 2. Setup
```bash
git clone https://github.com/Nag4535/market-intel-ai-agent
cd market-intel-ai-agent
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add OpenAI and Pinecone API keys to .env
```

### 3. Embed Articles into Pinecone
```bash
python src/embeddings/embed_articles.py
```

### 4. Test the Agent
```bash
python src/agent/market_agent.py
```

### 5. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

Open browser at:
```
http://localhost:8501
```

## Project Structure
```
market-intel-ai-agent/
├── src/
│   ├── embeddings/
│   │   └── embed_articles.py   # Delta Lake → Pinecone
│   ├── agent/
│   │   └── market_agent.py     # LangChain ReAct agent
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Agent Tools

### search_news
Semantic search over financial articles stored
in Pinecone. Finds relevant news by meaning
not just keywords.
```python
search_news("NVIDIA earnings beat expectations")
# Returns top 5 most semantically similar articles
```

### get_sentiment
Calls your fine-tuned FinBERT model to classify
sentiment of any financial text.
```python
get_sentiment("NVIDIA stock surges 10 percent")
# Returns: POSITIVE (confidence: 99.63%)
```

### get_pipeline_stats
Queries Delta Lake to get volume statistics
for any stock ticker.
```python
get_pipeline_stats("NVDA")
# Returns: article count, coverage percentage,
# most recent mention timestamp
```

## Example Interaction

**Question:**
```
What is the latest news and sentiment around NVIDIA?
```

**Agent Response:**
```
Recent news about NVIDIA indicates strong positive
sentiment driven by record revenue and AI chip demand.
Key headlines show stock surging after earnings beat
with 97%+ confidence scores. Pipeline data shows
NVIDIA mentioned in 63.8% of all articles suggesting
significant market attention. Some caution around
H100 GPU supply constraints.
```

## API Keys Required

| Service | URL | Cost |
|---|---|---|
| OpenAI | platform.openai.com | ~$2 total |
| Pinecone | pinecone.io | Free tier |
| LangSmith | smith.langchain.com | Free tier |

## Key Engineering Decisions

**Why RAG?**
GPT-4o alone has no knowledge of your specific
pipeline data. RAG grounds every answer in real
articles from your Kafka pipeline — no hallucination.

**Why ReAct agent?**
ReAct (Reasoning + Acting) lets the agent decide
which tools to use and in what order based on
the question. More flexible than hardcoded chains.

**Why Pinecone?**
Serverless vector database — no infrastructure
to manage. Free tier handles thousands of vectors
which is more than enough for this project.

## Part of Larger System

- [market-intel-data-pipeline](https://github.com/Nag4535/market-intel-data-pipeline) — data source
- [market-intel-mlops](https://github.com/Nag4535/market-intel-mlops) — sentiment model
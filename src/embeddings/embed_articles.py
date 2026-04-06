"""
Embedding Pipeline
Reads articles from your Delta Lake pipeline,
converts them into vector embeddings using OpenAI,
and stores them in Pinecone for semantic search.

Run with: python src/embeddings/embed_articles.py
"""

import os
import glob
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
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
EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)
DELTA_PATH       = os.getenv(
    "DELTA_PATH",
    "../market-intel-data-pipeline/data/delta/articles/news"
)

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIM = 1536


def load_articles() -> pd.DataFrame:
    """
    Load articles from Delta Lake parquet files.
    """
    files = glob.glob(f"{DELTA_PATH}/*.snappy.parquet")

    if not files:
        raise FileNotFoundError(
            f"No parquet files found at {DELTA_PATH}. "
            "Make sure Phase 1 pipeline is running."
        )

    log.info(f"Found {len(files)} parquet files")
    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)

    # Remove duplicates and empty titles
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.len() > 10]
    df = df.drop_duplicates(subset=["event_id"])

    log.info(f"Loaded {len(df)} unique articles")
    return df


def setup_pinecone() -> any:
    """
    Connect to Pinecone and create index if needed.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        log.info(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        log.info("Index created!")
    else:
        log.info(f"Index '{INDEX_NAME}' already exists")

    return pc.Index(INDEX_NAME)


def get_embeddings(
    texts: list[str],
    client: OpenAI,
) -> list[list[float]]:
    """
    Convert a list of texts into vector embeddings
    using OpenAI's text-embedding-3-small model.

    Each text becomes a list of 1536 numbers that
    represents its meaning in vector space.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_and_store(
    df: pd.DataFrame,
    index: any,
    client: OpenAI,
):
    """
    Embed all articles and store in Pinecone.
    Processes in batches of 100 to stay within
    API rate limits.
    """
    # Check which articles are already in Pinecone
    existing_ids = set()
    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        log.info(f"Pinecone currently has {total} vectors")
    except Exception:
        pass

    batch_size = 100
    total_upserted = 0

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]

        # Prepare texts for embedding
        texts = batch["title"].tolist()
        ids   = batch["event_id"].tolist()

        log.info(
            f"Embedding batch {i//batch_size + 1} "
            f"({len(texts)} articles)..."
        )

        try:
            # Get embeddings from OpenAI
            embeddings = get_embeddings(texts, client)

            # Prepare vectors for Pinecone
            vectors = []
            for j, (event_id, text, embedding) in enumerate(
                zip(ids, texts, embeddings)
            ):
                row = batch.iloc[j]
                vectors.append({
                    "id": event_id,
                    "values": embedding,
                    # Store metadata for retrieval
                    "metadata": {
                        "title":       text[:500],
                        "source":      str(row.get("source", "")),
                        "content_type": str(
                            row.get("content_type", "news")
                        ),
                        "ingested_at": str(
                            row.get("ingested_at", "")
                        ),
                    }
                })

            # Upsert to Pinecone
            index.upsert(vectors=vectors)
            total_upserted += len(vectors)
            log.info(
                f"Upserted {len(vectors)} vectors to Pinecone"
            )

        except Exception as e:
            log.error(f"Error in batch {i//batch_size + 1}: {e}")
            continue

    log.info(
        f"Done! Total vectors upserted: {total_upserted}"
    )
    return total_upserted


def run():
    """Main embedding pipeline."""
    log.info("Starting embedding pipeline...")

    # Load articles
    df = load_articles()

    # Setup clients
    client = OpenAI(api_key=OPENAI_API_KEY)
    index  = setup_pinecone()

    # Embed and store
    total = embed_and_store(df, index, client)

    log.info(f"Embedding pipeline complete!")
    log.info(f"Total vectors in Pinecone: {total}")
    log.info(
        "Next step: run the agent with "
        "python src/agent/market_agent.py"
    )


if __name__ == "__main__":
    run()
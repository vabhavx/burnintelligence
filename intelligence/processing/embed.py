"""
BurnTheLies Intelligence — Embedding Pipeline
Uses paraphrase-multilingual-MiniLM-L12-v2 for CPU-friendly multilingual embeddings.
50M params, 384 dimensions, ~800 items/minute on CPU.
"""

import json
import logging
import time
import numpy as np
from typing import Optional

logger = logging.getLogger("intelligence.embed")

# Lazy-loaded model (heavy import)
_model = None
_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_model():
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_model_name}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of texts into 384-dim vectors.
    Returns numpy array of shape (n, 384).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings


def embed_and_store(db_conn, post_ids: list[str] = None,
                    batch_size: int = 64) -> int:
    """
    Embed posts that don't have embeddings yet and store them.
    Returns count of newly embedded posts.
    """
    # Find posts without embeddings
    if post_ids:
        placeholders = ",".join("?" * len(post_ids))
        rows = db_conn.execute(f"""
            SELECT rp.id, rp.title, rp.content
            FROM raw_posts rp
            LEFT JOIN embeddings e ON e.post_id = rp.id
            WHERE e.post_id IS NULL AND rp.id IN ({placeholders})
        """, post_ids).fetchall()
    else:
        rows = db_conn.execute("""
            SELECT rp.id, rp.title, rp.content
            FROM raw_posts rp
            LEFT JOIN embeddings e ON e.post_id = rp.id
            WHERE e.post_id IS NULL
            ORDER BY rp.ingested_at DESC
            LIMIT 5000
        """).fetchall()

    if not rows:
        logger.info("No posts need embedding")
        return 0

    logger.info(f"Embedding {len(rows)} posts...")

    # Prepare texts — combine title + content for richer representation
    ids = [r["id"] for r in rows]
    texts = [f"{r['title'] or ''} {r['content']}" for r in rows]

    # Batch embed
    vectors = embed_texts(texts, batch_size=batch_size)

    # Store in batched transactions of 200 to amortize commit overhead.
    write_start = time.monotonic()
    WRITE_BATCH = 200
    batch: list[tuple[str, str]] = []
    batches_committed = 0

    def _flush(rows: list[tuple[str, str]]) -> None:
        with db_conn:
            db_conn.executemany(
                "INSERT OR REPLACE INTO embeddings (post_id, vector) VALUES (?, ?)",
                rows,
            )

    for post_id, vector in zip(ids, vectors):
        batch.append((post_id, json.dumps(vector.tolist())))
        if len(batch) >= WRITE_BATCH:
            _flush(batch)
            batches_committed += 1
            batch = []

    if batch:
        _flush(batch)
        batches_committed += 1

    elapsed = time.monotonic() - write_start
    logger.info(
        f"embedded {len(ids)} posts in {batches_committed} batches in {elapsed:.2f} seconds"
    )
    return len(ids)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_all_embeddings(db_conn) -> tuple[list[str], np.ndarray]:
    """
    Load all embeddings from DB for clustering.
    Returns (post_ids, vectors_matrix).
    """
    rows = db_conn.execute("""
        SELECT e.post_id, e.vector
        FROM embeddings e
        JOIN raw_posts rp ON rp.id = e.post_id
        WHERE rp.ingested_at >= datetime('now', '-48 hours')
        ORDER BY rp.ingested_at DESC
    """).fetchall()

    if not rows:
        return [], np.array([])

    ids = [r["post_id"] for r in rows]
    vectors = np.array([json.loads(r["vector"]) for r in rows])

    return ids, vectors

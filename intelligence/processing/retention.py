"""
Retention + bounded-growth policy.

Without this, a stranger who clones the repo and leaves it running will have a
stuck 50 GB database within 30 days. The policy caps growth, prunes stale
data, checkpoints the WAL, and pauses ingest when disk is critically low.

Runs as a scheduled maintenance stage inside the continuous pipeline cycle.
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("intelligence.retention")


# ─── Tunables ────────────────────────────────────────────────────────────────

POST_RETENTION_DAYS = 30        # archive raw_posts older than this
CLUSTER_RETENTION_DAYS = 14     # purge archived clusters + their members
EMBEDDING_RETENTION_DAYS = 30   # delete embeddings for archived posts
DNA_MATCHES_PER_CLUSTER = 100   # hard ceiling: keep only top-K per cluster side
MIN_FREE_DISK_GB = 1            # pause ingest below this
WAL_CHECKPOINT_INTERVAL = 3600  # seconds between forced checkpoints


_last_checkpoint = 0.0


# ─── Public API ───────────────────────────────────────────────────────────────


def maintenance_cycle(db_conn: sqlite3.Connection, *, force: bool = False) -> dict:
    """
    Run one maintenance pass. Idempotent; safe to call every pipeline cycle.
    All operations use small transactions to avoid locking the DB for more
    than a few seconds at a time.
    """
    global _last_checkpoint
    stats: dict = {}

    t0 = time.time()

    # 1. Disk watchdog — cheapest check, do first
    stats["disk"] = _check_disk()

    # 2. WAL checkpoint — keep the write-ahead log from ballooning
    now_ts = time.time()
    if force or (now_ts - _last_checkpoint) >= WAL_CHECKPOINT_INTERVAL:
        stats["wal"] = _checkpoint_wal(db_conn)
        _last_checkpoint = now_ts
    else:
        stats["wal"] = {"skipped": True}

    # 3. Archive old posts + delete stale embeddings
    stats["posts"] = _archive_old_posts(db_conn)
    stats["embeddings"] = _prune_embeddings(db_conn)

    # 4. Purge old archived clusters
    stats["clusters"] = _purge_old_clusters(db_conn)

    # 5. Cap dna_matches per cluster
    stats["dna"] = _cap_dna_matches(db_conn)

    stats["elapsed"] = round(time.time() - t0, 2)
    logger.info("Maintenance complete: %s", stats)
    return stats


def should_pause_ingest() -> bool:
    """Returns True if ingest should be paused due to disk pressure."""
    return _check_disk().get("free_gb", 999) < MIN_FREE_DISK_GB


# ─── Internals ────────────────────────────────────────────────────────────────


def _check_disk() -> dict:
    """Check free disk space on the volume containing the DB."""
    try:
        # The DB lives in intelligence/data/; check that volume
        db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        usage = shutil.disk_usage(os.path.abspath(db_dir))
        free_gb = usage.free / (1024 ** 3)
        return {
            "free_gb": round(free_gb, 2),
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "pct_used": round(100 * usage.used / usage.total, 1),
            "paused": free_gb < MIN_FREE_DISK_GB,
        }
    except Exception:
        return {"free_gb": 999, "error": "disk check failed"}


def _checkpoint_wal(db_conn: sqlite3.Connection) -> dict:
    """Force a WAL checkpoint to keep the WAL file bounded."""
    t0 = time.time()
    try:
        db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        elapsed = time.time() - t0
        # Read WAL size after checkpoint
        db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        wal_path = os.path.join(os.path.abspath(db_dir), "intelligence.db-wal")
        wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        return {
            "wal_size_mb": round(wal_size / (1024 ** 2), 1),
            "elapsed": round(elapsed, 2),
        }
    except Exception:
        return {"error": "checkpoint failed"}


def _archive_old_posts(db_conn: sqlite3.Connection) -> dict:
    """
    Mark posts older than POST_RETENTION_DAYS as archived so they stop
    being picked up by the clustering/embedding pipeline. The raw text
    stays for reference but the heavy embedding row will be cleaned
    separately.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=POST_RETENTION_DAYS)).isoformat()
    try:
        # Archive posts
        cur = db_conn.execute(
            "UPDATE raw_posts SET metadata = json_set("
            "  COALESCE(metadata, '{}'), '$.archived', 1, '$.archived_at', ?"
            ") WHERE ingested_at < ? AND (metadata->>'$.archived' IS NULL OR metadata->>'$.archived' = '0')",
            (datetime.now(timezone.utc).isoformat(), cutoff),
        )
        db_conn.commit()
        if cur.rowcount and cur.rowcount > 0:
            logger.info("Archived %d posts older than %s", cur.rowcount, cutoff)
        return {"archived": cur.rowcount or 0, "cutoff": cutoff}
    except Exception as e:
        logger.warning("Archive posts failed: %s", e)
        return {"error": str(e)}


def _prune_embeddings(db_conn: sqlite3.Connection) -> dict:
    """Delete embeddings for archived posts to reclaim space."""
    try:
        cur = db_conn.execute("""
            DELETE FROM embeddings WHERE post_id IN (
                SELECT id FROM raw_posts
                WHERE metadata->>'$.archived' = '1'
            )
        """)
        db_conn.commit()
        return {"deleted": cur.rowcount or 0}
    except Exception as e:
        logger.warning("Prune embeddings failed: %s", e)
        return {"error": str(e)}


def _purge_old_clusters(db_conn: sqlite3.Connection) -> dict:
    """
    Purge clusters that were archived more than CLUSTER_RETENTION_DAYS ago.
    Also cleans up their cluster_members rows and any dangling NVI snapshots
    and coordination signals.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CLUSTER_RETENTION_DAYS)).isoformat()
    stats = {}
    try:
        # Find clusters to purge
        old = db_conn.execute(
            "SELECT id FROM narrative_clusters WHERE status = 'archived' AND lifecycle_updated < ?",
            (cutoff,),
        ).fetchall()
        ids = [r[0] for r in old]

        if not ids:
            stats["purged"] = 0
            return stats

        placeholders = ",".join("?" * len(ids))

        # Delete in dependency order
        for table in [
            "cluster_members", "nvi_snapshots", "coordination_signals",
            "narrative_dna", "dna_matches",
            "evidence_packs", "amplification_graph_snapshots",
        ]:
            try:
                cur = db_conn.execute(
                    f"DELETE FROM {table} WHERE cluster_id IN ({placeholders})"
                    if table != "dna_matches" else
                    f"DELETE FROM {table} WHERE cluster_a IN ({placeholders}) OR cluster_b IN ({placeholders})",
                    ids + ids if table == "dna_matches" else ids,
                )
                db_conn.commit()
                if cur.rowcount:
                    stats.setdefault("detail", {})[table] = cur.rowcount
            except Exception:
                pass  # table may not exist or column names differ

        cur = db_conn.execute(
            f"DELETE FROM narrative_clusters WHERE id IN ({placeholders})", ids
        )
        db_conn.commit()
        stats["purged"] = cur.rowcount or 0
        logger.info("Purged %d old archived clusters", stats["purged"])
        return stats
    except Exception as e:
        logger.warning("Purge clusters failed: %s", e)
        return {"error": str(e)}


def _cap_dna_matches(db_conn: sqlite3.Connection) -> dict:
    """
    Keep only the top-K dna_matches per cluster (by match_score).
    Without this, dna_matches grows at O(n^2) and the table becomes
    the single largest file on disk within weeks.
    """
    try:
        # Find clusters that exceed the cap
        over = db_conn.execute("""
            SELECT cluster_a, COUNT(*) as cnt
            FROM dna_matches GROUP BY cluster_a HAVING cnt > ?
        """, (DNA_MATCHES_PER_CLUSTER,)).fetchall()

        deleted = 0
        for row in over:
            # Keep the top K by match score for this cluster
            keep = db_conn.execute(
                "SELECT id FROM dna_matches WHERE cluster_a = ? "
                "ORDER BY match_score DESC LIMIT ?",
                (row[0], DNA_MATCHES_PER_CLUSTER),
            ).fetchall()
            keep_ids = [r[0] for r in keep]
            if not keep_ids:
                continue
            placeholders = ",".join("?" * len(keep_ids))
            cur = db_conn.execute(
                f"DELETE FROM dna_matches WHERE cluster_a = ? AND id NOT IN ({placeholders})",
                [row[0]] + keep_ids,
            )
            db_conn.commit()
            deleted += cur.rowcount or 0

        if deleted:
            logger.info("Capped dna_matches: removed %d excess rows", deleted)
        return {"capped_deleted": deleted}
    except Exception as e:
        logger.warning("DNA cap failed: %s", e)
        return {"error": str(e)}

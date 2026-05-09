"""
BurnTheLies Intelligence — Retention

Bounds growth of high-volume tables (dna_matches, nvi_snapshots) and runs
periodic VACUUM. Pure functions: each returns counts, no logging side
effects. Orchestrated by run_retention_cycle.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

from .db import set_pipeline_state


DEFAULT_DNA_MIN_SCORE = float(os.environ.get("INTEL_DNA_MIN_SCORE", "0.50"))
DEFAULT_NVI_RETENTION_DAYS = int(os.environ.get("INTEL_NVI_RETENTION_DAYS", "90"))


def prune_dna_matches(conn: sqlite3.Connection,
                      min_score: float = DEFAULT_DNA_MIN_SCORE) -> int:
    """Delete dna_matches rows below the score threshold. Returns rowcount."""
    cur = conn.execute(
        "DELETE FROM dna_matches WHERE match_score < ?", (min_score,)
    )
    deleted = cur.rowcount or 0
    conn.commit()
    return deleted


def archive_old_nvi_snapshots(
    conn: sqlite3.Connection,
    retention_days: int = DEFAULT_NVI_RETENTION_DAYS,
) -> int:
    """Delete nvi_snapshots older than the cutoff, but always preserve the
    latest snapshot per cluster (so a cluster never loses its only history).
    Returns rowcount.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")

    cur = conn.execute(
        """
        DELETE FROM nvi_snapshots
        WHERE timestamp < ?
          AND id NOT IN (
              SELECT MAX(id) FROM nvi_snapshots GROUP BY cluster_id
          )
        """,
        (cutoff,),
    )
    deleted = cur.rowcount or 0
    conn.commit()
    return deleted


def vacuum_db(conn: sqlite3.Connection) -> None:
    """VACUUM the database. Caller decides cadence."""
    conn.commit()
    conn.execute("VACUUM")


def run_retention_cycle(conn: sqlite3.Connection) -> dict:
    """Run prune + archive, and VACUUM only on Sunday (UTC) to avoid daily
    lock thrash. Records counters into pipeline_state.
    """
    pruned = prune_dna_matches(conn)
    archived = archive_old_nvi_snapshots(conn)

    now = datetime.now(timezone.utc)
    vacuumed = False
    if now.weekday() == 6:  # Sunday
        vacuum_db(conn)
        vacuumed = True
        set_pipeline_state(conn, "last_vacuum_run", now.isoformat())

    set_pipeline_state(conn, "last_retention_run", now.isoformat())
    set_pipeline_state(conn, "last_dna_pruned_count", str(pruned))

    return {
        "pruned_dna": pruned,
        "archived_nvi": archived,
        "vacuumed": vacuumed,
    }

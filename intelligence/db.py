"""
BurnTheLies Intelligence — Database Layer
SQLite with vector storage for narrative clustering.
Zero dependencies beyond stdlib + numpy.
"""

import sqlite3
import json
import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "intelligence.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None):
    """Create all tables. Idempotent."""
    if conn is None:
        conn = get_connection()

    conn.executescript("""
        -- Raw ingested posts from all sources
        CREATE TABLE IF NOT EXISTS raw_posts (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,          -- 'gdelt', 'bluesky', '4chan', 'telegram'
            url TEXT,
            title TEXT,
            content TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            author TEXT,
            published_at TEXT,
            ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            metadata JSON,
            content_hash TEXT NOT NULL,
            UNIQUE(content_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_raw_posts_source ON raw_posts(source);
        CREATE INDEX IF NOT EXISTS idx_raw_posts_published ON raw_posts(published_at);
        CREATE INDEX IF NOT EXISTS idx_raw_posts_ingested ON raw_posts(ingested_at);

        -- Embeddings stored as JSON arrays (SQLite has no native vector type)
        CREATE TABLE IF NOT EXISTS embeddings (
            post_id TEXT PRIMARY KEY REFERENCES raw_posts(id),
            vector JSON NOT NULL,
            model TEXT NOT NULL DEFAULT 'paraphrase-multilingual-MiniLM-L12-v2',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        -- Narrative clusters discovered by HDBSCAN
        CREATE TABLE IF NOT EXISTS narrative_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            keywords JSON,                -- top c-TF-IDF terms
            summary TEXT,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            post_count INTEGER DEFAULT 0,
            source_diversity REAL DEFAULT 0.0,  -- Shannon entropy across sources
            language_spread JSON,          -- {lang: count}
            status TEXT DEFAULT 'active',  -- 'active', 'merged', 'dead'
            metadata JSON
        );

        CREATE INDEX IF NOT EXISTS idx_clusters_status ON narrative_clusters(status);
        CREATE INDEX IF NOT EXISTS idx_clusters_first_seen ON narrative_clusters(first_seen);

        -- Maps posts to clusters
        CREATE TABLE IF NOT EXISTS cluster_members (
            post_id TEXT REFERENCES raw_posts(id),
            cluster_id INTEGER REFERENCES narrative_clusters(id),
            confidence REAL NOT NULL,
            assigned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            PRIMARY KEY (post_id, cluster_id)
        );

        -- NVI snapshots — time series of narrative velocity
        CREATE TABLE IF NOT EXISTS nvi_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER REFERENCES narrative_clusters(id),
            timestamp TEXT NOT NULL,
            nvi_score REAL NOT NULL,
            burst_zscore REAL NOT NULL,
            spread_factor REAL NOT NULL,
            mutation_penalty REAL NOT NULL,
            coordination_mult REAL NOT NULL,
            raw_components JSON,
            alert_level TEXT DEFAULT 'normal'  -- 'normal', 'elevated', 'critical'
        );

        CREATE INDEX IF NOT EXISTS idx_nvi_cluster ON nvi_snapshots(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_nvi_timestamp ON nvi_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_nvi_alert ON nvi_snapshots(alert_level);

        -- Coordination signals (CIB detection)
        CREATE TABLE IF NOT EXISTS coordination_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER REFERENCES narrative_clusters(id),
            signal_type TEXT NOT NULL,      -- 'temporal_sync', 'content_identity', 'network_density'
            score REAL NOT NULL,
            evidence JSON NOT NULL,
            detected_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_coord_cluster ON coordination_signals(cluster_id);

        -- Evidence packs (Berkeley Protocol compliant)
        CREATE TABLE IF NOT EXISTS evidence_packs (
            id TEXT PRIMARY KEY,
            cluster_id INTEGER REFERENCES narrative_clusters(id),
            sha256_hash TEXT NOT NULL,
            pack_data JSON NOT NULL,
            generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            version INTEGER DEFAULT 1
        );

        -- System state for pipeline tracking
        CREATE TABLE IF NOT EXISTS pipeline_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        -- Cross-narrative links (campaign detection)
        CREATE TABLE IF NOT EXISTS narrative_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_a INTEGER REFERENCES narrative_clusters(id),
            cluster_b INTEGER REFERENCES narrative_clusters(id),
            link_type TEXT NOT NULL,
            strength REAL NOT NULL,
            evidence JSON NOT NULL,
            detected_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(cluster_a, cluster_b, link_type)
        );

        CREATE INDEX IF NOT EXISTS idx_links_a ON narrative_links(cluster_a);
        CREATE INDEX IF NOT EXISTS idx_links_b ON narrative_links(cluster_b);

        -- Multi-narrative campaigns
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            narrative_ids JSON NOT NULL,
            campaign_score REAL NOT NULL,
            first_detected TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            evidence JSON
        );

        CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);

        -- Narrative DNA fingerprints (multi-modal operational signatures)
        CREATE TABLE IF NOT EXISTS narrative_dna (
            cluster_id INTEGER PRIMARY KEY REFERENCES narrative_clusters(id),
            fingerprint JSON NOT NULL,         -- 84-dim concatenated vector
            dimensions JSON NOT NULL,          -- {stylometric, cadence, network, entity_bias}
            metadata JSON NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_dna_cluster ON narrative_dna(cluster_id);

        -- DNA cross-matches (same-operator campaign links)
        CREATE TABLE IF NOT EXISTS dna_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_a INTEGER REFERENCES narrative_clusters(id),
            cluster_b INTEGER REFERENCES narrative_clusters(id),
            match_score REAL NOT NULL,
            dimension_scores JSON NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'medium',
            detected_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(cluster_a, cluster_b)
        );

        CREATE INDEX IF NOT EXISTS idx_dna_matches_a ON dna_matches(cluster_a);
        CREATE INDEX IF NOT EXISTS idx_dna_matches_b ON dna_matches(cluster_b);

        -- Amplification graph snapshots
        CREATE TABLE IF NOT EXISTS amplification_graph_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TEXT NOT NULL,
            node_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            graph_metrics JSON NOT NULL,
            graph_data JSON
        );

        -- Human review labels for closed-loop validation
        CREATE TABLE IF NOT EXISTS human_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER REFERENCES narrative_clusters(id),
            verdict TEXT NOT NULL CHECK(verdict IN ('coordinated', 'organic', 'uncertain')),
            reviewer TEXT,
            notes TEXT,
            reviewed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_cluster ON human_reviews(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_verdict ON human_reviews(verdict);

        -- Source credibility scores
        CREATE TABLE IF NOT EXISTS source_scores (
            domain TEXT PRIMARY KEY,
            credibility_score REAL NOT NULL,
            evidence_basis TEXT NOT NULL,
            category TEXT,
            last_updated TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)

    # Add lifecycle columns to narrative_clusters (safe — ignores if already exist)
    for col_sql in [
        "ALTER TABLE narrative_clusters ADD COLUMN lifecycle_phase TEXT DEFAULT 'emergence'",
        "ALTER TABLE narrative_clusters ADD COLUMN lifecycle_updated TEXT",
        "ALTER TABLE narrative_clusters ADD COLUMN lifecycle_data JSON",
    ]:
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Indexes that hot queries depend on. Each guarded individually so a single
    # failure on a 4.9GB DB under disk pressure does not abort init.
    deferred_indexes = [
        ("idx_cluster_members_cluster",
         "CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id)"),
        ("idx_nvi_cluster_id_desc",
         "CREATE INDEX IF NOT EXISTS idx_nvi_cluster_id_desc ON nvi_snapshots(cluster_id, id DESC)"),
        ("idx_dna_matches_score",
         "CREATE INDEX IF NOT EXISTS idx_dna_matches_score ON dna_matches(match_score)"),
        ("idx_coord_signals_cluster_time",
         "CREATE INDEX IF NOT EXISTS idx_coord_signals_cluster_time ON coordination_signals(cluster_id, detected_at DESC)"),
        ("idx_evidence_packs_cluster",
         "CREATE INDEX IF NOT EXISTS idx_evidence_packs_cluster ON evidence_packs(cluster_id)"),
    ]
    # idx_nvi_cluster (cluster_id only) is strictly subsumed by the new
    # composite idx_nvi_cluster_id_desc — drop it so the planner picks the
    # composite for "latest snapshot per cluster" queries.
    try:
        conn.execute("DROP INDEX IF EXISTS idx_nvi_cluster")
    except sqlite3.OperationalError as e:
        logger.warning("could not drop redundant idx_nvi_cluster: %s", e)

    for name, sql in deferred_indexes:
        t0 = time.monotonic()
        logger.info("building index %s", name)
        try:
            conn.execute(sql)
            logger.info("index %s ready in %.2fs", name, time.monotonic() - t0)
        except sqlite3.OperationalError as e:
            logger.warning("index %s build failed after %.2fs: %s",
                           name, time.monotonic() - t0, e)

    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Wrap a unit of work in BEGIN IMMEDIATE / COMMIT, rolling back on error.

    Provided for new callers — existing autocommit-style call sites are NOT
    migrated here; that is a separate sprint.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def insert_post(conn: sqlite3.Connection, source: str, content: str,
                title: str = "", url: str = "", language: str = "en",
                author: str = "", published_at: str = "",
                metadata: dict = None) -> Optional[str]:
    """Insert a raw post. Returns post ID or None if duplicate."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    post_id = f"{source}_{content_hash[:16]}"

    try:
        conn.execute("""
            INSERT INTO raw_posts (id, source, url, title, content, language,
                                   author, published_at, metadata, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (post_id, source, url, title, content, language,
              author, published_at or datetime.now(timezone.utc).isoformat(),
              json.dumps(metadata or {}), content_hash))
        conn.commit()
        return post_id
    except sqlite3.IntegrityError:
        return None  # Duplicate


def store_embedding(conn: sqlite3.Connection, post_id: str, vector: list):
    """Store embedding vector as JSON array."""
    conn.execute("""
        INSERT OR REPLACE INTO embeddings (post_id, vector)
        VALUES (?, ?)
    """, (post_id, json.dumps(vector)))
    conn.commit()


def get_recent_posts(conn: sqlite3.Connection, hours: int = 24,
                     source: str = None, limit: int = 10000) -> list:
    """Get recent posts for clustering."""
    query = """
        SELECT id, source, title, content, language, published_at, metadata
        FROM raw_posts
        WHERE ingested_at >= datetime('now', ?)
    """
    params = [f'-{hours} hours']

    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY ingested_at DESC LIMIT ?"
    params.append(limit)

    return [dict(row) for row in conn.execute(query, params).fetchall()]


def upsert_cluster(conn: sqlite3.Connection, label: str, keywords: list,
                   post_count: int, source_diversity: float,
                   language_spread: dict, cluster_id: int = None) -> int:
    """Create or update a narrative cluster."""
    now = datetime.now(timezone.utc).isoformat()

    if cluster_id:
        conn.execute("""
            UPDATE narrative_clusters
            SET label=?, keywords=?, post_count=?, source_diversity=?,
                language_spread=?, last_updated=?
            WHERE id=?
        """, (label, json.dumps(keywords), post_count, source_diversity,
              json.dumps(language_spread), now, cluster_id))
        conn.commit()
        return cluster_id
    else:
        cursor = conn.execute("""
            INSERT INTO narrative_clusters
            (label, keywords, first_seen, last_updated, post_count,
             source_diversity, language_spread)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (label, json.dumps(keywords), now, now, post_count,
              source_diversity, json.dumps(language_spread)))
        conn.commit()
        return cursor.lastrowid


def insert_nvi_snapshot(conn: sqlite3.Connection, cluster_id: int,
                        nvi_score: float, burst_zscore: float,
                        spread_factor: float, mutation_penalty: float,
                        coordination_mult: float, raw_components: dict = None,
                        effective_alert_level: str = None):
    """Record an NVI measurement."""
    if effective_alert_level:
        alert = effective_alert_level
    else:
        alert = 'normal'
        if nvi_score >= 80:
            alert = 'critical'
        elif nvi_score >= 60:
            alert = 'elevated'

    conn.execute("""
        INSERT INTO nvi_snapshots
        (cluster_id, timestamp, nvi_score, burst_zscore, spread_factor,
         mutation_penalty, coordination_mult, raw_components, alert_level)
        VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?, ?, ?, ?, ?, ?, ?)
    """, (cluster_id, nvi_score, burst_zscore, spread_factor,
          mutation_penalty, coordination_mult,
          json.dumps(raw_components or {}), alert))
    conn.commit()


def insert_coordination_signal(conn: sqlite3.Connection, cluster_id: int,
                                signal_type: str, score: float,
                                evidence: dict):
    """Persist a coordination signal with its evidence."""
    conn.execute("""
        INSERT INTO coordination_signals (cluster_id, signal_type, score, evidence)
        VALUES (?, ?, ?, ?)
    """, (cluster_id, signal_type, score, json.dumps(evidence)))
    conn.commit()


def get_top_narratives(conn: sqlite3.Connection, limit: int = 20,
                       min_post_count: int = 0) -> list:
    """Get narratives ranked by latest NVI score.

    min_post_count filters out noise clusters (default 0 = no filter).
    """
    rows = conn.execute("""
        SELECT nc.id, nc.label, nc.keywords, nc.summary, nc.first_seen,
               nc.last_updated, nc.post_count, nc.source_diversity,
               nc.language_spread, nc.status, nc.metadata,
               nv.nvi_score, nv.burst_zscore, nv.spread_factor,
               nv.mutation_penalty, nv.coordination_mult,
               nv.alert_level, nv.timestamp as nvi_timestamp
        FROM narrative_clusters nc
        LEFT JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
        WHERE nc.status = 'active'
        AND nc.post_count >= ?
        ORDER BY COALESCE(nv.nvi_score, 0) DESC
        LIMIT ?
    """, (min_post_count, limit)).fetchall()
    return [dict(r) for r in rows]


def get_narrative_detail(conn: sqlite3.Connection, cluster_id: int) -> dict:
    """Get full narrative detail with timeline and posts."""
    cluster = conn.execute(
        "SELECT * FROM narrative_clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    if not cluster:
        return {}

    timeline = conn.execute("""
        SELECT timestamp, nvi_score, burst_zscore, spread_factor,
               mutation_penalty, coordination_mult, alert_level
        FROM nvi_snapshots
        WHERE cluster_id = ?
        ORDER BY timestamp ASC
    """, (cluster_id,)).fetchall()

    posts = conn.execute("""
        SELECT rp.id, rp.source, rp.title, rp.url, rp.content,
               rp.language, rp.published_at, cm.confidence
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
        ORDER BY rp.published_at DESC
        LIMIT 100
    """, (cluster_id,)).fetchall()

    signals = conn.execute("""
        SELECT signal_type, score, evidence, detected_at
        FROM coordination_signals
        WHERE cluster_id = ?
        ORDER BY detected_at DESC
    """, (cluster_id,)).fetchall()

    return {
        "cluster": dict(cluster),
        "timeline": [dict(t) for t in timeline],
        "posts": [dict(p) for p in posts],
        "coordination_signals": [dict(s) for s in signals],
    }


def get_pipeline_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM pipeline_state WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def set_pipeline_state(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_state (key, value, updated_at)
        VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    """, (key, value))
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get overall system statistics."""
    stats = {}
    stats["total_posts"] = conn.execute("SELECT COUNT(*) FROM raw_posts").fetchone()[0]
    stats["posts_24h"] = conn.execute(
        "SELECT COUNT(*) FROM raw_posts WHERE ingested_at >= datetime('now', '-24 hours')"
    ).fetchone()[0]
    stats["active_narratives"] = conn.execute(
        "SELECT COUNT(*) FROM narrative_clusters WHERE status = 'active'"
    ).fetchone()[0]
    stats["critical_alerts"] = conn.execute(
        """SELECT COUNT(DISTINCT ns.cluster_id) FROM nvi_snapshots ns
           JOIN narrative_clusters nc ON nc.id = ns.cluster_id
           WHERE ns.alert_level = 'critical'
           AND nc.status = 'active'
           AND ns.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = ns.cluster_id)"""
    ).fetchone()[0]

    sources = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM raw_posts GROUP BY source"
    ).fetchall()
    stats["sources"] = {r["source"]: r["cnt"] for r in sources}

    return stats


# ─── Cross-Narrative & Campaign Functions ────────────────────────────────────

def upsert_narrative_link(conn: sqlite3.Connection, cluster_a: int,
                           cluster_b: int, link_type: str, strength: float,
                           evidence: dict):
    """Insert or update a link between two narratives."""
    a, b = min(cluster_a, cluster_b), max(cluster_a, cluster_b)
    conn.execute("""
        INSERT INTO narrative_links (cluster_a, cluster_b, link_type, strength, evidence)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(cluster_a, cluster_b, link_type)
        DO UPDATE SET strength = excluded.strength, evidence = excluded.evidence,
                      detected_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    """, (a, b, link_type, strength, json.dumps(evidence)))
    conn.commit()


def get_narrative_links(conn: sqlite3.Connection, cluster_id: int) -> list:
    """Get all links involving a cluster."""
    rows = conn.execute("""
        SELECT nl.*, nc_a.label as label_a, nc_b.label as label_b
        FROM narrative_links nl
        JOIN narrative_clusters nc_a ON nc_a.id = nl.cluster_a
        JOIN narrative_clusters nc_b ON nc_b.id = nl.cluster_b
        WHERE nl.cluster_a = ? OR nl.cluster_b = ?
        ORDER BY nl.strength DESC
    """, (cluster_id, cluster_id)).fetchall()
    return [dict(r) for r in rows]


def upsert_campaign(conn: sqlite3.Connection, label: str,
                     narrative_ids: list[int], campaign_score: float,
                     evidence: dict = None) -> int:
    """Create or update a campaign."""
    now = datetime.now(timezone.utc).isoformat()
    ids_json = json.dumps(sorted(narrative_ids))

    existing = conn.execute(
        "SELECT id FROM campaigns WHERE narrative_ids = ? AND status = 'active'",
        (ids_json,)
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE campaigns SET label=?, campaign_score=?, last_updated=?, evidence=?
            WHERE id=?
        """, (label, campaign_score, now, json.dumps(evidence or {}), existing["id"]))
        conn.commit()
        return existing["id"]
    else:
        cursor = conn.execute("""
            INSERT INTO campaigns (label, narrative_ids, campaign_score,
                                   first_detected, last_updated, evidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (label, ids_json, campaign_score, now, now, json.dumps(evidence or {})))
        conn.commit()
        return cursor.lastrowid


def get_campaigns(conn: sqlite3.Connection, limit: int = 20) -> list:
    """Get active campaigns ranked by score."""
    rows = conn.execute("""
        SELECT * FROM campaigns WHERE status = 'active'
        ORDER BY campaign_score DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_campaign_detail(conn: sqlite3.Connection, campaign_id: int) -> dict:
    """Get full campaign detail."""
    campaign = conn.execute(
        "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
    ).fetchone()
    if not campaign:
        return {}

    campaign = dict(campaign)
    narrative_ids = json.loads(campaign.get("narrative_ids", "[]"))

    narratives = []
    for nid in narrative_ids:
        row = conn.execute("""
            SELECT nc.*, nv.nvi_score, nv.alert_level
            FROM narrative_clusters nc
            LEFT JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
                AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
            WHERE nc.id = ?
        """, (nid,)).fetchone()
        if row:
            narratives.append(dict(row))

    links = conn.execute("""
        SELECT * FROM narrative_links
        WHERE cluster_a IN ({ids}) AND cluster_b IN ({ids})
        ORDER BY strength DESC
    """.format(ids=",".join("?" * len(narrative_ids))),
        narrative_ids + narrative_ids
    ).fetchall()

    return {
        "campaign": campaign,
        "narratives": narratives,
        "links": [dict(l) for l in links],
    }


# ─── Source Credibility Functions ────────────────────────────────────────────

def upsert_source_score(conn: sqlite3.Connection, domain: str,
                         credibility_score: float, evidence_basis: str,
                         category: str = "unknown"):
    """Insert or update a source credibility score."""
    conn.execute("""
        INSERT INTO source_scores (domain, credibility_score, evidence_basis, category)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(domain)
        DO UPDATE SET credibility_score = excluded.credibility_score,
                      evidence_basis = excluded.evidence_basis,
                      category = excluded.category,
                      last_updated = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    """, (domain, credibility_score, evidence_basis, category))
    conn.commit()


def get_source_score(conn: sqlite3.Connection, domain: str) -> Optional[dict]:
    """Get credibility score for a source domain."""
    row = conn.execute(
        "SELECT * FROM source_scores WHERE domain = ?", (domain,)
    ).fetchone()
    return dict(row) if row else None


def get_source_scores_for_cluster(conn: sqlite3.Connection,
                                   cluster_id: int) -> list:
    """Get credibility scores for all sources in a cluster."""
    rows = conn.execute("""
        SELECT DISTINCT rp.source, ss.credibility_score, ss.category
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        LEFT JOIN source_scores ss ON ss.domain = rp.source
        WHERE cm.cluster_id = ?
    """, (cluster_id,)).fetchall()
    return [dict(r) for r in rows]


# ─── Lifecycle Functions ─────────────────────────────────────────────────────

def update_lifecycle_phase(conn: sqlite3.Connection, cluster_id: int,
                            phase: str, data: dict):
    """Update the lifecycle phase of a narrative cluster."""
    conn.execute("""
        UPDATE narrative_clusters
        SET lifecycle_phase = ?, lifecycle_updated = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
            lifecycle_data = ?
        WHERE id = ?
    """, (phase, json.dumps(data), cluster_id))
    conn.commit()


# ─── DNA Functions ──────────────────────────────────────────────────────────

def get_dna_fingerprint(conn, cluster_id: int) -> dict | None:
    """Get stored DNA fingerprint for a cluster."""
    row = conn.execute(
        "SELECT * FROM narrative_dna WHERE cluster_id = ?", (cluster_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["fingerprint"] = json.loads(d["fingerprint"])
    d["dimensions"] = json.loads(d["dimensions"])
    d["metadata"] = json.loads(d["metadata"])
    return d


def get_dna_matches(conn, cluster_id: int) -> list[dict]:
    """Get all DNA matches involving a cluster."""
    rows = conn.execute("""
        SELECT *, nc.label as matched_label
        FROM dna_matches dm
        JOIN narrative_clusters nc ON nc.id = CASE
            WHEN dm.cluster_a = ? THEN dm.cluster_b
            ELSE dm.cluster_a
        END
        WHERE dm.cluster_a = ? OR dm.cluster_b = ?
        ORDER BY dm.match_score DESC
    """, (cluster_id, cluster_id, cluster_id)).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["dimension_scores"] = json.loads(d["dimension_scores"])
        results.append(d)
    return results


def get_all_dna_matches(conn, min_score: float = 0.75, limit: int = 200) -> list[dict]:
    """Get high-confidence DNA matches across the system. LIMIT pushed into
    SQL — fetchall() on 23M+ rows would otherwise time out."""
    rows = conn.execute("""
        SELECT dm.*, nc_a.label as label_a, nc_b.label as label_b
        FROM dna_matches dm
        JOIN narrative_clusters nc_a ON nc_a.id = dm.cluster_a
        JOIN narrative_clusters nc_b ON nc_b.id = dm.cluster_b
        WHERE dm.match_score >= ?
        ORDER BY dm.match_score DESC
        LIMIT ?
    """, (min_score, int(limit))).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["dimension_scores"] = json.loads(d["dimension_scores"])
        results.append(d)
    return results


# ─── Human Review Functions ─────────────────────────────────────────────────

def insert_review(conn, cluster_id: int, verdict: str, reviewer: str = "",
                  notes: str = ""):
    """Record a human review verdict for closed-loop validation."""
    conn.execute("""
        INSERT INTO human_reviews (cluster_id, verdict, reviewer, notes)
        VALUES (?, ?, ?, ?)
    """, (cluster_id, verdict, reviewer, notes))
    conn.commit()


def get_reviews(conn, cluster_id: int = None, limit: int = 100) -> list[dict]:
    """Get human reviews, optionally filtered by cluster."""
    if cluster_id:
        rows = conn.execute("""
            SELECT * FROM human_reviews WHERE cluster_id = ? ORDER BY reviewed_at DESC
        """, (cluster_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM human_reviews ORDER BY reviewed_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_review_stats(conn) -> dict:
    """Get review statistics for precision tracking."""
    total = conn.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0]
    coordinated = conn.execute(
        "SELECT COUNT(*) FROM human_reviews WHERE verdict = 'coordinated'"
    ).fetchone()[0]
    organic = conn.execute(
        "SELECT COUNT(*) FROM human_reviews WHERE verdict = 'organic'"
    ).fetchone()[0]
    uncertain = conn.execute(
        "SELECT COUNT(*) FROM human_reviews WHERE verdict = 'uncertain'"
    ).fetchone()[0]

    # For precision: among clusters we flagged (NVI >= 50), how many were confirmed coordinated?
    confirmed = conn.execute("""
        SELECT COUNT(*) FROM human_reviews hr
        JOIN nvi_snapshots nv ON nv.cluster_id = hr.cluster_id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = hr.cluster_id)
        WHERE hr.verdict = 'coordinated' AND COALESCE(nv.nvi_score, 0) >= 50
    """).fetchone()[0]
    flagged = conn.execute("""
        SELECT COUNT(DISTINCT hr.cluster_id) FROM human_reviews hr
        JOIN nvi_snapshots nv ON nv.cluster_id = hr.cluster_id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = hr.cluster_id)
        WHERE COALESCE(nv.nvi_score, 0) >= 50
    """).fetchone()[0]
    precision = round(confirmed / max(flagged, 1), 4)

    # For recall: among confirmed coordinated, how many did we flag (NVI >= 50)?
    reviewed_coordinated = coordinated
    flagged_coordinated = conn.execute("""
        SELECT COUNT(DISTINCT hr.cluster_id) FROM human_reviews hr
        JOIN nvi_snapshots nv ON nv.cluster_id = hr.cluster_id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = hr.cluster_id)
        WHERE hr.verdict = 'coordinated' AND COALESCE(nv.nvi_score, 0) >= 50
    """).fetchone()[0]
    recall = round(flagged_coordinated / max(reviewed_coordinated, 1), 4)

    return {
        "total_reviews": total,
        "coordinated": coordinated,
        "organic": organic,
        "uncertain": uncertain,
        "precision": precision,
        "recall": recall,
        "f1": round(2 * precision * recall / max(precision + recall, 0.01), 4),
    }


# ─── Graph Snapshot Functions ───────────────────────────────────────────────

def store_graph_snapshot(conn, node_count: int, edge_count: int,
                         metrics: dict, graph_data: dict = None):
    """Store an amplification graph snapshot."""
    conn.execute("""
        INSERT INTO amplification_graph_snapshots
        (snapshot_at, node_count, edge_count, graph_metrics, graph_data)
        VALUES (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?, ?, ?, ?)
    """, (node_count, edge_count, json.dumps(metrics),
          json.dumps(graph_data) if graph_data else None))
    conn.commit()


def get_latest_graph_snapshot(conn) -> dict | None:
    """Get the most recent amplification graph snapshot."""
    row = conn.execute("""
        SELECT * FROM amplification_graph_snapshots
        ORDER BY id DESC LIMIT 1
    """).fetchone()
    if not row:
        return None
    d = dict(row)
    d["graph_metrics"] = json.loads(d["graph_metrics"])
    if d["graph_data"]:
        d["graph_data"] = json.loads(d["graph_data"])
    return d

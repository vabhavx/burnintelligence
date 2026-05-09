"""
BurnTheLies Intelligence — FastAPI Server
Public API serving human-readable narrative intelligence.
Every response includes interpretation (for humans), raw data (for verification),
confidence intervals, alternative hypotheses, and source credibility.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from intelligence.auth import require_api_key
from intelligence.db import (
    get_connection, init_db, get_top_narratives, get_narrative_detail,
    get_stats, get_narrative_links, get_campaigns, get_campaign_detail,
)
from intelligence import metrics as intel_metrics
from intelligence.processing.interpret import (
    interpret_narrative, clean_cluster_label, generate_source_context,
)

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    _SLOWAPI_OK = True
except Exception as _slowapi_exc:  # pragma: no cover
    logging.getLogger("intelligence.api").warning(
        "slowapi unavailable, rate limiting disabled: %s", _slowapi_exc
    )
    Limiter = None  # type: ignore
    _SLOWAPI_OK = False


_RATE_DEFAULT = os.getenv("INTEL_RATE_LIMIT_DEFAULT", "60/minute")
_RATE_EXPENSIVE = os.getenv("INTEL_RATE_LIMIT_EXPENSIVE", "10/minute")
_RATE_SEARCH = os.getenv("INTEL_RATE_LIMIT_SEARCH", "30/minute")

if _SLOWAPI_OK:
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
else:
    class _NoopLimiter:
        def limit(self, *_a, **_kw):
            def deco(fn):
                return fn
            return deco

    limiter = _NoopLimiter()  # type: ignore


_HEALTH_FILE = Path(os.getenv(
    "INTEL_HEALTH_FILE",
    str(Path.home() / ".burn_state" / "intel_health.json"),
))
_STALE_THRESHOLD = int(os.getenv("INTEL_STALE_THRESHOLD", "3600"))

logger = logging.getLogger("intelligence.api")

_db = None

# Cache enriched narratives: {cluster_id: (timestamp, enriched_dict, dominant_title_count)}
_enrich_cache: dict[int, tuple[float, dict, int]] = {}
_ENRICH_CACHE_TTL = 15  # seconds
_REBUILD_FLAG = "/tmp/intel_rebuild_ts"


def _last_rebuild_time() -> float:
    try:
        with open(_REBUILD_FLAG) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return 0.0


def get_db():
    global _db
    if _db is None:
        _db = get_connection()
        init_db(_db)
    return _db


def _parse_json_field(val, default=None):
    if default is None:
        default = []
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt JSON field (returning default): %s", val[:200] if len(val) > 200 else val)
            return default
    return val if val is not None else default


def _get_post_titles(db, cluster_id: int, dominant_language: str = "en") -> tuple[list[str], int]:
    """Get post titles for a cluster, preferring the dominant language.
    Returns (titles_for_label, dominant_lang_count). When no dominant-language
    titles exist, returns empty list so clean_cluster_label uses GDELT themes."""
    lang_filter = dominant_language if dominant_language not in ("unknown", "") else "en"
    rows = db.execute("""
        SELECT rp.language, rp.title FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ? AND rp.title IS NOT NULL AND rp.title != ''
        ORDER BY CASE WHEN rp.language = ? THEN 0 ELSE 1 END
        LIMIT 50
    """, (cluster_id, lang_filter)).fetchall()
    dominant_titles = [r["title"] for r in rows if r["language"] == lang_filter]
    return dominant_titles, len(dominant_titles)


def _get_cluster_themes(db, cluster_id: int) -> list[str]:
    """Get aggregated themes from cluster member metadata."""
    from collections import Counter
    from intelligence.processing.interpret import _clean_gdelt_theme

    rows = db.execute("""
        SELECT rp.metadata FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
        LIMIT 50
    """, (cluster_id,)).fetchall()

    theme_counts = Counter()
    skip_prefixes = {"TAX_", "WB_", "EPU_", "CRISISLEX_", "GENERAL_", "USPEC_"}
    for r in rows:
        meta = _parse_json_field(r["metadata"], {})
        if isinstance(meta, dict) and "themes" in meta:
            for t in meta["themes"][:10]:
                if any(t.startswith(p) for p in skip_prefixes):
                    continue
                clean = t.replace("_", " ").strip().title()
                if len(clean) > 3:
                    human = _clean_gdelt_theme(clean)
                    theme_counts[human] += 1

    return [t for t, _ in theme_counts.most_common(8)]


def _get_nvi_components(db, cluster_id: int) -> dict:
    """Read all NVI signal metadata from latest NVI snapshot raw_components.

    Refreshes the dna_match gate decision against the live ``dna_matches``
    table — the snapshot stamps a stale count when DNA hasn't run yet.
    """
    from intelligence.processing.nvi import refresh_dna_match_gate
    row = db.execute("""
        SELECT raw_components FROM nvi_snapshots
        WHERE cluster_id = ? ORDER BY id DESC LIMIT 1
    """, (cluster_id,)).fetchone()
    if row and row["raw_components"]:
        comps = _parse_json_field(row["raw_components"], {})
        if isinstance(comps, dict):
            comps = refresh_dna_match_gate(db, cluster_id, comps)
            return {
                "entity_concentration": comps.get("entity_concentration", 0.5),
                "is_topic_cluster": comps.get("is_topic_cluster", False),
                "shared_entity_count": comps.get("shared_entity_count", 0),
                "wire_service_syndication": comps.get("wire_service_syndication", False),
                "wire_service_source_fraction": comps.get("wire_service_source_fraction", 0.0),
                "alert_suppressed": comps.get("alert_suppressed", False),
                "confidence_probability": comps.get("confidence_probability", 0.5),
                "effective_alert_level": comps.get("effective_alert_level", "normal"),
                "ensemble_uncertain": comps.get("ensemble_uncertain", False),
                "ensemble_disagreement": comps.get("ensemble_disagreement", 0.0),
                "gdelt_batch_artifact": comps.get("gdelt_batch_artifact", False),
                "dna_match_count": comps.get("dna_match_count", -1),
                "dna_evidence_strong": comps.get("dna_evidence_strong", False),
                "high_conf_dna_match_count": comps.get("high_conf_dna_match_count", 0),
                "cross_topic_persistence": comps.get("cross_topic_persistence", False),
                "cross_language_anomaly": comps.get("cross_language_anomaly", False),
                "geographic_spread": comps.get("geographic_spread", False),
                "high_signal_topic": comps.get("high_signal_topic", False),
                "circadian_anomaly": comps.get("circadian_anomaly", False),
                "content_anomaly": comps.get("content_anomaly", False),
                "content_noise": comps.get("content_noise", False),
                "ensemble_perfect_agreement_red_flag": comps.get("ensemble_perfect_agreement_red_flag", False),
                # v5 gate trace (empty list/dict on legacy snapshots)
                "gates_applied": comps.get("gates_applied", []),
                "gate_reasoning": comps.get("gate_reasoning", {}),
                "raw_nvi": comps.get("raw_nvi", comps.get("nvi_score", 0.0)),
                "narrative_coherence": comps.get("narrative_coherence", 0.5),
                "unique_hash_ratio": comps.get("unique_hash_ratio", 0.0),
                "embedding_similarity_mean": comps.get("embedding_similarity_mean", 0.0),
            }
    return {"entity_concentration": 0.5, "is_topic_cluster": False, "shared_entity_count": 0,
            "wire_service_syndication": False, "wire_service_source_fraction": 0.0,
            "alert_suppressed": True, "confidence_probability": 0.0,
            "effective_alert_level": "normal",
            "ensemble_uncertain": False, "ensemble_disagreement": 0.0,
            "gdelt_batch_artifact": False, "dna_match_count": -1,
            "dna_evidence_strong": False, "high_conf_dna_match_count": 0,
            "cross_topic_persistence": False, "cross_language_anomaly": False,
            "geographic_spread": False, "high_signal_topic": False,
            "circadian_anomaly": False, "content_anomaly": False, "content_noise": False,
            "ensemble_perfect_agreement_red_flag": False,
            "gates_applied": [], "gate_reasoning": {}, "raw_nvi": 0.0,
            "narrative_coherence": 0.5, "unique_hash_ratio": 0.0,
            "embedding_similarity_mean": 0.0}


def _get_lifecycle_data(db, cluster_id: int) -> dict:
    """Get lifecycle data for a cluster."""
    row = db.execute("""
        SELECT lifecycle_phase, lifecycle_updated, lifecycle_data
        FROM narrative_clusters WHERE id = ?
    """, (cluster_id,)).fetchone()
    if not row:
        return {"phase": "emergence", "phase_label": "Emerging"}
    data = _parse_json_field(row["lifecycle_data"], {})
    if isinstance(data, dict):
        data["phase"] = row["lifecycle_phase"] or "emergence"
        return data
    return {"phase": row["lifecycle_phase"] or "emergence", "phase_label": "Emerging"}


def _get_campaign_info(db, cluster_id: int) -> dict | None:
    """Check if this cluster is part of an active campaign."""
    campaigns = db.execute("""
        SELECT id, label, narrative_ids, campaign_score
        FROM campaigns WHERE status = 'active'
    """).fetchall()

    for c in campaigns:
        ids = _parse_json_field(c["narrative_ids"], [])
        if cluster_id in ids:
            return {
                "campaign_id": c["id"],
                "campaign_label": c["label"],
                "campaign_score": c["campaign_score"],
                "narrative_count": len(ids),
            }
    return None


KEYWORD_STOPLIST = {
    "html", "htm", "www", "com", "org", "net", "http", "https",
    "php", "index", "news", "article", "page", "the", "del", "per", "con",
    "topics", "people", "organizations", "tone",
}


# Map v5 gate IDs to legacy falsification-criterion IDs surfaced in the API.
# Multiple gates can map to the same criterion (gate trace is the canonical
# v5 truth; this preserves the 4-criterion contract for legacy consumers).
_GATE_TO_CRITERION_ID = {
    "organic_viral_spread": 1,
    "normal_news_cycle": 2,
    "wire_service": 3,
    "insufficient_evidence": 4,
}

_CRITERION_DEFS = {
    1: "Organic spread pattern: mutation > 0.30 and diversity > 0.60",
    2: "Normal news cycle: coordination_mult < 1.10 and burst < 5",
    3: "Wire service syndication detected (content from news agencies, not independent actors)",
    4: "Insufficient evidence: fewer than 10 posts available for analysis",
}


def _build_falsification_block(nvi_comps: dict, post_count: int, mutation: float,
                                coord: float, burst: float, source_div: float) -> dict:
    """
    Build the legacy `falsification` block for the API response.

    Source of truth is `gates_applied` from the v5 gate pipeline; for
    snapshots predating v5 (no gates_applied key), fall back to recomputing
    each criterion from the raw signal values.
    """
    gates_applied = nvi_comps.get("gates_applied") or []
    gate_reasoning = nvi_comps.get("gate_reasoning") or {}

    triggered_ids: set[int] = set()
    if gates_applied:
        for gate_name in gates_applied:
            cid = _GATE_TO_CRITERION_ID.get(gate_name)
            if cid is not None:
                triggered_ids.add(cid)
    else:
        # Legacy fallback — recompute from raw signals.
        wire_service = nvi_comps.get("wire_service_syndication", False)
        if mutation > 0.30 and source_div > 0.60:
            triggered_ids.add(1)
        if coord < 1.10 and burst < 5:
            triggered_ids.add(2)
        if wire_service:
            triggered_ids.add(3)
        if post_count < 10:
            triggered_ids.add(4)

    falsification_criteria = [
        {
            "id": cid,
            "description": _CRITERION_DEFS[cid],
            "triggered": cid in triggered_ids,
        }
        for cid in sorted(_CRITERION_DEFS.keys())
    ]
    any_triggered = bool(triggered_ids)

    if any_triggered:
        parts = []
        if 1 in triggered_ids:
            parts.append("High content mutation and source diversity suggest organic editorial coverage")
        if 2 in triggered_ids:
            parts.append("Coordination multiplier and burst rate are consistent with routine news coverage")
        if 3 in triggered_ids:
            parts.append("Content originates from wire services (AP, Reuters, AFP) — standard syndication pattern")
        if 4 in triggered_ids:
            parts.append("Sample too small for meaningful narrative analysis")
        interpretation_text = " | ".join(parts) + "."
    else:
        interpretation_text = "No signal-strength mitigation criteria triggered."

    block = {
        "criteria": falsification_criteria,
        "any_triggered": any_triggered,
        "interpretation": interpretation_text,
    }
    if gates_applied:
        # Surface the canonical v5 trace alongside the legacy 4-criterion view.
        block["gates_applied"] = list(gates_applied)
        block["gate_reasoning"] = dict(gate_reasoning)
    return block


def _enrich_narrative(n: dict, db) -> tuple[dict, int]:
    """Add interpretation layer, confidence intervals, lifecycle, campaigns.
    Returns (enriched_dict, dominant_title_count) — count is internal-only."""
    cluster_id = n["id"]
    now = time.time()
    cached = _enrich_cache.get(cluster_id)
    if cached and (now - cached[0]) < _ENRICH_CACHE_TTL and cached[0] > _last_rebuild_time():
        return cached[1], cached[2] if len(cached) > 2 else 0

    keywords = _parse_json_field(n.get("keywords", "[]"), [])
    lang_spread = _parse_json_field(n.get("language_spread", "{}"), {})
    dominant_language = max(lang_spread.items(), key=lambda x: x[1])[0] if lang_spread else "unknown"

    nvi_score = n.get("nvi_score", 0) or 0
    burst = n.get("burst_zscore", 0) or 0
    spread = n.get("spread_factor", 0) or 0
    mutation = n.get("mutation_penalty", 0) or 0
    coord = n.get("coordination_mult", 0) or 0
    post_count = n.get("post_count", 0) or 0
    source_div = n.get("source_diversity", 0) or 0

    # Clean the label — use dominant language to avoid non-English headlines winning
    post_titles, dominant_title_count = _get_post_titles(db, n["id"], dominant_language)
    themes = _get_cluster_themes(db, n["id"])
    clean_label = clean_cluster_label(n.get("label", ""), keywords, post_titles, themes)

    # Generate interpretation (now includes confidence_interval + alternative_hypotheses)
    interpretation = interpret_narrative(
        nvi_score=nvi_score,
        burst_zscore=burst,
        spread_factor=spread,
        mutation_penalty=mutation,
        coordination_mult=coord,
        post_count=post_count,
        source_diversity=source_div,
    )

    # Parse cluster metadata for resolution info
    cluster_meta = _parse_json_field(n.get("metadata", "{}"), {})
    resolution = cluster_meta.get("resolution") if isinstance(cluster_meta, dict) else None
    resolution_confidence = cluster_meta.get("resolution_confidence") if isinstance(cluster_meta, dict) else None
    single_source_topic_bag = (
        cluster_meta.get("single_source_topic_bag") is True
        if isinstance(cluster_meta, dict) else False
    )

    # Lifecycle data
    lifecycle = _get_lifecycle_data(db, n["id"])

    # NVI components (entity concentration, topic cluster flag)
    nvi_comps = _get_nvi_components(db, n["id"])

    # Campaign membership
    campaign = _get_campaign_info(db, n["id"])

    # Source credibility
    source_context = []
    try:
        from intelligence.processing.source_credibility import get_cluster_source_breakdown
        breakdown = get_cluster_source_breakdown(db, n["id"])
        from intelligence.processing.interpret import generate_source_context
        scores = [{"category": cat, "count": cnt} for cat, cnt in breakdown.get("categories", {}).items()]
        source_context = generate_source_context(
            [{"category": s["category"]} for s in breakdown.get("sources", [])]
        )
    except Exception as e:
        logger.warning("Source credibility failed for cluster %s: %s", n["id"], e)
        breakdown = {"categories": {}, "weighted_credibility": 0.5}
        source_context = []

    result = {
        "id": n["id"],
        "label": clean_label,
        "themes": themes,
        "keywords": [kw for kw in keywords if kw.lower() not in KEYWORD_STOPLIST],
        "first_seen": n.get("first_seen", ""),
        "last_updated": n.get("last_updated", ""),
        "post_count": post_count,
        "source_diversity": source_div,
        "language_spread": lang_spread,
        "dominant_language": dominant_language,
        "resolution": resolution,
        "resolution_confidence": resolution_confidence,
        "interpretation": interpretation,
        "lifecycle": lifecycle,
        "campaign": campaign,
        "source_credibility": {
            "categories": breakdown.get("categories", {}),
            "weighted_credibility": breakdown.get("weighted_credibility", 0.5),
            "context": source_context,
        },
        "raw": {
            "nvi_score": nvi_score,
            "raw_nvi": nvi_comps.get("raw_nvi", nvi_score),
            "burst_zscore": burst,
            "spread_factor": spread,
            "mutation_penalty": mutation,
            "coordination_mult": coord,
            "alert_level": nvi_comps.get("effective_alert_level", n.get("alert_level", "normal") or "normal"),
            "effective_alert_level": nvi_comps.get("effective_alert_level", n.get("alert_level", "normal") or "normal"),
            "confidence_probability": nvi_comps.get("confidence_probability", 0.5),
            "alert_suppressed": nvi_comps.get("alert_suppressed", False),
            "wire_service_syndication": nvi_comps.get("wire_service_syndication", False),
            "wire_service_source_fraction": nvi_comps.get("wire_service_source_fraction", 0.0),
            "ensemble_uncertain": nvi_comps.get("ensemble_uncertain", False),
            "ensemble_disagreement": nvi_comps.get("ensemble_disagreement", 0.0),
            "ensemble_perfect_agreement_red_flag": nvi_comps.get("ensemble_perfect_agreement_red_flag", False),
            "gdelt_batch_artifact": nvi_comps.get("gdelt_batch_artifact", False),
            "single_source_topic_bag": single_source_topic_bag,
            "dna_match_count": nvi_comps.get("dna_match_count", -1),
            "dna_evidence_strong": nvi_comps.get("dna_evidence_strong", False),
            "timestamp": n.get("nvi_timestamp", ""),
            "entity_concentration": nvi_comps["entity_concentration"],
            "is_topic_cluster": nvi_comps["is_topic_cluster"],
            "shared_entity_count": nvi_comps["shared_entity_count"],
            # v5 gate trace
            "gates_applied": nvi_comps.get("gates_applied", []),
            "gate_reasoning": nvi_comps.get("gate_reasoning", {}),
            "narrative_coherence": nvi_comps.get("narrative_coherence", 0.5),
            "unique_hash_ratio": nvi_comps.get("unique_hash_ratio", 0.0),
            "embedding_similarity_mean": nvi_comps.get("embedding_similarity_mean", 0.0),
        },
    }

    # Falsification criteria — derived from the v5 gate trace when available,
    # falls back to recomputing from raw signals for legacy snapshots.
    result["falsification"] = _build_falsification_block(
        nvi_comps, post_count, mutation, coord, burst, source_div
    )

    _enrich_cache[cluster_id] = (time.time(), result, dominant_title_count)
    return result, dominant_title_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Intelligence API starting...")
    get_db()
    logger.info("Database initialized")
    yield
    logger.info("Intelligence API shutting down")


app = FastAPI(
    title="BurnTheLies Intelligence API",
    description="Real-time narrative manipulation detection with human-readable interpretation",
    version="5.4.0",
    lifespan=lifespan,
)

_cors_origins = [
    o.strip()
    for o in os.getenv("INTEL_CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

if _SLOWAPI_OK:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

intel_metrics.install(app)


@app.get("/")
async def root():
    return {
        "service": "BurnTheLies Intelligence Engine",
        "version": "5.4.0",
        "status": "operational",
        "capabilities": [
            "narrative_detection", "coordination_signals", "confidence_intervals",
            "alternative_hypotheses", "campaign_detection", "lifecycle_tracking",
            "source_credibility", "evidence_packs",
            "dna_fingerprinting", "graph_topology", "ensemble_nvi",
            "multi_resolution_clustering", "human_review",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/health")
async def health():
    db = get_db()
    stats = get_stats(db)
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }


@app.get("/api/health/pipeline")
async def pipeline_health(response: Response):
    """Pipeline freshness — reports whether the intelligence loop is recent.

    Reads ~/.burn_state/intel_health.json (written by the pipeline). Returns
    HTTP 503 when the file is missing or the last cycle is older than
    INTEL_STALE_THRESHOLD seconds (default 3600), so external monitors can
    alarm on staleness without parsing the body.
    """
    if not _HEALTH_FILE.exists():
        response.status_code = 503
        return {"healthy": False, "reason": "no health file"}

    try:
        data = json.loads(_HEALTH_FILE.read_text())
    except Exception as e:
        response.status_code = 503
        return {"healthy": False, "reason": f"unreadable health file: {e}"}

    last_completed = data.get("last_cycle_completed")
    seconds_since: float | None = None
    if last_completed:
        try:
            ts = datetime.fromisoformat(last_completed.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            seconds_since = (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            seconds_since = None

    healthy = seconds_since is not None and seconds_since < _STALE_THRESHOLD
    if not healthy:
        response.status_code = 503

    return {
        "healthy": healthy,
        "last_cycle_completed": last_completed,
        "seconds_since_last_cycle": seconds_since,
        "current_stage": data.get("current_stage"),
        "stale_threshold_seconds": _STALE_THRESHOLD,
        "stages": data.get("stage_status", {}),
    }


@app.get("/api/narratives", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_DEFAULT)
async def list_narratives(
    request: Request,
    response: Response,
    limit: int = Query(default=20, ge=1, le=100),
    alert_level: str = Query(default=None),
    phase: str = Query(default=None),
    lang: str = Query(default="en"),
    min_post_count: int = Query(default=8, ge=0),
    show_suppressed: bool = Query(default=False),
):
    """Narratives ranked by narrative velocity index, with full interpretation.

    lang=en (default) filters to English-dominant clusters. lang=all disables.
    min_post_count filters out noise clusters below the threshold (default 8).
    show_suppressed=false (default) hides confidence-gated and topic-bag clusters.
    """
    response.headers["Cache-Control"] = "public, max-age=15"
    db = get_db()
    # Fetch all active clusters — Python-side filters (suppressed, topic_bag, lang, dedup)
    # can discard most of them; we need the full pool to avoid missing valid signals.
    fetch_limit = 2000
    narratives = get_top_narratives(db, limit=fetch_limit, min_post_count=min_post_count)

    results = []
    seen_label_words: list[frozenset] = []

    seen_labels_exact: set[str] = set()

    def _is_duplicate_label(label: str) -> bool:
        # Always deduplicate exact matches (cross-resolution same-story clusters).
        norm = label.strip().lower()
        if norm in seen_labels_exact:
            return True
        seen_labels_exact.add(norm)
        # For long rich labels (>4 unique words) also deduplicate by word overlap.
        words = frozenset(w.lower() for w in label.split() if len(w) > 3)
        if len(words) > 4:
            for seen in seen_label_words:
                if not seen:
                    continue
                overlap = len(words & seen) / max(len(words | seen), 1)
                if overlap >= 0.65:
                    return True
            seen_label_words.append(words)
        return False

    for n in narratives:
        enriched, dominant_title_count = _enrich_narrative(n, db)
        raw = enriched["raw"]

        # Skip confidence-gated (below threshold) clusters unless show_suppressed
        if not show_suppressed and raw.get("alert_suppressed", False):
            continue

        # Skip incoherent topic-bag clusters (many domains, unrelated stories)
        cluster_meta = _parse_json_field(n.get("metadata", "{}"), {})
        if isinstance(cluster_meta, dict) and cluster_meta.get("topic_bag"):
            continue

        if alert_level and raw["effective_alert_level"] != alert_level:
            continue

        if phase and enriched.get("lifecycle", {}).get("phase") != phase:
            continue

        # BUG A: Include "unknown"-dominant clusters in English feed.
        # GDELT assigns "unknown" to .com/.org/.net TLDs — in practice these are
        # overwhelmingly English-language articles. "translated" = non-English source.
        if lang == "en" and enriched.get("dominant_language") not in ("en", "unknown"):
            continue
        if lang not in ("all", "en") and enriched.get("dominant_language", "unknown") != lang:
            continue

        # Skip clusters whose label is still non-English (dtc=0 fallback to translated titles).
        if lang == "en" and dominant_title_count == 0:
            label_text = enriched.get("label", "")
            non_ascii = sum(1 for c in label_text if ord(c) > 127)
            if non_ascii > 2:
                continue

        # Deduplicate cross-resolution clusters showing the same story
        if _is_duplicate_label(enriched.get("label", "")):
            continue

        results.append(enriched)
        if len(results) >= limit:
            break

    return {
        "narratives": results,
        "count": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/narratives/{cluster_id}", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_DEFAULT)
async def narrative_detail(request: Request, cluster_id: int, response: Response):
    """Full narrative detail with all intelligence layers."""
    response.headers["Cache-Control"] = "public, max-age=30"
    db = get_db()
    detail = get_narrative_detail(db, cluster_id)

    if not detail or not detail.get("cluster"):
        raise HTTPException(status_code=404, detail="Narrative not found")

    cluster = detail["cluster"]

    # Parse JSON fields in cluster
    for field in ["keywords", "language_spread", "metadata"]:
        cluster[field] = _parse_json_field(
            cluster.get(field, ""),
            {} if field != "keywords" else []
        )

    # Parse evidence JSON in coordination signals
    for sig in detail.get("coordination_signals", []):
        if isinstance(sig.get("evidence"), str):
            sig["evidence"] = _parse_json_field(sig["evidence"], {})

    # Build enriched response
    nvi_score = 0
    burst = spread = mutation = coord = 0.0
    timeline = detail.get("timeline", [])
    if timeline:
        latest = timeline[-1]
        nvi_score = latest.get("nvi_score", 0)
        burst = latest.get("burst_zscore", 0)
        spread = latest.get("spread_factor", 0)
        mutation = latest.get("mutation_penalty", 0)
        coord = latest.get("coordination_mult", 0)

    # Clean label — filter to dominant language to avoid non-English titles winning
    lang_spread_detail = _parse_json_field(cluster.get("language_spread", "{}"), {})
    dominant_lang_detail = max(lang_spread_detail.items(), key=lambda x: x[1])[0] if lang_spread_detail else "en"
    post_titles, _ = _get_post_titles(db, cluster_id, dominant_lang_detail)
    keywords = cluster.get("keywords", [])
    if isinstance(keywords, str):
        keywords = _parse_json_field(keywords, [])
    detail_themes = _get_cluster_themes(db, cluster_id)
    clean_label = clean_cluster_label(
        cluster.get("label", ""), keywords, post_titles, detail_themes
    )

    # Generate full interpretation
    interpretation = interpret_narrative(
        nvi_score=nvi_score,
        burst_zscore=burst,
        spread_factor=spread,
        mutation_penalty=mutation,
        coordination_mult=coord,
        post_count=cluster.get("post_count", 0),
        source_diversity=cluster.get("source_diversity", 0),
    )

    # Lifecycle
    lifecycle = _get_lifecycle_data(db, cluster_id)

    # NVI components (entity concentration, topic cluster flag)
    nvi_comps = _get_nvi_components(db, cluster_id)

    # Campaign
    campaign = _get_campaign_info(db, cluster_id)

    # Linked narratives
    links = get_narrative_links(db, cluster_id)

    # Source credibility
    try:
        from intelligence.processing.source_credibility import get_cluster_source_breakdown
        source_breakdown = get_cluster_source_breakdown(db, cluster_id)
    except Exception as e:
        logger.warning("Source credibility failed for cluster %s: %s", cluster_id, e)
        source_breakdown = {"categories": {}, "weighted_credibility": 0.5, "sources": []}

    source_context = generate_source_context(source_breakdown.get("sources", []))

    lang_spread_detail = cluster.get("language_spread", {})
    cluster["dominant_language"] = max(lang_spread_detail.items(), key=lambda x: x[1])[0] if lang_spread_detail else "unknown"
    cluster["label"] = clean_label
    cluster["keywords"] = [kw for kw in keywords if kw.lower() not in KEYWORD_STOPLIST]

    return {
        "narrative": cluster,
        "interpretation": interpretation,
        "themes": detail_themes,
        "lifecycle": lifecycle,
        "campaign": campaign,
        "entity_concentration": nvi_comps["entity_concentration"],
        "is_topic_cluster": nvi_comps["is_topic_cluster"],
        "shared_entity_count": nvi_comps["shared_entity_count"],
        "linked_narratives": links,
        "source_credibility": {
            "breakdown": source_breakdown,
            "context": source_context,
        },
        "timeline": timeline,
        "posts": detail.get("posts", []),
        "coordination_signals": detail.get("coordination_signals", []),
        "raw": {
            "nvi_score": nvi_score,
            "raw_nvi": nvi_comps.get("raw_nvi", nvi_score),
            "burst_zscore": burst,
            "spread_factor": spread,
            "mutation_penalty": mutation,
            "coordination_mult": coord,
            "alert_level": nvi_comps.get("effective_alert_level", "normal"),
            "effective_alert_level": nvi_comps.get("effective_alert_level", "normal"),
            "confidence_probability": nvi_comps.get("confidence_probability", 0.5),
            "alert_suppressed": nvi_comps.get("alert_suppressed", False),
            "wire_service_syndication": nvi_comps.get("wire_service_syndication", False),
            "ensemble_uncertain": nvi_comps.get("ensemble_uncertain", False),
            "ensemble_disagreement": nvi_comps.get("ensemble_disagreement", 0.0),
            "ensemble_perfect_agreement_red_flag": nvi_comps.get("ensemble_perfect_agreement_red_flag", False),
            "gdelt_batch_artifact": nvi_comps.get("gdelt_batch_artifact", False),
            "dna_match_count": nvi_comps.get("dna_match_count", -1),
            "dna_evidence_strong": nvi_comps.get("dna_evidence_strong", False),
            "entity_concentration": nvi_comps["entity_concentration"],
            "is_topic_cluster": nvi_comps["is_topic_cluster"],
            "shared_entity_count": nvi_comps["shared_entity_count"],
            "gates_applied": nvi_comps.get("gates_applied", []),
            "gate_reasoning": nvi_comps.get("gate_reasoning", {}),
            "narrative_coherence": nvi_comps.get("narrative_coherence", 0.5),
            "unique_hash_ratio": nvi_comps.get("unique_hash_ratio", 0.0),
            "embedding_similarity_mean": nvi_comps.get("embedding_similarity_mean", 0.0),
        },
        "falsification": _build_falsification_block(
            nvi_comps,
            cluster.get("post_count", 0),
            mutation, coord, burst,
            cluster.get("source_diversity", 0),
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/narratives/{cluster_id}/evidence", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_EXPENSIVE)
async def narrative_evidence_pack(request: Request, cluster_id: int):
    """Generate Berkeley Protocol compliant evidence pack."""
    from intelligence.evidence.generate import generate_evidence_pack

    db = get_db()
    pack = generate_evidence_pack(db, cluster_id)

    if not pack:
        raise HTTPException(status_code=404, detail="Cannot generate evidence pack")

    return pack


@app.get("/api/narratives/{cluster_id}/links", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_DEFAULT)
async def narrative_links_endpoint(request: Request, cluster_id: int):
    """Get all narratives linked to this one."""
    db = get_db()
    links = get_narrative_links(db, cluster_id)
    return {
        "cluster_id": cluster_id,
        "links": links,
        "count": len(links),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/narratives/{cluster_id}/verdict", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_DEFAULT)
async def narrative_verdict(request: Request, cluster_id: int):
    """Deterministic verdict based on NVI component values.

    Returns a human-readable verdict explaining whether the system trusts
    its own narrative velocity assessment, with supporting evidence and falsification
    criteria evaluation. No ML — pure threshold-based rules.
    """
    db = get_db()

    cluster = db.execute("""
        SELECT id, post_count, source_diversity
        FROM narrative_clusters WHERE id = ?
    """, (cluster_id,)).fetchone()

    if not cluster:
        raise HTTPException(status_code=404, detail="Narrative not found")

    nvi_comps = _get_nvi_components(db, cluster_id)
    post_count = cluster["post_count"] or 0
    source_div = cluster["source_diversity"] or 0

    # Get raw signals from latest NVI snapshot
    snapshot = db.execute("""
        SELECT nvi_score, mutation_penalty, burst_zscore, coordination_mult
        FROM nvi_snapshots
        WHERE cluster_id = ? ORDER BY id DESC LIMIT 1
    """, (cluster_id,)).fetchone()

    if snapshot:
        nvi_score = snapshot["nvi_score"] or 0
        mutation = snapshot["mutation_penalty"] or 0
        burst = snapshot["burst_zscore"] or 0
        coord = snapshot["coordination_mult"] or 0
    else:
        nvi_score = 0
        mutation = nvi_comps.get("entity_concentration", 0.5)
        burst = 0
        coord = 0

    confidence = nvi_comps.get("confidence_probability", 0.5)
    alert_suppressed = nvi_comps.get("alert_suppressed", False)
    effective_alert_level = nvi_comps.get("effective_alert_level", "normal")
    wire_service_syndication = nvi_comps.get("wire_service_syndication", False)
    gdelt_batch_artifact = nvi_comps.get("gdelt_batch_artifact", False)
    is_topic_cluster = nvi_comps.get("is_topic_cluster", False)
    ensemble_perfect_agreement_red_flag = nvi_comps.get("ensemble_perfect_agreement_red_flag", False)
    dna_evidence_strong = nvi_comps.get("dna_evidence_strong", False)
    dna_match_count = nvi_comps.get("dna_match_count", -1)

    evidence_for = []
    evidence_against = []

    # Deterministic verdict classification
    has_snapshot = snapshot is not None

    if not has_snapshot:
        verdict = "INSUFFICIENT_DATA"
        confidence = 0.0
        evidence_against.append("No NVI snapshot exists for this narrative")
    elif post_count < 5:
        verdict = "INSUFFICIENT_DATA"
        confidence = min(confidence, 0.3)
        evidence_against.append(f"Fewer than 5 posts ({post_count}) — insufficient data volume for reliable analysis")
    elif (
        alert_suppressed
        or (wire_service_syndication and mutation < 0.10 and source_div < 0.3)
        or gdelt_batch_artifact
        or (is_topic_cluster and ensemble_perfect_agreement_red_flag)
    ):
        verdict = "NARRATIVE_SIGNAL_REDUCED"
        confidence = confidence * 0.5
        if gdelt_batch_artifact:
            evidence_against.append("Temporal patterns originate from GDELT batch cycle — not real-time publishing")
        if alert_suppressed:
            evidence_against.append("Alert was suppressed by confidence gating")
        if wire_service_syndication and mutation < 0.10 and source_div < 0.3:
            evidence_against.append("Wire service content with near-zero mutation and low source diversity")
        if is_topic_cluster and ensemble_perfect_agreement_red_flag:
            evidence_against.append("Topic cluster with high ensemble agreement on small sample — likely data aggregation")

    elif (
        nvi_score > 60
        and not alert_suppressed
        and confidence > 0.65
        and (dna_evidence_strong or (mutation < 0.15 and source_div > 0.5))
    ):
        verdict = "NARRATIVE_SIGNAL_HIGH"
    elif (
        nvi_score > 60
        and not alert_suppressed
        and confidence > 0.65
        and (dna_evidence_strong or (mutation < 0.15 and source_div > 0.5))
    ):
        verdict = "NARRATIVE_SIGNAL_HIGH"
        evidence_for.append(f"NVI score {nvi_score:.1f} exceeds high-velocity threshold (60)")
        if dna_evidence_strong:
            evidence_for.append(f"Strong cross-cluster content fingerprint ({dna_match_count} matches) detected across independent clusters")
        if mutation < 0.15 and source_div > 0.5:
            evidence_for.append(f"Near-identical content (mutation={mutation:.2f}) across diverse sources (diversity={source_div:.2f})")
    else:
        verdict = "UNCERTAIN"
        if nvi_score <= 60:
            evidence_against.append(f"NVI score ({nvi_score:.1f}) below coordination threshold (60)")
        if confidence <= 0.65:
            evidence_against.append(f"Confidence probability ({confidence:.2f}) below 0.65 threshold")
        if not dna_evidence_strong:
            evidence_against.append("No cross-cluster content fingerprint matches found")
        evidence_for.append("Human review recommended — system cannot definitively classify this narrative")

    # Falsification criteria evaluation for verdict
    criteria_triggered = []
    criteria_satisfied = []

    if mutation > 0.30 and source_div > 0.60:
        criteria_triggered.append(
            f"Organic spread pattern detected: mutation={mutation:.2f} > 0.30 AND diversity={source_div:.2f} > 0.60"
        )
    else:
        if mutation <= 0.30:
            criteria_satisfied.append(
                f"Content mutation is low ({mutation:.2f} <= 0.30)"
            )
        if source_div <= 0.60:
            criteria_satisfied.append(
                f"Source diversity is narrow ({source_div:.2f} <= 0.60)"
            )
        if source_div > 0.60 and mutation <= 0.30:
            criteria_satisfied.append(
                f"High source diversity ({source_div:.2f}) with low mutation ({mutation:.2f}) suggests shared origin content"
            )

    if coord < 1.10 and burst < 5:
        criteria_triggered.append(
            f"Normal coverage timing: timing_mult={coord:.2f} < 1.10 AND burst={burst:.2f} < 5"
        )
    else:
        if coord >= 1.10:
            criteria_satisfied.append(
                f"Timing multiplier ({coord:.2f}) exceeds normal cycle range (1.10)"
            )
        if burst >= 5:
            criteria_satisfied.append(
                f"Burst z-score ({burst:.2f}) exceeds normal cycle range (5)"
            )

    if wire_service_syndication:
        criteria_triggered.append("Wire service syndication detected — content from news agencies, not independent actors")
    else:
        criteria_satisfied.append("No wire service syndication detected")

    if post_count < 10:
        criteria_triggered.append(f"Insufficient evidence: post_count={post_count} < 10")
    else:
        criteria_satisfied.append(f"Sufficient data volume (post_count={post_count})")

    # Recommended action
    action_map = {
        "NARRATIVE_SIGNAL_HIGH": "Escalate to human review. Cross-cluster fingerprints indicate broad narrative spread across sources.",
        "NARRATIVE_SIGNAL_REDUCED": "No immediate action required. Signal reliability reduced by quality factors (batch artifact, wire syndication, or topic cluster).",
        "UNCERTAIN": "Increase monitoring frequency. Allow additional data to accumulate before making a determination. DNA cross-matching may resolve ambiguity.",
        "INSUFFICIENT_DATA": "Collect more data. No reliable determination possible with current evidence volume.",
    }

    return {
        "cluster_id": cluster_id,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "nvi_score": nvi_score,
        "effective_alert_level": effective_alert_level,
        "alert_suppressed": alert_suppressed,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "falsification_criteria": {
            "criteria_triggered": criteria_triggered,
            "criteria_satisfied": criteria_satisfied,
        },
        "recommended_action": action_map.get(verdict, "Monitor and reassess"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/campaigns", dependencies=[Depends(require_api_key)])
async def list_campaigns(limit: int = Query(default=10, ge=1, le=50)):
    """Active multi-narrative campaigns."""
    db = get_db()
    campaigns = get_campaigns(db, limit=limit)

    enriched = []
    for c in campaigns:
        narrative_ids = _parse_json_field(c.get("narrative_ids", "[]"), [])
        c["narrative_ids"] = narrative_ids
        c["evidence"] = _parse_json_field(c.get("evidence", "{}"), {})
        enriched.append(c)

    return {
        "campaigns": enriched,
        "count": len(enriched),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/campaigns/{campaign_id}", dependencies=[Depends(require_api_key)])
async def campaign_detail_endpoint(campaign_id: int):
    """Full campaign detail with all linked narratives."""
    db = get_db()
    detail = get_campaign_detail(db, campaign_id)

    if not detail or not detail.get("campaign"):
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Enrich each narrative in the campaign
    enriched_narratives = []
    for n in detail.get("narratives", []):
        try:
            enriched, _ = _enrich_narrative(n, db)
            enriched_narratives.append(enriched)
        except Exception as e:
            logger.warning("Failed to enrich narrative %s in campaign: %s", n.get("id"), e)
            enriched_narratives.append({"id": n.get("id"), "label": n.get("label", "")})

    campaign = detail["campaign"]
    campaign["narrative_ids"] = _parse_json_field(campaign.get("narrative_ids", "[]"), [])
    campaign["evidence"] = _parse_json_field(campaign.get("evidence", "{}"), {})

    return {
        "campaign": campaign,
        "narratives": enriched_narratives,
        "links": detail.get("links", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/stats", dependencies=[Depends(require_api_key)])
async def system_stats(response: Response):
    response.headers["Cache-Control"] = "public, max-age=15"
    db = get_db()
    stats = get_stats(db)

    # Add new stats
    stats["coordination_signals"] = db.execute("""
        SELECT COUNT(*) FROM coordination_signals cs
        JOIN narrative_clusters nc ON nc.id = cs.cluster_id AND nc.status = 'active'
    """).fetchone()[0]
    stats["active_campaigns"] = db.execute(
        "SELECT COUNT(*) FROM campaigns WHERE status = 'active'"
    ).fetchone()[0]
    stats["source_scores"] = db.execute(
        "SELECT COUNT(*) FROM source_scores"
    ).fetchone()[0]
    stats["dna_fingerprints"] = db.execute("""
        SELECT COUNT(*) FROM narrative_dna nd
        JOIN narrative_clusters nc ON nc.id = nd.cluster_id AND nc.status = 'active'
    """).fetchone()[0]
    stats["dna_matches"] = db.execute("""
        SELECT COUNT(*) FROM dna_matches dm
        JOIN narrative_clusters ca ON ca.id = dm.cluster_a AND ca.status = 'active'
        JOIN narrative_clusters cb ON cb.id = dm.cluster_b AND cb.status = 'active'
        WHERE dm.match_score >= 0.75
    """).fetchone()[0]
    stats["human_reviews"] = db.execute(
        "SELECT COUNT(*) FROM human_reviews"
    ).fetchone()[0]

    # Baseline maturity — burst z-score and coordination_mult need ~24h of
    # historical post counts to produce meaningful values. Until then the
    # engine is "calibrating" and burst-based gates can't surface unusual
    # activity. Tell the dashboard so it can show a calibration banner.
    earliest_post = db.execute(
        "SELECT MIN(ingested_at) FROM raw_posts"
    ).fetchone()[0]
    if earliest_post:
        try:
            earliest_dt = datetime.fromisoformat(earliest_post.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - earliest_dt).total_seconds() / 3600
            stats["baseline_age_hours"] = round(age_hours, 1)
            stats["baseline_mature"] = age_hours >= 24
            stats["baseline_status"] = (
                "mature" if age_hours >= 24
                else "calibrating" if age_hours >= 6
                else "starting"
            )
        except (ValueError, TypeError):
            stats["baseline_status"] = "unknown"
            stats["baseline_mature"] = False
    else:
        stats["baseline_status"] = "no_data"
        stats["baseline_mature"] = False

    # Lifecycle distribution
    lifecycle_rows = db.execute("""
        SELECT lifecycle_phase, COUNT(*) as cnt
        FROM narrative_clusters WHERE status = 'active'
        GROUP BY lifecycle_phase
    """).fetchall()
    stats["lifecycle_distribution"] = {r["lifecycle_phase"] or "emergence": r["cnt"] for r in lifecycle_rows}

    # Multi-resolution stats
    resolution_rows = db.execute("""
        SELECT json_extract(metadata, '$.resolution') as res, COUNT(*) as cnt
        FROM narrative_clusters WHERE status = 'active'
        GROUP BY res
    """).fetchall()
    stats["resolution_distribution"] = {
        str(r["res"] or "unknown"): r["cnt"] for r in resolution_rows
    }

    # Alerts by dominant language
    from collections import Counter
    alert_lang_rows = db.execute("""
        SELECT nc.language_spread
        FROM narrative_clusters nc
        JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
        WHERE nc.status = 'active'
        AND nv.alert_level IN ('elevated', 'critical')
    """).fetchall()
    alert_langs: Counter = Counter()
    for r in alert_lang_rows:
        ls = _parse_json_field(r["language_spread"], {})
        if ls:
            dominant = max(ls.items(), key=lambda x: x[1])[0]
            alert_langs[dominant] += 1
    stats["alerts_by_language"] = dict(alert_langs)

    return {
        "stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/alerts", dependencies=[Depends(require_api_key)])
async def active_alerts(
    hours: int = Query(default=24, ge=1, le=168),
    lang: str = Query(default="en"),
):
    """Active alerts with human-readable interpretation. lang=en filters to English."""
    db = get_db()

    rows = db.execute("""
        SELECT nc.id, nc.label, nc.keywords, nc.post_count,
               nc.source_diversity, nc.language_spread,
               nv.nvi_score, nv.burst_zscore, nv.spread_factor,
               nv.mutation_penalty, nv.coordination_mult,
               nv.alert_level, nv.timestamp, nv.raw_components
        FROM narrative_clusters nc
        JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
        WHERE nc.status = 'active'
        AND nv.timestamp >= datetime('now', ?)
        AND nv.alert_level IN ('elevated', 'critical')
        ORDER BY nv.nvi_score DESC
    """, (f'-{hours} hours',)).fetchall()

    alerts = []
    for r in rows:
        enriched, _ = _enrich_narrative(dict(r), db)
        # Language filter: skip non-matching after enrichment
        if lang != "all":
            dom_lang = enriched.get("dominant_language", "unknown")
            if dom_lang == "translated":  # never count translated as English
                continue
            if dom_lang != lang:
                continue
        confidence_probability = enriched["raw"].get("confidence_probability", 0.5)
        alert_suppressed = enriched["raw"].get("alert_suppressed", False)
        if confidence_probability > 0.80:
            alert_quality = "high_confidence"
        elif confidence_probability > 0.65:
            alert_quality = "moderate_confidence"
        elif alert_suppressed:
            alert_quality = "suppressed_low_confidence"
        else:
            alert_quality = "low_confidence"
        enriched["alert_quality"] = alert_quality
        alerts.append(enriched)

    return {
        "alerts": alerts,
        "count": len(alerts),
        "window_hours": hours,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/analytics/timeline", dependencies=[Depends(require_api_key)])
async def analytics_timeline(hours: int = Query(default=168, ge=1, le=720)):
    """Hour-by-hour NVI aggregate scores for heatmap visualization."""
    db = get_db()

    rows = db.execute("""
        SELECT strftime('%Y-%m-%dT%H:00:00Z', timestamp) as hour,
               COUNT(DISTINCT cluster_id) as narrative_count,
               AVG(nvi_score) as avg_nvi,
               MAX(nvi_score) as max_nvi,
               SUM(CASE WHEN json_extract(raw_components, '$.effective_alert_level') = 'critical' THEN 1 ELSE 0 END) as critical_count
        FROM nvi_snapshots
        WHERE timestamp >= datetime('now', ?)
        GROUP BY hour
        ORDER BY hour ASC
    """, (f'-{hours} hours',)).fetchall()

    return {
        "timeline": [dict(r) for r in rows],
        "hours": hours,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/analytics/sources", dependencies=[Depends(require_api_key)])
async def analytics_sources():
    """Source participation in coordinated vs organic narratives."""
    db = get_db()

    rows = db.execute("""
        SELECT rp.source,
               COUNT(DISTINCT cm.cluster_id) as cluster_count,
               COUNT(DISTINCT rp.id) as post_count,
               ss.credibility_score,
               ss.category
        FROM raw_posts rp
        JOIN cluster_members cm ON cm.post_id = rp.id
        LEFT JOIN source_scores ss ON ss.domain = rp.source
        GROUP BY rp.source
        ORDER BY cluster_count DESC
        LIMIT 50
    """).fetchall()

    return {
        "sources": [dict(r) for r in rows],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/search", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_SEARCH)
async def search_narratives(
    request: Request,
    q: str = Query(..., min_length=2),
    min_nvi: float = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Search narratives by keyword."""
    db = get_db()

    # Search in cluster labels and keywords
    rows = db.execute("""
        SELECT nc.id, nc.label, nc.keywords, nc.post_count,
               nc.source_diversity, nc.first_seen, nc.last_updated,
               nv.nvi_score, nv.burst_zscore, nv.spread_factor,
               nv.mutation_penalty, nv.coordination_mult,
               nv.alert_level, nv.timestamp as nvi_timestamp
        FROM narrative_clusters nc
        LEFT JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
        WHERE nc.status = 'active'
        AND (nc.label LIKE ? OR nc.keywords LIKE ?)
        AND COALESCE(nv.nvi_score, 0) >= ?
        ORDER BY COALESCE(nv.nvi_score, 0) DESC
        LIMIT ?
    """, (f'%{q}%', f'%{q}%', min_nvi, limit)).fetchall()

    results = []
    for r in rows:
        enriched, _ = _enrich_narrative(dict(r), db)
        results.append(enriched)

    return {
        "query": q,
        "results": results,
        "count": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/operations", dependencies=[Depends(require_api_key)])
async def list_operations(limit: int = Query(default=10, ge=1, le=50)):
    """Active intelligence operations — campaigns with their narratives and alert status."""
    db = get_db()
    campaigns = get_campaigns(db, limit=limit)

    operations = []
    for c in campaigns:
        narrative_ids = _parse_json_field(c.get("narrative_ids", "[]"), [])
        c["narrative_ids"] = narrative_ids
        c["evidence"] = _parse_json_field(c.get("evidence", "{}"), {})

        # Count alerts within this operation's narratives
        critical = 0
        elevated = 0
        for nid in narrative_ids:
            row = db.execute("""
                SELECT alert_level FROM nvi_snapshots
                WHERE cluster_id = ? ORDER BY id DESC LIMIT 1
            """, (nid,)).fetchone()
            if row:
                if row["alert_level"] == "critical":
                    critical += 1
                elif row["alert_level"] == "elevated":
                    elevated += 1

        c["alert_summary"] = {
            "critical": critical,
            "elevated": elevated,
            "total_narratives": len(narrative_ids),
        }
        operations.append(c)

    return {
        "operations": operations,
        "count": len(operations),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── DNA Endpoints ──────────────────────────────────────────────────────────
# IMPORTANT: literal-path routes must be declared BEFORE parameterized routes,
# otherwise FastAPI matches `/api/dna/matches` as `cluster_id="matches"` and 422s.

@app.get("/api/dna/matches", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_EXPENSIVE)
async def list_dna_matches(
    request: Request,
    response: Response,
    min_score: float = Query(default=0.60, ge=0.50, le=1.0),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List all DNA matches across the system."""
    response.headers["Cache-Control"] = "public, max-age=30"
    from intelligence.db import get_all_dna_matches

    db = get_db()
    matches = get_all_dna_matches(db, min_score=min_score, limit=limit)

    return {
        "matches": matches,
        "count": len(matches),
        "min_score": min_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/dna/{cluster_id}", dependencies=[Depends(require_api_key)])
@limiter.limit(_RATE_EXPENSIVE)
async def get_dna_fingerprint(request: Request, cluster_id: int, response: Response):
    """Get the full DNA fingerprint for a narrative cluster."""
    response.headers["Cache-Control"] = "public, max-age=60"
    from intelligence.db import get_dna_fingerprint, get_dna_matches
    from intelligence.processing.dna import compute_dna_fingerprint, store_dna

    db = get_db()
    dna = get_dna_fingerprint(db, cluster_id)

    if not dna:
        # Compute on demand
        try:
            fingerprint = compute_dna_fingerprint(db, cluster_id)
            store_dna(db, cluster_id, fingerprint)
            dna = get_dna_fingerprint(db, cluster_id)
        except Exception as e:
            logger.warning(f"DNA computation failed: {e}")
            raise HTTPException(status_code=404, detail="Cannot compute DNA fingerprint")

    matches = get_dna_matches(db, cluster_id)

    return {
        "cluster_id": cluster_id,
        "dna": dna,
        "matches": matches,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/review", dependencies=[Depends(require_api_key)])
async def submit_review(
    cluster_id: int,
    verdict: str = Query(..., pattern="^(coordinated|organic|uncertain)$"),
    reviewer: str = Query(default=""),
    notes: str = Query(default=""),
):
    """Submit a human review verdict for closed-loop validation."""
    from intelligence.db import insert_review

    db = get_db()
    insert_review(db, cluster_id, verdict, reviewer, notes)

    return {
        "status": "recorded",
        "cluster_id": cluster_id,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/reviews", dependencies=[Depends(require_api_key)])
async def list_reviews(
    cluster_id: int = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get human reviews, optionally filtered by cluster."""
    from intelligence.db import get_reviews

    db = get_db()
    reviews = get_reviews(db, cluster_id=cluster_id, limit=limit)

    return {
        "reviews": reviews,
        "count": len(reviews),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/analytics/precision", dependencies=[Depends(require_api_key)])
async def precision_analytics(response: Response):
    """Get precision/recall/F1 from human review labels."""
    response.headers["Cache-Control"] = "public, max-age=60"
    from intelligence.db import get_review_stats

    db = get_db()
    stats = get_review_stats(db)

    return {
        "precision_analytics": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Graph Topology Endpoints ───────────────────────────────────────────────

@app.get("/api/graph/status", dependencies=[Depends(require_api_key)])
async def graph_status(response: Response):
    """Get the latest amplification graph snapshot and metrics."""
    response.headers["Cache-Control"] = "public, max-age=60"
    from intelligence.db import get_latest_graph_snapshot

    db = get_db()
    snapshot = get_latest_graph_snapshot(db)

    if not snapshot:
        return {
            "status": "no_snapshot",
            "message": "Graph engine has not run yet. Start continuous pipeline.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "graph_snapshot": snapshot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/graph/cluster/{cluster_id}", dependencies=[Depends(require_api_key)])
async def cluster_subgraph(cluster_id: int, response: Response):
    """Get the ego amplification graph for a specific narrative cluster."""
    response.headers["Cache-Control"] = "public, max-age=60"
    from intelligence.processing.graph_engine import get_cluster_subgraph, compute_graph_metrics

    db = get_db()
    G = get_cluster_subgraph(db, cluster_id)

    if G is None or G.number_of_nodes() < 2:
        return {
            "cluster_id": cluster_id,
            "node_count": G.number_of_nodes() if G else 0,
            "edge_count": G.number_of_edges() if G else 0,
            "message": "Insufficient data for ego graph",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    metrics = compute_graph_metrics(G)

    # Serialize graph as adjacency list
    graph_data = {
        "nodes": list(G.nodes()),
        "edges": [
            {"from": u, "to": v, "weight": d.get("weight", 1)}
            for u, v, d in G.edges(data=True)
        ],
    }

    return {
        "cluster_id": cluster_id,
        "graph": graph_data,
        "topology_metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


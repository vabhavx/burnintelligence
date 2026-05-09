"""
BurnTheLies Intelligence — Main Orchestrator
Runs the full pipeline: ingest → embed → cluster → NVI → lifecycle → cross-narrative → credibility.
"""

import asyncio
import logging
import sys
import signal
import time
from datetime import datetime, timezone

from intelligence import health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("intelligence/data/intelligence.log", mode="a"),
    ]
)
logger = logging.getLogger("intelligence.main")

# Pipeline intervals
GDELT_INTERVAL = 900       # 15 minutes (matches GDELT update cycle)
BLUESKY_DURATION = 300     # 5 minutes per firehose session
EMBED_INTERVAL = 600       # 10 minutes
CLUSTER_INTERVAL = 1800    # 30 minutes
NVI_INTERVAL = 900         # 15 minutes
CROSS_NARRATIVE_INTERVAL = 3600  # 1 hour (O(n^2) on clusters)
CREDIBILITY_INTERVAL = 7200     # 2 hours
DNA_INTERVAL = 1800        # 30 min — must match CLUSTER_INTERVAL so NVI sees fresh DNA
GRAPH_INTERVAL = 3600      # 1 hour (ties to cross-narrative)

# Per-stage timeouts (seconds)
STAGE_TIMEOUTS = {
    "gdelt": 120,
    "bluesky": 360,
    "embed": 900,
    "cluster": 1800,
    "nvi": 900,
    "lifecycle": 300,
    "cross_narrative": 600,
    "graph": 600,
    "dna": 7200,
    "credibility": 600,
    "maintenance": 300,
}

_shutdown = False


def handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


async def _run_stage(name: str, func, *args, **kwargs):
    """Run a stage with timeout + health tracking. Returns (ok, result)."""
    timeout = STAGE_TIMEOUTS[name]
    health.stage_started(name)
    try:
        if asyncio.iscoroutinefunction(func):
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs), timeout=timeout
            )
        health.stage_completed(name, "ok")
        return True, result
    except asyncio.TimeoutError:
        logger.error(f"{name} stage timed out after {timeout}s")
        health.stage_completed(name, "timeout", error=f"timeout after {timeout}s")
        return False, None
    except Exception as e:
        logger.error(f"{name} stage failed: {e}")
        health.stage_completed(name, "failed", error=str(e))
        return False, None


async def _gdelt(db_conn) -> dict:
    from intelligence.ingestors.gdelt import ingest_cycle
    stats = await ingest_cycle(db_conn)
    logger.info(f"GDELT: {stats.get('articles_stored', 0)} new articles")
    return stats


async def _bluesky(db_conn) -> dict:
    from intelligence.ingestors.bluesky import connect_firehose
    stats = await connect_firehose(
        db_conn,
        duration_seconds=BLUESKY_DURATION,
        max_posts=500,
    )
    logger.info(f"Bluesky: {stats.get('stored', 0)} signal posts from {stats.get('received', 0)} total")
    return stats


def _embed(db_conn) -> int:
    from intelligence.processing.embed import embed_and_store
    count = embed_and_store(db_conn)
    logger.info(f"Embedded {count} new posts")
    return count


def _cluster(db_conn):
    from intelligence.processing.cluster import cluster_narratives_multi_resolution
    clusters = cluster_narratives_multi_resolution(db_conn)
    logger.info(f"Discovered {len(clusters)} narrative clusters")
    return clusters


def _nvi(db_conn):
    from intelligence.processing.nvi import compute_all_nvi
    results = compute_all_nvi(db_conn)
    critical = [r for r in results if r.get("nvi_score", 0) >= 80]
    if critical:
        logger.warning(f"CRITICAL ALERTS: {len(critical)} narratives above NVI 80")
        for c in critical:
            logger.warning(f"  NVI={c['nvi_score']} cluster_id={c['cluster_id']}")
    return results


def _lifecycle(db_conn):
    from intelligence.processing.lifecycle import classify_all_lifecycles
    results = classify_all_lifecycles(db_conn)
    logger.info(f"Lifecycle classified for {len(results)} clusters")
    return results


def _cross_narrative(db_conn) -> dict:
    from intelligence.processing.cross_narrative import run_cross_narrative_cycle as _run
    stats = _run(db_conn)
    logger.info(f"Cross-narrative: {stats.get('links', 0)} links, {stats.get('campaigns', 0)} campaigns")
    return stats


def _credibility(db_conn) -> int:
    from intelligence.processing.source_credibility import seed_source_scores, compute_dynamic_adjustments
    seeded = seed_source_scores(db_conn)
    adjusted = compute_dynamic_adjustments(db_conn)
    logger.info(f"Source credibility: {seeded} seeded, {adjusted} dynamically adjusted")
    return adjusted


def _dna(db_conn) -> dict:
    from intelligence.processing.dna import run_dna_cycle as _run
    stats = _run(db_conn)
    logger.info(
        f"DNA: {stats.get('fingerprints_computed', 0)} fingerprints, "
        f"{stats.get('cross_matches', 0)} cross-matches"
    )
    return stats


def _graph(db_conn) -> dict:
    from intelligence.processing.graph_engine import run_graph_cycle as _run
    metrics = _run(db_conn)
    logger.info(
        f"Graph: {metrics.get('node_count', 0)} nodes, "
        f"{metrics.get('edge_count', 0)} edges, "
        f"coordination={metrics.get('is_coordination_topology', False)}"
    )
    return metrics


def _maintenance(db_conn) -> dict:
    from intelligence.processing.retention import maintenance_cycle as _run
    stats = _run(db_conn)
    logger.info("Maintenance: %s", stats)
    return stats


async def run_pipeline_once(db_conn):
    """Run full pipeline once with per-stage timeouts and health tracking."""
    logger.info("=" * 60)
    logger.info("PIPELINE CYCLE START")
    logger.info("=" * 60)

    health.cycle_started()
    start = time.time()
    failures = 0

    # Phase 1: Ingest (parallel) — but each stage tracked separately
    logger.info("Phase 1: Ingestion")
    gdelt_ok, gdelt_stats = await _run_stage("gdelt", _gdelt, db_conn)
    bluesky_ok, bluesky_stats = await _run_stage("bluesky", _bluesky, db_conn)
    failures += (not gdelt_ok) + (not bluesky_ok)

    # Phase 2: Embed
    logger.info("Phase 2: Embedding")
    embed_ok, embed_count = await _run_stage("embed", _embed, db_conn)
    failures += not embed_ok

    # Phase 3: Cluster — multi-resolution (3, 5, 10, 25)
    logger.info("Phase 3: Multi-Resolution Clustering")
    cluster_ok, clusters = await _run_stage("cluster", _cluster, db_conn)
    failures += not cluster_ok

    # Phase 4: Narrative DNA — must run BEFORE NVI so the dna_match gate
    # has live cross-cluster fingerprint counts (otherwise it stamps -1
    # into the snapshot and caps NVI at 25 for hours).
    logger.info("Phase 4: DNA Fingerprinting")
    dna_ok, dna_stats = await _run_stage("dna", _dna, db_conn)
    failures += not dna_ok

    # Phase 5: NVI — ensemble v2.0
    logger.info("Phase 5: Ensemble NVI Scoring")
    nvi_ok, nvi_results = await _run_stage("nvi", _nvi, db_conn)
    failures += not nvi_ok

    # Phase 6: Lifecycle classification
    logger.info("Phase 6: Lifecycle Classification")
    lifecycle_ok, lifecycle_results = await _run_stage("lifecycle", _lifecycle, db_conn)
    failures += not lifecycle_ok

    # Phase 7: Cross-narrative campaign detection
    logger.info("Phase 7: Cross-Narrative Analysis")
    cross_ok, cross_stats = await _run_stage("cross_narrative", _cross_narrative, db_conn)
    failures += not cross_ok

    # Phase 8: Amplification graph topology
    logger.info("Phase 8: Graph Topology Analysis")
    graph_ok, graph_metrics = await _run_stage("graph", _graph, db_conn)
    failures += not graph_ok

    # Phase 9: Source credibility
    logger.info("Phase 9: Source Credibility")
    cred_ok, credibility_count = await _run_stage("credibility", _credibility, db_conn)
    failures += not cred_ok

    # Phase 10: Maintenance (retention, WAL checkpoint, DNA cap, disk watchdog)
    logger.info("Phase 10: Maintenance")
    maint_ok, maint_stats = await _run_stage("maintenance", _maintenance, db_conn)
    if not maint_ok:
        logger.warning("Maintenance stage failed (non-critical)")

    elapsed = time.time() - start
    if failures == 0:
        status = "ok"
    elif failures >= 10:
        status = "failed"
    else:
        status = "partial"
    health.cycle_completed(elapsed, status)
    logger.info(f"PIPELINE CYCLE COMPLETE in {elapsed:.1f}s ({status}, {failures} failures)")

    try:
        from intelligence.db import set_pipeline_state, get_stats
        set_pipeline_state(db_conn, "last_cycle", datetime.now(timezone.utc).isoformat())
        set_pipeline_state(db_conn, "last_cycle_duration", f"{elapsed:.1f}")
        stats = get_stats(db_conn)
        logger.info(f"System stats: {stats}")
    except Exception as e:
        logger.error(f"Failed to record pipeline state: {e}")
        stats = {}

    return {
        "gdelt": gdelt_stats,
        "bluesky": bluesky_stats,
        "embeddings": embed_count,
        "clusters": len(clusters) if clusters else 0,
        "nvi_results": len(nvi_results) if nvi_results else 0,
        "lifecycle": len(lifecycle_results) if lifecycle_results else 0,
        "cross_narrative": cross_stats,
        "graph": graph_metrics,
        "dna": dna_stats,
        "credibility": credibility_count,
        "maintenance": maint_stats,
        "duration": elapsed,
        "status": status,
        "stats": stats,
    }


async def run_continuous(db_conn):
    """Run the pipeline continuously with staggered intervals."""
    logger.info("Starting continuous pipeline")

    last_gdelt = 0.0
    last_bluesky = 0.0
    last_embed = 0.0
    last_cluster = 0.0
    last_nvi = 0.0
    last_cross_narrative = 0.0
    last_credibility = 0.0
    last_dna = 0.0
    last_graph = 0.0
    last_maintenance = 0.0
    last_cycle_start = 0.0

    while not _shutdown:
        now = time.time()
        cycle_touched = False

        # GDELT every 15 min
        if now - last_gdelt >= GDELT_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("gdelt", _gdelt, db_conn)
            last_gdelt = now

        # Bluesky every 5+1 min
        if now - last_bluesky >= BLUESKY_DURATION + 60:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("bluesky", _bluesky, db_conn)
            last_bluesky = now

        # Embedding every 10 min
        if now - last_embed >= EMBED_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("embed", _embed, db_conn)
            last_embed = now

        # Multi-resolution clustering every 30 min
        if now - last_cluster >= CLUSTER_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("cluster", _cluster, db_conn)
            last_cluster = now

        # DNA every 30 min — must run BEFORE NVI so dna_match gate has live counts
        if now - last_dna >= DNA_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("dna", _dna, db_conn)
            last_dna = now

        # NVI every 15 min (after DNA so dna_match_count is fresh)
        if now - last_nvi >= NVI_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("nvi", _nvi, db_conn)
            await _run_stage("lifecycle", _lifecycle, db_conn)
            last_nvi = now

        # Cross-narrative every 1 hour
        if now - last_cross_narrative >= CROSS_NARRATIVE_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("cross_narrative", _cross_narrative, db_conn)
            last_cross_narrative = now

        # Source credibility every 2 hours
        if now - last_credibility >= CREDIBILITY_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("credibility", _credibility, db_conn)
            last_credibility = now

        # Graph every 1 hour
        if now - last_graph >= GRAPH_INTERVAL:
            if not cycle_touched:
                health.cycle_started()
                last_cycle_start = now
                cycle_touched = True
            await _run_stage("graph", _graph, db_conn)
            last_graph = now

        # Maintenance every 30 min
        if now - last_maintenance >= 1800:
            await _run_stage("maintenance", _maintenance, db_conn)
            last_maintenance = now

        if cycle_touched:
            elapsed = time.time() - last_cycle_start
            health.cycle_completed(elapsed, "ok")

        await asyncio.sleep(30)

    logger.info("Pipeline shutdown complete")


async def main():
    """Entry point."""
    from intelligence.db import get_connection, init_db
    from pathlib import Path

    # Ensure data directory exists
    Path("intelligence/data").mkdir(parents=True, exist_ok=True)

    db = get_connection()
    init_db(db)

    # Self-test: refuse to start if the gate pipeline is broken.
    from intelligence.processing.selftest import run_selftest
    selftest = run_selftest()
    if not selftest["ok"]:
        logger.critical("SELFTEST FAILED: %d/%d checks passed", selftest["passed"], selftest["checks"])
        for f in selftest.get("failures", []):
            logger.critical("  FAIL: %s → %s", f["name"], f)
        raise SystemExit("Pipeline startup blocked: selftest failed. Fix gate logic before starting.")
    logger.info("Selftest: %d/%d checks passed", selftest["passed"], selftest["checks"])

    mode = sys.argv[1] if len(sys.argv) > 1 else "once"

    if mode == "once":
        health.init("once")
        result = await run_pipeline_once(db)
        logger.info(f"Pipeline result: {result}")

    elif mode == "continuous":
        from intelligence import locking
        locking.acquire_lock_or_exit()
        health.init("continuous")
        try:
            import uvicorn
            from intelligence.api import app

            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            server = uvicorn.Server(config)

            await asyncio.gather(
                server.serve(),
                run_continuous(db),
            )
        finally:
            locking.release_lock()

    elif mode == "api":
        import uvicorn
        uvicorn.run("intelligence.api:app", host="0.0.0.0", port=8000, reload=True)

    else:
        logger.info("Usage: python -m intelligence.main [once|continuous|api]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

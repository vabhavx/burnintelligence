"""
BurnTheLies Intelligence — Evidence Pack Generator
Berkeley Protocol compliant digital evidence packages.
SHA256-signed, timestamped, forensically verifiable.

Enhanced with:
- Chain of custody documentation
- Full methodology description with limitations
- Alternative hypotheses
- Confidence intervals
- Falsification criteria
- Source credibility breakdown
- Narrative DNA fingerprint + cross-campaign matches
- Ensemble NVI scores + disagreement detection
- Amplification graph topology
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("intelligence.evidence")


VERDICT_MAP = {
    "gdelt_batch_artifact": "GDELT_BATCH_ARTIFACT",
    "insufficient_evidence": "INSUFFICIENT_EVIDENCE",
    "single_source_cluster": "SINGLE_SOURCE_CLUSTER",
    "wire_service": "WIRE_SERVICE_SYNDICATION",
    "dna_match": "NO_DNA_MATCH_FOUND",
    "ensemble_uncertainty": "ENSEMBLE_UNCERTAIN",
    "entity_concentration": "TOPIC_CLUSTER",
    "narrative_coherence": "INCOHERENT_NARRATIVE",
    "organic_viral_spread": "ORGANIC_VIRAL_SPREAD",
    "normal_news_cycle": "NORMAL_NEWS_CYCLE",
    "confidence_threshold": "LOW_CONFIDENCE",
}


def _build_falsification_assessment(comps: Optional[dict], nvi_timeline: list,
                                     post_count: int,
                                     cluster_metadata: Optional[dict] = None) -> dict:
    """
    Build falsification_assessment from the gate trace (comps["gates_applied"]
    and comps["gate_reasoning"]) — single source of truth shared with the alert
    pipeline. Avoids the prior bug where verdict was computed from raw signals
    and could contradict the gate-derived alert_level.
    """
    comps = comps or {}
    gates_fired = list(comps.get("gates_applied", []) or [])
    gate_reasoning = dict(comps.get("gate_reasoning", {}) or {})
    latest_nvi = nvi_timeline[-1] if nvi_timeline else {}
    alert_level = (latest_nvi or {}).get("alert_level", "normal")
    alert_suppressed = bool(comps.get("alert_suppressed", False))

    criteria = []
    for gate_name, reasoning in gate_reasoning.items():
        if not isinstance(reasoning, dict):
            continue
        criteria.append({
            "id": gate_name,
            "triggered": bool(reasoning.get("fired", False)),
            "description": reasoning.get("why", ""),
            "evidence": reasoning.get("evidence", {}),
        })

    # Topic-bag flag from clustering stage — single-source roundups of unrelated
    # articles sharing only a taxonomy code. Overrides all gate-derived verdicts.
    cluster_meta = cluster_metadata or {}
    if cluster_meta.get("single_source_topic_bag") in (True, "true"):
        verdict = "TOPIC_BAG"
    else:
        verdict = "NARRATIVE_ELEVATED"
        for g in gates_fired:
            if g in VERDICT_MAP:
                verdict = VERDICT_MAP[g]
                break

        if alert_suppressed and verdict == "NARRATIVE_ELEVATED":
            verdict = "ALERT_SUPPRESSED"

        if alert_level == "normal" and verdict == "NARRATIVE_ELEVATED":
            verdict = "NORMAL_COVERAGE"

    suppression_reason = None
    if alert_suppressed:
        suppression_reason = (
            f"Confidence probability {comps.get('confidence_probability', 'N/A')} "
            "below 0.65 threshold"
        )

    return {
        "criteria": criteria,
        "verdict": verdict,
        "gates_fired": gates_fired,
        "alert_level": alert_level,
        "suppression_reason": suppression_reason,
    }


def generate_evidence_pack(db_conn, cluster_id: int) -> dict:
    """
    Generate a Berkeley Protocol-aligned evidence pack for a narrative cluster.

    Structure:
    - metadata: pack ID, generation time, version, methodology
    - narrative: cluster info, NVI scores, timeline
    - evidence: source posts with hashes, coordination signals
    - chain_of_custody: full processing provenance
    - methodology: detection methods, limitations, falsification criteria
    - interpretation: confidence interval, alternative hypotheses
    - integrity: SHA256 chain ensuring tamper detection
    """
    from intelligence.db import get_narrative_detail
    from intelligence.processing.interpret import (
        interpret_narrative, compute_confidence_interval, _generate_alternatives,
    )
    from intelligence.processing.source_credibility import get_cluster_source_breakdown

    detail = get_narrative_detail(db_conn, cluster_id)
    if not detail or not detail.get("cluster"):
        return {}

    cluster = detail["cluster"]
    timeline = detail["timeline"]
    posts = detail["posts"]
    signals = detail["coordination_signals"]

    now = datetime.now(timezone.utc).isoformat()

    # Build evidence entries with individual content hashes
    evidence_entries = []
    for post in posts:
        content = post.get("content", "")
        entry = {
            "post_id": post["id"],
            "source": post["source"],
            "url": post.get("url", ""),
            "title": post.get("title", ""),
            "published_at": post.get("published_at", ""),
            "language": post.get("language", ""),
            "cluster_confidence": post.get("confidence", 0),
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }
        evidence_entries.append(entry)

    # NVI timeline with full component breakdown
    nvi_timeline = []
    for t in timeline:
        nvi_timeline.append({
            "timestamp": t.get("timestamp", ""),
            "nvi_score": t.get("nvi_score", 0),
            "burst_zscore": t.get("burst_zscore", 0),
            "spread_factor": t.get("spread_factor", 0),
            "mutation_penalty": t.get("mutation_penalty", 0),
            "coordination_mult": t.get("coordination_mult", 0),
            "alert_level": t.get("alert_level", "normal"),
        })

    # Coordination signals
    coord_evidence = []
    for sig in signals:
        evidence_data = sig.get("evidence", "{}")
        if isinstance(evidence_data, str):
            try:
                evidence_data = json.loads(evidence_data)
            except json.JSONDecodeError:
                evidence_data = {}
        coord_evidence.append({
            "signal_type": sig["signal_type"],
            "score": sig["score"],
            "evidence": evidence_data,
            "detected_at": sig.get("detected_at", ""),
        })

    # Parse cluster fields
    keywords = cluster.get("keywords", "[]")
    if isinstance(keywords, str):
        try:
            keywords = json.loads(keywords)
        except json.JSONDecodeError:
            keywords = []

    lang_spread = cluster.get("language_spread", "{}")
    if isinstance(lang_spread, str):
        try:
            lang_spread = json.loads(lang_spread)
        except json.JSONDecodeError:
            lang_spread = {}

    cluster_metadata = cluster.get("metadata", "{}")
    if isinstance(cluster_metadata, str):
        try:
            cluster_metadata = json.loads(cluster_metadata)
        except json.JSONDecodeError:
            cluster_metadata = {}

    # Get latest NVI values for interpretation
    nvi_score = 0
    burst = spread = mutation = coord = 0.0
    if nvi_timeline:
        latest = nvi_timeline[-1]
        nvi_score = latest.get("nvi_score", 0)
        burst = latest.get("burst_zscore", 0)
        spread = latest.get("spread_factor", 0)
        mutation = latest.get("mutation_penalty", 0)
        coord = latest.get("coordination_mult", 0)

    post_count = cluster.get("post_count", 0)
    source_div = cluster.get("source_diversity", 0)

    # Derive unique publication-domain count from post URLs for the
    # single-domain confidence cap. Mirrors nvi._extract_domains_from_posts
    # so both code paths agree on what counts as a domain.
    from urllib.parse import urlparse
    import re as _re
    _domains = set()
    for _p in posts:
        _u = _p.get("url") or ""
        if _u:
            try:
                _netloc = urlparse(_u).netloc
                _d = _re.sub(r"^www\.", "", _netloc.lower()) if _netloc else ""
                if _d and len(_d) > 3:
                    _domains.add(_d)
                    continue
            except (ValueError, TypeError):
                pass
        _src = _p.get("source")
        if _src:
            _domains.add(str(_src).lower())
    unique_domain_count = len(_domains)

    # Fetch NVI snapshot raw_components (for ensemble, confidence, falsification)
    comps = None
    ensemble_data = None
    latest_nvi_snapshot = db_conn.execute("""
        SELECT raw_components FROM nvi_snapshots
        WHERE cluster_id = ? ORDER BY id DESC LIMIT 1
    """, (cluster_id,)).fetchone()
    if latest_nvi_snapshot:
        try:
            comps = json.loads(latest_nvi_snapshot["raw_components"])
            # Refresh stale dna_match gate against live dna_matches table.
            # NVI snapshots stamp -1 when DNA hadn't run yet; this fixes the
            # "DNA fingerprint not yet computed" reasoning shown to users
            # hours after DNA actually computed.
            from intelligence.processing.nvi import refresh_dna_match_gate
            comps = refresh_dna_match_gate(db_conn, cluster_id, comps) or comps
            ensemble_data = {
                "ensemble_scores": comps.get("ensemble_scores", []),
                "ensemble_disagreement": comps.get("ensemble_disagreement", 0),
                "ensemble_uncertain": comps.get("ensemble_uncertain", False),
                "ensemble_perfect_agreement_red_flag": comps.get("ensemble_perfect_agreement_red_flag", False),
                "gdelt_batch_artifact": comps.get("gdelt_batch_artifact", False),
                "dna_match_count": comps.get("dna_match_count", 0),
                "dna_evidence_strong": comps.get("dna_evidence_strong", False),
            }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"cluster {cluster_id}: failed to parse NVI raw_components: {e}")

    # Generate confidence interval
    confidence = compute_confidence_interval(
        nvi_score, burst, spread, mutation, coord, post_count, source_div,
        dna_match_count=comps.get("dna_match_count", 0) if comps else 0,
        ensemble_red_flag=comps.get("ensemble_perfect_agreement_red_flag", False) if comps else False,
        gdelt_batch_artifact=comps.get("gdelt_batch_artifact", False) if comps else False,
        unique_domain_count=unique_domain_count,
        language_spread=lang_spread,
    )

    # Generate alternative hypotheses
    alternatives = _generate_alternatives(nvi_score, burst, spread, mutation, coord)

    # Source credibility breakdown
    source_breakdown = get_cluster_source_breakdown(db_conn, cluster_id)

    # DNA fingerprint
    dna_data = None
    dna_matches = []
    try:
        from intelligence.db import get_dna_fingerprint, get_dna_matches
        dna_data = get_dna_fingerprint(db_conn, cluster_id)
        dna_matches = get_dna_matches(db_conn, cluster_id)
    except Exception as e:
        logger.warning(f"cluster {cluster_id}: DNA enrichment failed: {e}")

    # Graph topology for this cluster
    cluster_graph = None
    try:
        from intelligence.processing.graph_engine import get_cluster_subgraph, compute_graph_metrics
        G = get_cluster_subgraph(db_conn, cluster_id)
        if G and G.number_of_nodes() >= 2:
            cluster_graph = {
                "metrics": compute_graph_metrics(G),
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
            }
    except Exception as e:
        logger.warning(f"cluster {cluster_id}: graph topology enrichment failed: {e}")

    # Assemble the pack
    pack = {
        "metadata": {
            "pack_version": "5.4.0",
            "generator": "BurnTheLies Intelligence Engine",
            "generated_at": now,
            "standard": "Berkeley Protocol on Digital Open Source Investigations (2022)",
            "cluster_id": cluster_id,
        },
        "alert_status": {
            "suppressed": bool(comps.get("alert_suppressed", False)) if comps else False,
            "reason": f"Confidence probability {comps.get('confidence_probability', 'N/A')} below 0.65 threshold" if (comps and comps.get("alert_suppressed")) else None,
            # v5: read from unified gate trace; fall back to legacy boolean
            # assembly for snapshots that predate the gate refactor.
            "gates_triggered": (
                list(comps.get("gates_applied"))
                if (comps and comps.get("gates_applied"))
                else [
                    gate for gate in [
                        "wire_service_syndication" if (comps and comps.get("wire_service_syndication")) else None,
                        "gdelt_batch_artifact" if (comps and comps.get("gdelt_batch_artifact")) else None,
                        "is_topic_cluster" if (comps and comps.get("is_topic_cluster")) else None,
                        "ensemble_perfect_agreement_red_flag" if (comps and comps.get("ensemble_perfect_agreement_red_flag")) else None,
                    ] if gate is not None
                ]
            ),
            "gate_reasoning": dict(comps.get("gate_reasoning", {})) if comps else {},
            "message": "The system detected unusual patterns but confidence is too low to alert. This evidence pack is for review, not action." if (comps and comps.get("alert_suppressed")) else "Alert passed confidence gate.",
        },
        "falsification_assessment": _build_falsification_assessment(
            comps, nvi_timeline, post_count, cluster_metadata
        ),
        "narrative": {
            "label": cluster.get("label", ""),
            "keywords": keywords,
            "summary": cluster.get("summary", ""),
            "first_seen": cluster.get("first_seen", ""),
            "last_updated": cluster.get("last_updated", ""),
            "post_count": post_count,
            "source_diversity": source_div,
            "language_spread": lang_spread,
            "latest_nvi": nvi_timeline[-1] if nvi_timeline else None,
        },
        "chain_of_custody": {
            "ingestion": {
                "source_apis": ["GDELT GKG v2 (15-minute updates)", "GDELT DOC 2.0 API", "Bluesky Jetstream (AT Protocol)"],
                "ingestion_method": "Automated pipeline with content_hash deduplication (SHA256)",
                "data_retention": "All raw posts preserved with original timestamps and metadata",
            },
            "processing": {
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (50M params, 384-dim)",
                "clustering_method": "Multi-resolution UMAP(n_components=5, metric=cosine) → HDBSCAN(min_cluster_size=[3,5,10,25], method=eom) with cross-resolution validation",
                "nvi_formula": "Ensemble NVI v4.1: 3 parallel coefficient sets (burst-heavy, spread-heavy, mutation-heavy) with weighted consensus. Hard gates for wire service syndication, topic clusters, and GDELT batch artifacts. Confidence-gated alerting with DNA match evidence.",
                "nvi_weights": {
                    "ensemble_burst_heavy": {"alpha": 0.35, "beta": 0.20, "gamma": 0.28, "delta": 0.20},
                    "ensemble_spread_heavy": {"alpha": 0.20, "beta": 0.35, "gamma": 0.28, "delta": 0.20},
                    "ensemble_mutation_heavy": {"alpha": 0.25, "beta": 0.25, "gamma": 0.30, "delta": 0.20},
                },
                "additional_signals": ["gdelt_batch_artifact_detection", "dna_fingerprint_cross_matching", "entity_concentration_gate", "ensemble_red_flag_detection"],
                "dna_fingerprinting": "84-dim multi-modal: stylometric(32) + cadence(16) + network_topology(12) + entity_bias(24)",
                "graph_analysis": "Directed source amplification graph with topology metrics",
                "scoring_timestamp": now,
            },
            "analysis": {
                "coordination_detection": "Temporal synchrony (inter-arrival CoV with GDELT batch artifact detection), content identity (embedding variance), source concentration (Shannon entropy with actual domain extraction), tone uniformity (sentiment variance), entity concentration (named entity overlap), DNA fingerprint matching (84-dim multi-modal cosine similarity)",
                "confidence_method": "Deterministic signal-strength aggregation with ensemble cross-validation and sample-size penalties",
                "interpretation_method": "Threshold-based rule engine, no ML or LLM generation",
                "dna_matching": "Weighted cosine similarity across 4 modalities. Threshold >= 0.75 = same operator.",
            },
        },
        "methodology": {
            "detection_method": (
                "Statistical anomaly detection across four dimensions: "
                "publication timing regularity (coefficient of variation of inter-arrival times), "
                "content similarity (embedding centroid variance), "
                "source ecosystem diversity (Shannon entropy), "
                "and emotional framing uniformity (sentiment variance). "
                "Signals are combined via a weighted sigmoid function to produce the narrative velocity index (NVI)."
            ),
            "limitations": [
                "Cannot distinguish intentional narrative push from coincidental editorial alignment",
                "Wire service (AP/Reuters/AFP) distribution produces identical content that mimics high-velocity signals",
                "Small sample sizes (<15 posts) significantly reduce confidence in all metrics",
                "GDELT metadata may contain misattributed themes or persons from automated extraction",
                "Bluesky data is limited to public posts; private coordination is invisible to this system",
                "English-language content is overrepresented due to GDELT DOC API default query language",
                "Temporal analysis assumes UTC timestamps; timezone misattribution can create false timing signals",
            ],
        },
        "interpretation": {
            "confidence_interval": confidence,
            "alternative_hypotheses": alternatives,
            "source_credibility": {
                "weighted_credibility": source_breakdown.get("weighted_credibility", 0.5),
                "category_breakdown": source_breakdown.get("categories", {}),
                "source_count": source_breakdown.get("source_count", 0),
            },
        },
        "nvi_timeline": nvi_timeline,
        "evidence": {
            "post_count": len(evidence_entries),
            "posts": evidence_entries,
        },
        "coordination_analysis": {
            "signal_count": len(coord_evidence),
            "signals": coord_evidence,
        },
        "ensemble_nvi": ensemble_data if ensemble_data else {
            "note": "Ensemble NVI data not yet available. Run NVI cycle first."
        },
        "narrative_dna": dna_data if dna_data else {
            "note": "DNA fingerprint not yet computed. Run DNA cycle first."
        },
        "dna_matches": {
            "count": len(dna_matches),
            "matches": [
                {
                    "matched_cluster_id": m.get("matched_cluster_id") or (
                        m["cluster_b"] if m["cluster_a"] == cluster_id else m["cluster_a"]
                    ),
                    "match_score": m.get("match_score", 0),
                    "confidence": m.get("confidence", "low"),
                    "dimension_scores": m.get("dimension_scores", {}),
                    "matched_label": m.get("matched_label", ""),
                }
                for m in dna_matches[:10]
            ],
        } if dna_matches else {"count": 0, "matches": []},
        "amplification_graph": cluster_graph if cluster_graph else {
            "note": "Graph topology not available. Run graph cycle first."
        },
    }

    # Compute integrity hash over the entire pack
    pack_json = json.dumps(pack, sort_keys=True, ensure_ascii=False)
    pack_hash = hashlib.sha256(pack_json.encode()).hexdigest()

    pack["integrity"] = {
        "sha256": pack_hash,
        "hash_method": "SHA256 over JSON-serialized pack (sort_keys=True, ensure_ascii=False)",
        "verification": "To verify: remove 'integrity' key, serialize with same params, compute SHA256",
    }

    # Store in database
    pack_id = f"EP-{cluster_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    try:
        db_conn.execute("""
            INSERT OR REPLACE INTO evidence_packs (id, cluster_id, sha256_hash, pack_data)
            VALUES (?, ?, ?, ?)
        """, (pack_id, cluster_id, pack_hash, json.dumps(pack)))
        db_conn.commit()
    except Exception as e:
        logger.error(f"Failed to store evidence pack: {e}")

    pack["metadata"]["pack_id"] = pack_id
    logger.info(f"Evidence pack {pack_id} generated: {len(evidence_entries)} posts, hash={pack_hash[:16]}...")

    return pack


def verify_evidence_pack(pack: dict) -> bool:
    """Verify the integrity of an evidence pack."""
    stored_hash = pack.get("integrity", {}).get("sha256")
    if not stored_hash:
        return False

    # Remove integrity block and recompute
    pack_copy = {k: v for k, v in pack.items() if k != "integrity"}
    recomputed = hashlib.sha256(
        json.dumps(pack_copy, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    return recomputed == stored_hash

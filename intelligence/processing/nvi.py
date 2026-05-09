"""
BurnTheLies Intelligence — NVI (Narrative Velocity Index) Engine

NVI Formula:
    NVI(t) = σ(α·B(t) + β·S(t) - γ·M(t) + δ·T(t)) × C(t)

Where:
    B(t) = Burst z-score — how fast a narrative is accelerating vs baseline
    S(t) = Spread factor — cross-platform/language/geography diversity
    M(t) = Mutation penalty — how much the narrative is morphing (high = organic)
    T(t) = Tone uniformity — how uniform the emotional framing is (low variance = suspicious)
    C(t) = Coordination multiplier — temporal/content synchrony signals
    σ   = Sigmoid normalization → [0, 100]

This is the core engine. Every number here must be traceable to evidence.
"""

import json
import logging
import math
import numpy as np
from collections import Counter
from math import log2, exp
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("intelligence.nvi")

# Optional Prometheus metrics — no-op if module is unavailable.
try:
    from intelligence import metrics as _metrics  # type: ignore
except Exception:  # broad: metrics is best-effort instrumentation, never fatal
    _metrics = None

# ─── Ensemble NVI — Three coefficient sets for cross-validation ─────────────────
# Disagreement between sets >20 points → flagged "uncertain" (possible evasion)
# Weighted mean used as final NVI. This defeats single-signal evasion strategies.
ENSEMBLE_CONFIGS = [
    {  # Set A: Burst-heavy — catches fast-breaking coordinated spikes
        "name": "burst_heavy",
        "alpha": 0.35, "beta": 0.20, "gamma": 0.28, "delta": 0.20,
        "weight": 0.33,
    },
    {  # Set B: Spread-heavy — catches cross-platform amplification
        "name": "spread_heavy",
        "alpha": 0.20, "beta": 0.35, "gamma": 0.28, "delta": 0.20,
        "weight": 0.33,
    },
    {  # Set C: Mutation-heavy — catches copy-paste coordination
        "name": "mutation_heavy",
        "alpha": 0.25, "beta": 0.25, "gamma": 0.30, "delta": 0.20,
        "weight": 0.34,
    },
]

# Legacy defaults (used by ensemble as baseline)
ALPHA = 0.30
BETA = 0.25
GAMMA = 0.15
DELTA = 0.20
COORD_BASE = 1.0

# Ensemble disagreement threshold — if max(NVI) - min(NVI) > this, flag as uncertain
ENSEMBLE_DISAGREEMENT_THRESHOLD = 20.0


def compute_nvi(db_conn, cluster_id: int) -> dict:
    """
    Compute NVI for a single narrative cluster.
    Returns full breakdown dict. Persists coordination signals.

    Pipeline (v5):
        1. Load posts (with content_hash, title for new features).
        2. Compute core signals B/S/M/T/C and existing detectors.
        3. Compute new cluster-level features (hash diversity, inter-arrival
           stats, narrative coherence).
        4. Compute ensemble raw NVI from coefficient sets.
        5. Compute confidence (independent signal-strength function).
        6. Pack everything into ClusterFeatures and run apply_falsification_gates().
        7. Final NVI = 0 if gate 1 fired, else min(raw_nvi, gate_cap).
        8. Persist snapshot with full gate trace; emit one coordination signal
           per fired gate.
    """
    from intelligence.db import insert_nvi_snapshot, insert_coordination_signal
    try:
        from intelligence.db import transaction as _db_transaction
    except ImportError:
        _db_transaction = None
    from intelligence.processing.gates import (
        ClusterFeatures,
        apply_falsification_gates,
        compute_inter_arrival_stats,
        compute_narrative_coherence,
        compute_unique_hash_ratio,
        compute_wire_hash_diversity_signal,
    )

    posts = db_conn.execute("""
        SELECT rp.id, rp.source, rp.url, rp.title, rp.language,
               rp.published_at, rp.ingested_at, rp.metadata, rp.content_hash
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
        ORDER BY rp.published_at ASC
    """, (cluster_id,)).fetchall()

    if len(posts) < 3:
        return {"nvi_score": 0, "insufficient_data": True}

    posts = [dict(p) for p in posts]

    # ── Core signals (unchanged from v4.1) ──
    burst = _compute_burst(posts)
    spread = _compute_spread(posts)
    mutation, mutation_evidence = _compute_mutation(db_conn, cluster_id)
    tone_uniformity = _compute_tone_uniformity(posts)
    coordination, coord_signals = _compute_coordination(posts)

    # ── New v5 cluster-level features ──
    embedding_similarity_mean = float(
        mutation_evidence.get("centroid_similarity_mean", 0.0)
    )
    unique_hash_ratio = compute_unique_hash_ratio(posts)
    inter_arrival_mean, inter_arrival_std = compute_inter_arrival_stats(posts)
    narrative_coherence = compute_narrative_coherence(posts)

    # ── Tier 1+ Anomaly signals ──
    # Circadian anomaly: posts clustered during off-hours (1-5 AM UTC).
    # Coordinated content often publishes outside normal news cycles.
    off_hour_count = 0
    for p in posts:
        try:
            pub = p.get("published_at", "")
            if pub:
                hour = int(pub[11:13])
            else:
                continue
            if hour in (1, 2, 3, 4, 5):
                off_hour_count += 1
        except (ValueError, IndexError):
            pass
    circadian_anomaly = off_hour_count >= max(3, len(posts) * 0.4) if len(posts) >= 5 else False

    # Content anomaly: GDELT tone values that are extreme outliers.
    # GDELT tone scale: activity_density mean=22.9 median=22.6, negative_tone
    # mean=4.2 median=3.5, self_reference mean=0.6 median=0.1.
    # Top-5% thresholds: ad>35, nt>12, sr<0.3 — very extreme combinations.
    tone_anomalies = 0
    for p in posts:
        meta = p.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if isinstance(meta, dict):
            tone = meta.get("tone", {}) or {}
            ad = tone.get("activity_density", 0) or 0
            nt = tone.get("negative", 0) or 0
            sr = tone.get("self_reference", 0) or 0
            if ad > 35 and nt > 12 and sr < 0.3:
                tone_anomalies += 1
    content_anomaly = tone_anomalies >= max(2, len(posts) * 0.3) if len(posts) >= 5 else False

    # ── Content noise detection ──
    # Reject clusters whose post titles match clickbait/listicle/obituary/
    # classifieds patterns. These are not news and should not surface.
    import re as _re
    _NOISE_TITLE_PAT = _re.compile(
        r'\d+\s+(Science|Best|Worst|Things|Ways|Reasons|Tips|Secrets|Facts|Photos|Pictures|Signs|Jobs)'
        r'|(Obituari|Police\s+Report|Sports\s+Roundup|News\s+Quiz|Weekly\s+Roundup|Briefly|Classified)'
        r'|(Top\s+\d+|Top\s+Ten|\d+\s+Things|Quiz:)',
        _re.IGNORECASE
    )
    noise_count = 0
    for p in posts:
        title = p.get("title", "") or ""
        if len(title) < 20:
            noise_count += 1
        elif _NOISE_TITLE_PAT.search(title):
            noise_count += 1
    content_noise = noise_count >= max(1, int(len(posts) * 0.5)) if posts else False

    # ── Multi-domain topic bag flag (set by cluster.py) ──
    # When cluster.py detects a cross-domain topic bag, force the coherence
    # gate to fire so these clusters don't surface as false positive alerts.
    cluster_meta = db_conn.execute(
        "SELECT metadata FROM narrative_clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    if cluster_meta and cluster_meta[0]:
        try:
            meta = json.loads(cluster_meta[0]) if isinstance(cluster_meta[0], str) else cluster_meta[0]
            if meta.get("topic_bag"):
                narrative_coherence = 0.0  # force coherence gate to fire
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Existing detectors (now feed into ClusterFeatures, not inline gates) ──
    wire_service_detection = _detect_wire_service_syndication(posts, mutation)
    wire_signal_known_syndicators = wire_service_detection.get("is_syndication", False)

    gdelt_sources = {"gdelt_gkg", "gdelt_doc"}
    gdelt_post_count = sum(1 for p in posts if p.get("source", "") in gdelt_sources)
    gdelt_fraction = gdelt_post_count / len(posts)

    entity_concentration, shared_entity_count = _compute_entity_concentration(posts)
    gdelt_batch_artifact = _detect_gdelt_batch_artifact(posts, gdelt_fraction)

    # When GDELT batch artifact fires, nullify the coordination multiplier so
    # the ensemble below isn't inflated by the batch-cycle artifact. Gates 8/9
    # are already configured to skip when gate 1 fires (see gates.GATES).
    if gdelt_batch_artifact:
        coordination = 1.0

    # Source diversity normalized to [0,1] (matches _compute_spread's scaling).
    _cluster_domains = _extract_domains_from_posts(posts)
    source_entropy_raw = _shannon_entropy(_cluster_domains)
    source_diversity_normalized = min(1.0, source_entropy_raw / 4.0)
    unique_domain_count = len({d for d in _cluster_domains if d})

    # ── Tier 1+ Anomaly signals (computed after source_diversity_normalized) ──
    # Cross-language: when the same cluster spans 3+ real languages
    real_langs = {p.get("language", "") for p in posts
                  if p.get("language", "") not in ("translated", "unknown", "")}
    cross_language_anomaly = len(real_langs) >= 3 and source_diversity_normalized > 0.3

    # Geographic spread: uses V2EnhancedLocations (GKG column 8) which has
    # human-readable country codes. V1Locations (column 9, stored in metadata)
    # only has numeric location IDs — unusable without a lookup table.
    # Set to False until V2EnhancedLocations parsing is implemented.
    geographic_spread = False

    # High-signal topic: manipulation-specific themes (INFORMATION_WARFARE,
    # DISINFORMATION, PROPAGANDA, CONSPIRACY, etc.) spreading across sources.
    # Uses a NARROW subset — ELECTION, MILITARY, CORRUPTION, PROTEST are too
    # broad and appear in most GKG news clusters.
    _MANIPULATION_THEMES = {
        "INFORMATION_WARFARE", "DISINFORMATION", "PROPAGANDA",
        "CONSPIRACY", "WEAPONIZED_NARRATIVE", "INFLUENCE_OPERATION",
        "MEDIA_CENSORSHIP",
    }
    high_signal_count = 0
    for p in posts:
        meta = p.get("metadata")
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}
        post_themes = set((meta or {}).get("themes", []) or [])
        if post_themes & _MANIPULATION_THEMES:
            high_signal_count += 1
    high_signal_topic = (
        high_signal_count >= max(3, len(posts) * 0.3)
        and source_diversity_normalized > 0.3
        and len(posts) >= 5
    )

    dna_match_count = _check_dna_match_count(db_conn, cluster_id)
    dna_evidence_strong = dna_match_count >= 5
    high_conf_dna_match_count = _check_dna_match_count_highconf(db_conn, cluster_id)
    cross_topic_persistence = _check_cross_topic_persistence(db_conn, cluster_id)

    # Language spread — topic bags often have diverse languages
    _cluster_langs = [p.get("language", "unknown") for p in posts]
    lang_counts = Counter(_cluster_langs)
    dominant_lang_fraction = max(lang_counts.values()) / len(posts) if posts else 0
    language_count = len(lang_counts)

    # NEW: hash-diversity wire-service signal (rewritten copy across many outlets).
    wire_signal_hash_diversity = compute_wire_hash_diversity_signal(
        unique_hash_ratio=unique_hash_ratio,
        embedding_similarity_mean=embedding_similarity_mean,
        source_diversity=source_diversity_normalized,
        dna_match_count=dna_match_count,
        post_count=len(posts),
    )

    # ── Ensemble NVI Computation ──
    ensemble_scores = []
    for cfg in ENSEMBLE_CONFIGS:
        score = _compute_raw_nvi(
            burst, spread, mutation, tone_uniformity, coordination,
            cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["delta"],
        )
        ensemble_scores.append(score)

    raw_nvi = round(float(np.average(
        ensemble_scores,
        weights=[c["weight"] for c in ENSEMBLE_CONFIGS],
    )), 2)
    ensemble_min = min(ensemble_scores)
    ensemble_max = max(ensemble_scores)
    ensemble_disagreement = ensemble_max - ensemble_min
    ensemble_perfect_agreement_red_flag = (
        ensemble_disagreement < 1.0 and len(posts) < 15
    )
    ensemble_uncertain = ensemble_disagreement > ENSEMBLE_DISAGREEMENT_THRESHOLD

    # ── Confidence (independent signal-strength function on raw NVI) ──
    from intelligence.processing.interpret import compute_confidence_interval
    confidence_check = compute_confidence_interval(
        nvi_score=raw_nvi,
        burst=burst,
        spread=spread,
        mutation=mutation,
        coord=coordination,
        post_count=len(posts),
        source_diversity=source_entropy_raw,
        dna_match_count=dna_match_count,
        ensemble_red_flag=ensemble_perfect_agreement_red_flag,
        gdelt_batch_artifact=gdelt_batch_artifact,
        unique_domain_count=unique_domain_count,
        language_spread=dict(lang_counts),
    )
    confidence_probability = confidence_check["probability"]
    if dna_evidence_strong and confidence_probability < 0.65:
        confidence_probability = min(0.70, confidence_probability + 0.10)

    # ── Apply falsification gate pipeline ──
    features = ClusterFeatures(
        cluster_id=cluster_id,
        post_count=len(posts),
        burst=burst,
        spread=spread,
        mutation=mutation,
        coordination=coordination,
        tone_uniformity=tone_uniformity,
        entity_concentration=entity_concentration,
        shared_entity_count=shared_entity_count,
        unique_hash_ratio=unique_hash_ratio,
        embedding_similarity_mean=embedding_similarity_mean,
        inter_arrival_mean=inter_arrival_mean,
        inter_arrival_std=inter_arrival_std,
        gdelt_fraction=gdelt_fraction,
        source_diversity=source_diversity_normalized,
        unique_domain_count=unique_domain_count,
        gdelt_batch_artifact=gdelt_batch_artifact,
        wire_signal_known_syndicators=wire_signal_known_syndicators,
        wire_signal_hash_diversity=wire_signal_hash_diversity,
        dna_match_count=dna_match_count,
        high_conf_dna_match_count=high_conf_dna_match_count,
        cross_topic_persistence=cross_topic_persistence,
        circadian_anomaly=circadian_anomaly,
        content_anomaly=content_anomaly,
        content_noise=content_noise,
        cross_language_anomaly=cross_language_anomaly,
        geographic_spread=geographic_spread,
        high_signal_topic=high_signal_topic,
        ensemble_disagreement=ensemble_disagreement,
        ensemble_perfect_agreement_red_flag=ensemble_perfect_agreement_red_flag,
        narrative_coherence=narrative_coherence,
        confidence_probability=confidence_probability,
        dominant_lang_fraction=dominant_lang_fraction,
        language_count=language_count,
    )
    gate_result = apply_falsification_gates(features)

    if gate_result.nvi_zero:
        nvi_score = 0.0
    else:
        # Apply cap first (quality ceiling), then boost floor (velocity signal).
        # Boost CAN override caps for legitimate clusters — coordinated campaigns
        # often have modest post counts but strong DNA evidence. Topic bags are
        # handled by specific guards in each boost gate, not by blanket capping.
        capped = min(raw_nvi, gate_result.nvi_cap)
        nvi_score = float(round(max(capped, gate_result.nvi_floor), 2))

    # Invariant: nvi_score must respect cap when no boost overrides it.
    # When both cap and floor fire, floor may exceed cap by design.
    is_topic_cluster = "entity_concentration" in gate_result.gates_applied
    alert_suppressed = gate_result.alert_suppressed
    if alert_suppressed:
        effective_alert_level = "normal"
    elif gate_result.force_alert_level:
        # Boost gate forces a level — apply it regardless of NVI score
        effective_alert_level = gate_result.force_alert_level
    else:
        effective_alert_level = (
            "critical" if nvi_score >= 80 else "elevated" if nvi_score >= 60 else "normal"
        )

    result = {
        "nvi_score": nvi_score,
        "raw_nvi": raw_nvi,
        "burst_zscore": round(burst, 4),
        "spread_factor": round(spread, 4),
        "mutation_penalty": round(mutation, 4),
        "tone_uniformity": round(tone_uniformity, 4),
        "coordination_mult": round(coordination, 4),
        "post_count": len(posts),
        # Entity concentration
        "entity_concentration": round(entity_concentration, 4),
        "is_topic_cluster": is_topic_cluster,
        "shared_entity_count": shared_entity_count,
        # Ensemble metadata
        "ensemble_scores": [float(s) for s in ensemble_scores],
        "ensemble_disagreement": float(round(ensemble_disagreement, 2)),
        "ensemble_uncertain": bool(ensemble_uncertain),
        "ensemble_perfect_agreement_red_flag": bool(ensemble_perfect_agreement_red_flag),
        # GDELT
        "gdelt_batch_artifact": bool(gdelt_batch_artifact),
        "gdelt_fraction": float(round(gdelt_fraction, 4)),
        # DNA
        "dna_match_count": dna_match_count,
        "dna_evidence_strong": bool(dna_evidence_strong),
        "high_conf_dna_match_count": high_conf_dna_match_count,
        "cross_topic_persistence": cross_topic_persistence,
        "cross_language_anomaly": cross_language_anomaly,
        "geographic_spread": geographic_spread,
        "high_signal_topic": high_signal_topic,
        "circadian_anomaly": circadian_anomaly,
        "content_anomaly": content_anomaly,
        "content_noise": content_noise,
        # Wire service (legacy boolean preserved for backward-compat consumers)
        "wire_service_syndication": wire_signal_known_syndicators,
        "wire_service_source_fraction": wire_service_detection.get("wire_fraction", 0.0),
        "wire_service_downgraded": "wire_service" in gate_result.gates_applied,
        # NEW v5 features
        "unique_hash_ratio": round(unique_hash_ratio, 4),
        "embedding_similarity_mean": round(embedding_similarity_mean, 4),
        "inter_arrival_mean_seconds": (
            None if math.isnan(inter_arrival_mean) else round(inter_arrival_mean, 2)
        ),
        "inter_arrival_std_seconds": (
            None if math.isnan(inter_arrival_std) else round(inter_arrival_std, 2)
        ),
        "narrative_coherence": round(narrative_coherence, 4),
        # Gate trace (NEW — primary interface for downstream consumers)
        "gates_applied": list(gate_result.gates_applied),
        "gate_reasoning": dict(gate_result.gate_reasoning),
        # Confidence
        "confidence_probability": float(round(confidence_probability, 3)),
        "alert_suppressed": bool(alert_suppressed),
        "effective_alert_level": effective_alert_level,
        # Source diversity (normalized 0-1)
        "source_diversity": round(source_diversity_normalized, 4),
        "unique_domain_count": unique_domain_count,
        # Language diversity
        "dominant_lang_fraction": round(dominant_lang_fraction, 4),
        "language_count": language_count,
    }

    # ── Build coordination-signal list (no DB writes yet) ──
    # Temporal signals from _compute_coordination (legacy) + per-gate signals
    # derived from the gate pipeline result.
    all_signals = list(coord_signals)

    if mutation_evidence and mutation_evidence.get("similarity_variance", 1) < 0.02:
        all_signals.append({
            "signal_type": "content_identity",
            "score": 1.0 - mutation_evidence["similarity_variance"] * 50,
            "evidence": mutation_evidence,
        })

    # One signal per fired gate (replaces the v4.1 ad-hoc per-gate signals).
    for gate_name, reasoning in gate_result.gate_reasoning.items():
        if not reasoning.get("fired"):
            continue
        all_signals.append({
            "signal_type": f"gate_{gate_name}",
            "score": 1.0,
            "evidence": {
                "interpretation": reasoning.get("why", ""),
                **reasoning.get("evidence", {}),
            },
        })

    # Source concentration signal (kept — orthogonal to gates, useful for ops).
    source_concentration_entropy = _shannon_entropy([p["source"] for p in posts])
    if source_concentration_entropy < 0.5 and len(posts) >= 5:
        source_counts = Counter(p["source"] for p in posts)
        dominant = source_counts.most_common(1)[0]
        all_signals.append({
            "signal_type": "source_concentration",
            "score": 1.0 - source_concentration_entropy,
            "evidence": {
                "dominant_source": dominant[0],
                "dominant_count": dominant[1],
                "total_sources": len(source_counts),
                "entropy": round(source_concentration_entropy, 4),
            },
        })

    if tone_uniformity > 0.7:
        all_signals.append({
            "signal_type": "tone_uniformity",
            "score": tone_uniformity,
            "evidence": {
                "uniformity_score": round(tone_uniformity, 4),
                "post_count": len(posts),
            },
        })

    if bool(ensemble_uncertain):
        all_signals.append({
            "signal_type": "ensemble_uncertain",
            "score": float(ensemble_disagreement) / 100.0,
            "evidence": {
                "ensemble_scores": [float(s) for s in ensemble_scores],
                "disagreement": float(ensemble_disagreement),
                "interpretation": "Detection methods disagree significantly. Possible evasion or ambiguous signal.",
            },
        })

    # ── Persist snapshot + signals atomically ──
    # Snapshot and per-gate signals must commit together: a snapshot without
    # its signals leaves the UI showing "elevated NVI / no evidence" — the
    # exact failure mode the platform exists to avoid.
    def _persist():
        insert_nvi_snapshot(
            db_conn, cluster_id,
            nvi_score=nvi_score,
            burst_zscore=burst,
            spread_factor=spread,
            mutation_penalty=mutation,
            coordination_mult=coordination,
            raw_components=result,
            effective_alert_level=effective_alert_level,
        )
        for sig in all_signals:
            insert_coordination_signal(
                db_conn, cluster_id,
                signal_type=sig["signal_type"],
                score=sig["score"],
                evidence=sig["evidence"],
            )
        # Refresh source_diversity and post_count on the cluster record — the
        # values set at creation time are stale once the cluster grows.
        db_conn.execute(
            "UPDATE narrative_clusters SET source_diversity = ?, post_count = ?, last_updated = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
            (source_diversity_normalized, len(posts), cluster_id),
        )

    if _db_transaction is not None:
        try:
            with _db_transaction(db_conn):
                _persist()
        except Exception:
            logger.exception(
                "NVI persist failed (transaction rolled back) for cluster_id=%s",
                cluster_id,
            )
            raise
    else:
        # Fallback: manual transaction control if db.transaction helper absent.
        try:
            db_conn.execute("BEGIN IMMEDIATE")
            _persist()
            db_conn.commit()
        except Exception:
            db_conn.rollback()
            logger.exception(
                "NVI persist failed (manual rollback) for cluster_id=%s",
                cluster_id,
            )
            raise

    # ── Prometheus instrumentation (best-effort, never fatal) ──
    if _metrics is not None:
        try:
            for gate_name, reasoning in gate_result.gate_reasoning.items():
                if reasoning.get("fired"):
                    _metrics.record_gate_fired(gate_name)
            if effective_alert_level in ("critical", "elevated"):
                _metrics.record_alert(effective_alert_level)
        except Exception:
            # broad: instrumentation must never break the pipeline
            logger.exception(
                "Metrics emission failed for cluster_id=%s", cluster_id,
            )

    return result


def _compute_burst(posts: list[dict]) -> float:
    """
    Burst z-score: how much the post rate in the last hour
    deviates from the rolling average.

    Returns 0.0 if timestamps are batch-ingestion artifacts (all within 60s),
    since we cannot distinguish coordinated bursts from batch data loading.
    """
    timestamps = []
    for p in posts:
        try:
            ts = datetime.fromisoformat(p["published_at"].replace("Z", "+00:00"))
            timestamps.append(ts)
        except (ValueError, TypeError, AttributeError):
            try:
                ts = datetime.fromisoformat(p["ingested_at"].replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, TypeError, AttributeError):
                # Best-effort: malformed timestamps are skipped; downstream
                # handles the empty-list case. Hot path — log at debug only.
                logger.debug(
                    "Skipping post %s: unparseable published_at/ingested_at",
                    p.get("id"),
                )

    if not timestamps:
        return 0.0

    # Batch artifact detection: if all timestamps span < 60s, can't compute real burst
    ts_spread = (max(timestamps) - min(timestamps)).total_seconds()
    if ts_spread < 60:
        return 0.0  # Insufficient temporal data — likely batch ingestion

    # Bucket posts into 15-minute windows
    windows = {}
    for ts in timestamps:
        bucket = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
        bucket_key = bucket.isoformat()
        windows[bucket_key] = windows.get(bucket_key, 0) + 1

    counts = list(windows.values())
    if len(counts) < 2:
        return 0.0

    mean = np.mean(counts)
    std = np.std(counts)

    if std < 0.01:
        return 0.0

    # Latest window count
    latest = counts[-1]
    z_score = (latest - mean) / std

    # Cap at ±10 to prevent extreme values from dominating
    return float(np.clip(z_score, -10.0, 10.0))


def _compute_spread(posts: list[dict]) -> float:
    """
    Spread factor: Shannon entropy across publication domains and languages.
    Higher = more organic spread. But combined with high burst = suspicious.

    Extracts actual publication domains from article URLs for GDELT posts.
    """
    import re
    from urllib.parse import urlparse
    source_ids = []
    for p in posts:
        src = p.get("source", "unknown")
        if src in ("gdelt_gkg", "gdelt_doc"):
            # Extract actual publication domain from the article URL
            url = p.get("url", "") or ""
            try:
                netloc = urlparse(url).netloc
                domain = re.sub(r'^www\.', '', netloc.lower()) if netloc else ""
            except (ValueError, AttributeError, TypeError):
                # Malformed URL → fall back to "unknown"; not fatal for spread.
                domain = ""
            source_ids.append(domain if domain else "unknown")
        else:
            source_ids.append(src)

    languages = [p.get("language", "en") for p in posts]

    source_entropy = _shannon_entropy(source_ids)
    lang_entropy = _shannon_entropy(languages)

    # Normalize: max ~4 bits for sources across many domains
    spread = (source_entropy / 4.0) * 0.7 + (lang_entropy / 4.0) * 0.3

    # Single-language penalty: clusters that are all one language (e.g. all
    # Japanese, all Romanian) are statistically far more likely to be topic
    # bags than cross-language coordination.
    if lang_entropy < 0.5:  # effectively single-language
        spread *= 0.5

    return min(spread, 1.0)


def _compute_mutation(db_conn, cluster_id: int) -> tuple[float, dict]:
    """
    Mutation penalty: how much variation exists within the cluster.
    Low mutation (near-identical content) = coordinated. High = organic discourse.
    Uses embedding variance within the cluster.

    Returns (mutation_score, evidence_dict).
    """
    members = db_conn.execute("""
        SELECT e.vector FROM embeddings e
        JOIN cluster_members cm ON cm.post_id = e.post_id
        WHERE cm.cluster_id = ?
    """, (cluster_id,)).fetchall()

    if len(members) < 2:
        return 0.5, {}

    vectors = np.array([json.loads(m["vector"]) for m in members])

    # Compute pairwise cosine similarity variance
    centroid = vectors.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

    similarities = []
    for v in vectors:
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        similarities.append(float(np.dot(v_norm, centroid_norm)))

    # High variance = high mutation = organic = penalty applied
    # Low variance = low mutation = coordinated = low penalty
    variance = float(np.var(similarities))
    mean_sim = float(np.mean(similarities))

    # Map: low variance (0.0) → low penalty (0.1), high variance (0.1+) → high penalty (1.0)
    mutation = min(1.0, variance * 10)

    evidence = {
        "centroid_similarity_mean": round(mean_sim, 6),
        "similarity_variance": round(variance, 6),
        "sample_count": len(members),
    }

    return mutation, evidence


def _compute_tone_uniformity(posts: list[dict]) -> float:
    """
    Tone uniformity: if all posts have near-identical sentiment framing,
    that signals coordination. Organic discourse has varied emotion.

    Uses GDELT tone data stored in post metadata.
    """
    polarity_values = []
    tone_values = []

    for p in posts:
        meta_raw = p.get("metadata", "{}")
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            meta = meta_raw or {}

        tone = meta.get("tone", {})
        if isinstance(tone, dict):
            if "polarity" in tone:
                polarity_values.append(float(tone["polarity"]))
            if "tone" in tone:
                tone_values.append(float(tone["tone"]))

    # Need at least 5 data points for meaningful analysis
    if len(polarity_values) < 5:
        return 0.0  # No signal — neutral contribution

    polarity_var = float(np.var(polarity_values))
    tone_var = float(np.var(tone_values)) if tone_values else polarity_var

    # Combine: low variance = high uniformity = suspicious
    combined_var = polarity_var * 0.6 + tone_var * 0.4
    uniformity = max(0.0, 1.0 - min(1.0, combined_var / 2.0))

    return uniformity


def _compute_entity_concentration(posts: list[dict]) -> tuple[float, int]:
    """
    Measures whether cluster articles share specific named entities (a real coordinated narrative)
    or merely cover the same broad topic with independent events (a false positive).

    Key distinction:
      - Topic cluster: daily drug busts in 15 countries — different police, different traffickers,
        different locations. Entity overlap ≈ 0. NOT a coordinated narrative.
      - Real narrative: 20 articles about "Trump election fraud" — all reference Trump, DOJ, FBI.
        Entity overlap is high.

    Returns (concentration_score [0,1], shared_entity_count).
    If entity_coverage < 30% (insufficient metadata), returns 0.5 (neutral — don't gate).
    """
    post_entity_sets = []
    posts_with_entities = 0

    for p in posts:
        meta_raw = p.get("metadata", "{}")
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        else:
            meta = meta_raw or {}

        persons_str = meta.get("persons", "") or ""
        orgs_str = meta.get("organizations", "") or ""

        persons = {x.strip().lower() for x in persons_str.split(";")
                   if x.strip() and len(x.strip()) > 2 and "#" not in x}
        orgs = {x.strip().lower() for x in orgs_str.split(";")
                if x.strip() and len(x.strip()) > 2 and "#" not in x}

        entities = persons | orgs
        if entities:
            posts_with_entities += 1
            post_entity_sets.append(entities)

    # Need ≥30% of posts with entity data to make a determination
    # If data is sparse (mostly DOC API articles with no persons/orgs), return neutral
    entity_coverage = posts_with_entities / len(posts) if posts else 0
    if entity_coverage < 0.30 or len(post_entity_sets) < 3:
        return 0.5, 0  # Insufficient entity metadata — don't gate

    # Count how frequently each entity appears across posts
    entity_freq = Counter()
    for entity_set in post_entity_sets:
        for entity in entity_set:
            entity_freq[entity] += 1

    if not entity_freq:
        return 0.0, 0  # No entities at all → pure topic cluster

    n_posts = len(post_entity_sets)

    # Top entity: fraction of posts mentioning it
    _, top_count = entity_freq.most_common(1)[0]
    top_fraction = top_count / n_posts

    # Entities shared by ≥20% of posts
    shared_entities = sum(1 for _, c in entity_freq.items() if c / n_posts >= 0.20)

    # Weighted score: top entity coverage (60%) + breadth of shared entities (40%)
    concentration = top_fraction * 0.6 + min(1.0, shared_entities / 3.0) * 0.4

    return float(min(1.0, concentration)), shared_entities


def _compute_coordination(posts: list[dict]) -> tuple[float, list[dict]]:
    """
    Coordination multiplier C(t): detects temporal synchrony.
    If posts arrive in suspicious bursts at regular intervals = amplify NVI.

    Returns (coord_mult, list of signal dicts).
    """
    signals = []

    if len(posts) < 5:
        return COORD_BASE, signals

    # Extract timestamps
    timestamps = []
    for p in posts:
        try:
            ts = datetime.fromisoformat(p["published_at"].replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError):
            # Best-effort: malformed timestamps are skipped; if too few remain
            # we return COORD_BASE below. Hot path — debug log only.
            logger.debug(
                "_compute_coordination: skipping post %s with bad published_at",
                p.get("id"),
            )

    if len(timestamps) < 5:
        return COORD_BASE, signals

    timestamps.sort()

    # Compute inter-arrival times
    deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    if not deltas:
        return COORD_BASE, signals

    # Coefficient of variation of inter-arrival times
    # Low CoV = suspiciously regular timing
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    if mean_delta < 1:
        return COORD_BASE, signals

    cov = std_delta / mean_delta

    # Very regular timing (CoV < 0.3) → coordination signal
    if cov < 0.3:
        coord_mult = 1.0 + (0.3 - cov) * 3  # Up to 1.9x multiplier
        regularity_score = (0.3 - cov) * 3
    elif cov < 0.5:
        coord_mult = 1.0 + (0.5 - cov) * 1  # Mild amplification
        regularity_score = (0.5 - cov) * 1
    else:
        coord_mult = COORD_BASE  # Normal variation
        regularity_score = 0.0

    # Cap at 2x
    coord_mult = min(2.0, coord_mult)

    # Emit temporal_sync signal if meaningful
    if cov < 0.5:
        signals.append({
            "signal_type": "temporal_sync",
            "score": round(regularity_score, 4),
            "evidence": {
                "inter_arrival_mean_seconds": round(float(mean_delta), 2),
                "inter_arrival_std_seconds": round(float(std_delta), 2),
                "coefficient_of_variation": round(float(cov), 4),
                "regularity_score": round(regularity_score, 4),
                "sample_size": len(timestamps),
            },
        })

    return coord_mult, signals


# ─── Wire Service Syndication Detection ──────────────────────────────────────

# Top-tier wire services — their distribution explains near-identical content
_WIRE_SERVICES = {
    "apnews.com", "reuters.com", "afp.com", "upi.com", "efe.com", "dpa.de",
    "kyodonews.jp", "xinhuanet.com", "bbc.co.uk", "bbc.com",
}

# Major news agency / syndication domains that distribute to regional outlets
_MAJOR_SYNDICATORS = {
    "ap.org", "reutersmedia.com", "afpforum.com", "pressassociation.com",
    "pa.media", "pa.press.net",
}


def _extract_domains_from_posts(posts: list[dict]) -> list[str]:
    """Extract actual publication domains from post URLs."""
    import re as _re
    from urllib.parse import urlparse as _urlparse
    domains = []
    for p in posts:
        source = p.get("source", "")
        url = (p.get("url") or "").strip()
        if source in ("gdelt_gkg", "gdelt_doc") and url:
            try:
                netloc = _urlparse(url).netloc
                domain = _re.sub(r'^www\.', '', netloc.lower()) if netloc else ""
                if domain and len(domain) > 3:
                    domains.append(domain)
                    continue
            except (ValueError, AttributeError, TypeError):
                # Malformed URL → fall through to source-id fallback below.
                logger.debug(
                    "_extract_domains_from_posts: bad url for post %s",
                    p.get("id"),
                )
        domains.append(source)
    return domains


def _detect_shared_story_path(posts: list[dict], min_posts: int = 5) -> dict:
    """
    Universal syndication signal: detect clusters where multiple different
    domains all publish the exact same URL path (story ID / slug).

    This is the strongest structural evidence of centralized content
    distribution — it requires no knowledge of which publisher networks
    exist in any country. If 10 domains all point to /story/9241136/ or
    /article/trump-tariffs or /p/breaking-news-xyz, they pulled it from
    a shared CMS. The path is the fingerprint, not the domain.

    Returns:
        {
            "is_shared_path": bool,
            "shared_path": str,          # the path that appears across domains
            "domain_count": int,         # how many unique domains share it
            "coverage": float,           # fraction of posts sharing the path
        }
    """
    from urllib.parse import urlparse as _urlparse
    import re as _re

    if len(posts) < min_posts:
        return {"is_shared_path": False, "shared_path": "", "domain_count": 0, "coverage": 0.0}

    path_domain_pairs: list[tuple[str, str]] = []
    for p in posts:
        url = (p.get("url") or "").strip()
        if not url:
            continue
        try:
            parsed = _urlparse(url)
            netloc = _re.sub(r'^www\.', '', parsed.netloc.lower())
            path = parsed.path.rstrip("/")
            if netloc and len(netloc) > 3 and path and len(path) > 3:
                path_domain_pairs.append((path, netloc))
        except (ValueError, AttributeError, TypeError):
            continue

    if not path_domain_pairs:
        return {"is_shared_path": False, "shared_path": "", "domain_count": 0, "coverage": 0.0}

    # Count how many unique domains share each path
    from collections import Counter as _Counter
    path_counts: Counter = _Counter()
    path_domains: dict = {}
    for path, domain in path_domain_pairs:
        path_counts[path] += 1
        path_domains.setdefault(path, set()).add(domain)

    # Find the most-shared path
    best_path, best_count = path_counts.most_common(1)[0]
    unique_domain_count = len(path_domains[best_path])
    coverage = best_count / len(posts)

    # Syndication: same path on 3+ distinct domains covering ≥60% of cluster posts
    is_shared = unique_domain_count >= 3 and coverage >= 0.60

    return {
        "is_shared_path": is_shared,
        "shared_path": best_path,
        "domain_count": unique_domain_count,
        "coverage": round(coverage, 3),
    }


def _detect_wire_service_syndication(posts: list[dict], mutation_penalty: float) -> dict:
    """
    Detect wire service syndication — where a central CMS or wire agency
    distributes a story and multiple independent outlets publish it.
    This is NOT coordination.

    Two universal signals, either alone is sufficient:

    Signal A — Structural (URL path): Multiple different domains serve the
    exact same URL path. This proves a shared CMS origin regardless of
    country, language, or publisher network. No domain lists required.

    Signal B — Known wire agency domains: AP, Reuters, AFP, etc. dominate
    the cluster. Limited to the handful of global wire services; we do not
    maintain country-specific regional outlet lists (too narrow, always
    incomplete, creates false precision).

    Mutation threshold raised to 0.25 — syndicated content often has minor
    regional edits (byline, dateline, local paragraph) that push mutation
    above 0.15 even though it's the same article.

    Returns:
        {"is_syndication": bool, "signal": str, "wire_fraction": float, "total_domains": int}
    """
    if len(posts) < 5:
        return {"is_syndication": False, "signal": "", "wire_fraction": 0.0, "total_domains": len(posts)}

    # Signal A: shared story path (universal — works for any CMS / any country)
    path_result = _detect_shared_story_path(posts)
    if path_result["is_shared_path"] and mutation_penalty < 0.40:
        # Mutation threshold is relaxed for path-based detection: same story,
        # minor regional editing is expected. Only reject if heavily rewritten.
        return {
            "is_syndication": True,
            "signal": "shared_story_path",
            "wire_fraction": 1.0,
            "total_domains": path_result["domain_count"],
            "shared_path": path_result["shared_path"],
            "path_coverage": path_result["coverage"],
        }

    # Content must be near-identical for Signal B (known wire domains)
    if mutation_penalty >= 0.25:
        return {"is_syndication": False, "signal": "", "wire_fraction": 0.0, "total_domains": 0}

    domains = _extract_domains_from_posts(posts)
    unique_domains = list(set(domains))
    n_domains = len(unique_domains)

    if n_domains < 3:
        return {"is_syndication": False, "signal": "", "wire_fraction": 0.0, "total_domains": n_domains}

    # Signal B: known wire agency or syndication domains dominate the cluster
    wire_service_domains = [
        d for d in unique_domains
        if d.lower() in _WIRE_SERVICES or d.lower() in _MAJOR_SYNDICATORS
    ]
    wire_fraction = len(wire_service_domains) / n_domains if n_domains > 0 else 0

    is_syndication = wire_fraction >= 0.80
    return {
        "is_syndication": is_syndication,
        "signal": "known_wire_domains" if is_syndication else "",
        "wire_fraction": round(wire_fraction, 3),
        "total_domains": n_domains,
    }


# ─── GDELT Batch Artifact Detection ─────────────────────────────────────────

def _detect_gdelt_batch_artifact(posts: list[dict], gdelt_fraction: float = 1.0) -> bool:
    """
    Detect when temporal synchrony signals are actually GDELT's 15-minute
    batch publishing cycle, not real-world coordination.

    GDELT publishes articles in 15-minute batches. When most/all posts in a
    cluster are from GDELT, any temporal regularity (low CoV, burst patterns)
    is a DATA PIPELINE ARTIFACT, not evidence of coordination.

    Path A (strict):  All posts are GDELT — CoV < 0.15 or near-900s
                      multiple with CoV < 0.25.
    Path B (broad):   >80% posts are GDELT — CoV < 0.2 or near-900s
                      multiple with CoV < 0.30.

    Returns True if temporal signals should be NULLIFIED.
    """
    if len(posts) < 5:
        return False

    # ── Extract timestamps (shared by both paths) ──
    timestamps = []
    for p in posts:
        try:
            ts = datetime.fromisoformat(p["published_at"].replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError):
            # Best-effort: malformed timestamps skipped; if too few remain we
            # return False below (no batch-artifact claim without data).
            logger.debug(
                "_detect_gdelt_batch_artifact: skipping post %s with bad published_at",
                p.get("id"),
            )

    if len(timestamps) < 5:
        return False

    timestamps.sort()
    deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    if not deltas:
        return False

    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    if mean_delta < 1:
        return False

    cov = std_delta / mean_delta

    def _batch_signature(cov: float, cov_threshold: float, near_multiple_threshold: float) -> bool:
        """Check if temporal pattern matches GDELT's 15-min batch cycle."""
        if cov < cov_threshold:
            return True
        remainder = mean_delta % 900
        if min(remainder, 900 - remainder) < 60 and cov < near_multiple_threshold:
            return True
        return False

    gdelt_sources = {"gdelt_gkg", "gdelt_doc"}
    sources = set(p.get("source", "") for p in posts)

    # Path A: Strict — all posts from GDELT
    if sources.issubset(gdelt_sources):
        if _batch_signature(cov, 0.15, 0.25):
            return True

    # Path B: Broad — >80% GDELT posts, relaxed thresholds
    if gdelt_fraction >= 0.80:
        if _batch_signature(cov, 0.2, 0.30):
            return True

    return False


def _check_dna_match_count(db_conn, cluster_id: int) -> int:
    """
    Check how many other clusters share DNA fingerprint matches with this one.
    Real coordinated campaigns leave persistent operational fingerprints.

    Queries the live ``dna_matches`` table first — if matches exist, the DNA
    cycle has run for this cluster (matches are inserted by the same cycle
    that creates ``narrative_dna`` rows). Falls back to checking
    ``narrative_dna`` only when no matches are found, distinguishing
    "DNA computed, found nothing" (return 0) from "DNA pending" (return -1).

    Returns -1 if DNA fingerprint hasn't been computed yet, 0+ for match count.
    """
    try:
        row = db_conn.execute("""
            SELECT COUNT(*) as cnt FROM dna_matches
            WHERE (cluster_a = ? OR cluster_b = ?)
            AND match_score >= 0.75
        """, (cluster_id, cluster_id)).fetchone()
        match_count = int(row["cnt"]) if row else 0
        if match_count > 0:
            return match_count

        # No matches — check if DNA cycle has run at all
        has_dna = db_conn.execute(
            "SELECT 1 FROM narrative_dna WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        return 0 if has_dna else -1
    except Exception:
        logger.exception("DNA match query failed for cluster_id=%s", cluster_id)
        return -1


def _check_dna_match_count_highconf(db_conn, cluster_id: int) -> int:
    """High-confidence DNA matches at ≥0.90 cosine similarity.

    At ≥0.90, the 4-modality fingerprint (stylometric 0.30 + cadence 0.30
    + network 0.20 + entity_bias 0.20 weighted cosine) requires very strong
    alignment across ALL dimensions. Wire-syndicated identical content
    typically scores 0.75−0.85 because each outlet's CMS adds editor
    fingerprints (cadence, network) that differ. Genuine same-operator
    persistence (same author, same posting rhythm, same amplifier network)
    consistently scores ≥0.90.

    Returns 0+ for match count (never -1 — this is only called when DNA
    has been confirmed to exist).
    """
    try:
        row = db_conn.execute("""
            SELECT COUNT(*) as cnt FROM dna_matches
            WHERE (cluster_a = ? OR cluster_b = ?)
            AND match_score >= 0.90
        """, (cluster_id, cluster_id)).fetchone()
        return int(row["cnt"]) if row else 0
    except Exception:
        logger.exception("High-conf DNA query failed for cluster_id=%s", cluster_id)
        return 0


def _check_cross_topic_persistence(db_conn, cluster_id: int) -> bool:
    """Check if high-confidence DNA matches show cross-topic persistence.

    Reads ``dimension_scores`` JSON from matches at ≥0.90. If ≥2 matches
    have high stylometric (>0.85) + cadence (>0.80) similarity but LOW
    entity_bias similarity (<0.50), the same operator is writing about
    DIFFERENT topics across clusters — the strongest Tier 1+ signal of
    persistent coordinated authorship.
    """
    try:
        rows = db_conn.execute("""
            SELECT json_extract(dimension_scores, '$.stylometric') as st,
                   json_extract(dimension_scores, '$.cadence') as ca,
                   json_extract(dimension_scores, '$.entity_bias') as eb
            FROM dna_matches
            WHERE (cluster_a = ? OR cluster_b = ?)
            AND match_score >= 0.90
        """, (cluster_id, cluster_id)).fetchall()
        cross_topic_count = 0
        for r in rows:
            st = r["st"]
            ca = r["ca"]
            eb = r["eb"]
            if (st is not None and ca is not None and eb is not None
                    and st > 0.85 and ca > 0.80 and eb < 0.50):
                cross_topic_count += 1
        return cross_topic_count >= 2
    except Exception:
        return False


def refresh_dna_match_gate(db_conn, cluster_id: int, comps: dict) -> dict:
    """Refresh the stale ``dna_match`` gate decision using the live
    ``dna_matches`` table.

    NVI snapshots stamp ``dna_match_count = -1`` when DNA hasn't run yet, but
    the DNA cycle runs at a different cadence. Once DNA matches exist, the
    snapshot's stamped count stays stale until the next cluster rebuild —
    which means evidence packs and the dashboard show the wrong gate decision
    for hours. This helper rewrites the gate result in-place using current
    ``dna_matches`` truth.

    Mirrors the logic in ``gates._gate_dna_match`` so behavior is consistent.
    """
    if not isinstance(comps, dict):
        return comps
    gate_reasoning = comps.get("gate_reasoning") or {}
    stamped = comps.get("dna_match_count", -1)
    live_count = _check_dna_match_count(db_conn, cluster_id)
    if live_count == stamped:
        return comps

    comps["dna_match_count"] = live_count
    comps["dna_evidence_strong"] = live_count >= 5

    post_count = int(comps.get("post_count", 0) or 0)
    if live_count == -1:
        if post_count >= 10:
            gate_reasoning["dna_match"] = {
                "fired": True, "cap": 25.0,
                "why": ("DNA fingerprint not yet computed. Pending DNA cycle, "
                        "capping at 25 to prevent premature alerts."),
                "evidence": {"post_count": post_count, "dna_match_count": -1},
            }
        else:
            gate_reasoning["dna_match"] = {
                "fired": False,
                "why": "DNA fingerprint not yet computed; small cluster covered by insufficient_evidence.",
            }
    elif live_count == 0 and post_count < 20:
        gate_reasoning["dna_match"] = {
            "fired": True,
            "why": (f"Small cluster ({post_count} posts) with zero DNA matches "
                    "across other campaigns. Real coordinated operators leave "
                    "persistent fingerprints; one-off false positives don't."),
            "evidence": {"post_count": post_count, "dna_match_count": 0},
        }
    else:
        gate_reasoning["dna_match"] = {
            "fired": False,
            "why": (f"Either large enough ({post_count} posts) or has DNA matches "
                    f"({live_count})."),
        }
    comps["gate_reasoning"] = gate_reasoning
    return comps


# ─── Ensemble NVI Computation ───────────────────────────────────────────────

def _compute_raw_nvi(burst: float, spread: float, mutation: float,
                     tone_uniformity: float, coordination: float,
                     alpha: float, beta: float, gamma: float, delta: float) -> float:
    """Compute NVI with specific coefficient weights. Returns [0, 100].

    Neutral cluster (all signals near zero): raw_score ≈ 0, sigmoid(0) = 0.5,
    NVI ≈ 50 * coordination. Normal coverage stabilizes around NVI 40-50.
    Elevated threshold at 60, critical at 80.
    """
    raw_score = alpha * burst + beta * spread - gamma * mutation + delta * tone_uniformity
    nvi_raw = _sigmoid(raw_score) * coordination
    return round(min(100, max(0, nvi_raw * 100)), 2)


def _shannon_entropy(items: list) -> float:
    """Compute Shannon entropy of a discrete distribution."""
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def _sigmoid(x: float) -> float:
    """Sigmoid normalization to [0, 1]."""
    return 1.0 / (1.0 + exp(-x))


def compute_all_nvi(db_conn) -> list[dict]:
    """Compute NVI for all active clusters."""
    clusters = db_conn.execute("""
        SELECT id FROM narrative_clusters WHERE status = 'active'
    """).fetchall()

    results = []
    for c in clusters:
        result = compute_nvi(db_conn, c["id"])
        result["cluster_id"] = c["id"]
        results.append(result)

    # Sort by NVI descending
    results.sort(key=lambda x: x["nvi_score"], reverse=True)

    critical = sum(1 for r in results if r["nvi_score"] >= 80)
    elevated = sum(1 for r in results if 50 <= r["nvi_score"] < 80)
    logger.info(f"NVI computed for {len(results)} clusters: {critical} critical, {elevated} elevated")

    return results

"""
BurnTheLies Intelligence — Cross-Narrative Campaign Detection

Detects when multiple narrative clusters are part of the same coordinated campaign.
Four algorithms: source overlap, temporal correlation, theme affinity, amplification chains.
Connected components assembly into campaigns.

All deterministic. No ML. Every link traceable to evidence.
"""

import json
import logging
import numpy as np
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger("intelligence.cross_narrative")


def run_cross_narrative_cycle(db_conn) -> dict:
    """
    Full cross-narrative analysis cycle.
    Returns stats dict.
    """
    from intelligence.db import upsert_narrative_link, upsert_campaign

    # Get active clusters with NVI scores
    clusters = db_conn.execute("""
        SELECT nc.id, nc.label, nc.keywords, nc.post_count, nc.first_seen,
               nv.nvi_score
        FROM narrative_clusters nc
        LEFT JOIN nvi_snapshots nv ON nv.cluster_id = nc.id
            AND nv.id = (SELECT MAX(id) FROM nvi_snapshots WHERE cluster_id = nc.id)
        WHERE nc.status = 'active' AND nc.post_count >= 3
        ORDER BY COALESCE(nv.nvi_score, 0) DESC
        LIMIT 100
    """).fetchall()

    clusters = [dict(c) for c in clusters]
    if len(clusters) < 2:
        logger.info("Not enough active clusters for cross-narrative analysis")
        return {"links": 0, "campaigns": 0}

    logger.info(f"Cross-narrative analysis on {len(clusters)} clusters")

    # Preload source sets and keywords for each cluster
    cluster_data = {}
    for c in clusters:
        cid = c["id"]
        # Extract actual publication domains from article URLs (GDELT stores domain in URL)
        # For Bluesky posts, use the author handle
        import re
        from urllib.parse import urlparse

        gdelt_urls = db_conn.execute("""
            SELECT DISTINCT rp.url FROM cluster_members cm
            JOIN raw_posts rp ON rp.id = cm.post_id
            WHERE cm.cluster_id = ? AND rp.source IN ('gdelt_gkg', 'gdelt_doc')
              AND rp.url IS NOT NULL AND length(rp.url) > 5
        """, (cid,)).fetchall()
        bluesky_sources = db_conn.execute("""
            SELECT DISTINCT rp.author FROM cluster_members cm
            JOIN raw_posts rp ON rp.id = cm.post_id
            WHERE cm.cluster_id = ? AND rp.source = 'bluesky'
              AND rp.author IS NOT NULL
        """, (cid,)).fetchall()

        gdelt_domains = set()
        for r in gdelt_urls:
            url = r["url"] or ""
            try:
                netloc = urlparse(url).netloc
                domain = re.sub(r'^www\.', '', netloc.lower())
                if domain and len(domain) > 3:
                    gdelt_domains.add(domain)
            except (ValueError, AttributeError, TypeError) as e:
                logger.debug(f"urlparse failed cluster={cid} url={url!r}: {e}")

        source_set = gdelt_domains | {f"bsky:{r['author']}" for r in bluesky_sources}

        kw_raw = c.get("keywords", "[]")
        if isinstance(kw_raw, str):
            try:
                keywords = set(json.loads(kw_raw))
            except (json.JSONDecodeError, TypeError):
                keywords = set()
        else:
            keywords = set(kw_raw) if kw_raw else set()

        cluster_data[cid] = {
            "sources": source_set,
            "keywords": keywords,
            "nvi": c.get("nvi_score") or 0,
            "label": c.get("label", ""),
            "first_seen": c.get("first_seen", ""),
            "post_count": c.get("post_count", 0),
        }

    # Pairwise analysis
    link_count = 0
    all_links = []
    cluster_ids = list(cluster_data.keys())

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a, b = cluster_ids[i], cluster_ids[j]
            da, db_ = cluster_data[a], cluster_data[b]

            # Algorithm 1: Source Overlap
            link = _detect_source_overlap(a, b, da, db_)
            if link:
                upsert_narrative_link(db_conn, **link)
                all_links.append(link)
                link_count += 1

            # Algorithm 2: Theme Affinity
            link = _detect_theme_affinity(a, b, da, db_)
            if link:
                upsert_narrative_link(db_conn, **link)
                all_links.append(link)
                link_count += 1

            # Algorithm 3: Amplification Chain
            link = _detect_amplification_chain(db_conn, a, b, da, db_)
            if link:
                upsert_narrative_link(db_conn, **link)
                all_links.append(link)
                link_count += 1

    # Algorithm 4: Temporal Correlation (needs NVI time series)
    temporal_links = _detect_temporal_correlations(db_conn, cluster_ids, cluster_data)
    for link in temporal_links:
        upsert_narrative_link(db_conn, **link)
        all_links.append(link)
        link_count += 1

    # Algorithm 5: DNA-Based Links (from stored DNA matches)
    dna_link_count = _add_dna_links(db_conn, all_links, cluster_data, cluster_ids)

    # Assemble campaigns from connected components (includes DNA links)
    campaign_count = _assemble_campaigns(db_conn, all_links, cluster_data)

    logger.info(f"Cross-narrative: {link_count} source/theme/temporal links, {dna_link_count} DNA links, {campaign_count} campaigns")
    return {"links": link_count + dna_link_count, "campaigns": campaign_count, "dna_links": dna_link_count}


# Wire services that appear across all clusters — exclude from source overlap analysis
_WIRE_SERVICE_DOMAINS = {
    "reuters.com", "apnews.com", "afp.com", "bbc.co.uk", "bbc.com",
    "theguardian.com", "nytimes.com", "wsj.com", "bloomberg.com",
    "washingtonpost.com", "cnn.com", "aljazeera.com", "ft.com",
    "cnbc.com", "thehill.com", "politico.com", "axios.com",
}


def _detect_source_overlap(a: int, b: int, da: dict, db_: dict) -> dict | None:
    """Detect source overlap between two clusters using actual publication domains."""
    sources_a = da["sources"] - _WIRE_SERVICE_DOMAINS
    sources_b = db_["sources"] - _WIRE_SERVICE_DOMAINS
    if len(sources_a) < 2 or len(sources_b) < 2:
        return None

    intersection = sources_a & sources_b
    union = sources_a | sources_b
    jaccard = len(intersection) / len(union) if union else 0

    # Higher threshold: 0.5 to avoid wire-service-driven false positives
    # AND require both clusters to have elevated NVI (coordination independently confirmed)
    if jaccard > 0.5 and da["nvi"] >= 40 and db_["nvi"] >= 40:
        return {
            "cluster_a": min(a, b),
            "cluster_b": max(a, b),
            "link_type": "source_overlap",
            "strength": round(jaccard, 4),
            "evidence": {
                "jaccard_index": round(jaccard, 4),
                "shared_sources": list(intersection)[:20],
                "sources_a_count": len(sources_a),
                "sources_b_count": len(sources_b),
            },
        }
    return None


def _detect_theme_affinity(a: int, b: int, da: dict, db_: dict) -> dict | None:
    """Detect theme/keyword overlap between clusters with different labels."""
    kw_a, kw_b = da["keywords"], db_["keywords"]
    if not kw_a or not kw_b:
        return None

    # Normalize to lowercase and filter noise before comparison
    kw_a_lower = {k.lower() for k in kw_a}
    kw_b_lower = {k.lower() for k in kw_b}

    # Filter out noise tokens that are common to all clusters (URL artifacts + GDELT generic terms)
    _KW_NOISE = {"news", "html", "net", "del", "www", "com", "org", "article",
                 "releases", "index", "source", "gdelt", "gkg"}
    kw_a_lower -= _KW_NOISE
    kw_b_lower -= _KW_NOISE

    if not kw_a_lower or not kw_b_lower:
        return None

    intersection = kw_a_lower & kw_b_lower
    union = kw_a_lower | kw_b_lower
    jaccard = len(intersection) / len(union) if union else 0

    # Must have different labels AND higher threshold (0.4) to avoid noise-driven links
    if jaccard > 0.4 and da["label"] != db_["label"]:
        return {
            "cluster_a": min(a, b),
            "cluster_b": max(a, b),
            "link_type": "theme_affinity",
            "strength": round(jaccard, 4),
            "evidence": {
                "jaccard_index": round(jaccard, 4),
                "shared_keywords": list(intersection)[:15],
                "keywords_a": list(kw_a)[:10],
                "keywords_b": list(kw_b)[:10],
            },
        }
    return None


def _detect_amplification_chain(db_conn, a: int, b: int,
                                 da: dict, db_: dict) -> dict | None:
    """Detect if cluster B is an amplification/reframe of cluster A."""
    # B must have appeared after A
    try:
        ts_a = datetime.fromisoformat(da["first_seen"].replace("Z", "+00:00"))
        ts_b = datetime.fromisoformat(db_["first_seen"].replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    # Determine which came first
    if ts_a > ts_b:
        a, b = b, a
        da, db_ = db_, da
        ts_a, ts_b = ts_b, ts_a

    # Must not be simultaneous (need at least 30 min gap)
    if (ts_b - ts_a).total_seconds() < 1800:
        return None

    # Check source superset: B's sources should include most of A's
    if not da["sources"] or not db_["sources"]:
        return None

    overlap = da["sources"] & db_["sources"]
    if len(overlap) < len(da["sources"]) * 0.5:
        return None  # B doesn't share enough of A's sources

    # Check embedding similarity between clusters
    similarity = _cluster_embedding_similarity(db_conn, a, b)
    if similarity is None or similarity < 0.5:
        return None

    strength = (similarity * 0.5 + len(overlap) / max(len(da["sources"]), 1) * 0.5)

    if strength > 0.4:
        return {
            "cluster_a": min(a, b),
            "cluster_b": max(a, b),
            "link_type": "amplification_chain",
            "strength": round(strength, 4),
            "evidence": {
                "time_gap_hours": round((ts_b - ts_a).total_seconds() / 3600, 2),
                "embedding_similarity": round(similarity, 4),
                "source_overlap_ratio": round(len(overlap) / max(len(da["sources"]), 1), 4),
                "earlier_cluster": a,
                "later_cluster": b,
            },
        }
    return None


def _detect_temporal_correlations(db_conn, cluster_ids: list[int],
                                   cluster_data: dict) -> list[dict]:
    """Detect NVI time series correlations between clusters."""
    links = []

    # Load NVI time series for each cluster
    series = {}
    for cid in cluster_ids:
        rows = db_conn.execute("""
            SELECT timestamp, nvi_score FROM nvi_snapshots
            WHERE cluster_id = ? ORDER BY timestamp ASC
        """, (cid,)).fetchall()
        if len(rows) >= 3:
            series[cid] = {r["timestamp"]: r["nvi_score"] for r in rows}

    if len(series) < 2:
        return links

    # Find common timestamps between pairs
    series_ids = list(series.keys())
    for i in range(len(series_ids)):
        for j in range(i + 1, len(series_ids)):
            a, b = series_ids[i], series_ids[j]
            common_ts = set(series[a].keys()) & set(series[b].keys())

            if len(common_ts) < 3:
                continue

            sorted_ts = sorted(common_ts)
            vals_a = [series[a][t] for t in sorted_ts]
            vals_b = [series[b][t] for t in sorted_ts]

            # Pearson correlation
            if np.std(vals_a) < 0.01 or np.std(vals_b) < 0.01:
                continue

            corr = float(np.corrcoef(vals_a, vals_b)[0, 1])

            if corr > 0.7:
                links.append({
                    "cluster_a": min(a, b),
                    "cluster_b": max(a, b),
                    "link_type": "temporal_correlation",
                    "strength": round(corr, 4),
                    "evidence": {
                        "pearson_correlation": round(corr, 4),
                        "common_timestamps": len(common_ts),
                        "time_range": f"{sorted_ts[0]} to {sorted_ts[-1]}",
                    },
                })

    return links


def _add_dna_links(db_conn, all_links: list, cluster_data: dict,
                    cluster_ids: list[int]) -> int:
    """
    Add DNA-based links from the narrative_dna matches table.
    DNA links connect narratively different but operationally identical campaigns.
    These are the highest-quality links — same operator, different content.
    """
    count = 0
    existing_pairs = set()
    for link in all_links:
        existing_pairs.add((link["cluster_a"], link["cluster_b"]))

    # Get DNA matches where both clusters are in our current active set
    if len(cluster_ids) < 2:
        return 0

    placeholders = ",".join("?" * len(cluster_ids))
    rows = db_conn.execute(f"""
        SELECT dm.cluster_a, dm.cluster_b, dm.match_score, dm.dimension_scores, dm.confidence
        FROM dna_matches dm
        WHERE dm.cluster_a IN ({placeholders}) AND dm.cluster_b IN ({placeholders})
          AND dm.match_score >= 0.60
        ORDER BY dm.match_score DESC
    """, cluster_ids + cluster_ids).fetchall()

    for r in rows:
        a, b = r["cluster_a"], r["cluster_b"]
        if (min(a, b), max(a, b)) in existing_pairs:
            continue

        dim_scores = json.loads(r["dimension_scores"]) if isinstance(r["dimension_scores"], str) else r["dimension_scores"]

        link = {
            "cluster_a": min(a, b),
            "cluster_b": max(a, b),
            "link_type": "dna_match",
            "strength": round(r["match_score"], 4),
            "evidence": {
                "dna_match_score": r["match_score"],
                "dimension_scores": dim_scores,
                "confidence": r["confidence"],
            },
        }

        try:
            from intelligence.db import upsert_narrative_link
            upsert_narrative_link(db_conn, **link)
            all_links.append(link)
            existing_pairs.add((min(a, b), max(a, b)))
            count += 1
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Skipping DNA link pair=({a},{b}): malformed data — {e}",
                exc_info=False,
            )
            continue
        except Exception:
            logger.exception(
                f"Unexpected error inserting DNA link pair=({a},{b})"
            )

    if count:
        logger.info(f"Added {count} DNA-based narrative links")
    return count


def _cluster_embedding_similarity(db_conn, cluster_a: int,
                                   cluster_b: int) -> float | None:
    """Compute cosine similarity between cluster centroids."""
    centroid_a = _get_cluster_centroid(db_conn, cluster_a)
    centroid_b = _get_cluster_centroid(db_conn, cluster_b)

    if centroid_a is None or centroid_b is None:
        return None

    norm_a = np.linalg.norm(centroid_a)
    norm_b = np.linalg.norm(centroid_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return None

    return float(np.dot(centroid_a, centroid_b) / (norm_a * norm_b))


def _get_cluster_centroid(db_conn, cluster_id: int) -> np.ndarray | None:
    """Get the mean embedding vector for a cluster."""
    rows = db_conn.execute("""
        SELECT e.vector FROM embeddings e
        JOIN cluster_members cm ON cm.post_id = e.post_id
        WHERE cm.cluster_id = ?
        LIMIT 50
    """, (cluster_id,)).fetchall()

    if not rows:
        return None

    vectors = np.array([json.loads(r["vector"]) for r in rows])
    return vectors.mean(axis=0)


def _assemble_campaigns(db_conn, links: list[dict],
                         cluster_data: dict) -> int:
    """
    Use union-find to group linked narratives into campaigns.
    Only links with strength > 0.4 are considered.
    """
    from intelligence.db import upsert_campaign

    strong_links = [l for l in links if l["strength"] > 0.4]
    if not strong_links:
        return 0

    # Union-Find
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for link in strong_links:
        union(link["cluster_a"], link["cluster_b"])

    # Group into components
    components = {}
    for node in parent:
        root = find(node)
        if root not in components:
            components[root] = set()
        components[root].add(node)

    # Only campaigns with 2+ narratives
    campaign_count = 0
    for root, members in components.items():
        if len(members) < 2:
            continue

        member_list = sorted(members)
        # Label from highest-NVI narrative
        best_nvi = 0
        best_label = "Multi-Narrative Campaign"
        for m in member_list:
            if m in cluster_data and cluster_data[m]["nvi"] > best_nvi:
                best_nvi = cluster_data[m]["nvi"]
                best_label = cluster_data[m]["label"]

        # Campaign score: mean NVI of members
        nvi_values = [cluster_data[m]["nvi"] for m in member_list if m in cluster_data]
        campaign_score = np.mean(nvi_values) if nvi_values else 0

        # Gather link evidence
        campaign_links = [
            l for l in strong_links
            if l["cluster_a"] in members and l["cluster_b"] in members
        ]
        evidence = {
            "link_count": len(campaign_links),
            "link_types": list(set(l["link_type"] for l in campaign_links)),
            "avg_link_strength": round(
                np.mean([l["strength"] for l in campaign_links]), 4
            ) if campaign_links else 0,
        }

        upsert_campaign(
            db_conn,
            label=best_label,
            narrative_ids=member_list,
            campaign_score=round(float(campaign_score), 2),
            evidence=evidence,
        )
        campaign_count += 1

    return campaign_count

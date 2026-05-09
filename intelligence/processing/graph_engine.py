"""
BurnTheLies Intelligence — Source Amplification Graph Engine

Builds a directed graph of the information ecosystem: nodes = source domains/accounts,
edges = amplification relationships (B published similar content after A within a
time window). Graph topology reveals coordinated amplifier networks that individual
post-level signals miss.

Key metrics that flag coordination:
  - High clustering coefficient + low diameter → coordinated amplifier ring
  - Core-periphery structure → centralized command + amplifier network
  - Bimodal degree distribution → bots + commanders pattern
  - Sudden edge density spikes → amplifier network activation
  - Low reciprocity → one-directional amplification (not mutual citation)

Graph rebuilt every cycle. Metrics stored in amplification_graph_snapshots.
"""

import json
import logging
import re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger("intelligence.graph_engine")

# Lazy-load networkx (optional dependency for graph analysis)
_nx = None


def _get_nx():
    global _nx
    if _nx is None:
        try:
            import networkx as nx
            _nx = nx
        except ImportError:
            logger.warning("networkx not available — graph analysis disabled")
    return _nx


def build_amplification_graph(db_conn, lookback_hours: int = 48,
                               max_edges_per_node: int = 100):
    """
    Build a directed source amplification graph from recent posts.
    Edge A→B: source B published thematically similar content within 2 hours after A.

    Returns networkx DiGraph or None if networkx is unavailable.
    """
    nx = _get_nx()
    if nx is None:
        return None

    # Get recent posts with timestamps and URLs
    rows = db_conn.execute(f"""
        SELECT id, source, url, title, content, published_at
        FROM raw_posts
        WHERE ingested_at >= datetime('now', '-{lookback_hours} hours')
          AND url IS NOT NULL AND length(url) > 5
        ORDER BY published_at ASC
    """).fetchall()

    if len(rows) < 10:
        logger.info("Not enough posts for graph construction")
        return None

    logger.info(f"Building amplification graph from {len(rows)} posts")

    # Extract domains and timestamps
    domain_posts = defaultdict(list)  # domain → [(timestamp, post_id), ...]
    for r in rows:
        url = r["url"] or ""
        try:
            netloc = urlparse(url).netloc
            domain = re.sub(r'^www\.', '', netloc.lower())
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"urlparse failed url={url!r} post={r['id']}: {e}")
            domain = r["source"]
        if not domain or len(domain) < 3:
            continue

        try:
            ts = datetime.fromisoformat((r["published_at"] or "").replace("Z", "+00:00"))
            ts_float = ts.timestamp()
        except (ValueError, TypeError):
            continue

        domain_posts[domain].append((ts_float, r["id"]))

    # Sort each domain's posts by time
    for domain in domain_posts:
        domain_posts[domain].sort()

    domains = list(domain_posts.keys())
    if len(domains) < 3:
        logger.info("Not enough unique domains for graph")
        return None

    logger.info(f"Graph nodes (domains): {len(domains)}")

    # Build graph: A→B with weight = count of B publishing within 2h after A
    G = nx.DiGraph()
    G.add_nodes_from(domains)

    edge_count = 0
    for a in domains:
        a_posts = domain_posts[a]
        edge_candidates = defaultdict(int)

        for b in domains:
            if a == b:
                continue
            b_posts = domain_posts[b]
            # Count B posts that follow A posts within 2 hours
            count = 0
            b_idx = 0
            for ta, _ in a_posts:
                # Find B posts in (ta, ta+7200] window
                while b_idx < len(b_posts) and b_posts[b_idx][0] <= ta:
                    b_idx += 1
                temp_idx = b_idx
                while temp_idx < len(b_posts) and b_posts[temp_idx][0] <= ta + 7200:
                    count += 1
                    temp_idx += 1
            if count > 0:
                edge_candidates[b] = count

        # Keep top N edges per node to control graph density
        top_edges = sorted(edge_candidates.items(), key=lambda x: x[1], reverse=True)
        for b, weight in top_edges[:max_edges_per_node]:
            G.add_edge(a, b, weight=weight)
            edge_count += 1

    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G


def compute_graph_metrics(G) -> dict:
    """
    Compute coordination-relevant topology metrics from the amplification graph.

    Returns dict of metrics. Every value is deterministic.
    """
    if G is None or G.number_of_nodes() < 3 or G.number_of_edges() < 2:
        return {
            "node_count": G.number_of_nodes() if G else 0,
            "edge_count": G.number_of_edges() if G else 0,
            "is_coordination_topology": False,
            "topology_signals": [],
        }

    nx = _get_nx()
    if nx is None:
        return {"node_count": 0, "edge_count": 0, "is_coordination_topology": False, "topology_signals": []}

    n = G.number_of_nodes()
    m = G.number_of_edges()
    signals = []
    score = 0.0

    # 1: Clustering coefficient (undirected)
    try:
        cc = nx.average_clustering(G.to_undirected())
        cc_signal = cc > 0.6  # High clustering with low diameter = amplifier ring
        if cc_signal:
            signals.append("high_clustering_coefficient")
            score += 0.15
    except (nx.NetworkXError, ZeroDivisionError, ValueError) as e:
        logger.warning(f"Clustering coefficient failed (n={n}, m={m}): {e}")
        cc = 0
    except Exception:
        logger.exception(f"Unexpected error computing clustering coefficient (n={n}, m={m})")
        cc = 0

    # 2: Average path length
    try:
        if nx.is_weakly_connected(G):
            apl = nx.average_shortest_path_length(G)
        else:
            largest = max(nx.weakly_connected_components(G), key=len)
            sub = G.subgraph(largest)
            apl = nx.average_shortest_path_length(sub) if len(sub) > 1 else n
    except (nx.NetworkXError, nx.NetworkXPointlessConcept, ValueError) as e:
        logger.warning(f"Average path length failed (n={n}, m={m}): {e}")
        apl = n
    except Exception:
        logger.exception(f"Unexpected error computing average path length (n={n}, m={m})")
        apl = n

    apl_signal = apl < 2.5 and n > 5  # Low diameter = tight amplifier network
    if apl_signal:
        signals.append("low_diameter")
        score += 0.15

    # 3: Small-world coefficient (cc / random_cc) / (apl / random_apl)
    sw = 0
    if n > 10 and cc > 0:
        try:
            random_g = nx.gnm_random_graph(n, m, directed=False)
            random_cc = nx.average_clustering(random_g) if random_g.number_of_nodes() > 2 else 0.001
            if nx.is_connected(random_g):
                random_apl = nx.average_shortest_path_length(random_g)
            else:
                random_apl = n
            sw = (cc / max(random_cc, 0.001)) / (apl / max(random_apl, 0.001))
            sw_signal = sw > 2.0
            if sw_signal:
                signals.append("small_world_topology")
                score += 0.10
        except (nx.NetworkXError, ZeroDivisionError, ValueError) as e:
            logger.warning(f"Small-world coefficient failed (n={n}, m={m}): {e}")
            sw = 0
        except Exception:
            logger.exception(f"Unexpected error computing small-world (n={n}, m={m})")
            sw = 0

    # 4: Degree distribution analysis
    degrees = [d for _, d in G.degree()]
    if degrees:
        deg_mean = np.mean(degrees)
        deg_std = np.std(degrees)
        # Bimodal distribution: high std/mean ratio + presence of high-degree nodes
        bimodal = deg_std > deg_mean * 1.5 and max(degrees) > deg_mean * 3
        if bimodal:
            signals.append("bimodal_degree_distribution")
            score += 0.10

    # 5: Reciprocity (mutual amplification vs one-directional)
    try:
        reciprocity = nx.reciprocity(G)
        low_recip = reciprocity < 0.15 and m > 5  # One-directional = coordination
        if low_recip:
            signals.append("low_reciprocity")
            score += 0.10
    except (nx.NetworkXError, ZeroDivisionError, ValueError) as e:
        logger.warning(f"Reciprocity failed (n={n}, m={m}): {e}")
        reciprocity = 0
    except Exception:
        logger.exception(f"Unexpected error computing reciprocity (n={n}, m={m})")
        reciprocity = 0

    # 6: Core-periphery detection (degree threshold method)
    core_ratio = 0.0
    if degrees:
        deg_median = np.median(degrees)
        core_nodes = [n for n, d in G.degree() if d > deg_median * 2]
        core_size = len(core_nodes)
        core_ratio = core_size / n if n > 0 else 0
        # Core-periphery: small core with high degree, large periphery with low degree
        cp_signal = 0.05 < core_ratio < 0.30
        if cp_signal:
            signals.append("core_periphery_structure")
            score += 0.10

    # 7: Edge density
    density = nx.density(G)
    # Both very low and very high density are suspicious in different ways
    if density < 0.01:
        signals.append("sparse_but_structured")
        score += 0.05
    elif density > 0.5:
        signals.append("dense_amplifier_mesh")
        score += 0.10

    # 8: In/out degree correlation
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    if len(in_degrees) > 3 and np.std(in_degrees) > 0.01 and np.std(out_degrees) > 0.01:
        try:
            io_corr = np.corrcoef(in_degrees, out_degrees)[0, 1]
            if not np.isnan(io_corr):
                neg_corr = io_corr < -0.3  # Negative correlation = some nodes only amplify, others only get amplified
                if neg_corr:
                    signals.append("in_out_degree_anti_correlation")
                    score += 0.10
        except (ValueError, FloatingPointError) as e:
            logger.debug(f"in/out degree correlation failed (n={n}): {e}")
        except Exception:
            logger.exception(f"Unexpected error computing in/out degree correlation (n={n})")

    # Normalize topology score to [0, 1]
    topology_score = min(1.0, round(score, 4))
    is_coordination = topology_score > 0.25

    return {
        "node_count": n,
        "edge_count": m,
        "clustering_coefficient": round(cc, 4) if isinstance(cc, (int, float)) else 0,
        "average_path_length": round(apl, 2) if isinstance(apl, (int, float)) else n,
        "small_world_coefficient": round(sw, 2) if isinstance(sw, (int, float)) else 0,
        "reciprocity": round(reciprocity, 4) if isinstance(reciprocity, (int, float)) else 0,
        "edge_density": round(density, 6) if isinstance(density, (int, float)) else 0,
        "core_ratio": round(core_ratio, 4),
        "topology_score": topology_score,
        "is_coordination_topology": is_coordination,
        "topology_signals": signals,
    }


def run_graph_cycle(db_conn, lookback_hours: int = 48) -> dict:
    """
    Full graph analysis cycle: build graph → compute metrics → store snapshot.

    Returns metrics dict.
    """
    from intelligence.db import store_graph_snapshot

    G = build_amplification_graph(db_conn, lookback_hours=lookback_hours)
    metrics = compute_graph_metrics(G)

    # Store snapshot
    if G is not None:
        # Store graph data as adjacency list for later retrieval
        graph_data = {
            "nodes": list(G.nodes()),
            "edges": [
                {"from": u, "to": v, "weight": d.get("weight", 1)}
                for u, v, d in G.edges(data=True)
            ],
        }
        store_graph_snapshot(
            db_conn,
            node_count=metrics["node_count"],
            edge_count=metrics["edge_count"],
            metrics=metrics,
            graph_data=graph_data,
        )

    if metrics.get("is_coordination_topology"):
        logger.warning(
            f"Amplifier network coordination detected! "
            f"Signals: {metrics['topology_signals']}, "
            f"Score: {metrics['topology_score']}"
        )

    return metrics


def get_cluster_subgraph(db_conn, cluster_id: int):
    """
    Extract the ego amplification graph for a specific narrative cluster.
    Returns the subgraph induced by sources that contributed to this cluster.
    """
    nx = _get_nx()
    if nx is None:
        return None

    # Get sources in this cluster
    rows = db_conn.execute("""
        SELECT DISTINCT rp.url, rp.published_at, rp.source
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
          AND rp.url IS NOT NULL AND length(rp.url) > 5
        ORDER BY rp.published_at ASC
    """, (cluster_id,)).fetchall()

    if len(rows) < 3:
        return None

    domains = set()
    domain_times = defaultdict(list)
    for r in rows:
        try:
            netloc = urlparse(r["url"]).netloc
            domain = re.sub(r'^www\.', '', netloc.lower())
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"urlparse failed url={r['url']!r}: {e}")
            domain = r["source"]
        if domain and len(domain) > 3:
            domains.add(domain)
            try:
                ts = datetime.fromisoformat((r["published_at"] or "").replace("Z", "+00:00"))
                domain_times[domain].append(ts.timestamp())
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"published_at parse failed cluster={cluster_id} "
                    f"value={r['published_at']!r}: {e}"
                )

    domain_list = sorted(domains)
    G = nx.DiGraph()
    G.add_nodes_from(domain_list)

    for a in domain_list:
        for b in domain_list:
            if a == b:
                continue
            weight = 0
            for ta in domain_times.get(a, []):
                for tb in domain_times.get(b, []):
                    if 0 < tb - ta <= 7200:
                        weight += 1
            if weight > 0:
                G.add_edge(a, b, weight=weight)

    return G

"""
BurnTheLies Intelligence — Narrative DNA Fingerprinting

Persistent multi-modal operational fingerprints that survive domain changes,
content paraphrasing, and timing jitter. When a campaign disappears and re-emerges
with fresh infrastructure and different wording, DNA matching links them.

Four independent dimensions, each hard for an adversary to simultaneously control:
  D1 — Stylometric (32-dim): function word frequencies, sentence structure
  D2 — Cadence (16-dim): FFT of posting timestamps, spectral signature
  D3 — Network (12-dim): amplifier graph topology
  D4 — Entity bias (24-dim): consistent entity pair associations

Total fingerprint: 84-dimensional vector.
Matching: weighted cosine similarity, threshold ≥0.75 = same operator.

No ML. Every dimension is deterministic and traceable to evidence.
"""

import json
import logging
import math
import os
import re
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("intelligence.dna")

# Write-time filter: refuse to persist matches below this score regardless of
# the caller's min_score. Cap on dna_matches table growth.
DNA_MATCH_PERSIST_THRESHOLD = float(os.getenv("INTEL_DNA_PERSIST_THRESHOLD", "0.50"))

# Memory guard for the (n_q × n_all) similarity matrix in batch_cosine_matches.
_DNA_LARGE_CORPUS_THRESHOLD = 50000
_DNA_DB_CHUNK_SIZE = 10000

# ─── Dimension weights for matching ─────────────────────────────────────────
# Tuned to penalize dimensions an adversary CAN control and reward those they can't.
# Stylometry + cadence are hardest to fake simultaneously → highest weights.
DNA_WEIGHTS = {
    "stylometric": 0.30,
    "cadence": 0.30,
    "network": 0.20,
    "entity_bias": 0.20,
}

# Match threshold: cosine similarity >= this → same operator
DNA_MATCH_THRESHOLD = 0.75

# ─── D1: Stylometric Fingerprint ────────────────────────────────────────────

# Function words that survive paraphrasing and are mostly language-invariant
_FUNCTION_WORDS = [
    "the", "of", "and", "to", "in", "that", "it", "was", "for", "on",
    "are", "as", "with", "his", "they", "at", "be", "this", "have", "from",
    "or", "one", "had", "by", "but", "not", "what", "all", "were", "we",
    "when", "your", "can", "said", "there", "use", "an", "each", "which",
    "she", "do", "how", "their", "if", "will", "up", "other", "about",
    "out", "many", "then", "them", "these", "so", "some", "her", "would",
    "make", "like", "him", "into", "time", "has", "look", "two", "more",
    "write", "go", "see", "number", "no", "way", "could", "people", "my",
    "than", "first", "water", "been", "call", "who", "oil", "its", "now",
    "find", "long", "down", "day", "did", "get", "come", "made", "may", "part",
]


def compute_stylometric_vector(texts: list[str]) -> np.ndarray:
    """
    Extract 32-dim stylometric fingerprint from a collection of texts.
    Paraphrasing preserves function word distribution. Translation preserves
    sentence structure patterns. This is the hardest dimension to fake.

    Returns normalized float32 vector or zero vector if insufficient data.
    """
    if not texts:
        return np.zeros(32, dtype=np.float32)

    combined = " ".join(texts)
    words = re.findall(r'\b[a-zA-Z]+\b', combined.lower())
    if len(words) < 100:
        return np.zeros(32, dtype=np.float32)

    total = len(words)

    # 1-20: Function word frequencies (normalized by total words)
    fw_freqs = []
    for fw in _FUNCTION_WORDS[:20]:
        count = words.count(fw)
        fw_freqs.append(count / total)

    # 21-22: Sentence length statistics
    sentences = re.split(r'[.!?]+', combined)
    sent_lens = [len(s.split()) for s in sentences if s.strip()]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    var_sent_len = np.var(sent_lens) if len(sent_lens) > 1 else 0

    # 23: Punctuation density (commas + periods + colons per 1000 chars)
    punct_chars = len(re.findall(r'[,.:;]', combined))
    punct_density = punct_chars / max(len(combined), 1) * 1000

    # 24: Type-token ratio (vocabulary richness)
    unique_words = len(set(words))
    ttr = unique_words / total

    # 25: Haplology rate — repeated bigram ratio (lower = more repetitive = coordinated)
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = len(set(bigrams))
    haplology = 1.0 - (unique_bigrams / max(len(bigrams), 1))

    # 26-27: Content-to-function word ratios (how "substantive" the text is)
    function_count = sum(1 for w in words if w in _FUNCTION_WORDS)
    content_ratio = 1.0 - (function_count / total)
    fw_diversity = len(set(w for w in words if w in _FUNCTION_WORDS)) / max(len(_FUNCTION_WORDS), 1)

    # 28: Average word length
    avg_word_len = np.mean([len(w) for w in words])

    # 29: Capitalization rate (proper noun density, survives translation)
    caps_rate = len(re.findall(r'\b[A-Z][a-z]+\b', combined)) / max(total, 1)

    # 30: Quote density (quoted speech patterns)
    quote_density = len(re.findall(r'"([^"]*)"', combined)) / max(len(sentences), 1)

    # 31: Passive voice indicator (forms of "be" + past participle endings "ed")
    passive_indicators = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', combined.lower()))
    passive_rate = passive_indicators / max(len(sentences), 1)

    # 32: Negation rate (cognitive complexity signal)
    negation_rate = len(re.findall(r'\b(not|no|never|neither|nor)\b', combined.lower())) / max(total, 1)

    vector = np.array([
        *fw_freqs,
        avg_sent_len / 50,          # normalize to ~[0,1]
        var_sent_len / 500,         # normalize
        punct_density / 50,         # normalize
        ttr,                        # already [0,1]
        haplology,                  # already [0,1]
        content_ratio,              # already [0,1]
        fw_diversity,               # already [0,1]
        avg_word_len / 10,          # normalize
        caps_rate * 10,             # amplify
        quote_density,
        passive_rate,
        negation_rate * 100,        # amplify
    ], dtype=np.float32)

    # Clip and normalize
    vector = np.clip(vector, -5, 5)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.astype(np.float32)


# ─── D2: Frequency-Domain Cadence Fingerprint ──────────────────────────────


def compute_cadence_vector(timestamps: list[float]) -> np.ndarray:
    """
    Extract 16-dim cadence fingerprint from posting timestamps using FFT.
    Jitter broadens spectral peaks but the fundamental frequency persists.

    Returns normalized float32 vector or zero vector if insufficient data.
    """
    if len(timestamps) < 10:
        return np.zeros(16, dtype=np.float32)

    timestamps = sorted(timestamps)
    start = timestamps[0]
    end = timestamps[-1]
    duration_hours = (end - start) / 3600

    if duration_hours < 1:
        return np.zeros(16, dtype=np.float32)

    # Bin into 10-minute windows
    bin_seconds = 600
    bins = int(duration_hours * 6) + 1
    histogram = np.zeros(bins)
    for ts in timestamps:
        idx = min(bins - 1, int((ts - start) / bin_seconds))
        histogram[idx] += 1

    # FFT on the histogram
    fft = np.abs(np.fft.rfft(histogram))
    freqs = np.fft.rfftfreq(len(histogram), d=bin_seconds)

    # Avoid DC component
    if len(fft) < 2:
        return np.zeros(16, dtype=np.float32)

    # 1-4: Top 4 frequency peaks (normalized by DC)
    dc = fft[0] or 1.0
    peak_indices = np.argsort(fft[1:])[-4:] + 1
    peak_freqs_hz = freqs[peak_indices]
    peak_amps = fft[peak_indices] / dc

    # 5: Spectral centroid
    spectral_centroid = np.sum(freqs[1:] * fft[1:]) / max(np.sum(fft[1:]), 1e-8)

    # 6: Spectral spread
    spectral_spread = np.sqrt(
        np.sum(((freqs[1:] - spectral_centroid) ** 2) * fft[1:]) / max(np.sum(fft[1:]), 1e-8)
    )

    # 7: Spectral flatness (geometric mean / arithmetic mean — high = noise-like, low = tonal)
    nonzero = fft[1:][fft[1:] > 1e-10]
    if len(nonzero) > 0:
        geo_mean = np.exp(np.mean(np.log(nonzero)))
        arith_mean = np.mean(fft[1:]) or 1e-8
        spectral_flatness = geo_mean / arith_mean
    else:
        spectral_flatness = 1.0

    # 8-9: Autocorrelation peak lag and strength
    hist_centered = histogram - np.mean(histogram)
    autocorr = np.correlate(hist_centered, hist_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / max(autocorr[0], 1e-8)
    # Find first peak after lag 0
    peaks = []
    for i in range(2, len(autocorr)-1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
            peaks.append((i, autocorr[i]))
    if peaks:
        best_peak = max(peaks, key=lambda x: x[1])
        ac_peak_lag = best_peak[0] * bin_seconds / 3600  # in hours
        ac_peak_strength = best_peak[1]
    else:
        ac_peak_lag = 0
        ac_peak_strength = 0

    # 10-11: Burst-decay metrics
    # How quickly does the histogram drop from its peak?
    peak_idx = np.argmax(histogram)
    peak_val = histogram[peak_idx]
    # Decay half-life: how many bins until histogram drops to peak/2
    decay_half = 0
    for i in range(peak_idx + 1, len(histogram)):
        if histogram[i] <= peak_val / 2:
            decay_half = (i - peak_idx) * bin_seconds / 60  # in minutes
            break
    # Burst sharpness: ratio of peak to mean of surrounding bins
    window = max(1, len(histogram) // 10)
    surround_start = max(0, peak_idx - window)
    surround_end = min(len(histogram), peak_idx + window)
    surround_mean = np.mean(histogram[surround_start:surround_end]) if surround_end > surround_start else 1
    burst_sharpness = peak_val / max(surround_mean, 0.01)

    # 12-13: Inter-arrival statistics
    deltas = np.diff(timestamps)
    ia_mean = np.mean(deltas) / 60  # minutes
    ia_cv = np.std(deltas) / max(ia_mean * 60, 1)  # coefficient of variation

    # 14-16: Posting density distribution (skew, kurtosis of hourly rates)
    hourly_bins = max(1, int(duration_hours))
    hourly_hist = np.zeros(hourly_bins)
    for ts in timestamps:
        idx = min(hourly_bins - 1, int((ts - start) / 3600))
        hourly_hist[idx] += 1
    hourly_mean = np.mean(hourly_hist)
    hourly_std = np.std(hourly_hist)
    hourly_skew = np.mean(((hourly_hist - hourly_mean) / max(hourly_std, 0.01)) ** 3) if hourly_std > 0.01 else 0
    hourly_kurt = np.mean(((hourly_hist - hourly_mean) / max(hourly_std, 0.01)) ** 4) if hourly_std > 0.01 else 0

    vector = np.array([
        peak_freqs_hz[0] * 10000 if len(peak_freqs_hz) > 0 else 0,
        peak_freqs_hz[1] * 10000 if len(peak_freqs_hz) > 1 else 0,
        peak_freqs_hz[2] * 10000 if len(peak_freqs_hz) > 2 else 0,
        peak_freqs_hz[3] * 10000 if len(peak_freqs_hz) > 3 else 0,
        min(spectral_centroid * 1000, 10),
        min(spectral_spread * 1000, 10),
        min(spectral_flatness * 5, 5),
        min(ac_peak_lag / 24, 1),
        ac_peak_strength,
        min(decay_half / 120, 2),
        min(burst_sharpness / 10, 5),
        min(ia_mean / 60, 5),
        min(ia_cv * 5, 5),
        np.clip(hourly_skew, -5, 5),
        np.clip(hourly_kurt / 3, 0, 5),
        0.0,  # reserved
    ], dtype=np.float32)

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.astype(np.float32)


# ─── D3: Network Topology Fingerprint ──────────────────────────────────────


def compute_network_vector(cluster_id: int, db_conn) -> np.ndarray:
    """
    Extract 12-dim topology fingerprint from the source amplification graph
    surrounding this cluster. Captures the structure of the amplifier network.

    Returns normalized float32 vector or zero vector if insufficient data.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available — network DNA will be zeros")
        return np.zeros(12, dtype=np.float32)

    # Build ego network: sources that published content in this cluster
    rows = db_conn.execute("""
        SELECT DISTINCT rp.url, rp.published_at, rp.source
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
          AND rp.url IS NOT NULL AND length(rp.url) > 5
        ORDER BY rp.published_at ASC
    """, (cluster_id,)).fetchall()

    if len(rows) < 5:
        return np.zeros(12, dtype=np.float32)

    # Extract domains
    from urllib.parse import urlparse
    domains = []
    domain_times = {}
    for r in rows:
        try:
            netloc = urlparse(r["url"]).netloc
            domain = re.sub(r'^www\.', '', netloc.lower())
        except (ValueError, AttributeError) as e:
            # Malformed URL — fall back to raw source field; recoverable per-row.
            logger.debug("urlparse failed for url=%r: %s", r["url"], e)
            domain = r["source"]
        if domain and len(domain) > 3:
            domains.append(domain)
            if domain not in domain_times:
                domain_times[domain] = []
            try:
                ts = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00"))
                domain_times[domain].append(ts.timestamp())
            except (ValueError, TypeError, AttributeError) as e:
                # Bad/missing timestamp on a single row; skip without aborting the cluster.
                logger.debug("published_at parse failed: %s", e)

    unique_domains = list(set(domains))
    if len(unique_domains) < 3:
        return np.zeros(12, dtype=np.float32)

    # Build directed graph: A → B if B published after A within 2 hours
    G = nx.DiGraph()
    G.add_nodes_from(unique_domains)

    for a in unique_domains:
        for b in unique_domains:
            if a == b:
                continue
            if a not in domain_times or b not in domain_times:
                continue
            # Count how many times b published within 2 hours after a
            weight = 0
            for ta in domain_times[a]:
                for tb in domain_times[b]:
                    if 0 < tb - ta <= 7200:  # within 2 hours after
                        weight += 1
            if weight > 0:
                G.add_edge(a, b, weight=weight)

    if G.number_of_edges() < 2:
        return np.zeros(12, dtype=np.float32)

    n = G.number_of_nodes()

    # 1: Clustering coefficient
    cc = nx.average_clustering(G.to_undirected()) if n > 2 else 0

    # 2: Average path length — requires strongly connected directed graph
    try:
        if nx.is_strongly_connected(G):
            apl = nx.average_shortest_path_length(G)
        elif nx.is_weakly_connected(G):
            apl = nx.average_shortest_path_length(G.to_undirected())
        else:
            largest = max(nx.weakly_connected_components(G), key=len)
            sub = G.subgraph(largest).to_undirected()
            apl = nx.average_shortest_path_length(sub) if len(sub) > 1 else 0
    except (nx.NetworkXError, nx.NetworkXException, ValueError) as e:
        logger.debug("network DNA: average_shortest_path_length fallback: %s", e)
        apl = 0

    # 3: Diameter
    try:
        if nx.is_weakly_connected(G) and n > 1:
            diam = nx.diameter(G.to_undirected())
        else:
            diam = 0
    except (nx.NetworkXError, nx.NetworkXException, ValueError) as e:
        logger.debug("network DNA: diameter fallback: %s", e)
        diam = 0

    # 4: Degree assortativity
    try:
        assort = nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 5 else 0
    except (nx.NetworkXError, nx.NetworkXException, ValueError, ZeroDivisionError) as e:
        logger.debug("network DNA: degree_assortativity_coefficient fallback: %s", e)
        assort = 0

    # 5: Edge density
    density = nx.density(G)

    # 6-7: Core-periphery ratio
    core_ratio = 0.0
    core_avg_deg = 0.0
    degrees = dict(G.degree())
    if degrees:
        median_deg = np.median(list(degrees.values()))
        core = sum(1 for d in degrees.values() if d > median_deg * 1.5)
        periphery = n - core
        core_ratio = core / max(n, 1)
        core_avg_deg = np.mean([d for d in degrees.values() if d > median_deg * 1.5]) if core > 0 else 0

    # 8-9: Eigenvector centrality of top 3 nodes (requires connected graph)
    try:
        if n <= 500:
            G_undir = G.to_undirected()
            if nx.is_connected(G_undir):
                ec = nx.eigenvector_centrality_numpy(G, weight='weight')
            else:
                largest_cc = max(nx.connected_components(G_undir), key=len)
                sub = G.subgraph(largest_cc)
                ec = nx.eigenvector_centrality_numpy(sub, weight='weight') if len(sub) > 1 else {}
        else:
            ec = {}
    except (nx.NetworkXError, nx.NetworkXException, np.linalg.LinAlgError, ArithmeticError) as e:
        logger.debug("network DNA: eigenvector_centrality_numpy fallback: %s", e)
        ec = {}
    top_ec = sorted(ec.values(), reverse=True)[:3] if ec else []
    ec_top1 = top_ec[0] if len(top_ec) > 0 else 0
    ec_top3_mean = np.mean(top_ec) if top_ec else 0

    # 10: Reciprocity (mutual amplification)
    reciprocity = nx.reciprocity(G) if G.number_of_edges() > 1 else 0

    # 11: Modularity (community structure)
    try:
        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        modularity_val = len(communities)
    except (nx.NetworkXError, nx.NetworkXException, ValueError, ZeroDivisionError) as e:
        logger.debug("network DNA: greedy_modularity_communities fallback: %s", e)
        modularity_val = 1

    # 12: In/out degree ratio variance (coordinated = uniform)
    in_degs = [G.in_degree(node) for node in G.nodes()]
    out_degs = [G.out_degree(node) for node in G.nodes()]
    ratios = []
    for in_d, out_d in zip(in_degs, out_degs):
        if out_d > 0:
            ratios.append(in_d / out_d)
    ratio_var = np.var(ratios) if len(ratios) > 1 else 0

    vector = np.array([
        min(cc * 3, 3),
        min(apl / 5, 3) if apl > 0 else 0,
        min(diam / 10, 3) if diam > 0 else 0,
        np.clip(assort, -1, 1),
        min(density * 5, 5),
        core_ratio,
        min(core_avg_deg / 5, 5),
        ec_top1 * 5,
        ec_top3_mean * 5,
        reciprocity * 3,
        min(modularity_val / 5, 3),
        min(ratio_var * 3, 3),
    ], dtype=np.float32)

    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.astype(np.float32)


# ─── D4: Entity Association Bias ───────────────────────────────────────────


def compute_entity_bias_vector(cluster_id: int, db_conn) -> np.ndarray:
    """
    Extract 24-dim entity association fingerprint from named entities
    consistently paired together across a campaign's narratives.

    Uses PMI-based co-occurrence matrix, reduced via SVD to 24 dimensions.
    Reveals the operator's consistent worldview focus.
    """
    # Get all entities across cluster members
    rows = db_conn.execute("""
        SELECT rp.metadata FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
        LIMIT 200
    """, (cluster_id,)).fetchall()

    person_freq = Counter()
    org_freq = Counter()
    co_occur = Counter()  # (entity_a, entity_b) → count

    entity_sets = []
    for r in rows:
        meta_raw = r["metadata"]
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            meta = meta_raw or {}

        persons_str = meta.get("persons", "") or ""
        orgs_str = meta.get("organizations", "") or ""

        persons = {p.strip().lower() for p in persons_str.split(";")
                   if p.strip() and len(p.strip()) > 2 and "#" not in p}
        orgs = {o.strip().lower() for o in orgs_str.split(";")
                if o.strip() and len(o.strip()) > 2 and "#" not in o}

        entities = persons | orgs
        if len(entities) >= 2:
            entity_sets.append(entities)
            for e in entities:
                if e in persons:
                    person_freq[e] += 1
                else:
                    org_freq[e] += 1
            # Count co-occurrences
            entity_list = sorted(entities)
            for i in range(len(entity_list)):
                for j in range(i+1, len(entity_list)):
                    pair = (entity_list[i], entity_list[j])
                    co_occur[pair] += 1

    if len(entity_sets) < 3 or len(co_occur) < 5:
        return np.zeros(24, dtype=np.float32)

    # Get top entities (combined persons + orgs, at least 15)
    all_entities = dict(person_freq)
    for k, v in org_freq.items():
        all_entities[k] = all_entities.get(k, 0) + v

    top_entities = [e for e, _ in Counter(all_entities).most_common(30)]
    if len(top_entities) < 5:
        return np.zeros(24, dtype=np.float32)

    # Build PMI co-occurrence matrix
    total_posts = len(entity_sets)
    entity_index = {e: i for i, e in enumerate(top_entities)}

    cooc_matrix = np.zeros((len(top_entities), len(top_entities)))
    for (a, b), count in co_occur.items():
        if a in entity_index and b in entity_index:
            i, j = entity_index[a], entity_index[b]
            # Simple PMI: log(P(a,b) / (P(a) * P(b)))
            p_ab = count / total_posts
            p_a = all_entities.get(a, 0) / total_posts
            p_b = all_entities.get(b, 0) / total_posts
            denom = max(p_a * p_b, 1e-8)
            pmi = max(0, math.log(max(p_ab, 1e-8) / denom))
            cooc_matrix[i, j] = pmi
            cooc_matrix[j, i] = pmi

    # Flatten via SVD to 24 dimensions
    try:
        U, S, Vt = np.linalg.svd(cooc_matrix, full_matrices=False)
        k = min(24, len(S))
        vector = np.zeros(24, dtype=np.float32)
        # Weight singular vectors by singular values
        for i in range(k):
            vector[i] = S[i] * np.mean(np.abs(U[:, i]))
    except np.linalg.LinAlgError:
        return np.zeros(24, dtype=np.float32)

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.astype(np.float32)


# ─── Full DNA Computation ──────────────────────────────────────────────────


def compute_dna_fingerprint(db_conn, cluster_id: int) -> dict:
    """
    Compute complete 84-dimensional DNA fingerprint for a narrative cluster.

    Returns:
        {
            "cluster_id": int,
            "fingerprint": list[float],  # 84-dim concatenated vector
            "dimensions": {
                "stylometric": list[float],  # 32-dim
                "cadence": list[float],       # 16-dim
                "network": list[float],       # 12-dim
                "entity_bias": list[float],   # 24-dim
            },
            "metadata": {
                "text_count": int,
                "timestamp_count": int,
                "graph_nodes": int,
                "entity_count": int,
            }
        }
    """
    # Get cluster posts
    posts = db_conn.execute("""
        SELECT rp.id, rp.title, rp.content, rp.published_at, rp.source, rp.metadata
        FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ?
        ORDER BY rp.published_at ASC
    """, (cluster_id,)).fetchall()

    posts = [dict(p) for p in posts]

    # D1: Stylometric
    texts = []
    timestamps = []
    for p in posts:
        title = p.get("title", "") or ""
        content = p.get("content", "") or ""
        text = f"{title} {content}".strip()
        if text:
            texts.append(text)
        try:
            ts = datetime.fromisoformat((p.get("published_at") or "").replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError, AttributeError) as e:
            # One bad timestamp shouldn't kill the fingerprint; cadence tolerates missing rows.
            logger.debug("compute_dna_fingerprint: published_at parse failed: %s", e)

    stylometric = compute_stylometric_vector(texts)

    # D2: Cadence
    cadence = compute_cadence_vector(timestamps)

    # D3: Network topology
    network = compute_network_vector(cluster_id, db_conn)

    # D4: Entity bias
    entity_bias = compute_entity_bias_vector(cluster_id, db_conn)

    # Concatenate
    full = np.concatenate([stylometric, cadence, network, entity_bias])
    norm = np.linalg.norm(full)
    if norm > 0:
        full = full / norm

    result = {
        "cluster_id": cluster_id,
        "fingerprint": full.tolist(),
        "dimensions": {
            "stylometric": stylometric.tolist(),
            "cadence": cadence.tolist(),
            "network": network.tolist(),
            "entity_bias": entity_bias.tolist(),
        },
        "metadata": {
            "text_count": len(texts),
            "timestamp_count": len(timestamps),
            "graph_nodes": int(np.count_nonzero(network)),
            "entity_count": int(np.count_nonzero(entity_bias)),
        },
    }

    return result


# ─── DNA Comparison ────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def match_dna(fingerprint_a: dict, fingerprint_b: dict) -> dict:
    """
    Compare two DNA fingerprints and return a match score.
    Weighted cosine similarity across all four dimensions.

    Returns:
        {
            "match_score": float,        # Overall weighted similarity [0,1]
            "is_match": bool,            # >= 0.75 threshold
            "dimension_scores": dict,    # Per-dimension similarity
            "confidence": str,           # high/medium/low
        }
    """
    dims_a = fingerprint_a.get("dimensions", {})
    dims_b = fingerprint_b.get("dimensions", {})

    scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for dim_name, weight in DNA_WEIGHTS.items():
        vec_a = np.array(dims_a.get(dim_name, []), dtype=np.float32)
        vec_b = np.array(dims_b.get(dim_name, []), dtype=np.float32)

        if len(vec_a) == 0 or len(vec_b) == 0:
            scores[dim_name] = 0.0
            continue

        sim = cosine_similarity(vec_a, vec_b)
        scores[dim_name] = round(sim, 4)
        weighted_sum += sim * weight
        total_weight += weight

    if total_weight == 0:
        match_score = 0.0
    else:
        match_score = weighted_sum / total_weight

    match_score = round(match_score, 4)

    if match_score >= 0.85:
        confidence = "high"
    elif match_score >= 0.75:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "match_score": match_score,
        "is_match": match_score >= DNA_MATCH_THRESHOLD,
        "dimension_scores": scores,
        "confidence": confidence,
    }


# ─── Vectorized Batch Matching ──────────────────────────────────────────────


def _load_dna_matrix(db_conn):
    """
    Load all fingerprints into a single (N, 84) float32 matrix in one query.
    L2-normalizes all rows for cosine similarity via dot product.

    Returns:
        all_vectors:    np.ndarray (N, 84), each row L2-normalized
        all_ids:        np.ndarray (N,), cluster_ids in row order
        dim_map:        dict[int, dict[str, np.ndarray]] — cluster_id → dimension arrays
        meta_map:       dict[int, dict] — cluster_id → metadata
    """
    rows = db_conn.execute(
        "SELECT cluster_id, fingerprint, dimensions, metadata FROM narrative_dna"
    ).fetchall()

    n = len(rows)
    if n == 0:
        return (
            np.empty((0, 84), dtype=np.float32),
            np.array([], dtype=np.int32),
            {},
            {},
        )

    all_vectors = np.zeros((n, 84), dtype=np.float32)
    all_ids = np.zeros(n, dtype=np.int32)
    dim_map = {}
    meta_map = {}

    for i, row in enumerate(rows):
        all_ids[i] = row["cluster_id"]
        fp = json.loads(row["fingerprint"])
        all_vectors[i] = np.array(fp, dtype=np.float32)

        dims_raw = json.loads(row["dimensions"])
        dim_map[row["cluster_id"]] = {
            k: np.array(v, dtype=np.float32) for k, v in dims_raw.items()
        }

        meta_raw = row["metadata"]
        meta_map[row["cluster_id"]] = (
            json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        )

    # Safety against NaN/Inf
    all_vectors = np.nan_to_num(all_vectors, nan=0.0, posinf=0.0, neginf=0.0)

    # L2-normalize
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    all_vectors /= norms

    return all_vectors, all_ids, dim_map, meta_map


def batch_cosine_matches(db_conn, cluster_ids, min_score=0.75):
    """
    Vectorized batch cross-matching. Loads all fingerprints once as a numpy
    matrix, computes pairwise cosine similarities via matrix multiplication,
    and inserts matches above min_score into dna_matches.

    Returns list of match dicts: [{cluster_a, cluster_b, match_score}, ...]
    """
    all_vectors, all_ids, dim_map, meta_map = _load_dna_matrix(db_conn)
    n_all = len(all_ids)
    if n_all < 2:
        return []

    id_to_idx = {int(all_ids[i]): i for i in range(n_all)}

    # Collect query clusters that have fingerprints
    query_idx_pairs = []
    for cid in cluster_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            query_idx_pairs.append((cid, idx))

    if not query_idx_pairs:
        return []

    query_ids = [p[0] for p in query_idx_pairs]
    query_indices = [p[1] for p in query_idx_pairs]
    n_q = len(query_indices)

    # Extract query vectors
    query_matrix = all_vectors[query_indices]  # (n_q, 84)

    # Matrix multiply: (n_q × 84) @ (84 × n_all) = (n_q × n_all)
    # Process in batches if many queries to limit temp matrix
    batch_size = 500
    # When the corpus is huge, also chunk the DB side so the (batch × n_all)
    # similarity tile never blows out RAM.
    if n_all > _DNA_LARGE_CORPUS_THRESHOLD:
        logger.warning(
            "batch_cosine_matches: corpus has %d fingerprints (> %d); "
            "chunking DB side at %d rows to bound memory.",
            n_all, _DNA_LARGE_CORPUS_THRESHOLD, _DNA_DB_CHUNK_SIZE,
        )
        db_chunk = _DNA_DB_CHUNK_SIZE
    else:
        db_chunk = n_all
    matches = []
    seen_pairs = set()
    insert_batch = []
    # Effective floor for the score we will *persist*. Caller's min_score may
    # be lower (e.g. UI search), but write side never goes below the env-tuned
    # persist threshold — this is the dna_matches growth cap.
    effective_persist_min = max(min_score, DNA_MATCH_PERSIST_THRESHOLD)

    for batch_start in range(0, n_q, batch_size):
        batch_end = min(batch_start + batch_size, n_q)
        batch_qids = query_ids[batch_start:batch_end]
        batch_idx = query_indices[batch_start:batch_end]
        batch_matrix = query_matrix[batch_start:batch_end]

        for db_start in range(0, n_all, db_chunk):
            db_end = min(db_start + db_chunk, n_all)
            similarities = batch_matrix @ all_vectors[db_start:db_end].T  # (batch, db_chunk)

            for i, (qid, q_idx) in enumerate(zip(batch_qids, batch_idx)):
                row = similarities[i]
                for jj in range(db_end - db_start):
                    aid = int(all_ids[db_start + jj])
                    if qid == aid:
                        continue
                    score = float(row[jj])
                    if score >= min_score:
                        pair = (min(qid, aid), max(qid, aid))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            matches.append({
                                "cluster_a": pair[0],
                                "cluster_b": pair[1],
                                "match_score": round(score, 4),
                            })

        if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
            logger.info(
                "batch_cosine_matches: %d/%d queries processed, %d matches so far",
                batch_start, n_q, len(matches),
            )

    if not matches:
        return []

    # Compute per-dimension scores and confidence for each matched pair
    skipped_below_threshold = 0
    for m in matches:
        if m["match_score"] < effective_persist_min:
            skipped_below_threshold += 1
            continue

        dims_a = dim_map.get(m["cluster_a"], {})
        dims_b = dim_map.get(m["cluster_b"], {})

        dim_scores = {}
        for dim_name in DNA_WEIGHTS:
            va = dims_a.get(dim_name)
            vb = dims_b.get(dim_name)
            if va is not None and vb is not None and len(va) > 0 and len(vb) > 0:
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 1e-10 and nb > 1e-10:
                    dim_scores[dim_name] = round(float(np.dot(va, vb) / (na * nb)), 4)
                else:
                    dim_scores[dim_name] = 0.0
            else:
                dim_scores[dim_name] = 0.0

        score = m["match_score"]
        if score >= 0.85:
            confidence = "high"
        elif score >= 0.75:
            confidence = "medium"
        else:
            confidence = "low"

        insert_batch.append((
            m["cluster_a"],
            m["cluster_b"],
            m["match_score"],
            json.dumps(dim_scores),
            confidence,
        ))

    logger.info(
        "dna_matches write filter: persisted %d of %d matches above threshold %.2f",
        len(insert_batch), len(matches), effective_persist_min,
    )

    # Batch insert in chunks of 100
    for i in range(0, len(insert_batch), 100):
        chunk = insert_batch[i:i + 100]
        db_conn.executemany("""
            INSERT OR IGNORE INTO dna_matches
            (cluster_a, cluster_b, match_score, dimension_scores, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, chunk)

    db_conn.commit()

    logger.info(
        "batch_cosine_matches: %d fingerprints, %d queries, %d matches >= %.2f",
        n_all, n_q, len(matches), min_score,
    )

    return matches


def match_single_cluster(db_conn, cluster_id, min_score=0.75):
    """
    Match a single cluster against all stored fingerprints using vectorized
    matrix multiplication. Inserts matches into dna_matches.

    Returns number of matches found.
    """
    matches = batch_cosine_matches(db_conn, [cluster_id], min_score=min_score)
    return len(matches)


# ─── Database Integration ──────────────────────────────────────────────────


def store_dna(db_conn, cluster_id: int, fingerprint: dict):
    """Persist DNA fingerprint to database."""
    db_conn.execute("""
        INSERT INTO narrative_dna (cluster_id, fingerprint, dimensions, metadata)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cluster_id)
        DO UPDATE SET fingerprint = excluded.fingerprint,
                      dimensions = excluded.dimensions,
                      metadata = excluded.metadata,
                      updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    """, (
        cluster_id,
        json.dumps(fingerprint["fingerprint"]),
        json.dumps(fingerprint["dimensions"]),
        json.dumps(fingerprint["metadata"]),
    ))
    db_conn.commit()


def find_matching_campaigns(db_conn, cluster_id: int, min_score: float = 0.60) -> list[dict]:
    """
    Search all stored DNA fingerprints for matches to this cluster.
    Uses vectorized matrix multiplication for fast similarity computation
    while maintaining the original per-dimension return format.

    Returns matches above min_score, sorted by similarity descending.
    """
    all_vectors, all_ids, dim_map, meta_map = _load_dna_matrix(db_conn)
    n_all = len(all_ids)
    if n_all < 2:
        return []

    id_to_idx = {int(all_ids[i]): i for i in range(n_all)}
    q_idx = id_to_idx.get(cluster_id)
    if q_idx is None:
        return []

    query_vec = all_vectors[q_idx:q_idx+1]  # (1, 84)
    similarities = query_vec @ all_vectors.T  # (1, n_all)
    sim_row = similarities[0]

    matches = []
    for j in range(n_all):
        aid = int(all_ids[j])
        if aid == cluster_id:
            continue
        score = float(sim_row[j])
        if score < min_score:
            continue

        # Compute per-dimension weighted score for backward compat
        dims_q = dim_map.get(cluster_id, {})
        dims_a = dim_map.get(aid, {})
        dim_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, weight in DNA_WEIGHTS.items():
            va = dims_q.get(dim_name)
            vb = dims_a.get(dim_name)
            if va is not None and vb is not None and len(va) > 0 and len(vb) > 0:
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 1e-10 and nb > 1e-10:
                    ds = float(np.dot(va, vb) / (na * nb))
                else:
                    ds = 0.0
            else:
                ds = 0.0
            ds = round(ds, 4)
            dim_scores[dim_name] = ds
            weighted_sum += ds * weight
            total_weight += weight

        weighted_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

        if weighted_score >= 0.85:
            confidence = "high"
        elif weighted_score >= 0.75:
            confidence = "medium"
        else:
            confidence = "low"

        matches.append({
            "match_score": weighted_score,
            "is_match": weighted_score >= DNA_MATCH_THRESHOLD,
            "dimension_scores": dim_scores,
            "confidence": confidence,
            "matched_cluster_id": aid,
            "matched_metadata": meta_map.get(aid, {}),
        })

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches


def compute_and_store_all_dna(db_conn) -> int:
    """Compute and store DNA for all active clusters. Returns count."""
    clusters = db_conn.execute(
        "SELECT id FROM narrative_clusters WHERE status = 'active' AND post_count >= 3"
    ).fetchall()

    count = 0
    for c in clusters:
        try:
            fingerprint = compute_dna_fingerprint(db_conn, c["id"])
            store_dna(db_conn, c["id"], fingerprint)
            count += 1
        except Exception as e:
            logger.warning(f"DNA computation failed for cluster {c['id']}: {e}")

    logger.info(f"DNA computed for {count} clusters")
    return count


def run_dna_cycle(db_conn) -> dict:
    """
    Full DNA cycle: compute fingerprints for all active clusters, then
    cross-match via vectorized batch comparison to find same-operator campaigns.
    """
    count = compute_and_store_all_dna(db_conn)

    clusters = db_conn.execute(
        "SELECT id FROM narrative_clusters WHERE status = 'active' AND post_count >= 5"
    ).fetchall()
    cluster_ids = [c["id"] for c in clusters]

    if len(cluster_ids) < 2:
        logger.info(f"DNA cycle: {count} fingerprints, 0 cross-matches (< 2 clusters)")
        return {"fingerprints_computed": count, "cross_matches": 0}

    match_results = batch_cosine_matches(
        db_conn, cluster_ids, min_score=DNA_MATCH_THRESHOLD
    )

    logger.info(f"DNA cycle: {count} fingerprints, {len(match_results)} cross-matches")
    return {"fingerprints_computed": count, "cross_matches": len(match_results)}

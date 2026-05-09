"""
BurnTheLies Intelligence — Narrative Clustering
UMAP dimensionality reduction → HDBSCAN density clustering → c-TF-IDF labeling.
Discovers narrative clusters from raw post embeddings.
"""

import json
import logging
import re
import numpy as np
from collections import Counter
from math import log2
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from intelligence.processing.interpret import clean_cluster_label

logger = logging.getLogger("intelligence.cluster")


# ─── Label-generation filters (single-source-of-truth) ────────────────────

LABEL_STOPWORDS = {
    "the", "a", "an", "of", "for", "they", "them", "their",
    "is", "are", "was", "were", "be", "been",
    "to", "in", "on", "at", "by", "with", "from", "as",
    "it", "its", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "also",
    "more", "most", "less", "such", "into", "than",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "there", "here", "up", "down", "out", "over", "under",
    "again", "further", "once",
    "e", "i", "ii", "iii", "iv", "v",
}

# GDELT taxonomy-code prefixes that must never reach an editorial label.
# GDELT taxonomy-code prefixes that must never reach an editorial label.
#
# Group 1 — GKG-internal codes (rarely appear as natural English):
#   broad [a-z_0-9]* suffix is safe because wb/ungp/fncact etc. are not
#   valid substrings of English words.
# Group 2 — codes that ARE common English word fragments (tax, eth, env, …):
#   require underscore-compound form so we don't eat real words like
#   "taxation", "ethics", "environment", "education", "naturally".
GKG_CODE_RE = re.compile(
    r'^(wb|ungp|crisislex|fncact|epu|armedconflict|govt)[a-z_0-9]*$'
    r'|^(tax|eth|env|econ|edu|natural)(_[a-z_0-9]+)?$',
    re.IGNORECASE,
)

MOJIBAKE_CHARS = {"â", "Ã", "Â", "ð", "¢", "ã", "â", "ä", "å", "æ", "è", "é", "ê", "ë", "ì", "í", "î", "ï"}

# Em-dash separators GDELT appends as a metadata suffix on titles.
# Ordered: â (mojibake em-dash, most common in GKG), — (correct em-dash), - (plain dash).
GKG_TITLE_SUFFIX_SEPARATORS = (" â ", " — ", " - ")


def _normalize_title(title: str) -> str:
    """
    Strip GKG metadata suffix from a title.
    GDELT titles have the form '<real title> â <theme1> â <theme2> ...' — the
    FIRST separator marks where the real title ends and taxonomy codes begin.
    Everything from the first separator onward is stripped.
    """
    if not title:
        return ""
    s = title
    for sep in GKG_TITLE_SUFFIX_SEPARATORS:
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
            break  # one separator found and stripped; stop
    return s.strip()


def _filter_label_tokens(tokens):
    """Apply the label-token filter pipeline. Lowercases everything."""
    out = []
    for raw in tokens:
        t = raw.lower()
        if len(t) < 3:
            continue
        if t in LABEL_STOPWORDS:
            continue
        if any(ch in MOJIBAKE_CHARS for ch in t):
            continue
        if GKG_CODE_RE.match(t):
            continue
        # Fallback: GKG codes in themes often survive as UPPERCASE; recheck.
        if GKG_CODE_RE.match(raw.upper()):
            continue
        out.append(t)
    return out


def generate_label(titles, cluster_id: int = 0) -> str:
    """
    Build an editorial cluster label from member post titles.
    Strips GKG suffix, filters stop-words / taxonomy codes / mojibake,
    requires >=2 distinct lexical tokens, falls back to 'Untitled Cluster #N'.
    """
    counts = Counter()
    for title in titles or []:
        normalized = _normalize_title(title)
        # Tokenise on Latin letters incl. Latin-1 supplement so mojibake survives
        # to the filter step (where filter-e drops it).
        raw_tokens = re.findall(r"[A-Za-zÀ-ÿ]+", normalized)
        for t in _filter_label_tokens(raw_tokens):
            counts[t] += 1

    top = [t for t, _ in counts.most_common(8)]
    if len(set(top)) < 2:
        return f"Untitled Cluster #{cluster_id}"
    return " ".join(t.title() for t in top)


def _single_source_topic_bag(sources, titles) -> bool:
    """
    Detect HDBSCAN bagging single-domain unrelated posts under shared GKG themes.
    Flag if dominant_domain_fraction > 0.85 AND post_count >= 8 AND
    mean pairwise title-token Jaccard < 0.15.
    """
    if not sources or len(sources) < 8:
        return False
    counts = Counter(sources)
    total = sum(counts.values())
    if total < 8:
        return False
    if (max(counts.values()) / total) <= 0.85:
        return False

    token_sets = []
    for title in titles:
        norm = _normalize_title(title)
        toks = {t.lower() for t in re.findall(r"\w+", norm) if len(t) >= 2}
        if toks:
            token_sets.append(toks)
    if len(token_sets) < 2:
        return False

    jaccards = []
    for i in range(len(token_sets)):
        a = token_sets[i]
        for j in range(i + 1, len(token_sets)):
            b = token_sets[j]
            union = a | b
            if not union:
                continue
            jaccards.append(len(a & b) / len(union))
    if not jaccards:
        return False
    return (sum(jaccards) / len(jaccards)) < 0.15


def _multi_domain_topic_bag(sources, titles) -> bool:
    """
    Detect HDBSCAN bagging multi-domain unrelated posts under shared GKG themes.

    Complement to _single_source_topic_bag. Where that catches single-domain
    editorial coverage (cluster 39038 class), this catches cross-domain topic
    bags — 100 Japanese entertainment articles from 50 different .jp sites,
    or 80 Romanian politics articles from 30 Romanian outlets, all about
    different events but sharing language + broad GKG themes.

    Triggers when:
      - dominant_domain_fraction <= 0.15 (wide domain spread rules out single_source)
      - post_count >= 15 (small clusters are handled by insufficient_evidence gate)
      - mean pairwise title-token Jaccard < 0.12 (near-zero lexical overlap
        proves these are different stories, not the same story told differently)
    """
    if not sources or len(sources) < 15:
        return False
    counts = Counter(sources)
    total = sum(counts.values())
    dominant_frac = max(counts.values()) / total
    if dominant_frac > 0.15:  # a single domain dominates — handled by single_source
        return False

    token_sets = []
    for title in titles:
        norm = _normalize_title(title)
        # Unicode \w+ catches all scripts (Hindi, Arabic, CJK, Cyrillic, Latin).
        # Latin-only [A-Za-zÀ-ÿ]+ silently produced empty sets for non-Latin
        # titles, making the Jaccard test unreachable for entire language families.
        toks = {t.lower() for t in re.findall(r"\w+", norm) if len(t) >= 2}
        if toks:
            token_sets.append(toks)
    if len(token_sets) < 2:
        return False

    jaccards = []
    for i in range(len(token_sets)):
        a = token_sets[i]
        for j in range(i + 1, len(token_sets)):
            b = token_sets[j]
            union = a | b
            if not union:
                continue
            jaccards.append(len(a & b) / len(union))
    if not jaccards:
        return False
    mean_jaccard = sum(jaccards) / len(jaccards)
    return mean_jaccard < 0.12


def cluster_narratives(db_conn, min_cluster_size: int = 5,
                       min_samples: int = 3) -> list[dict]:
    """
    Run full clustering pipeline on recent embeddings.
    Returns list of discovered cluster dicts.
    """
    from intelligence.processing.embed import get_all_embeddings
    from intelligence.db import upsert_cluster

    post_ids, vectors = get_all_embeddings(db_conn)

    if len(post_ids) < min_cluster_size * 2:
        logger.info(f"Not enough posts for clustering ({len(post_ids)})")
        return []

    logger.info(f"Clustering {len(post_ids)} posts...")

    # Step 1: UMAP dimensionality reduction (384 → 5)
    import umap
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=min(15, len(post_ids) - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)

    # Step 2: HDBSCAN density clustering
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

    # Archive all previous clusters and membership — full rebuild each run
    db_conn.execute("UPDATE narrative_clusters SET status='archived' WHERE status='active'")
    db_conn.execute("DELETE FROM cluster_members")
    # Purge DNA data for archived clusters so counts and matches stay current.
    # Without this, dna_matches accumulates unboundedly across cluster rebuilds.
    db_conn.execute("""
        DELETE FROM narrative_dna
        WHERE cluster_id IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.execute("""
        DELETE FROM dna_matches
        WHERE cluster_a IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
           OR cluster_b IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.commit()
    logger.info("Archived previous clusters. Rebuilt DNA tables. Rebuilding from scratch.")

    # Step 3: Build cluster metadata
    # Get post data for labeling
    post_data = {}
    for pid in post_ids:
        row = db_conn.execute(
            "SELECT source, url, title, content, language, published_at, metadata FROM raw_posts WHERE id = ?",
            (pid,)
        ).fetchone()
        if row:
            post_data[pid] = dict(row)

    clusters = []
    for cluster_id_raw in set(labels):
        if cluster_id_raw == -1:
            continue

        mask = labels == cluster_id_raw
        member_ids = [post_ids[i] for i in range(len(post_ids)) if mask[i]]
        member_probs = [float(probabilities[i]) for i in range(len(post_ids)) if mask[i]]

        # Extract metadata
        sources = []
        languages = []
        texts = []
        titles = []
        all_themes = []
        all_persons = []
        all_orgs = []
        for pid in member_ids:
            pd = post_data.get(pid, {})
            source_raw = pd.get("source", "unknown")
            url = pd.get("url", "") or ""
            sources.append(_extract_actual_source(source_raw, url))
            languages.append(pd.get("language") or "unknown")
            title_str = pd.get("title", "") or ""
            titles.append(title_str)
            texts.append(f"{title_str} {pd.get('content', '')}")

            # Parse metadata for structured signals
            meta_raw = pd.get("metadata", "{}")
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            else:
                meta = meta_raw or {}

            if "themes" in meta and isinstance(meta["themes"], list):
                all_themes.extend(meta["themes"][:10])
            if "persons" in meta and isinstance(meta["persons"], str):
                all_persons.extend([p.strip() for p in meta["persons"].split(";") if p.strip()])
            if "organizations" in meta and isinstance(meta["organizations"], str):
                all_orgs.extend([o.strip() for o in meta["organizations"].split(";") if o.strip()])

        # Source diversity (Shannon entropy across actual publication domains)
        source_counts = Counter(sources)
        total = sum(source_counts.values())
        source_diversity = -sum(
            (c / total) * log2(c / total) for c in source_counts.values() if c > 0
        )

        # Language spread
        lang_spread = dict(Counter(languages))

        # Generate label: prefer real post titles (Strategy 1 in clean_cluster_label)
        # over GDELT entity extraction. A real headline communicates more than
        # "person1, person2 — org — Theme" entity soup.
        keywords = _extract_keywords(texts, top_n=8)
        label = clean_cluster_label(
            raw_label="", keywords=keywords, post_titles=titles, themes=all_themes
        )
        if label.startswith("Developing Narrative"):
            metadata_label = _generate_metadata_label(all_themes, all_persons, all_orgs)
            if metadata_label:
                label = metadata_label
            else:
                label = generate_label(titles, cluster_id=int(cluster_id_raw))

        # Store in database
        db_cluster_id = upsert_cluster(
            db_conn,
            label=label,
            keywords=keywords,
            post_count=len(member_ids),
            source_diversity=source_diversity,
            language_spread=lang_spread,
        )

        # Guard: deduplicate labels across clusters. If another cluster already
        # owns this label (e.g. same GKG themes on different articles), append
        # the cluster id to disambiguate.
        dup_count = db_conn.execute(
            "SELECT COUNT(*) FROM narrative_clusters WHERE label = ? AND id != ?",
            (label, db_cluster_id),
        ).fetchone()[0]
        if dup_count > 0:
            label = f"{label} #{db_cluster_id}"
            db_conn.execute(
                "UPDATE narrative_clusters SET label = ? WHERE id = ?",
                (label, db_cluster_id),
            )
            db_conn.commit()
            logger.info("Deduplicated label for cluster %s → %s", db_cluster_id, label)

        # Flag single-domain topic-bag clusters for downstream gate suppression.
        single_source = _single_source_topic_bag(sources, titles)
        if single_source:
            try:
                db_conn.execute("""
                    UPDATE narrative_clusters
                    SET metadata = json_set(
                        COALESCE(metadata, '{}'),
                        '$.single_source_topic_bag', json('true')
                    )
                    WHERE id = ?
                """, (db_cluster_id,))
                logger.warning(
                    f"Cluster {db_cluster_id} flagged single_source_topic_bag "
                    f"(posts={len(member_ids)}, dominant_domain={Counter(sources).most_common(1)[0]})"
                )
            except Exception:
                logger.exception(f"Failed to flag single_source_topic_bag on cluster {db_cluster_id}")

        # Flag multi-domain topic-bag clusters (complement to single_source).
        multi_topic = _multi_domain_topic_bag(sources, titles)
        if multi_topic:
            try:
                db_conn.execute("""
                    UPDATE narrative_clusters
                    SET metadata = json_set(
                        COALESCE(metadata, '{}'),
                        '$.topic_bag', json('true')
                    )
                    WHERE id = ?
                """, (db_cluster_id,))
                logger.warning(
                    f"Cluster {db_cluster_id} flagged multi_domain_topic_bag "
                    f"(posts={len(member_ids)}, domains={len(Counter(sources))}, "
                    f"dominant_frac={max(Counter(sources).values())/sum(Counter(sources).values()):.2f})"
                )
            except Exception:
                logger.exception(f"Failed to flag multi_domain_topic_bag on cluster {db_cluster_id}")

        # Store cluster membership
        for pid, prob in zip(member_ids, member_probs):
            try:
                db_conn.execute("""
                    INSERT OR REPLACE INTO cluster_members (post_id, cluster_id, confidence)
                    VALUES (?, ?, ?)
                """, (pid, db_cluster_id, prob))
            except Exception as e:
                logger.warning(
                    f"Skipping cluster_members insert post={pid} cluster={db_cluster_id}: {e}",
                    exc_info=False,
                )
        db_conn.commit()

        clusters.append({
            "id": db_cluster_id,
            "label": label,
            "keywords": keywords,
            "post_count": len(member_ids),
            "source_diversity": source_diversity,
            "language_spread": lang_spread,
            "single_source_topic_bag": single_source,
        })

    logger.info(f"Stored {len(clusters)} narrative clusters")
    return clusters


def _extract_actual_source(source: str, url: str) -> str:
    """Extract the actual publication domain from a URL for GDELT posts."""
    if source in ("gdelt_gkg", "gdelt_doc") and url:
        try:
            netloc = urlparse(url).netloc
            domain = re.sub(r'^www\.', '', netloc.lower()) if netloc else ""
            if domain and len(domain) > 3:
                return domain
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"urlparse failed for url={url!r}: {e}")
    return source


def _extract_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    """
    Simple keyword extraction via term frequency.
    (Full c-TF-IDF comes with BERTopic integration later)
    """
    from collections import Counter
    import re

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "and", "but", "or",
        "if", "while", "that", "this", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "these", "those",
        "http", "https", "www", "com", "org", "themes", "via",
        # URL fragment noise from GDELT content (URL + themes format)
        "html", "net", "del", "php", "asp", "jsp", "htm",
        "news", "article", "articles", "releases", "release",
        "index", "page", "post", "posts", "tag", "tags",
        "category", "section", "topic", "latest", "view",
        "read", "more", "click", "here", "link", "full",
        # Common domain extensions/fragments
        "co", "uk", "au", "ca", "gov", "edu", "io",
        # Spanish/French/German URL words that appear in GDELT
        "del", "los", "las", "una", "por", "con", "sur", "les",
        "der", "die", "das", "von", "und", "fur", "des",
        # GDELT internal terms
        "gdelt", "gkg", "source", "record",
    }

    word_counts = Counter()
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_counts.update(w for w in words if w not in stopwords)

    return [word for word, _ in word_counts.most_common(top_n)]


def _generate_metadata_label(themes: list, persons: list, orgs: list) -> str:
    """Generate a meaningful label from GKG metadata (themes, persons, orgs)."""
    # Clean themes — remove GDELT internal prefixes
    skip_prefixes = {
        "TAX_", "WB_", "EPU_", "CRISISLEX_", "GENERAL_", "USPEC_",
        "ARMEDCONFLICT", "HEALTH_", "MEDIA_", "LEADERSHIP", "LEGISLATIVE",
        "GOVT_", "MILITARY_", "DIPLOMACY_", "REFUGEE_", "HUMANRIGHTS_",
        "ECON_", "ENV_", "ETH_", "EDU_", "NATURAL_", "UNGP",
    }
    skip_upper = {p.upper() for p in skip_prefixes}
    clean_themes = []
    for t in themes:
        tu = t.upper()
        if any(tu.startswith(p) for p in skip_upper):
            continue
        clean = t.replace("_", " ").strip().title()
        if len(clean) > 3:
            clean_themes.append(clean)

    top_themes = [t for t, _ in Counter(clean_themes).most_common(3)]

    # Filter out GDELT location format strings (contain # with location codes)
    # These appear when old data has V2EnhancedLocations stored as persons
    clean_persons = [
        p for p in persons
        if p and "#" not in p and len(p) > 2 and len(p) < 60
    ]
    clean_orgs = [
        o for o in orgs
        if o and "#" not in o and len(o) > 2 and len(o) < 60
    ]

    top_persons = [p for p, _ in Counter(clean_persons).most_common(2)]
    top_orgs = [o for o, _ in Counter(clean_orgs).most_common(2)]

    parts = []
    if top_persons:
        parts.append(", ".join(top_persons[:2]))
    if top_orgs and len(parts) < 2:
        parts.append(", ".join(top_orgs[:1]))
    if top_themes:
        parts.append(top_themes[0])

    if parts:
        return " — ".join(parts[:3])
    return ""


# ─── Multi-Resolution Clustering ───────────────────────────────────────────

MULTI_RESOLUTION_SIZES = [3, 5, 10, 25]


def cluster_narratives_multi_resolution(db_conn) -> list[dict]:
    """
    Run HDBSCAN at 4 resolution levels (min_cluster_size = 3, 5, 10, 25).
    Cross-validate: clusters appearing at multiple resolutions get higher confidence.
    Catches both small stealth campaigns (3-4 posts) and major operations (25+).

    Resolution semantic:
      3  — "whisper": smallest detectable signal, higher noise
      5  — "signal": current default, balanced
      10 — "narrative": medium confidence, likely real narratives
      25 — "campaign": large-scale, high-confidence operations only
    """
    from intelligence.processing.embed import get_all_embeddings
    from intelligence.db import upsert_cluster

    post_ids, vectors = get_all_embeddings(db_conn)

    if len(post_ids) < MULTI_RESOLUTION_SIZES[0] * 2:
        logger.info(f"Not enough posts for multi-resolution clustering ({len(post_ids)})")
        return []

    logger.info(f"Multi-resolution clustering {len(post_ids)} posts at sizes {MULTI_RESOLUTION_SIZES}")

    # Step 1: UMAP once, reuse for all resolutions
    import umap
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=min(15, len(post_ids) - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)

    # Step 2: HDBSCAN at each resolution
    import hdbscan
    resolution_results = {}
    all_labels = {}

    for min_size in MULTI_RESOLUTION_SIZES:
        if len(post_ids) < min_size * 2:
            logger.info(f"Skipping resolution {min_size} — insufficient data")
            continue

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(1, min_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced)
        probabilities = clusterer.probabilities_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(f"Resolution {min_size}: {n_clusters} clusters, {n_noise} noise")

        resolution_results[min_size] = {
            "labels": labels,
            "probabilities": probabilities,
            "n_clusters": n_clusters,
        }
        all_labels[min_size] = labels

    # Archive previous clusters and rebuild
    db_conn.execute("UPDATE narrative_clusters SET status='archived' WHERE status='active'")
    db_conn.execute("DELETE FROM cluster_members")
    db_conn.execute("""
        DELETE FROM narrative_dna
        WHERE cluster_id IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.execute("""
        DELETE FROM dna_matches
        WHERE cluster_a IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
           OR cluster_b IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.execute("""
        DELETE FROM nvi_snapshots
        WHERE cluster_id IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.execute("""
        DELETE FROM coordination_signals
        WHERE cluster_id IN (SELECT id FROM narrative_clusters WHERE status = 'archived')
    """)
    db_conn.commit()
    # Signal api.py to flush its enrich cache (flag file approach — no shared memory needed)
    try:
        import time as _time
        with open("/tmp/intel_rebuild_ts", "w") as _f:
            _f.write(str(_time.time()))
    except OSError:
        pass
    logger.info("Archived previous clusters. Rebuilding multi-resolution.")

    # Step 3: Get post data for labeling
    post_data = {}
    for pid in post_ids:
        row = db_conn.execute(
            "SELECT source, url, title, content, language, published_at, metadata FROM raw_posts WHERE id = ?",
            (pid,)
        ).fetchone()
        if row:
            post_data[pid] = dict(row)

    # Step 4: Build clusters at each resolution
    all_clusters = []
    cluster_id_map = {}  # (resolution, internal_label) → db_cluster_id

    for min_size, result in resolution_results.items():
        labels = result["labels"]
        probabilities = result["probabilities"]

        for internal_label in set(labels):
            if internal_label == -1:
                continue

            mask = labels == internal_label
            member_ids = [post_ids[i] for i in range(len(post_ids)) if mask[i]]
            member_probs = [float(probabilities[i]) for i in range(len(post_ids)) if mask[i]]

            # Extract metadata and build label (same as single-resolution)
            sources = []
            languages = []
            texts = []
            titles = []
            all_themes = []
            all_persons = []
            all_orgs = []

            for pid in member_ids:
                pd = post_data.get(pid, {})
                source_raw = pd.get("source", "unknown")
                url = pd.get("url", "") or ""
                sources.append(_extract_actual_source(source_raw, url))
                languages.append(pd.get("language") or "unknown")
                title_str = pd.get("title", "") or ""
                titles.append(title_str)
                texts.append(f"{title_str} {pd.get('content', '')}")

                meta_raw = pd.get("metadata", "{}")
                if isinstance(meta_raw, str):
                    try:
                        meta = json.loads(meta_raw)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                else:
                    meta = meta_raw or {}

                if "themes" in meta and isinstance(meta["themes"], list):
                    all_themes.extend(meta["themes"][:10])
                if "persons" in meta and isinstance(meta["persons"], str):
                    all_persons.extend([p.strip() for p in meta["persons"].split(";") if p.strip()])
                if "organizations" in meta and isinstance(meta["organizations"], str):
                    all_orgs.extend([o.strip() for o in meta["organizations"].split(";") if o.strip()])

            # Source diversity (now uses actual domains, not 'gdelt_gkg')
            source_counts = Counter(sources)
            total = sum(source_counts.values())
            source_diversity = -sum(
                (c / total) * log2(c / total) for c in source_counts.values() if c > 0
            ) if total > 0 else 0

            lang_spread = dict(Counter(languages))

            # Label: prefer real post titles over GDELT entity extraction.
            keywords = _extract_keywords(texts, top_n=8)
            label = clean_cluster_label(
                raw_label="", keywords=keywords, post_titles=titles, themes=all_themes
            )
            if label.startswith("Developing Narrative"):
                metadata_label = _generate_metadata_label(all_themes, all_persons, all_orgs)
                if metadata_label:
                    label = metadata_label
                else:
                    label = generate_label(titles, cluster_id=int(internal_label))

            # Store with resolution metadata
            db_cluster_id = upsert_cluster(
                db_conn,
                label=label,
                keywords=keywords,
                post_count=len(member_ids),
                source_diversity=source_diversity,
                language_spread=lang_spread,
            )

            # Guard: deduplicate labels across clusters.
            dup_count = db_conn.execute(
                "SELECT COUNT(*) FROM narrative_clusters WHERE label = ? AND id != ?",
                (label, db_cluster_id),
            ).fetchone()[0]
            if dup_count > 0:
                label = f"{label} #{db_cluster_id}"
                db_conn.execute(
                    "UPDATE narrative_clusters SET label = ? WHERE id = ?",
                    (label, db_cluster_id),
                )
                db_conn.commit()
                logger.info(
                    "Deduplicated label for cluster %s (resolution=%s) → %s",
                    db_cluster_id, min_size, label,
                )

            # Flag single-domain topic-bag clusters for downstream gate suppression.
            single_source = _single_source_topic_bag(sources, titles)
            if single_source:
                try:
                    db_conn.execute("""
                        UPDATE narrative_clusters
                        SET metadata = json_set(
                            COALESCE(metadata, '{}'),
                            '$.single_source_topic_bag', json('true')
                        )
                        WHERE id = ?
                    """, (db_cluster_id,))
                    logger.warning(
                        f"Cluster {db_cluster_id} (resolution={min_size}) flagged "
                        f"single_source_topic_bag — posts={len(member_ids)}, "
                        f"dominant_domain={Counter(sources).most_common(1)[0]}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed to flag single_source_topic_bag on cluster {db_cluster_id}"
                    )

            # Flag multi-domain topic-bag clusters (complement to single_source).
            multi_topic = _multi_domain_topic_bag(sources, titles)
            if multi_topic:
                try:
                    db_conn.execute("""
                        UPDATE narrative_clusters
                        SET metadata = json_set(
                            COALESCE(metadata, '{}'),
                            '$.topic_bag', json('true')
                        )
                        WHERE id = ?
                    """, (db_cluster_id,))
                    logger.warning(
                        f"Cluster {db_cluster_id} (resolution={min_size}) flagged "
                        f"multi_domain_topic_bag — posts={len(member_ids)}, "
                        f"domains={len(Counter(sources))}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed to flag multi_domain_topic_bag on cluster {db_cluster_id}"
                    )

            # Store membership
            for pid, prob in zip(member_ids, member_probs):
                try:
                    db_conn.execute("""
                        INSERT OR REPLACE INTO cluster_members (post_id, cluster_id, confidence)
                        VALUES (?, ?, ?)
                    """, (pid, db_cluster_id, prob))
                except Exception as e:
                    logger.warning(
                        f"Skipping cluster_members insert post={pid} "
                        f"cluster={db_cluster_id} resolution={min_size}: {e}",
                        exc_info=False,
                    )

            # Update cluster metadata with resolution info
            db_conn.execute("""
                UPDATE narrative_clusters
                SET metadata = json_set(
                    COALESCE(metadata, '{}'),
                    '$.resolution', ?,
                    '$.resolution_name', ?
                )
                WHERE id = ?
            """, (min_size, _resolution_name(min_size), db_cluster_id))

            db_conn.commit()

            cluster_id_map[(min_size, internal_label)] = db_cluster_id
            all_clusters.append({
                "id": db_cluster_id,
                "resolution": min_size,
                "resolution_name": _resolution_name(min_size),
                "label": label,
                "keywords": keywords,
                "post_count": len(member_ids),
                "source_diversity": source_diversity,
                "language_spread": lang_spread,
                "single_source_topic_bag": single_source,
            })

    # Step 5: Cross-validate — compute resolution confidence
    all_clusters = _stitch_resolution_clusters(db_conn, all_clusters, cluster_id_map,
                                                resolution_results, post_ids)

    logger.info(f"Multi-resolution: {len(all_clusters)} total clusters across {len(MULTI_RESOLUTION_SIZES)} levels")
    return all_clusters


def _resolution_name(min_size: int) -> str:
    """Human-readable resolution name."""
    return {3: "whisper", 5: "signal", 10: "narrative", 25: "campaign"}.get(min_size, f"r{min_size}")


def _stitch_resolution_clusters(db_conn, all_clusters: list[dict],
                                 cluster_id_map: dict, resolution_results: dict,
                                 post_ids: list[str]) -> list[dict]:
    """
    Cross-validate clusters across resolutions by member overlap.
    A cluster appearing at resolution 3 that substantially overlaps with one at
    resolution 5 gets a confidence boost. Isolated single-resolution clusters
    get a penalty.

    Updates each cluster's metadata with resolution_confidence [0, 1].
    """
    if len(MULTI_RESOLUTION_SIZES) < 2:
        return all_clusters

    # Build post→cluster mappings per resolution
    post_to_cluster = {}  # (resolution, post_id) → cluster_db_id
    for (res, internal_label), db_id in cluster_id_map.items():
        if res in resolution_results:
            labels = resolution_results[res]["labels"]
            for i, label in enumerate(labels):
                if label == internal_label:
                    post_to_cluster[(res, post_ids[i])] = db_id

    # For each resolution pair (lower→higher), compute overlap
    resolution_pairs = []
    sizes_sorted = sorted(MULTI_RESOLUTION_SIZES)
    for i in range(len(sizes_sorted)):
        for j in range(i + 1, len(sizes_sorted)):
            resolution_pairs.append((sizes_sorted[i], sizes_sorted[j]))

    cluster_overlaps = {}  # (db_id_low, db_id_high) → overlap_ratio

    for low_res, high_res in resolution_pairs:
        # Get clusters at each resolution
        low_clusters = set()
        high_clusters = set()
        cluster_members = {}  # db_id → set of post_ids

        for (res, internal_label), db_id in cluster_id_map.items():
            if res == low_res:
                low_clusters.add(db_id)
            elif res == high_res:
                high_clusters.add(db_id)

        for (res, pid), db_id in post_to_cluster.items():
            if res == low_res or res == high_res:
                if db_id not in cluster_members:
                    cluster_members[db_id] = set()
                cluster_members[db_id].add(pid)

        # Compute overlaps
        for low_id in low_clusters:
            for high_id in high_clusters:
                low_members = cluster_members.get(low_id, set())
                high_members = cluster_members.get(high_id, set())
                if not low_members or not high_members:
                    continue
                intersection = low_members & high_members
                # Jaccard: intersection / union
                union = low_members | high_members
                jaccard = len(intersection) / len(union) if union else 0
                cluster_overlaps[(low_id, high_id)] = jaccard

    # Compute resolution confidence for each cluster
    for cluster in all_clusters:
        db_id = cluster["id"]
        res = cluster["resolution"]

        # Find all overlaps with this cluster
        overlaps = []
        for (low_id, high_id), jaccard in cluster_overlaps.items():
            if low_id == db_id or high_id == db_id:
                overlaps.append(jaccard)

        if len(overlaps) >= 2:
            # Confirmed across multiple resolutions → high confidence
            confidence = min(1.0, 0.6 + 0.2 * len(overlaps) + 0.1 * max(overlaps))
        elif len(overlaps) == 1:
            # Confirmed at one other resolution → medium confidence
            confidence = min(0.8, 0.4 + 0.3 * overlaps[0])
        else:
            # Single resolution only → low confidence, possible noise
            confidence = 0.25 if res <= 3 else 0.35

        cluster["resolution_confidence"] = round(confidence, 4)

        # Update DB metadata
        try:
            db_conn.execute("""
                UPDATE narrative_clusters
                SET metadata = json_set(
                    COALESCE(metadata, '{}'),
                    '$.resolution_confidence', ?,
                    '$.cross_resolution_overlaps', ?
                )
                WHERE id = ?
            """, (confidence, len(overlaps), db_id))
            db_conn.commit()
        except Exception:
            logger.exception(
                f"Failed to update resolution_confidence for cluster {db_id} "
                f"(resolution={res})"
            )

    return all_clusters


def incremental_assign(db_conn, new_post_ids: list[str],
                       threshold: float = 0.65) -> dict:
    """
    Assign new posts to existing clusters without full re-clustering.
    Uses centroid similarity.
    """
    from intelligence.processing.embed import cosine_similarity

    if not new_post_ids:
        return {"assigned": 0, "unassigned": 0}

    # Get existing cluster centroids
    clusters = db_conn.execute("""
        SELECT nc.id, nc.label FROM narrative_clusters nc
        WHERE nc.status = 'active'
    """).fetchall()

    if not clusters:
        return {"assigned": 0, "unassigned": len(new_post_ids)}

    # Compute centroids
    centroids = {}
    for c in clusters:
        members = db_conn.execute("""
            SELECT e.vector FROM embeddings e
            JOIN cluster_members cm ON cm.post_id = e.post_id
            WHERE cm.cluster_id = ?
        """, (c["id"],)).fetchall()

        if members:
            vecs = np.array([json.loads(m["vector"]) for m in members])
            centroids[c["id"]] = vecs.mean(axis=0)

    stats = {"assigned": 0, "unassigned": 0}

    for pid in new_post_ids:
        emb_row = db_conn.execute(
            "SELECT vector FROM embeddings WHERE post_id = ?", (pid,)
        ).fetchone()
        if not emb_row:
            stats["unassigned"] += 1
            continue

        vec = np.array(json.loads(emb_row["vector"]))
        best_cluster = None
        best_sim = 0

        for cid, centroid in centroids.items():
            sim = cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_cluster and best_sim >= threshold:
            db_conn.execute("""
                INSERT OR REPLACE INTO cluster_members (post_id, cluster_id, confidence)
                VALUES (?, ?, ?)
            """, (pid, best_cluster, best_sim))
            stats["assigned"] += 1
        else:
            stats["unassigned"] += 1

    db_conn.commit()
    return stats

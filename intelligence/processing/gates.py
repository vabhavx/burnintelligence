"""
BurnTheLies Intelligence — Falsification Gate Pipeline (v5.4)

18 gates organized by type. The pipeline implements a Popperian falsification
approach: each gate attempts to DISPROVE the narrative significance hypothesis
rather than prove coordination (which is epistemologically impossible from
open-source data alone). What survives all 18 falsification attempts is the
residue — clusters that COULD be coordinated and merit human review.

Gate priority order (specificity and destructiveness):

  Terminal (zeroes NVI immediately):
    1.  gdelt_batch_artifact          → NVI = 0 (15-min GDELT batch cadence, not real event)

  Suppression caps (structure-based):
    2.  content_noise                 suppress_only (listicles, obituaries, classifieds)
    3.  insufficient_evidence         cap 40/70 (<5 posts → 40, 5-9 → 70)
    4.  single_source_cluster         cap 15 (1 domain or Shannon entropy <0.20)
    5.  wire_service                  cap 20/25 (known syndicator or hash diversity)
    6.  dna_match                     cap 25 (no DNA fingerprint matches, <20 posts)

  Anomaly boosts (raise NVI floor):
    7.  cross_language                floor 65 (3+ real languages)
    8.  geographic_spread             floor 60 (locations in 3+ countries)
    9.  high_signal_topic             floor 55 (disinformation/propaganda themes)
   10.  circadian_anomaly             floor 50/65 (>40% posts during 1-5 AM UTC)
   11.  content_anomaly               floor 50 (high density + negative tone + low self-ref)
   12.  cross_cluster_velocity         floor 65/80 (DNA evidence across clusters)

  Quality caps (content-based):
   13.  ensemble_uncertainty          cap 35 (3 ensemble configs agree too perfectly)
   14.  entity_concentration          cap 35 (≤1 shared entity across ≥15 posts)
   15.  narrative_coherence           cap 20 (entity continuity + token Jaccard <0.22)
   16.  organic_viral_spread          cap 30 (high mutation + high diversity)
   17.  normal_news_cycle             cap 35 (low coordination + low burst + <15 posts)

  Suppression:
   18.  confidence_threshold          suppress_only (confidence <0.65)

Final NVI = max(min(raw_nvi, strictest_cap), highest_floor)
Boost gates can override cap gates. Real signal survives structural criticism.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── Lightweight English stopword list for token Jaccard ────────────────────
_STOP_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "will", "with", "but", "not", "they",
    "their", "them", "we", "our", "you", "your", "i", "my", "me", "she",
    "her", "his", "him", "us", "if", "so", "no", "yes", "do", "does", "did",
    "say", "says", "said", "what", "which", "who", "how", "when", "where",
    "why", "than", "then", "into", "over", "under", "after", "before",
    "about", "against", "between", "during", "without", "within", "above",
    "below", "up", "down", "out", "off", "again", "further", "all", "any",
    "more", "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "can", "could", "should", "would", "may", "might", "must",
    "just", "now", "new", "one", "two", "three", "four", "five",
    "news", "today", "yesterday", "tomorrow",
}

_TOKEN_RE = re.compile(r"[\w][\w'-]+", re.UNICODE)

# Entities so generic they appear across unrelated stories — drop from
# entity-continuity computation to prevent inflated coherence scores when
# clusters merely share a superpower or tech-megacap mention.
_GENERIC_ENTITIES = {
    "european union", "united states", "european commission", "government",
    "apple", "samsung", "google", "amazon", "microsoft", "facebook", "meta",
    "president", "minister", "prime minister", "mayor", "governor", "senator",
    "spokesperson", "official", "police", "court", "parliament", "congress",
    "reuters", "associated press", "bbc", "cnn", "white house", "kremlin",
    "china", "russia", "india", "japan", "france", "germany", "ukraine",
    "united nations", "nato", "world bank", "imf", "world health organization",
}


# ─── Data classes ───────────────────────────────────────────────────────────


@dataclass
class ClusterFeatures:
    """Inputs to the gate pipeline. Pure data, no I/O at gate evaluation time."""

    cluster_id: int
    post_count: int

    # Core signals (already in v4.1)
    burst: float
    spread: float
    mutation: float
    coordination: float
    tone_uniformity: float

    # Composite/derived (already in v4.1)
    entity_concentration: float
    shared_entity_count: int

    # NEW for v5: cluster-level features
    unique_hash_ratio: float
    embedding_similarity_mean: float
    inter_arrival_mean: float            # seconds; math.nan if undefined
    inter_arrival_std: float             # seconds; math.nan if undefined

    # Source/identity (already in v4.1, plumbed into features)
    gdelt_fraction: float
    source_diversity: float              # Shannon entropy across publication domains, normalized [0,1]

    # Pre-computed boolean signals (computed during feature extraction with full
    # access to posts/timestamps; gates just compose them to keep gate logic pure).
    gdelt_batch_artifact: bool
    wire_signal_known_syndicators: bool
    wire_signal_hash_diversity: bool

    # DNA matching
    dna_match_count: int                 # -1 = not yet computed → gate skips

    # Ensemble metadata
    ensemble_disagreement: float
    ensemble_perfect_agreement_red_flag: bool

    # Narrative coherence (NEW): GKG-metadata-based, [0,1]; 0.5 = insufficient data
    narrative_coherence: float

    # Number of distinct publication domains (raw count; complements source_diversity)
    unique_domain_count: int = 5

    # Language diversity (topic bags often have diverse languages)
    dominant_lang_fraction: float = 1.0
    language_count: int = 1

    # High-confidence DNA (≥0.90 cosine — operator identity, not wire content)
    high_conf_dna_match_count: int = 0
    cross_topic_persistence: bool = False

    # Anomaly signals (Tier 1+)
    circadian_anomaly: bool = False   # >40% posts during off-hours (1-5 AM UTC)
    content_anomaly: bool = False     # Unusual tone patterns (high activity + high negative + low self-ref)
    content_noise: bool = False       # Clickbait/listicle/obituary/classified content
    cross_language_anomaly: bool = False  # 3+ real languages in same cluster
    geographic_spread: bool = False       # Locations in 3+ countries
    high_signal_topic: bool = False       # Disinfo/propaganda themes with velocity

    # Confidence (computed downstream of NVI; attached for gate 10)
    confidence_probability: float = 0.5


@dataclass
class GateResult:
    """Result of running the gate pipeline."""

    nvi_cap: float = 100.0
    nvi_zero: bool = False
    gates_applied: list[str] = field(default_factory=list)
    gate_reasoning: dict[str, dict] = field(default_factory=dict)
    alert_suppressed: bool = False
    # Positive boost: lifts NVI floor when external evidence (e.g. DNA matches
    # across other clusters) confirms coordination beyond what cluster-local
    # signals (burst, spread, mutation) capture. Suppression caps still apply.
    nvi_floor: float = 0.0
    force_alert_level: Optional[str] = None


@dataclass
class Gate:
    """Static definition of a single gate in the pipeline."""

    name: str
    fn: Callable[[ClusterFeatures], tuple[bool, dict]]
    cap: Optional[float] = None
    # Boost gates raise the NVI floor instead of capping it. Used to surface
    # clusters that have external coordination evidence (cross-cluster DNA
    # matches) which raw cluster-local NVI signals can't capture on their own.
    boost: bool = False
    terminal: bool = False               # gate 1 zeroes NVI and short-circuits
    suppress_only: bool = False          # gate 10 suppresses alert without capping NVI
    skip_if: set[str] = field(default_factory=set)


# ─── Helper feature computations (called from nvi._extract_cluster_features) ─


def compute_unique_hash_ratio(posts: list[dict]) -> float:
    """
    Fraction of posts with distinct content_hash. Wire-service rewrites
    yield ratio ≈ 1.0; identical syndication yields ratio < 0.5.
    """
    if not posts:
        return 0.0
    hashes = [p.get("content_hash") for p in posts if p.get("content_hash")]
    if not hashes:
        return 0.0
    return len(set(hashes)) / len(hashes)


def compute_inter_arrival_stats(posts: list[dict]) -> tuple[float, float]:
    """
    Returns (mean, std) of inter-arrival times in seconds.
    Returns (nan, nan) when fewer than 2 valid timestamps.
    """
    from datetime import datetime

    timestamps = []
    for p in posts:
        ts_str = p.get("published_at") or p.get("ingested_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError):
            continue

    if len(timestamps) < 2:
        return math.nan, math.nan

    timestamps.sort()
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if not deltas:
        return math.nan, math.nan

    n = len(deltas)
    mean = sum(deltas) / n
    if n < 2:
        return mean, 0.0
    var = sum((d - mean) ** 2 for d in deltas) / n
    return mean, math.sqrt(var)


def _parse_metadata(p: dict) -> dict:
    raw = p.get("metadata", "{}")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw or {}


def _post_themes(meta: dict) -> set[str]:
    themes = meta.get("themes") or []
    if isinstance(themes, str):
        # GDELT GKG sometimes stores semicolon-delimited
        themes = [t for t in themes.split(";") if t.strip()]
    return {str(t).strip().lower() for t in themes if str(t).strip()}


def _post_persons_orgs(meta: dict) -> tuple[set[str], set[str]]:
    persons_raw = meta.get("persons", "") or ""
    orgs_raw = meta.get("organizations", "") or ""
    if isinstance(persons_raw, list):
        persons_raw = ";".join(str(x) for x in persons_raw)
    if isinstance(orgs_raw, list):
        orgs_raw = ";".join(str(x) for x in orgs_raw)
    persons = {
        x.strip().lower()
        for x in str(persons_raw).split(";")
        if x.strip() and len(x.strip()) > 2 and "#" not in x
    }
    orgs = {
        x.strip().lower()
        for x in str(orgs_raw).split(";")
        if x.strip() and len(x.strip()) > 2 and "#" not in x
    }
    return persons, orgs


def _title_tokens(post: dict) -> set[str]:
    title = (post.get("title") or "")
    if not title:
        return set()
    tokens = {t.lower() for t in _TOKEN_RE.findall(title) if len(t) > 2}
    return tokens - _STOP_TOKENS


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _pairwise_jaccard_mean(sets: list[set]) -> float:
    """Mean pairwise Jaccard. Returns 0.0 if fewer than 2 non-empty sets."""
    nonempty = [s for s in sets if s]
    if len(nonempty) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            total += _jaccard(nonempty[i], nonempty[j])
            pairs += 1
    if pairs == 0:
        return 0.0
    return total / pairs


def compute_narrative_coherence(posts: list[dict]) -> float:
    """
    Distinct from entity_concentration. Where entity_concentration asks
    "does the same person appear in N% of posts?", coherence asks
    "do these posts tell the same story?"

    Formula (v5.5, entity-weighted — token jaccard reduced to 0.30):
        coherence = 0.70 * entity_continuity
                  + 0.30 * token_jaccard_mean

    The 50/50 split (v5.4) punished the *correct* pattern: many independent
    outlets reporting the same story in their own words. Entity continuity
    (shared protagonists/orgs) is the genuine "is this one story?" signal;
    title-token overlap is mostly stopword noise plus phrasing variation.
    Cluster 716 (12 outlets reporting the Canvas LMS breach) computed 0.27
    under v5.4 — gate fired and capped NVI at 20. Under v5.5 it computes
    ~0.40 — gate does not fire.

    Theme Jaccard remains removed (rewards topic bags).

    Coverage rule: if <30% of posts have persons/orgs metadata,
    return 0.5 (neutral, gate doesn't fire).
    """
    if len(posts) < 3:
        return 0.5

    themes_per_post: list[set[str]] = []
    persons_per_post: list[set[str]] = []
    orgs_per_post: list[set[str]] = []
    tokens_per_post: list[set[str]] = []
    posts_with_meta = 0

    for p in posts:
        meta = _parse_metadata(p)
        themes = _post_themes(meta)
        persons, orgs = _post_persons_orgs(meta)
        tokens = _title_tokens(p)

        if themes or persons or orgs:
            posts_with_meta += 1

        themes_per_post.append(themes)
        persons_per_post.append(persons)
        orgs_per_post.append(orgs)
        tokens_per_post.append(tokens)

    coverage = posts_with_meta / len(posts)
    if coverage < 0.30:
        return 0.5  # insufficient metadata → don't gate

    token_jaccard_mean = _pairwise_jaccard_mean(tokens_per_post)

    # Entity continuity: top-3 most-frequent persons OR orgs across cluster,
    # then fraction of posts mentioning at least one of them.
    person_freq: Counter = Counter()
    org_freq: Counter = Counter()
    for s in persons_per_post:
        person_freq.update(e for e in s if e not in _GENERIC_ENTITIES)
    for s in orgs_per_post:
        org_freq.update(e for e in s if e not in _GENERIC_ENTITIES)

    top_persons = {e for e, _ in person_freq.most_common(3)}
    top_orgs = {e for e, _ in org_freq.most_common(3)}
    top_entities = top_persons | top_orgs

    if not top_entities:
        entity_continuity = 0.0
    else:
        hits = 0
        for persons, orgs in zip(persons_per_post, orgs_per_post):
            if (persons | orgs) & top_entities:
                hits += 1
        entity_continuity = hits / len(posts)

    coherence = (
        0.70 * entity_continuity
        + 0.30 * token_jaccard_mean
    )
    return float(max(0.0, min(1.0, coherence)))


def compute_wire_hash_diversity_signal(
    unique_hash_ratio: float,
    embedding_similarity_mean: float,
    source_diversity: float,
    dna_match_count: int,
    post_count: int,
) -> bool:
    """
    Detect "wire service distributed but rewritten by N outlets" pattern.

    Triggers when:
      - unique_hash_ratio > 0.90 (nearly every post is byte-distinct)
      - embedding_similarity_mean > 0.55 (semantically similar — calibrated
        for translated wire where embedding drops to 0.55−0.75 across
        languages; uhr + sd + pc provide orthogonal confirmation)
      - source_diversity > 0.65 (5+ distinct outlets, Shannon entropy on domains)
      - post_count >= 7 (lowered from 10 for small but clear wire patterns)

    Note: unlike earlier versions, this does NOT check ``dna_match_count`` —
    wire-distributed content naturally produces cross-cluster DNA matches
    (identical text → identical fingerprint), so excluding clusters with
    DNA matches would defeat the detector's purpose.
    """
    if post_count < 7:
        return False

    # Primary check: strong signals across diverse sources.
    # Catches wire syndication with well-distributed outlet coverage.
    if (
        unique_hash_ratio > 0.90
        and embedding_similarity_mean > 0.55
        and source_diversity > 0.65
    ):
        return True

    # Broader check: catch wire-distributed clusters where source diversity
    # is moderate (3-5 uneven outlets) but content identity is still unmistakable.
    # uhr=1.00 + esm>0.40 across any mult-source cluster = shared origin.
    # Broader check: moderate source diversity but strong content identity.
    # Catch clusters where content is near-identical across a narrower set of
    # outlets (e.g. a PTI wire story on 5 Indian news sites with the same text).
    if (
        post_count >= 7
        and unique_hash_ratio > 0.85
        and embedding_similarity_mean > 0.55
        and source_diversity > 0.40
    ):
        return True

    return False


# ─── Individual gate functions ──────────────────────────────────────────────


def _gate_gdelt_batch(f: ClusterFeatures) -> tuple[bool, dict]:
    if f.gdelt_batch_artifact:
        return True, {
            "fired": True,
            "why": (
                "Temporal regularity originates from GDELT's 15-minute batch "
                "publishing cycle, not real-world coordination. NVI zeroed."
            ),
            "evidence": {
                "gdelt_fraction": round(f.gdelt_fraction, 3),
                "inter_arrival_mean_seconds": (
                    None if math.isnan(f.inter_arrival_mean)
                    else round(f.inter_arrival_mean, 1)
                ),
                "inter_arrival_std_seconds": (
                    None if math.isnan(f.inter_arrival_std)
                    else round(f.inter_arrival_std, 1)
                ),
            },
        }
    return False, {"fired": False, "why": "Temporal pattern is not a GDELT batch signature."}


def _gate_insufficient_evidence(f: ClusterFeatures) -> tuple[bool, dict]:
    cap = 70 if f.post_count >= 5 else 40
    if f.post_count < 10:
        return True, {
            "fired": True,
            "why": (
                f"Only {f.post_count} posts. Below the 10-post threshold "
                f"for reliable narrative classification — NVI capped at {cap:.0f}."
            ),
            "cap": cap,
            "evidence": {"post_count": f.post_count},
        }
    return False, {"fired": False, "why": f"Sufficient sample ({f.post_count} posts)."}


def _gate_single_source_cluster(f: ClusterFeatures) -> tuple[bool, dict]:
    """
    A cluster drawn from a single publication domain (or with near-zero domain
    entropy) cannot be 'coordination' — it is one outlet's editorial coverage.
    The 39038 false positive (21 posts, all www.chip.de, all unrelated topics)
    is the canonical case.
    """
    if f.unique_domain_count <= 1 or f.source_diversity < 0.20:
        return True, {
            "fired": True,
            "why": (
                f"Cluster has only {f.unique_domain_count} unique domains "
                f"(Shannon entropy {f.source_diversity:.3f}). Single-source "
                "clusters cannot be coordination — they are one outlet's "
                "editorial coverage."
            ),
            "evidence": {
                "unique_domain_count": f.unique_domain_count,
                "source_diversity": round(f.source_diversity, 3),
            },
        }
    return False, {
        "fired": False,
        "why": (
            f"Cluster spans {f.unique_domain_count} domains "
            f"(diversity {f.source_diversity:.3f})."
        ),
    }


def _gate_dna_match(f: ClusterFeatures) -> tuple[bool, dict]:
    # -1 means DNA fingerprint hasn't been computed yet. For small clusters,
    # insufficient_evidence already covers it. For larger clusters (>=10 posts)
    # we apply a soft cap to prevent premature COORDINATION alerts before the
    # DNA cycle has had a chance to either confirm or deny operator persistence.
    if f.dna_match_count == -1:
        if f.post_count >= 10:
            return True, {
                "fired": True,
                "cap": 25.0,
                "why": (
                    "DNA fingerprint not yet computed. Pending DNA cycle, "
                    "capping at 25 to prevent premature alerts."
                ),
                "evidence": {
                    "post_count": f.post_count,
                    "dna_match_count": f.dna_match_count,
                },
            }
        return False, {
            "fired": False,
            "why": "DNA fingerprint not yet computed; small cluster covered by insufficient_evidence.",
        }
    if f.post_count < 20 and f.dna_match_count == 0:
        return True, {
            "fired": True,
            "why": (
                f"Small cluster ({f.post_count} posts) with zero DNA matches "
                "across other campaigns. Real coordinated operators leave "
                "persistent fingerprints; one-off false positives don't."
            ),
            "evidence": {
                "post_count": f.post_count,
                "dna_match_count": f.dna_match_count,
            },
        }
    return False, {
        "fired": False,
        "why": (
            f"Either large enough ({f.post_count} posts) or has DNA matches "
            f"({f.dna_match_count})."
        ),
    }


def _gate_wire_service(f: ClusterFeatures) -> tuple[bool, dict]:
    """
    Two signals feed this gate:
      A) wire_signal_known_syndicators — domain-list match + low mutation
      B) wire_signal_hash_diversity — high hash diversity + high embedding
                                       similarity + diverse sources + no DNA

    Either alone caps at 25. Both together (rewritten wire copy with low
    mutation across many outlets) caps at 20 — strongest signature.
    """
    a = f.wire_signal_known_syndicators
    b = f.wire_signal_hash_diversity
    if not (a or b):
        return False, {"fired": False, "why": "No wire-service signature detected."}

    if a and b:
        return True, {
            "fired": True,
            "cap": 20.0,
            "why": (
                "Dual wire-service signature: known syndicator domains AND "
                "high-hash-diversity rewrite pattern. NVI capped at 20 "
                "(strongest signal)."
            ),
            "evidence": {
                "unique_hash_ratio": round(f.unique_hash_ratio, 3),
                "embedding_similarity_mean": round(f.embedding_similarity_mean, 3),
                "source_diversity": round(f.source_diversity, 3),
                "dna_match_count": f.dna_match_count,
            },
        }

    if a:
        return True, {
            "fired": True,
            "why": (
                "Known wire-service / regional syndicator domains dominate "
                "this cluster with near-identical content. NVI capped at 25."
            ),
            "evidence": {
                "wire_signal_known_syndicators": True,
                "mutation": round(f.mutation, 3),
            },
        }

    # b only
    return True, {
        "fired": True,
        "why": (
            "Wire-service rewrite pattern: nearly all posts have unique "
            "content hashes (rewritten) but high embedding similarity "
            "(same story), distributed across diverse sources with no DNA "
            "matches. NVI capped at 25."
        ),
        "evidence": {
            "unique_hash_ratio": round(f.unique_hash_ratio, 3),
            "embedding_similarity_mean": round(f.embedding_similarity_mean, 3),
            "source_diversity": round(f.source_diversity, 3),
            "dna_match_count": f.dna_match_count,
        },
    }


def _gate_ensemble_uncertainty(f: ClusterFeatures) -> tuple[bool, dict]:
    if f.ensemble_disagreement < 1.0 and f.post_count < 15:
        return True, {
            "fired": True,
            "why": (
                f"All three ensemble weighting schemes produced near-identical "
                f"NVI scores (disagreement={f.ensemble_disagreement:.2f}) on a "
                f"small sample ({f.post_count} posts). They are all measuring "
                "the same artifact, not independent signals. NVI capped at 35."
            ),
            "evidence": {
                "ensemble_disagreement": round(f.ensemble_disagreement, 3),
                "post_count": f.post_count,
            },
        }
    return False, {
        "fired": False,
        "why": (
            f"Either disagreement is healthy ({f.ensemble_disagreement:.2f}) "
            f"or sample is large enough ({f.post_count} posts)."
        ),
    }


def _gate_entity_concentration(f: ClusterFeatures) -> tuple[bool, dict]:
    """
    Real coordination shares named entities (people, orgs) across posts.
    Topic bags don't — each post in a Romanian news roundup mentions
    different people. The evidence pack shows all 7 posts mentioning the
    same event with high entity overlap.

    Triggers when shared_entity_count <= 1 AND post_count >= 15.
    Small clusters (< 15 posts) are left to insufficient_evidence.
    """
    if f.shared_entity_count <= 1 and f.post_count >= 15:
        return True, {
            "fired": True,
            "why": (
                f"Only {f.shared_entity_count} shared named entities across "
                f"{f.post_count} posts. Real coordination shares named people "
                "and organizations; topic bags share only a subject taxonomy. "
                "NVI capped at 35."
            ),
            "evidence": {
                "shared_entity_count": f.shared_entity_count,
                "post_count": f.post_count,
                "entity_concentration": round(f.entity_concentration, 4),
            },
        }
    return False, {
        "fired": False,
        "why": (
            f"Entity overlap acceptable ({f.shared_entity_count} shared entities "
            f"across {f.post_count} posts)."
        ),
    }


def _gate_narrative_coherence(f: ClusterFeatures) -> tuple[bool, dict]:
    # 0.5 is the sentinel for insufficient metadata — neutral, don't fire.
    if f.narrative_coherence == 0.5:
        return False, {"fired": False, "why": "Insufficient GKG metadata to assess coherence."}
    if f.narrative_coherence < 0.22:
        return True, {
            "fired": True,
            "why": (
                f"Narrative coherence {f.narrative_coherence:.2f} below 0.22 threshold "
                "(0.70·entity_continuity + 0.30·title-tokens). "
                "Posts lack a common cast of named protagonists — they are not "
                "telling one story. NVI capped at 20."
            ),
            "evidence": {"narrative_coherence": round(f.narrative_coherence, 3)},
        }
    return False, {
        "fired": False,
        "why": f"Cluster is narratively coherent ({f.narrative_coherence:.2f}).",
    }


def _gate_organic_viral(f: ClusterFeatures) -> tuple[bool, dict]:
    if f.mutation > 0.30 and f.source_diversity > 0.60:
        return True, {
            "fired": True,
            "why": (
                f"High content mutation ({f.mutation:.2f}) combined with high "
                f"source diversity ({f.source_diversity:.2f}) indicates organic "
                "viral spread — independent rewrites by independent outlets. "
                "NVI capped at 30."
            ),
            "evidence": {
                "mutation": round(f.mutation, 3),
                "source_diversity": round(f.source_diversity, 3),
            },
        }
    return False, {"fired": False, "why": "Pattern is not consistent with organic viral spread."}


def _gate_normal_news_cycle(f: ClusterFeatures) -> tuple[bool, dict]:
    # Only suppress tiny, genuinely low-burst clusters. The old threshold
    # (burst < 5) fired on 42% of all clusters because temporal_sync only
    # activates on ~0.4% — coordination_mult defaults to 1.0 for everything
    # else. Require all three conditions to avoid mass suppression.
    if f.coordination < 1.05 and f.burst < 1.5 and f.post_count < 15:
        # Bypass: high source diversity (many distinct outlets reporting the
        # same story) is itself a coordination signal that this gate would
        # otherwise mute. 0.7 ≈ 5+ near-equally-weighted domains in Shannon
        # entropy normalized to [0,1].
        if f.source_diversity >= 0.7:
            return False, {
                "fired": False,
                "why": (
                    f"Bypassed: source_diversity {f.source_diversity:.2f} "
                    "indicates many distinct outlets — not a normal local cycle."
                ),
            }
        # Bypass: cross-cluster DNA evidence supersedes timing/burst suppression.
        # If the same operator fingerprint shows in 5+ other clusters, this is
        # a coordinated campaign even when local burst is quiet.
        if f.dna_match_count is not None and f.dna_match_count >= 5:
            return False, {
                "fired": False,
                "why": (
                    f"Bypassed: {f.dna_match_count} cross-cluster DNA matches "
                    "indicate coordinated authorship beyond local timing."
                ),
            }
        return True, {
            "fired": True,
            "why": (
                f"Coordination multiplier ({f.coordination:.2f}), burst "
                f"({f.burst:.1f}), and post count ({f.post_count}) are all "
                "below minimum thresholds for meaningful signal. NVI capped at 35."
            ),
            "evidence": {
                "coordination_mult": round(f.coordination, 3),
                "burst_zscore": round(f.burst, 3),
                "post_count": f.post_count,
            },
        }
    return False, {"fired": False, "why": "Timing pattern exceeds normal news cycle."}


def _gate_confidence_threshold(f: ClusterFeatures) -> tuple[bool, dict]:
    if f.confidence_probability < 0.65:
        return True, {
            "fired": True,
            "why": (
                f"Confidence probability {f.confidence_probability:.2f} is "
                "below the 0.65 alert threshold. Alert suppressed; system "
                "refuses to cry wolf on weak evidence."
            ),
            "evidence": {"confidence_probability": round(f.confidence_probability, 3)},
        }
    return False, {
        "fired": False,
        "why": f"Confidence ({f.confidence_probability:.2f}) clears the 0.65 threshold.",
    }


# ─── Content Quality Gate ──────────────────────────────────────────────────

def _gate_content_noise(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect clusters made of content noise — listicles, obituaries,
    classifieds, clickbait, very short non-news titles.

    These are filtered at the API level (suppressed from feed) so the
    dashboard only shows substantive narratives.
    """
    if not f.content_noise:
        return False, {"fired": False, "why": "Content is substantive, not noise."}
    return True, {
        "fired": True,
        "suppress": True,
        "why": (
            "Content appears to be non-news material: listicle, obituary, "
            "classified, or clickbait. Suppressed from main feed."
        ),
        "evidence": {"content_noise": f.content_noise},
    }


# ─── Tier 1+ Anomaly Gates ─────────────────────────────────────────────────

def _gate_circadian_anomaly(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect off-hour publishing patterns characteristic of coordinated
    inauthentic behavior or automated content operations.

    Normal news outlets publish during business hours (8 AM - 8 PM local).
    Coordinated campaigns often publish at 1-5 AM UTC to evade editorial
    oversight, reach specific timezones, or batch through automated systems.

    When >40% of a cluster's posts are during off-hours AND the source
    diversity exceeds 0.3 (ruling out single-source data ingestion errors),
    this is a strong behavioral anomaly signal.
    """
    if not f.circadian_anomaly or f.source_diversity < 0.3:
        return False, {
            "fired": False,
            "why": "Off-hour publishing pattern not detected.",
        }
    if f.post_count >= 10:
        floor = 65.0
        force_level = "elevated"
        explanation = (
            f"Off-hour publishing anomaly: {f.post_count} posts during 1-5 AM UTC "
            "across multiple sources. Coordinated operations frequently publish "
            "outside normal news cycles."
        )
    else:
        floor = 50.0
        force_level = "elevated"
        explanation = (
            "Off-hour publishing pattern detected on small cluster. "
            "Elevating NVI for review."
        )
    return True, {
        "fired": True, "floor": floor, "force_alert_level": force_level,
        "why": explanation,
        "evidence": {
            "post_count": f.post_count,
            "source_diversity": round(f.source_diversity, 3),
            "circadian_anomaly": f.circadian_anomaly,
        },
    }


def _gate_content_anomaly(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect unusual tone/activity patterns that suggest automated or
    manipulated content.

    High activity_density (>3) + high negative tone (>2) + low self_reference
    (<1) is characteristic of content designed to provoke emotional response
    with minimal original authoring -- a common CIB pattern.

    Fires as a mild boost so this signal can accumulate with other evidence.
    """
    if not f.content_anomaly:
        return False, {
            "fired": False,
            "why": "Content tone anomaly not detected.",
        }
    return True, {
        "fired": True, "floor": 50.0, "force_alert_level": "elevated",
        "why": (
            "Content tone anomaly: high activity density combined with high "
            "negative tone and low self-reference. This pattern is associated "
            "with automated or emotionally-manipulated content."
        ),
        "evidence": {"content_anomaly": f.content_anomaly},
    }


# ─── Anomaly Gates ─────────────────────────────────────────────────────────

def _gate_cross_language(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect clusters spanning 3+ real languages (not GDELT auto-translation).

    A story appearing in German, Turkish, and Arabic simultaneously is a
    genuinely cross-cultural narrative event — unusual and worth flagging.
    """
    if not f.cross_language_anomaly or f.source_diversity < 0.3:
        return False, {"fired": False, "why": "Cross-language pattern not detected."}
    return True, {
        "fired": True, "floor": 65.0, "force_alert_level": "elevated",
        "why": (
            f"Cross-language narrative: spans {f.language_count} distinct languages "
            "across multiple sources. This level of language diversity is "
            "characteristic of cross-cultural news events."
        ),
        "evidence": {
            "language_count": f.language_count,
            "source_diversity": round(f.source_diversity, 3),
        },
    }


def _gate_geographic_spread(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect stories covering events in 3+ different countries.

    Most news stays within one country. A story about a Spanish hantavirus
    outbreak covered by outlets from UK, Germany, France is geographically
    anomalous — it indicates a story crossing national boundaries.
    """
    if not f.geographic_spread:
        return False, {"fired": False, "why": "Geographic spread not detected."}
    return True, {
        "fired": True, "floor": 60.0, "force_alert_level": "elevated",
        "why": (
            "Geographic spread anomaly: this story references locations in "
            "multiple countries, indicating cross-border narrative spread."
        ),
        "evidence": {"source_diversity": round(f.source_diversity, 3)},
    }


def _gate_high_signal_topic(f: ClusterFeatures) -> tuple[bool, dict]:
    """Detect clusters about disinformation/propaganda topics that have
    cross-source velocity. The story BEING about coordination AND spreading
    across sources IS the signal.
    """
    if not f.high_signal_topic:
        return False, {"fired": False, "why": "High-signal topic pattern not detected."}
    return True, {
        "fired": True, "floor": 55.0, "force_alert_level": "elevated",
        "why": (
            "High-signal topic with cross-source spread: this narrative "
            "about disinformation, propaganda, or influence operations "
            "is being carried by multiple independent sources."
        ),
        "evidence": {
            "source_diversity": round(f.source_diversity, 3),
            "post_count": f.post_count,
        },
    }


# ─── Boost gate: cross-cluster velocity ────────────────────────────────────

def _gate_cross_cluster_velocity(f: ClusterFeatures) -> tuple[bool, dict]:
    """Lift NVI floor when cross-cluster content relationships indicate
    elevated narrative velocity beyond what cluster-local signals capture.

    Uses TWO tiers of DNA evidence:

    **Tier 1 — Cross-topic persistence (strongest signal):** When the same
    operator fingerprint appears across clusters about DIFFERENT topics
    (high stylometric+network similarity + low entity_bias similarity),
    this is the same author covering unrelated stories for multiple outlets.
    Forces critical regardless of source diversity.

    **Tier 2 — High-confidence DNA count (≥0.90):** At ≥0.90 cosine the 4-
    modality weighted fingerprint requires alignment across ALL dimensions.
    Wire-syndicated identical content scores 0.75−0.85 (CMS-specific editor
    fingerprints differ in cadence/network). ≥0.90 means genuine operator
    persistence.

    **Guards:**
    - Wire syndication: identical text on many outlets (false DNA signal)
    - Topic bags: GDELT taxonomy overlap, not real stories

    **Thresholds (high_conf match count):**
    - n ≥ 20 high-conf matches: critical (NVI floor 80)
    - n ≥ 5 high-conf matches: elevated (NVI floor 65)
    - n < 5: not enough high-confidence evidence
    """
    hc = f.high_conf_dna_match_count

    # Guard 1: Wire syndication
    if f.wire_signal_known_syndicators or f.wire_signal_hash_diversity:
        return False, {
            "fired": False,
            "why": (
                "Content relationships reflect wire-syndicated distribution, "
                "not operator persistence."
            ),
            "evidence": {
                "dna_match_count": f.dna_match_count,
                "high_conf_dna_match_count": hc,
                "wire_signal_known_syndicators": f.wire_signal_known_syndicators,
                "wire_signal_hash_diversity": f.wire_signal_hash_diversity,
            },
        }

    # Guard 2: Topic bags
    if f.shared_entity_count <= 1 and f.narrative_coherence < 0.30:
        return False, {
            "fired": False,
            "why": (
                "Topic bag — GKG theme overlap, not a real narrative."
            ),
            "evidence": {
                "dna_match_count": f.dna_match_count,
                "high_conf_dna_match_count": hc,
                "shared_entity_count": f.shared_entity_count,
                "narrative_coherence": round(f.narrative_coherence, 3),
            },
        }

    # Tier 1: Cross-topic persistence — strongest signal, bypasses source diversity
    if f.cross_topic_persistence:
        return True, {
            "fired": True,
            "floor": 80.0,
            "force_alert_level": "critical",
            "why": (
                "Cross-topic operator persistence detected: same writing "
                "fingerprint appears across DISTINCT topics in multiple "
                "clusters. Forces critical velocity floor (NVI floor 80)."
            ),
            "evidence": {
                "dna_match_count": f.dna_match_count,
                "high_conf_dna_match_count": hc,
                "cross_topic_persistence": f.cross_topic_persistence,
            },
        }

    # Tier 2: High-confidence match count
    if hc < 5:
        return False, {
            "fired": False,
            "why": (
                f"Insufficient high-confidence matches "
                f"(count={hc}, requires ≥5 at ≥0.90 cosine)."
            ),
            "evidence": {
                "dna_match_count": f.dna_match_count,
                "high_conf_dna_match_count": hc,
            },
        }

    # Source diversity — relaxed for high-confidence matches
    if f.source_diversity < 0.5:
        return False, {
            "fired": False,
            "why": (
                f"Source diversity {f.source_diversity:.2f} below 0.5 — "
                "not enough independent outlets."
            ),
            "evidence": {
                "dna_match_count": f.dna_match_count,
                "high_conf_dna_match_count": hc,
                "source_diversity": round(f.source_diversity, 3),
            },
        }

    if hc >= 20:
        floor = 80.0
        force_level = "critical"
        explanation = (
            f"Strong high-confidence operator persistence: {hc} matches "
            f"at ≥0.90 across {f.source_diversity:.2f} diverse sources. "
            "Setting critical velocity floor (NVI floor 80)."
        )
    else:
        floor = 65.0
        force_level = "elevated"
        explanation = (
            f"Operator persistence detected: {hc} high-confidence matches "
            f"at ≥0.90 across {f.source_diversity:.2f} diverse sources. "
            "Setting elevated velocity floor (NVI floor 65)."
        )
    return True, {
        "fired": True,
        "floor": floor,
        "force_alert_level": force_level,
        "why": explanation,
        "evidence": {
            "dna_match_count": f.dna_match_count,
            "high_conf_dna_match_count": hc,
            "source_diversity": round(f.source_diversity, 3),
        },
    }


# ─── Pipeline declaration ───────────────────────────────────────────────────

GATES: list[Gate] = [
    Gate("gdelt_batch_artifact",   _gate_gdelt_batch,            terminal=True),
    Gate("content_noise",          _gate_content_noise,          suppress_only=True),
    Gate("insufficient_evidence",  _gate_insufficient_evidence,  cap=40.0),
    Gate("single_source_cluster",  _gate_single_source_cluster,  cap=15.0),
    Gate("wire_service",           _gate_wire_service,           cap=25.0),
    Gate("dna_match",              _gate_dna_match,              cap=25.0),
    Gate("cross_language",         _gate_cross_language,         boost=True),
    Gate("geographic_spread",      _gate_geographic_spread,      boost=True),
    Gate("high_signal_topic",      _gate_high_signal_topic,      boost=True),
    Gate("circadian_anomaly",      _gate_circadian_anomaly,      boost=True),
    Gate("content_anomaly",        _gate_content_anomaly,        boost=True),
    Gate("cross_cluster_velocity", _gate_cross_cluster_velocity, boost=True),
    Gate("ensemble_uncertainty",   _gate_ensemble_uncertainty,   cap=35.0),
    Gate("entity_concentration",   _gate_entity_concentration,   cap=35.0),
    Gate("narrative_coherence",    _gate_narrative_coherence,    cap=20.0),
    Gate("organic_viral_spread",   _gate_organic_viral,          cap=30.0),
    Gate("normal_news_cycle",      _gate_normal_news_cycle,      cap=35.0),
    Gate("confidence_threshold",   _gate_confidence_threshold,   suppress_only=True),
]


def apply_falsification_gates(features: ClusterFeatures) -> GateResult:
    """
    Run gates in declared priority order. Accumulates the strictest cap.
    Gate 1 is terminal: when fired it sets nvi_zero and short-circuits all
    subsequent NVI caps (confidence gate still runs to mark suppression).
    """
    result = GateResult()
    fired_set: set[str] = set()

    for gate in GATES:
        if fired_set & gate.skip_if:
            result.gate_reasoning[gate.name] = {
                "fired": False,
                "why": (
                    f"Skipped because {sorted(fired_set & gate.skip_if)} already fired."
                ),
            }
            continue

        # If gate 1 already zeroed NVI, skip subsequent capping gates but still
        # evaluate confidence (suppress_only) so downstream consumers know
        # whether the alert would have been suppressed independently.
        if result.nvi_zero and not gate.suppress_only:
            result.gate_reasoning[gate.name] = {
                "fired": False,
                "why": "Skipped because gdelt_batch_artifact already zeroed NVI.",
            }
            continue

        fired, reasoning = gate.fn(features)
        result.gate_reasoning[gate.name] = reasoning

        if not fired:
            continue

        fired_set.add(gate.name)
        result.gates_applied.append(gate.name)

        if gate.terminal:
            result.nvi_zero = True
            continue

        if gate.suppress_only:
            result.alert_suppressed = True
            continue

        if gate.boost:
            floor = reasoning.get("floor")
            if floor is not None and float(floor) > result.nvi_floor:
                result.nvi_floor = float(floor)
            forced = reasoning.get("force_alert_level")
            if forced:
                # Critical wins over elevated if multiple boost gates fire.
                if result.force_alert_level != "critical":
                    result.force_alert_level = forced
            continue

        # Use gate-supplied cap when present (e.g. wire dual-signal cap=20),
        # otherwise fall back to declared cap.
        cap_value = reasoning.get("cap", gate.cap)
        if cap_value is None:
            continue
        if cap_value < result.nvi_cap:
            result.nvi_cap = float(cap_value)

    return result


__all__ = [
    "ClusterFeatures",
    "GateResult",
    "Gate",
    "GATES",
    "apply_falsification_gates",
    "compute_unique_hash_ratio",
    "compute_inter_arrival_stats",
    "compute_narrative_coherence",
    "compute_wire_hash_diversity_signal",
]

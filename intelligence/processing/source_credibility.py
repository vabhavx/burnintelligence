"""
BurnTheLies Intelligence — Source Credibility Scoring

Deterministic scoring system for source domains.
NOT an ML model — based on curated registry + observed behavioral signals.

Categories:
- wire_service (0.85-0.95): AP, Reuters, AFP
- major_editorial (0.70-0.85): NYT, BBC, Guardian
- state_media (0.30-0.50): RT, Xinhua, TASS
- independent_verified (0.60-0.75): ProPublica, Bellingcat
- regional_outlet (0.55): UK/US regional news outlets, heavy wire copy
- unknown (0.50): default for unregistered domains
"""

import logging
from collections import Counter

logger = logging.getLogger("intelligence.source_credibility")

# ─── Curated Source Registry ─────────────────────────────────────────────────
# Manually vetted. Each entry: domain_fragment -> (category, base_score)
# Domain fragments match via substring (e.g. "reuters" matches "reuters.com")

_SOURCE_REGISTRY = {
    # Wire Services
    "apnews": ("wire_service", 0.92),
    "reuters": ("wire_service", 0.92),
    "afp.com": ("wire_service", 0.90),
    "upi.com": ("wire_service", 0.85),
    "efe.com": ("wire_service", 0.85),
    "dpa.de": ("wire_service", 0.85),
    "kyodonews": ("wire_service", 0.85),

    # Major English-Language Editorial
    "nytimes": ("major_editorial", 0.85),
    "washingtonpost": ("major_editorial", 0.82),
    "theguardian": ("major_editorial", 0.82),
    "bbc.com": ("major_editorial", 0.85),
    "bbc.co.uk": ("major_editorial", 0.85),
    "economist": ("major_editorial", 0.83),
    "ft.com": ("major_editorial", 0.82),
    "wsj.com": ("major_editorial", 0.82),
    "npr.org": ("major_editorial", 0.80),
    "pbs.org": ("major_editorial", 0.80),
    "latimes": ("major_editorial", 0.78),
    "chicagotribune": ("major_editorial", 0.76),
    "usatoday": ("major_editorial", 0.74),
    "cnn.com": ("major_editorial", 0.73),
    "nbcnews": ("major_editorial", 0.73),
    "cbsnews": ("major_editorial", 0.73),
    "abcnews": ("major_editorial", 0.73),
    "politico": ("major_editorial", 0.76),
    "thehill": ("major_editorial", 0.74),
    "axios.com": ("major_editorial", 0.75),

    # Major International Editorial
    "lemonde": ("major_editorial", 0.82),
    "spiegel": ("major_editorial", 0.80),
    "elpais": ("major_editorial", 0.78),
    "corriere": ("major_editorial", 0.76),
    "asahi": ("major_editorial", 0.78),
    "yomiuri": ("major_editorial", 0.76),
    "haaretz": ("major_editorial", 0.78),
    "scmp.com": ("major_editorial", 0.74),
    "aljazeera": ("major_editorial", 0.72),
    "dw.com": ("major_editorial", 0.78),
    "france24": ("major_editorial", 0.76),
    "thehindu": ("major_editorial", 0.74),
    "timesofindia": ("major_editorial", 0.70),
    "ndtv.com": ("major_editorial", 0.70),
    "abc.net.au": ("major_editorial", 0.80),
    "cbc.ca": ("major_editorial", 0.78),
    "globeandmail": ("major_editorial", 0.76),
    "smh.com.au": ("major_editorial", 0.75),
    "nzherald": ("major_editorial", 0.72),
    "straitstimes": ("major_editorial", 0.72),

    # State-Affiliated Media
    "rt.com": ("state_media", 0.35),
    "sputniknews": ("state_media", 0.30),
    "tass.com": ("state_media", 0.40),
    "xinhua": ("state_media", 0.40),
    "globaltimes": ("state_media", 0.35),
    "chinadaily": ("state_media", 0.38),
    "cgtn.com": ("state_media", 0.35),
    "presstv": ("state_media", 0.30),
    "telesur": ("state_media", 0.35),
    "granma.cu": ("state_media", 0.30),
    "kcna.kp": ("state_media", 0.25),
    "voanews": ("state_media", 0.50),  # US gov-funded but editorially independent
    "rferl": ("state_media", 0.50),

    # Verified Independent / Investigative
    "propublica": ("independent_verified", 0.85),
    "bellingcat": ("independent_verified", 0.85),
    "theintercept": ("independent_verified", 0.75),
    "motherjones": ("independent_verified", 0.72),
    "revealnews": ("independent_verified", 0.78),
    "icij.org": ("independent_verified", 0.85),
    "occrp.org": ("independent_verified", 0.82),
    "mediapart": ("independent_verified", 0.78),
    "correctiv": ("independent_verified", 0.78),

    # Fact-Checking
    "snopes.com": ("independent_verified", 0.80),
    "politifact": ("independent_verified", 0.78),
    "factcheck.org": ("independent_verified", 0.80),
    "fullfact": ("independent_verified", 0.78),

    # Known Unreliable / Partisan
    "infowars": ("unreliable", 0.15),
    "breitbart": ("unreliable", 0.25),
    "naturalnews": ("unreliable", 0.10),
    "thegatewaypundit": ("unreliable", 0.15),
    "dailywire": ("unreliable", 0.30),
    "oann.com": ("unreliable", 0.25),
    "newsmax": ("unreliable", 0.30),
    "dailykos": ("unreliable", 0.35),
    "occupydemocrats": ("unreliable", 0.20),
    "rawstory": ("unreliable", 0.35),
    "zerohedge": ("unreliable", 0.20),

    # Regional News Outlets (Wire Service Subscribers)
    # Legitimate outlets but content is predominantly wire copy — moderate credibility
    "eveningstandard": ("regional_outlet", 0.55),
    "manchestereveningnews": ("regional_outlet", 0.55),
    "liverpoolecho": ("regional_outlet", 0.55),
    "belfasttelegraph": ("regional_outlet", 0.55),
    "walesonline": ("regional_outlet", 0.55),
    "birminghammail": ("regional_outlet", 0.55),
    "chroniclelive": ("regional_outlet", 0.55),
    "yorkshirepost": ("regional_outlet", 0.55),
    "glasgowtimes": ("regional_outlet", 0.55),
    "edinburghnews": ("regional_outlet", 0.55),
    "expressandstar": ("regional_outlet", 0.55),
    "shropshirestar": ("regional_outlet", 0.55),
    "thecourier": ("regional_outlet", 0.55),
    "dorsetecho": ("regional_outlet", 0.55),
    "theargus": ("regional_outlet", 0.55),
    "eadt": ("regional_outlet", 0.55),
    "ipswichstar": ("regional_outlet", 0.55),
    "thenorthernecho": ("regional_outlet", 0.55),
    "sunderlandecho": ("regional_outlet", 0.55),
    "wirralglobe": ("regional_outlet", 0.55),
    "lancashiretelegraph": ("regional_outlet", 0.55),
    "oxfordmail": ("regional_outlet", 0.55),
    "swindonadvertiser": ("regional_outlet", 0.55),
    "worcesternews": ("regional_outlet", 0.55),
    "herefordtimes": ("regional_outlet", 0.55),
    "southwalesargus": ("regional_outlet", 0.55),
    "greenocktelegraph": ("regional_outlet", 0.55),
    "bournemouthecho": ("regional_outlet", 0.55),
    "dailyecho": ("regional_outlet", 0.55),
    "portsmouth.co.uk": ("regional_outlet", 0.55),
    "thestar.co.uk": ("regional_outlet", 0.55),
    "lep.co.uk": ("regional_outlet", 0.55),
    "gazettelive.co.uk": ("regional_outlet", 0.55),
    "yorkpress.co.uk": ("regional_outlet", 0.55),
    "wiltshiretimes": ("regional_outlet", 0.55),
    "countypress": ("regional_outlet", 0.55),
    "bucksfreepress": ("regional_outlet", 0.55),
    "guardian-series": ("regional_outlet", 0.55),
    "newsandstar": ("regional_outlet", 0.55),
    "cumbriacrack": ("regional_outlet", 0.55),
    "cambridge-news": ("regional_outlet", 0.55),
    "norfolkeveningnews": ("regional_outlet", 0.55),
    "norwichgazette": ("regional_outlet", 0.55),
    "kentonline": ("regional_outlet", 0.55),
    "kentlive": ("regional_outlet", 0.55),
    "essexlive": ("regional_outlet", 0.55),
    "surreycomet": ("regional_outlet", 0.55),
    "getwestlondon": ("regional_outlet", 0.55),
    "mylondon": ("regional_outlet", 0.55),
    "bristolpost": ("regional_outlet", 0.55),
    "gloucestershirelive": ("regional_outlet", 0.55),
    "somersetlive": ("regional_outlet", 0.55),
    "devonlive": ("regional_outlet", 0.55),
    "cornwalllive": ("regional_outlet", 0.55),
    "plymouthherald": ("regional_outlet", 0.55),
    "nottinghampost": ("regional_outlet", 0.55),
    "leicestermercury": ("regional_outlet", 0.55),
    "derbytelegraph": ("regional_outlet", 0.55),
    "staffslive": ("regional_outlet", 0.55),
    "stokesentinel": ("regional_outlet", 0.55),
    "coventrytelegraph": ("regional_outlet", 0.55),
    "bedfordshirelive": ("regional_outlet", 0.55),
    "hulldailymail": ("regional_outlet", 0.55),
    # US regional wire subscribers (Advance Local / wire-heavy outlets)
    "pennlive.com": ("regional_outlet", 0.55),
    "nj.com": ("regional_outlet", 0.55),
    "mlive.com": ("regional_outlet", 0.55),
    "silive.com": ("regional_outlet", 0.55),
    "masslive.com": ("regional_outlet", 0.55),
    "lehighvalleylive.com": ("regional_outlet", 0.55),
    "al.com": ("regional_outlet", 0.55),
    "cleveland.com": ("regional_outlet", 0.55),
    "oregonlive.com": ("regional_outlet", 0.55),
    "syracuse.com": ("regional_outlet", 0.55),
}


# Regional news outlets known to heavily syndicate wire service copy.
# These are legitimate outlets but their content is predominantly wire distribution.
_REGIONAL_NEWS_WIRE_SUBSCRIBERS = {
    # UK regional outlets (from nvi.py _UK_REGIONAL_NEWS_DOMAINS)
    "eveningstandard", "manchestereveningnews", "liverpoolecho",
    "belfasttelegraph", "walesonline", "birminghammail", "chroniclelive",
    "yorkshirepost", "glasgowtimes", "edinburghnews", "thecourier",
    "dorsetecho", "expressandstar", "theargus", "eadt", "ipswichstar",
    "thenorthernecho", "shropshirestar", "sunderlandecho", "wirralglobe",
    "lancashiretelegraph", "oxfordmail", "swindonadvertiser",
    "worcesternews", "herefordtimes", "southwalesargus", "greenocktelegraph",
    "bournemouthecho", "dailyecho", "portsmouth.co.uk", "thestar.co.uk",
    "lep.co.uk", "gazettelive.co.uk", "yorkpress.co.uk", "wiltshiretimes",
    "countypress", "bucksfreepress", "guardian-series",
    "newsandstar", "cumbriacrack", "cambridge-news",
    "norfolkeveningnews", "norwichgazette", "kentonline", "kentlive",
    "essexlive", "surreycomet", "getwestlondon", "mylondon",
    "bristolpost", "gloucestershirelive", "somersetlive",
    "devonlive", "cornwalllive", "plymouthherald", "nottinghampost",
    "leicestermercury", "derbytelegraph", "staffslive", "stokesentinel",
    "coventrytelegraph", "bedfordshirelive", "hulldailymail",
    # US regional Advance Local / wire-heavy outlets
    "pennlive.com", "nj.com", "mlive.com", "silive.com", "masslive.com",
    "lehighvalleylive.com", "al.com", "cleveland.com", "oregonlive.com",
    "syracuse.com",
}


def get_source_category(domain: str) -> tuple[str, float]:
    """
    Look up a source domain in the registry.
    Returns (category, base_score). Defaults to ("unknown", 0.50).
    """
    domain_lower = domain.lower().strip()

    for fragment, (category, score) in _SOURCE_REGISTRY.items():
        if fragment in domain_lower:
            return category, score

    return "unknown", 0.50


def _extract_domain(url: str) -> str:
    """Extract domain from a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception as e:
        logger.warning(f"Skipping credibility update for {url!r}: {e}")
        return ""


def seed_source_scores(db_conn):
    """
    Seed the source_scores table with domains extracted from post URLs.
    Maps each domain against the curated registry.
    """
    from intelligence.db import upsert_source_score

    # Extract unique domains from URLs in raw_posts
    rows = db_conn.execute(
        "SELECT DISTINCT url FROM raw_posts WHERE url IS NOT NULL AND url != ''"
    ).fetchall()

    domains_seen = set()
    seeded = 0
    for row in rows:
        domain = _extract_domain(row["url"])
        if not domain or domain in domains_seen or len(domain) < 3:
            continue
        domains_seen.add(domain)
        category, base_score = get_source_category(domain)
        upsert_source_score(db_conn, domain, base_score, "curated_registry", category)
        seeded += 1

    # Also seed the ingestor-level sources
    for src in ["gdelt_doc", "gdelt_gkg", "bluesky"]:
        upsert_source_score(db_conn, src, 0.50, "ingestor_platform", "platform")

    logger.info(f"Seeded {seeded} source scores from {len(domains_seen)} unique domains")
    return seeded


def compute_dynamic_adjustments(db_conn):
    """
    Adjust source scores based on observed behavior in the intelligence data.
    Three signals:
    1. narrative_participation_rate: fraction of coordinated narratives this source appears in
    2. content_originality: average mutation penalty in clusters containing this source
    3. timing_independence: average coordination multiplier for clusters with this source
    """
    from intelligence.db import upsert_source_score

    # Get all sources with their cluster participation
    sources = db_conn.execute(
        "SELECT DISTINCT source FROM raw_posts"
    ).fetchall()

    # Total coordinated narratives (NVI >= 60)
    total_coordinated = db_conn.execute("""
        SELECT COUNT(DISTINCT nv.cluster_id) FROM nvi_snapshots nv
        WHERE nv.nvi_score >= 60
        AND nv.id IN (SELECT MAX(id) FROM nvi_snapshots GROUP BY cluster_id)
    """).fetchone()[0]

    if total_coordinated == 0:
        return 0

    adjusted = 0
    for row in sources:
        domain = row["source"]
        category, base_score = get_source_category(domain)

        # 1. Narrative participation rate
        source_coordinated = db_conn.execute("""
            SELECT COUNT(DISTINCT cm.cluster_id) FROM cluster_members cm
            JOIN raw_posts rp ON rp.id = cm.post_id
            JOIN nvi_snapshots nv ON nv.cluster_id = cm.cluster_id
            WHERE rp.source = ?
            AND nv.nvi_score >= 60
            AND nv.id IN (SELECT MAX(id) FROM nvi_snapshots GROUP BY cluster_id)
        """, (domain,)).fetchone()[0]

        participation_rate = source_coordinated / total_coordinated if total_coordinated > 0 else 0

        # 2. Content originality (avg mutation in source's clusters)
        avg_mutation = db_conn.execute("""
            SELECT AVG(nv.mutation_penalty) FROM nvi_snapshots nv
            JOIN cluster_members cm ON cm.cluster_id = nv.cluster_id
            JOIN raw_posts rp ON rp.id = cm.post_id
            WHERE rp.source = ?
            AND nv.id IN (SELECT MAX(id) FROM nvi_snapshots GROUP BY cluster_id)
        """, (domain,)).fetchone()[0] or 0.5

        # 3. Timing independence (avg coordination mult in source's clusters)
        avg_coord = db_conn.execute("""
            SELECT AVG(nv.coordination_mult) FROM nvi_snapshots nv
            JOIN cluster_members cm ON cm.cluster_id = nv.cluster_id
            JOIN raw_posts rp ON rp.id = cm.post_id
            WHERE rp.source = ?
            AND nv.id IN (SELECT MAX(id) FROM nvi_snapshots GROUP BY cluster_id)
        """, (domain,)).fetchone()[0] or 1.0

        # Dynamic adjustment formula
        adjusted_score = base_score
        adjusted_score *= (1.0 - 0.2 * min(1.0, participation_rate * 5))
        adjusted_score *= (0.8 + 0.2 * min(1.0, avg_mutation))
        adjusted_score *= (1.0 - 0.1 * max(0, avg_coord - 1.0))

        # Clamp
        adjusted_score = max(0.05, min(0.99, adjusted_score))

        evidence = f"curated_registry+dynamic(participation={participation_rate:.3f},mutation={avg_mutation:.3f},coord={avg_coord:.3f})"
        upsert_source_score(db_conn, domain, round(adjusted_score, 4), evidence, category)
        adjusted += 1

    logger.info(f"Dynamically adjusted {adjusted} source scores")
    return adjusted


def get_cluster_source_breakdown(db_conn, cluster_id: int) -> dict:
    """
    Get a credibility breakdown for all sources in a cluster.
    Extracts actual media domains from post URLs for accurate scoring.
    """
    rows = db_conn.execute("""
        SELECT DISTINCT rp.url FROM cluster_members cm
        JOIN raw_posts rp ON rp.id = cm.post_id
        WHERE cm.cluster_id = ? AND rp.url IS NOT NULL AND rp.url != ''
    """, (cluster_id,)).fetchall()

    if not rows:
        return {
            "categories": {},
            "weighted_credibility": 0.5,
            "source_count": 0,
            "sources": [],
        }

    categories = Counter()
    total_credibility = 0
    source_details = []
    seen_domains = set()

    for r in rows:
        domain = _extract_domain(r["url"])
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)

        # Look up in source_scores table first
        score_row = db_conn.execute(
            "SELECT credibility_score, category FROM source_scores WHERE domain = ?",
            (domain,)
        ).fetchone()

        if score_row:
            cat = score_row["category"] or "unknown"
            cred = score_row["credibility_score"]
        else:
            cat, cred = get_source_category(domain)

        categories[cat] += 1
        total_credibility += cred
        source_details.append({
            "domain": domain,
            "category": cat,
            "credibility": round(cred, 3),
        })

    weighted = total_credibility / len(source_details) if source_details else 0.5

    return {
        "categories": dict(categories),
        "weighted_credibility": round(weighted, 4),
        "source_count": len(source_details),
        "sources": sorted(source_details, key=lambda x: x["credibility"], reverse=True),
    }

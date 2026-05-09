"""
BurnTheLies Intelligence — Interpretation Engine

Translates raw NVI signals into human-readable assessments.
Every statement is a DETERMINISTIC derivation from measured data.
No generation. No fabrication. Pure threshold-based translation.

If a 12-year-old cannot understand it in 10 seconds, rewrite it.
"""

from typing import Optional


# ─── Human Metric Names ──────────────────────────────────────────────────────

METRIC_NAMES = {
    "nvi_score": "Narrative Velocity",
    "burst_zscore": "Speed of Spread",
    "spread_factor": "Source Independence",
    "mutation_penalty": "Content Similarity",
    "coordination_mult": "Timing Pattern",
}

METRIC_DESCRIPTIONS = {
    "nvi_score": "How fast and widely this narrative is spreading across sources, from 0 (minimal) to 100 (critical velocity).",
    "burst_zscore": "How fast this narrative is spreading compared to normal. Higher means faster.",
    "spread_factor": "How many independent sources are carrying this. Higher means more diverse.",
    "mutation_penalty": "How similar the content is across sources. Low means near-identical (shared origin). High means varied (independent reporting).",
    "coordination_mult": "Whether sources are publishing at unusually regular intervals.",
}


# ─── Alert Level Labels ──────────────────────────────────────────────────────

ALERT_LABELS = {
    "critical": {
        "label": "Signal: Critical",
        "color": "red",
        "icon": "critical",
    },
    "elevated": {
        "label": "Signal: Elevated",
        "color": "amber",
        "icon": "elevated",
    },
    "normal": {
        "label": "Normal Coverage",
        "color": "green",
        "icon": "normal",
    },
}


# ─── Interpretation Functions ─────────────────────────────────────────────────

def interpret_narrative(nvi_score: float, burst_zscore: float,
                        spread_factor: float, mutation_penalty: float,
                        coordination_mult: float, post_count: int = 0,
                        source_diversity: float = 0.0) -> dict:
    """
    Generate a complete 4-layer interpretation from raw NVI components.

    Returns:
        {
            "alert": { label, color, icon },
            "layer1_insight": str,      # Plain English, one block
            "layer2_risk": [str],       # Risk consequences
            "layer3_action": [str],     # What to do
            "metrics": [{ name, human_name, value, human_value, explanation }],
            "confidence_note": str,     # Honest assessment of our own confidence
        }

    Every string is deterministically derived from the input numbers.
    Nothing is fabricated.
    """

    # ── Determine alert level ──
    if nvi_score >= 80:
        alert = ALERT_LABELS["critical"]
    elif nvi_score >= 50:
        alert = ALERT_LABELS["elevated"]
    else:
        alert = ALERT_LABELS["normal"]

    # ── Layer 1: Immediate Insight ──
    insight = _generate_insight(nvi_score, burst_zscore, spread_factor,
                                 mutation_penalty, coordination_mult, post_count)

    # ── Layer 2: Risk Interpretation ──
    risks = _generate_risks(nvi_score, burst_zscore, spread_factor,
                             mutation_penalty, coordination_mult)

    # ── Layer 3: Action Directive ──
    actions = _generate_actions(nvi_score, burst_zscore, mutation_penalty,
                                 spread_factor)

    # ── Metrics with human translations ──
    metrics = [
        {
            "name": "nvi_score",
            "human_name": METRIC_NAMES["nvi_score"],
            "value": nvi_score,
            "human_value": _humanize_coordination(nvi_score),
            "explanation": _explain_nvi(nvi_score),
        },
        {
            "name": "burst_zscore",
            "human_name": METRIC_NAMES["burst_zscore"],
            "value": burst_zscore,
            "human_value": _humanize_speed(burst_zscore),
            "explanation": _explain_burst(burst_zscore),
        },
        {
            "name": "spread_factor",
            "human_name": METRIC_NAMES["spread_factor"],
            "value": spread_factor,
            "human_value": _humanize_independence(spread_factor),
            "explanation": _explain_spread(spread_factor),
        },
        {
            "name": "mutation_penalty",
            "human_name": METRIC_NAMES["mutation_penalty"],
            "value": mutation_penalty,
            "human_value": _humanize_similarity(mutation_penalty),
            "explanation": _explain_mutation(mutation_penalty),
        },
        {
            "name": "coordination_mult",
            "human_name": METRIC_NAMES["coordination_mult"],
            "value": coordination_mult,
            "human_value": _humanize_timing(coordination_mult),
            "explanation": _explain_coordination(coordination_mult),
        },
    ]

    # ── Confidence note (honesty about our own limitations) ──
    confidence = _assess_confidence(post_count, source_diversity, spread_factor)

    # ── Confidence interval (quantitative) ──
    confidence_interval = compute_confidence_interval(
        nvi_score, burst_zscore, spread_factor, mutation_penalty,
        coordination_mult, post_count, source_diversity
    )

    # ── Alternative hypotheses (intellectual honesty) ──
    alternatives = _generate_alternatives(
        nvi_score, burst_zscore, spread_factor, mutation_penalty, coordination_mult
    )

    return {
        "alert": alert,
        "layer1_insight": insight,
        "layer2_risk": risks,
        "layer3_action": actions,
        "metrics": metrics,
        "confidence_note": confidence,
        "confidence_interval": confidence_interval,
        "alternative_hypotheses": alternatives,
    }


# ─── Layer 1: Insight Generation ─────────────────────────────────────────────

def _generate_insight(nvi: float, burst: float, spread: float,
                       mutation: float, coord: float, post_count: int) -> str:
    """One clear paragraph. No jargon."""

    # High velocity + near-identical content across sources
    if nvi >= 80 and mutation < 0.15:
        return (
            "This topic is spreading with near-identical reporting "
            "across multiple sources. The content shows very little variation, "
            "which typically indicates shared source material (wire service, "
            "press release, or embargoed briefing). "
            f"{post_count} sources are carrying this narrative."
        )

    # High velocity + varied content = organic spread
    if nvi >= 80 and mutation >= 0.15:
        return (
            "This topic is generating broad attention across multiple sources. "
            "Content varies significantly between sources, "
            f"suggesting independent editorial coverage. {post_count} sources tracked."
        )

    # High burst but low spread = single-source surge
    if burst > 10 and spread < 0.15:
        return (
            "A sudden surge of reports from a narrow set of sources. "
            "The information has not yet spread to independent outlets. "
            "This could be a breaking story that has not been independently confirmed, "
            "or a planned release from related outlets."
        )

    # Moderate velocity
    if nvi >= 50:
        return (
            "This topic shows above-normal activity patterns. "
            f"Monitoring {post_count} sources for further development."
        )

    # Elevated timing regularity
    if coord > 1.3:
        return (
            "Sources are publishing at unusually regular intervals on this topic. "
            "This timing pattern can occur during scheduled news cycles or "
            "when embargoed content is released simultaneously."
        )

    # Normal
    return (
        f"This topic is showing normal activity patterns across {post_count} sources."
    )


# ─── Layer 2: Risk Generation ────────────────────────────────────────────────

def _generate_risks(nvi: float, burst: float, spread: float,
                     mutation: float, coord: float) -> list[str]:
    """Translate signals into consequences people understand."""
    risks = []

    if nvi >= 80:
        risks.append("Early reports may be incomplete, misleading, or biased.")
        risks.append("Public perception is being shaped rapidly — first impressions will dominate.")

    if mutation < 0.1:
        risks.append(
            "Content is nearly identical across sources. "
            "This means you may be reading the same claim repeated, not confirmed."
        )

    if spread < 0.15:
        risks.append(
            "Independent confirmation is weak. "
            "Few unrelated sources are covering this."
        )

    if burst > 15:
        risks.append(
            "The speed of spread is extremely high. "
            "Information is moving faster than verification."
        )

    if coord > 1.5:
        risks.append(
            "Publishing timing is unusually synchronized. "
            "This can indicate embargoed or scheduled releases."
        )

    if nvi >= 50 and not risks:
        risks.append("Activity is above normal. Situation is developing.")

    if not risks:
        risks.append("No elevated risk signals at this time.")

    return risks


# ─── Layer 3: Action Generation ──────────────────────────────────────────────

def _generate_actions(nvi: float, burst: float, mutation: float,
                       spread: float) -> list[str]:
    """Tell the user what to do. Concrete, specific."""
    actions = []

    if nvi >= 80:
        actions.append("Wait for independent confirmation before trusting specific details.")
        actions.append("Look for reporting from unrelated sources with different wording.")
        actions.append("Be cautious sharing this information — it may change significantly.")

    if mutation < 0.1:
        actions.append(
            "Check whether sources are independently reporting or copying the same original."
        )

    if spread < 0.15 and nvi >= 50:
        actions.append("Seek out sources from different countries or media ecosystems.")

    if burst > 20:
        actions.append("The story is moving very fast. Key details are likely to change within hours.")

    if nvi < 50:
        actions.append("No special caution needed at this time.")

    return actions


# ─── Metric Humanization ─────────────────────────────────────────────────────

def _humanize_coordination(nvi: float) -> str:
    if nvi >= 90: return "Extreme"
    if nvi >= 80: return "Very High"
    if nvi >= 60: return "High"
    if nvi >= 40: return "Moderate"
    if nvi >= 20: return "Low"
    return "Minimal"


def _humanize_speed(burst: float) -> str:
    if burst >= 30: return "Explosive"
    if burst >= 15: return "Very Fast"
    if burst >= 5: return "Fast"
    if burst >= 2: return "Above Normal"
    if burst >= 0: return "Normal"
    return "Declining"


def _humanize_independence(spread: float) -> str:
    if spread >= 0.7: return "Highly Independent"
    if spread >= 0.4: return "Moderately Independent"
    if spread >= 0.15: return "Limited Independence"
    return "Single Ecosystem"


def _humanize_similarity(mutation: float) -> str:
    # Note: LOW mutation = HIGH similarity (suspicious)
    if mutation < 0.05: return "Nearly Identical"
    if mutation < 0.1: return "Very Similar"
    if mutation < 0.2: return "Similar"
    if mutation < 0.4: return "Moderately Varied"
    return "Highly Varied"


def _humanize_timing(coord: float) -> str:
    if coord >= 1.7: return "Highly Synchronized"
    if coord >= 1.3: return "Suspicious Timing"
    if coord >= 1.1: return "Slightly Regular"
    return "Normal Timing"


# ─── Metric Explanations ─────────────────────────────────────────────────────

def _explain_nvi(nvi: float) -> str:
    if nvi >= 80:
        return "Multiple strong signals indicate high narrative velocity across sources."
    if nvi >= 50:
        return "Above-normal narrative velocity detected."
    return "This narrative is spreading through normal channels."


def _explain_burst(burst: float) -> str:
    if burst >= 15:
        return "This topic appeared in sources far faster than normal. Information is outpacing verification."
    if burst >= 5:
        return "Spreading faster than typical news. Worth monitoring."
    return "Spreading at a normal rate."


def _explain_spread(spread: float) -> str:
    if spread >= 0.4:
        return "Multiple independent source types are carrying this. That is a sign of genuine interest."
    if spread >= 0.15:
        return "Limited to a few types of sources. Independent confirmation is developing."
    return "Concentrated in a single source ecosystem. Cross-check with unrelated outlets."


def _explain_mutation(mutation: float) -> str:
    if mutation < 0.1:
        return (
            "Sources are using nearly identical language. This typically means they share "
            "a common origin — a press release, wire service, or embargoed briefing."
        )
    if mutation < 0.25:
        return "Moderate similarity between reports. Some independent analysis is present."
    return "Content varies significantly. Sources appear to be reporting independently."


def _explain_coordination(coord: float) -> str:
    if coord >= 1.5:
        return (
            "Sources published at suspiciously regular intervals. "
            "This pattern is uncommon in natural news cycles."
        )
    if coord >= 1.1:
        return "Slight timing regularity detected. Could be coincidence or scheduled publishing."
    return "No unusual timing patterns. Sources published at varied intervals."


# ─── Confidence Assessment ────────────────────────────────────────────────────

def _assess_confidence(post_count: int, source_diversity: float,
                        spread: float) -> str:
    """Honest assessment of how much to trust our own analysis."""

    if post_count < 10:
        return (
            "Low confidence: This assessment is based on very few sources. "
            "Results may change significantly as more data arrives."
        )

    if source_diversity < 0.3 and spread < 0.15:
        return (
            "Moderate confidence: Data comes primarily from one source type. "
            "Signal reliability is limited without broader source diversity."
        )

    if post_count >= 50 and source_diversity >= 0.5:
        return (
            "High confidence: Assessment is based on a large sample across "
            "multiple independent source types."
        )

    return (
        "Moderate confidence: Reasonable data volume but limited source diversity. "
        "Treat specific numbers as estimates, not measurements."
    )


# ─── Confidence Interval (Quantitative Honesty) ─────────────────────────────

def compute_confidence_interval(nvi_score: float, burst: float, spread: float,
                                 mutation: float, coord: float,
                                 post_count: int = 0,
                                 source_diversity: float = 0.0,
                                 dna_match_count: int = 0,
                                 ensemble_red_flag: bool = False,
                                 gdelt_batch_artifact: bool = False,
                                 unique_domain_count: Optional[int] = None,
                                 language_spread: Optional[dict] = None) -> dict:
    """
    Compute a deterministic probability estimate for narrative velocity.
    NOT ML — this is a signal-strength aggregation function.

    Returns confidence interval with limiting factors.
    """
    from math import sqrt

    # ── Falsification criteria (informational only in v5) ───────────────
    # In v4.1 these capped probability at 0.40-0.50. In v5 the equivalent
    # conditions are enforced as hard NVI caps in gates.py (gates 2/4/8/9),
    # so capping confidence here too would be double jeopardy. The list is
    # kept for legacy consumers (evidence pack, /verdict endpoint) that
    # surface "criteria triggered" alongside the gate trace.
    falsification_criteria_triggered = []

    if mutation > 0.30 and source_diversity > 0.60:
        falsification_criteria_triggered.append(
            "C1: Organic spread — mutation={:.2f} + source_diversity={:.2f} indicate genuine editorial variation across outlets".format(
                mutation, source_diversity))

    if coord < 1.1 and burst < 5 and mutation >= 0.1 and spread >= 0.15:
        falsification_criteria_triggered.append(
            "C2: Normal coverage pattern — coord={:.2f} + burst={:.1f} match standard daily news rhythm "
            "AND content variation (mutation={:.2f}>=0.1, spread={:.2f}>=0.15). "
            "This looks like routine coverage.".format(
                coord, burst, mutation, spread))

    if gdelt_batch_artifact:
        falsification_criteria_triggered.append(
            "C3: Wire service syndication — temporal signals originate from batch distribution cycle")

    if post_count < 5:
        falsification_criteria_triggered.append(
            "C4: Insufficient evidence — post_count={} is below minimum threshold for reliable analysis".format(post_count))

    probability = 0.55  # Base: better than coin flip since NVI was computed from real signals

    # Strong signals each add ~0.12
    if burst > 10:
        probability += 0.12
    elif burst > 5:
        probability += 0.07

    if mutation < 0.1:
        probability += 0.12
    elif mutation < 0.2:
        probability += 0.07

    if coord > 1.5:
        probability += 0.12
    elif coord > 1.2:
        probability += 0.07

    if spread < 0.15:
        probability += 0.12
    elif spread < 0.3:
        probability += 0.07

    # NVI itself is a composite — boost for high composite scores
    if nvi_score >= 80:
        probability += 0.08
    elif nvi_score >= 60:
        probability += 0.05

    # Data quality — use a single factor (worst applicable) instead of stacking
    limiting_factors = []
    quality_factor = 1.0

    if post_count < 10:
        quality_factor = min(quality_factor, 0.88)
        limiting_factors.append("Small sample size (fewer than 10 posts)")
        # Wire service pattern on tiny sample = almost certainly false
        if source_diversity < 0.3 and mutation < 0.1:
            probability *= 0.90
            limiting_factors.append(
                "Wire service pattern on tiny sample — near-identical content from few sources suggests syndicated origin")
    elif post_count < 25:
        quality_factor = min(quality_factor, 0.93)
        limiting_factors.append("Limited sample size")

    if source_diversity < 0.3:
        quality_factor = min(quality_factor, 0.90)
        limiting_factors.append("Low cross-platform diversity — mostly single source type")

    if spread > 0.5 and mutation > 0.3:
        quality_factor = min(quality_factor, 0.88)
        limiting_factors.append("High content variation suggests organic spread")

    # ── GDELT batch artifact penalty ──
    # When all temporal signals come from GDELT's 15-min batch pipeline,
    # the coordination evidence is fundamentally compromised.
    if gdelt_batch_artifact:
        quality_factor = min(quality_factor, 0.82)
        limiting_factors.append(
            "Temporal signals originate from GDELT's 15-minute batch cycle — not real-time publishing")

    # ── Ensemble perfect agreement (v5: handled as NVI gate, not confidence multiplier) ──
    # In v4.1 this multiplied quality_factor by 0.85. In v5, gate 5
    # (ensemble_uncertainty) caps NVI at 35 directly when this fires on
    # small samples, so we keep the limiting_factors note for transparency
    # but don't double-penalise the confidence number.
    if ensemble_red_flag:
        limiting_factors.append(
            "All three ensemble weighting schemes produced identical scores on a small sample — "
            "they're measuring the same artifact, not independent signals.")

    # ── DNA match evidence ──
    # Real coordinated campaigns leave persistent operational fingerprints.
    # Zero DNA matches on small clusters strongly suggests false positive.
    if dna_match_count == -1:
        # Not yet computed — neutral, no penalty or bonus
        pass
    elif dna_match_count == 0 and post_count < 20:
        quality_factor = min(quality_factor, 0.75)
        limiting_factors.append(
            "No DNA fingerprint matches — no persistent content relationship across clusters")
    elif dna_match_count == 0 and post_count < 50:
        quality_factor = min(quality_factor, 0.85)
        limiting_factors.append(
            "No DNA fingerprint matches — limited cross-cluster content relationship evidence")
    elif dna_match_count >= 10:
        # Strong DNA evidence — persistent fingerprint across many clusters
        probability += 0.10
        limiting_factors.append(
            "Strong DNA evidence — persistent content fingerprint across 10+ clusters")
    elif dna_match_count >= 5:
        probability += 0.07
    elif dna_match_count >= 1:
        probability += 0.04

    probability *= quality_factor

    # Cap at 0.95 — never claim certainty. Falsification caps removed in v5
    # (gates.py enforces equivalent constraints as hard NVI caps).
    probability = min(0.95, max(0.05, probability))

    # Single-source / low-diversity confidence floors. Gates already cap NVI
    # for these conditions, but consumers (verdict, alert UI) read the
    # confidence number independently — without these caps a 21-post
    # single-domain cluster can still pass the 0.65 alert threshold.
    if source_diversity is not None and source_diversity < 0.20:
        probability *= 0.50
        limiting_factors.append("Single-source cluster — confidence halved")
    if unique_domain_count is not None and unique_domain_count <= 1:
        probability = min(probability, 0.40)
        limiting_factors.append("Only one domain present — hard confidence cap at 0.40")

    # Language-based confidence adjustment. Cross-language coordination is a
    # stronger signal than single-language clusters (which are more likely to
    # be coincidental topic overlap, especially for non-English content).
    if language_spread:
        langs = [k for k, v in language_spread.items() if isinstance(v, (int, float)) and v > 0]
        if len(langs) > 1 and "translated" not in langs:
            probability *= 1.1
            limiting_factors.append(
                f"Cross-language cluster ({len(langs)} languages) — "
                "multi-language spread widens the signal; confidence boosted 10%"
            )

    probability = min(0.95, max(0.05, probability))

    # Compute bounds using sample-size-aware intervals
    effective_n = max(3, post_count)
    margin = 1.0 / sqrt(effective_n)
    lower = max(0.05, min(probability, probability * (1.0 - margin)))
    upper = min(0.99, max(probability, probability * (1.0 + 0.5 * margin)))

    # Sample adequacy
    if post_count >= 50 and source_diversity >= 0.4:
        adequacy = "sufficient"
    elif post_count >= 15:
        adequacy = "moderate"
    elif post_count >= 5:
        adequacy = "marginal"
    else:
        adequacy = "insufficient"

    if not limiting_factors:
        limiting_factors.append("No significant limitations identified")

    return {
        "probability": round(probability, 3),
        "lower_bound": round(lower, 3),
        "upper_bound": round(upper, 3),
        "sample_adequacy": adequacy,
        "limiting_factors": limiting_factors,
        "falsification_criteria_triggered": falsification_criteria_triggered,
    }


# ─── Alternative Hypotheses (Intellectual Honesty) ───────────────────────────

def _generate_alternatives(nvi: float, burst: float, spread: float,
                            mutation: float, coord: float) -> list[str]:
    """
    Generate alternative explanations for the same data.
    This is what makes the platform honest — it argues against itself.
    Every hypothesis is derivable from the signal values.
    """
    alternatives = []

    if mutation < 0.15:
        alternatives.append(
            "Wire service distribution: Low content variation is also consistent with "
            "AP/Reuters/AFP syndication, where many outlets publish the same wire copy."
        )

    if coord > 1.3:
        alternatives.append(
            "Editorial scheduling: Regular publication timing can reflect shared editorial "
            "deadlines across newsrooms in similar time zones, not necessarily publishing orchestration."
        )

    if burst > 10 and spread > 0.3:
        alternatives.append(
            "Genuine breaking news: High speed and broad reach can occur naturally "
            "when multiple independent newsrooms cover the same developing event."
        )

    if spread < 0.2:
        alternatives.append(
            "Niche topic: Concentrated sources may simply mean this topic is only covered "
            "by specialized outlets, not that coverage is being suppressed or restricted."
        )

    if nvi >= 70 and mutation > 0.25:
        alternatives.append(
            "Independent editorial coverage: High velocity with varied content "
            "suggests multiple outlets independently covering a genuinely trending topic."
        )

    if burst > 5 and burst < 15:
        alternatives.append(
            "News cycle effect: Moderate speed increases are common during peak "
            "publishing hours (9-11am, 2-4pm) in major media markets."
        )

    # Always include a base honesty statement
    if not alternatives:
        alternatives.append(
            "Normal activity: Current signal levels are within expected ranges."
        )

    return alternatives


# ─── Source Context Interpretation ───────────────────────────────────────────

def generate_source_context(source_scores: list[dict]) -> list[str]:
    """Generate human-readable source credibility assessment for a narrative."""
    if not source_scores:
        return ["Source credibility data not yet available for this narrative."]

    total = len(source_scores)
    categories = {}
    for s in source_scores:
        cat = s.get("category") or "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    context = []

    wire_count = categories.get("wire_service", 0)
    editorial_count = categories.get("major_editorial", 0)
    state_count = categories.get("state_media", 0)
    unknown_count = categories.get("unknown", 0)

    credible_pct = round((wire_count + editorial_count) / total * 100) if total else 0
    state_pct = round(state_count / total * 100) if total else 0
    unknown_pct = round(unknown_count / total * 100) if total else 0

    if credible_pct >= 60:
        context.append(
            f"{credible_pct}% of sources are established editorial organizations "
            f"or wire services. This narrative has strong institutional backing."
        )
    elif state_pct >= 30:
        context.append(
            f"{state_pct}% of sources carrying this narrative are classified as "
            f"state-affiliated media. Consider the editorial independence of these sources."
        )
    elif unknown_pct >= 50:
        context.append(
            f"{unknown_pct}% of sources are unverified or unknown outlets. "
            f"Independent confirmation from established media is limited."
        )
    else:
        context.append(
            f"Sources include a mix of {total} outlets across "
            f"{len(categories)} credibility categories."
        )

    return context


# ─── Cluster Label Cleaning ──────────────────────────────────────────────────

# Web artifacts that should never appear in narrative labels
URL_ARTIFACTS = {
    "html", "htm", "www", "com", "org", "net", "http", "https",
    "php", "asp", "aspx", "jsp", "index", "page", "article",
    "news", "story", "post", "blog", "the", "and", "for",
    "from", "with", "that", "this", "del", "per", "con",
    "les", "des", "den", "der", "die", "das", "und", "von",
    "info", "site", "media", "online", "digital", "press",
    "noticias", "journal", "gazette", "herald", "tribune",
    "times", "daily", "weekly", "topics", "people", "organizations",
    "tone", "themes",
}


# GDELT theme code → human-readable mapping
_GDELT_THEME_MAP = {
    "Soc Generalcrime": "Crime", "Soc Pointsofinterest": "Infrastructure",
    "Soc Pointsofinterest Hospital": "Hospitals", "Soc Pointsofinterest Airport": "Airports",
    "Soc Pointsofinterest School": "Schools", "Ungp Forests Rivers Oceans": "Environment",
    "Manmade Disaster Implied": "Industrial Disaster", "Econ Taxation": "Taxation",
    "Drug Trade": "Drug Trade", "Crime Illegal Drugs": "Illegal Drugs",
    "Information Warfare": "Information Warfare", "Media Censorship": "Censorship",
    "Armed Conflict": "Armed Conflict", "Armedconflict": "Armed Conflict",
    "Political Turmoil": "Political Turmoil", "Cyber Attack": "Cyber Attack",
    "Affect": "Humanitarian Impact", "Leader": "Political Leaders",
    "Protest": "Protests", "Military": "Military", "Election": "Elections",
    "Corruption": "Corruption", "Sanctions": "Sanctions",
    "Crisislex T03 Dead": "Casualties", "Medical": "Healthcare",
    "Education": "Education", "Religion": "Religion",
    "Evacuation": "Evacuations", "Arrest": "Arrests",
    "Security Services": "Security Forces",
}


def _clean_gdelt_theme(theme: str) -> str:
    """Convert GDELT theme codes to human-readable text."""
    if theme in _GDELT_THEME_MAP:
        return _GDELT_THEME_MAP[theme]
    # Strip common prefixes
    for prefix in ["Soc ", "Econ ", "Ungp ", "Crisislex "]:
        if theme.startswith(prefix):
            theme = theme[len(prefix):]
    return theme


def _is_domain_name(s: str) -> bool:
    """Check if a string looks like a domain name."""
    import re
    return bool(re.match(
        r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$', s.strip()
    ))


def clean_cluster_label(raw_label: str, keywords: list[str],
                         post_titles: list[str] = None,
                         themes: list[str] = None) -> str:
    """
    Generate a meaningful human-readable label from cluster data.
    Falls back through multiple strategies:
    1. Use themes (most reliable for GDELT data)
    2. Use the most informative post title (not domain names)
    3. Use cleaned keywords
    4. Use a generic but honest label
    """
    import html as _html
    from collections import Counter

    # Strategy 1: Prefer the most representative real article title. Themes
    # produce labels like "Corruption — Kill — Media Msm" — accurate codes,
    # useless to readers. A real headline always communicates more.
    if post_titles:
        good_titles = []
        for t in post_titles:
            if not t or len(t) < 15:
                continue
            if t.startswith("http") or _is_domain_name(t):
                continue
            if any(t.lower().endswith(ext) for ext in [".html", ".htm", ".php"]):
                continue
            # Decode any leftover HTML entities so the label reads as text.
            decoded = _html.unescape(t) if ("&#" in t or "&amp;" in t) else t
            # Reject titles that are still mostly entity-encoded after decode
            # (defensive — should be rare once ingest unescapes upstream).
            if "&#x" in decoded or "&#X" in decoded:
                continue
            # Strip wire-syndication suffix ("Title | Source Name")
            if " | " in decoded:
                decoded = decoded.split(" | ")[0].strip()
            if not decoded or len(decoded) < 10:
                continue
            good_titles.append(decoded.strip())
        if good_titles:
            title_counts = Counter(good_titles)
            best_title = title_counts.most_common(1)[0][0]
            if len(best_title) > 80:
                best_title = best_title[:77] + "…"
            return best_title

    # Strategy 2: Themes (fallback when no usable titles — common for
    # translation-only GKG records).
    if themes:
        clean_themes = [
            _clean_gdelt_theme(t) for t in themes
            if len(t) > 3
            and t.lower() not in URL_ARTIFACTS
            and not _is_domain_name(t)
        ]
        seen = set()
        unique_themes = []
        for t in clean_themes:
            if t not in seen:
                seen.add(t)
                unique_themes.append(t)
        if len(unique_themes) >= 1:
            return " — ".join(unique_themes[:3])

    # Strategy 3: Clean keywords and build a label
    clean_kw = [
        kw for kw in keywords
        if kw.lower() not in URL_ARTIFACTS
        and len(kw) > 2
        and not kw.startswith("http")
        and not kw.isdigit()
        and not _is_domain_name(kw)
    ]

    if len(clean_kw) >= 2:
        return " — ".join(clean_kw[:3]).title()

    # Strategy 4: Generic honest label
    return "Developing Narrative (Insufficient Data for Classification)"

"""
Synthetic adversarial benchmark for the v5.2 falsification gate pipeline.

Four gates, each catching a specific false-positive class with high
empirical discrimination. This benchmark proves the gate logic is correct
against hand-crafted adversarial scenarios.

Run with:
    python -m intelligence.validation.synthetic_benchmark
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from intelligence.processing.gates import (
    ClusterFeatures,
    apply_falsification_gates,
)


@dataclass
class Scenario:
    name: str
    description: str
    features: ClusterFeatures
    expected_gates: Set[str]
    expected_alert_suppressed: bool
    expected_max_nvi_cap: float
    is_true_positive: bool = False


def _base_features(**overrides) -> ClusterFeatures:
    defaults = dict(
        cluster_id=0, post_count=20,
        burst=1.0, spread=0.5, mutation=0.20, coordination=1.05,
        tone_uniformity=0.5, entity_concentration=0.30, shared_entity_count=3,
        unique_hash_ratio=1.0, embedding_similarity_mean=0.65,
        inter_arrival_mean=1800.0, inter_arrival_std=600.0,
        gdelt_fraction=0.7, source_diversity=0.55,
        gdelt_batch_artifact=False,
        wire_signal_known_syndicators=False,
        wire_signal_hash_diversity=False,
        dna_match_count=2, ensemble_disagreement=2.5,
        ensemble_perfect_agreement_red_flag=False,
        narrative_coherence=0.50, unique_domain_count=10,
        confidence_probability=0.75,
    )
    defaults.update(overrides)
    return ClusterFeatures(**defaults)


SCENARIOS: list[Scenario] = [
    # ── Gate 1: gdelt_batch_artifact ──
    Scenario(
        name="gdelt_batch_artifact",
        description=(
            "5 posts arriving at exactly 900-second intervals with zero "
            "variance — GDELT 15-min batching creates fake temporal sync."
        ),
        features=_base_features(
            cluster_id=1, post_count=5,
            inter_arrival_mean=900.0, inter_arrival_std=0.0,
            gdelt_fraction=1.0, gdelt_batch_artifact=True,
            coordination=1.0,
        ),
        expected_gates={"gdelt_batch_artifact"},
        expected_alert_suppressed=True,
        expected_max_nvi_cap=0.0,
    ),
    # ── Gate 2: insufficient_evidence ──
    Scenario(
        name="insufficient_evidence",
        description="6 posts across 4 domains — too small. confidence_threshold "
                    "also fires (prob=0.55 < 0.65) suppressing alert.",
        features=_base_features(
            cluster_id=2, post_count=6,
            unique_domain_count=4, source_diversity=0.40,
            confidence_probability=0.55,
        ),
        expected_gates={"insufficient_evidence", "confidence_threshold"},
        expected_alert_suppressed=True,
        expected_max_nvi_cap=40.0,
    ),
    # ── Gate 3: single_source_cluster ──
    Scenario(
        name="single_source_chip_de",
        description=(
            "Canonical cluster-39038: 21 posts, all from www.chip.de. "
            "Single-domain. Multiple gates fire: single_source (cap 15), "
            "dna_match (-1, cap 25), narrative_coherence (0.22<0.30, cap 20), "
            "normal_news_cycle (cap 35), confidence_threshold (0.40<0.65, "
            "suppressed). Min cap = 15, alert suppressed."
        ),
        features=_base_features(
            cluster_id=3, post_count=21,
            unique_domain_count=1, source_diversity=0.0,
            embedding_similarity_mean=0.88, mutation=0.03,
            narrative_coherence=0.22, dna_match_count=-1,
            confidence_probability=0.40,
        ),
        expected_gates={"single_source_cluster", "confidence_threshold"},
        expected_alert_suppressed=True,
        expected_max_nvi_cap=15.0,
    ),
    # ── Gate 4: wire_service ──
    Scenario(
        name="wire_service_syndication_known",
        description=(
            "AP/Reuters story syndicated by 25 outlets — low mutation, "
            "high domain diversity, known syndicator domains."
        ),
        features=_base_features(
            cluster_id=4, post_count=30,
            unique_domain_count=25, source_diversity=0.85,
            mutation=0.05, embedding_similarity_mean=0.92,
            wire_signal_known_syndicators=True, dna_match_count=0,
        ),
        expected_gates={"wire_service"},
        expected_alert_suppressed=False,
        expected_max_nvi_cap=25.0,
    ),
    Scenario(
        name="wire_service_hash_diversity",
        description=(
            "Wire-service rewritten by 30 outlets — every post byte-distinct "
            "but semantically near-identical. Cluster 5122/31610 class."
        ),
        features=_base_features(
            cluster_id=5, post_count=50,
            unique_domain_count=30, source_diversity=0.78,
            unique_hash_ratio=0.96, embedding_similarity_mean=0.84,
            wire_signal_hash_diversity=True, dna_match_count=0, mutation=0.10,
        ),
        expected_gates={"wire_service"},
        expected_alert_suppressed=False,
        expected_max_nvi_cap=25.0,
    ),
    # ── Combined: two gates fire ──
    Scenario(
        name="single_source_wire_combo",
        description=(
            "Edge case: wire-service pattern detected but from a single "
            "source. The stricter cap (single_source=15) should win."
        ),
        features=_base_features(
            cluster_id=6, post_count=15,
            unique_domain_count=1, source_diversity=0.0,
            wire_signal_hash_diversity=True,
            embedding_similarity_mean=0.86, dna_match_count=0,
        ),
        expected_gates={"single_source_cluster"},
        expected_alert_suppressed=False,
        expected_max_nvi_cap=15.0,  # stricter of 15 and 25
    ),
    # ── MULTIPLE GATES: insufficient + low diversity ──
    Scenario(
        name="small_single_source",
        description="5 posts, one domain — both insufficient_evidence and single_source fire.",
        features=_base_features(
            cluster_id=7, post_count=5,
            unique_domain_count=1, source_diversity=0.0,
            coordination=1.02,
        ),
        expected_gates={"insufficient_evidence", "single_source_cluster"},
        expected_alert_suppressed=False,
        expected_max_nvi_cap=15.0,  # min(40, 15)
    ),
    # ── TRUE POSITIVE ──
    Scenario(
        name="TRUE_COORDINATION_should_alert",
        description=(
            "THE CRITICAL POSITIVE TEST. 25 posts, 12 distinct domains, "
            "tightly synchronized, high coherence, low mutation, DNA matches "
            "4 prior campaigns. NO falsification gate should fire. Raw NVI "
            "must pass through uncapped. This is what the system exists to find."
        ),
        features=_base_features(
            cluster_id=8, post_count=25,
            unique_domain_count=12, source_diversity=0.70,
            burst=4.5, mutation=0.08, coordination=1.40,
            embedding_similarity_mean=0.91,
            narrative_coherence=0.78, entity_concentration=0.65,
            shared_entity_count=8, dna_match_count=4,
            ensemble_disagreement=2.8, confidence_probability=0.85,
            unique_hash_ratio=0.92,
        ),
        expected_gates=set(),
        expected_alert_suppressed=False,
        expected_max_nvi_cap=100.0,
        is_true_positive=True,
    ),
    # ── Normal organic news (no gate fires) ──
    Scenario(
        name="normal_organic_news",
        description=(
            "100 posts, 50 domains, moderate spread, normal temporal pattern. "
            "Doesn't trip any gate — should flow through with raw NVI."
        ),
        features=_base_features(
            cluster_id=9, post_count=100,
            unique_domain_count=50, source_diversity=0.70,
            coordination=1.05, burst=2.0, mutation=0.25,
            narrative_coherence=0.45, entity_concentration=0.20,
            dna_match_count=1, embedding_similarity_mean=0.60,
            unique_hash_ratio=0.94,
        ),
        expected_gates=set(),
        expected_alert_suppressed=False,
        expected_max_nvi_cap=100.0,
    ),
    # ── GDELT batch but diverse sources ──
    Scenario(
        name="gdelt_batch_terminal_zero",
        description=(
            "GDELT batch artifact detection is terminal — NVI is zeroed "
            "regardless of other features. Ensures the gate 1 path works."
        ),
        features=_base_features(
            cluster_id=10, post_count=10,
            unique_domain_count=15, source_diversity=0.70,
            gdelt_batch_artifact=True, gdelt_fraction=1.0,
            burst=30.0, mutation=0.50, coordination=5.0,
            wire_signal_hash_diversity=True, embedding_similarity_mean=0.90,
        ),
        expected_gates={"gdelt_batch_artifact"},
        expected_alert_suppressed=True,
        expected_max_nvi_cap=0.0,
    ),
]


def evaluate() -> dict:
    results = []
    gate_universe = {"gdelt_batch_artifact", "insufficient_evidence",
                     "single_source_cluster", "wire_service"}

    tp = {g: 0 for g in gate_universe}
    fp = {g: 0 for g in gate_universe}
    fn = {g: 0 for g in gate_universe}

    for s in SCENARIOS:
        gr = apply_falsification_gates(s.features)
        actual_gates = set(gr.gates_applied)

        for g in gate_universe:
            in_actual = g in actual_gates
            in_expected = g in s.expected_gates
            if in_actual and in_expected:
                tp[g] += 1
            elif in_actual and not in_expected:
                fp[g] += 1
            elif not in_actual and in_expected:
                fn[g] += 1

        expected_subset = s.expected_gates.issubset(actual_gates)
        effective_suppressed = gr.alert_suppressed or gr.nvi_zero
        effective_cap = 0.0 if gr.nvi_zero else gr.nvi_cap

        if s.is_true_positive:
            scenario_pass = (
                not gr.alert_suppressed and not gr.nvi_zero
                and gr.nvi_cap >= 50.0 and len(actual_gates) == 0
            )
        else:
            scenario_pass = (
                expected_subset
                and effective_suppressed == s.expected_alert_suppressed
                and effective_cap <= s.expected_max_nvi_cap + 0.01
            )

        results.append({
            "name": s.name,
            "expected_gates": sorted(s.expected_gates),
            "actual_gates": sorted(actual_gates),
            "expected_suppressed": s.expected_alert_suppressed,
            "actual_suppressed": gr.alert_suppressed,
            "nvi_zero": gr.nvi_zero,
            "expected_max_cap": s.expected_max_nvi_cap,
            "actual_cap": effective_cap,
            "pass": scenario_pass,
        })

    per_gate_metrics = {}
    for g in sorted(gate_universe):
        precision = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) else 1.0
        recall = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_gate_metrics[g] = {
            "tp": tp[g], "fp": fp[g], "fn": fn[g],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    n_pass = sum(1 for r in results if r["pass"])
    return {
        "scenarios_total": len(results),
        "scenarios_pass": n_pass,
        "accuracy": round(n_pass / len(results), 3),
        "per_scenario": results,
        "per_gate": per_gate_metrics,
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# Synthetic Adversarial Benchmark — v5.2 Falsification Gates\n")
    lines.append(
        f"**{report['scenarios_pass']} / {report['scenarios_total']} scenarios "
        f"correctly classified ({100 * report['accuracy']:.1f}% accuracy)**\n"
    )
    lines.append("## Per-scenario results\n")
    lines.append("| Scenario | Expected | Actual | Suppr. E/A | Cap E/A | Pass |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["per_scenario"]:
        eg = ", ".join(r["expected_gates"]) or "(none — true positive)"
        ag = ", ".join(r["actual_gates"]) or "(none)"
        sup = f"{r['expected_suppressed']}/{r['actual_suppressed']}"
        cap = f"{r['expected_max_cap']}/{r['actual_cap']}"
        ok = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| `{r['name']}` | {eg} | {ag} | {sup} | {cap} | {ok} |")

    lines.append("\n## Per-gate precision / recall / F1\n")
    lines.append("| Gate | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g, m in sorted(report["per_gate"].items()):
        lines.append(
            f"| `{g}` | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = evaluate()
    print(render_markdown(report))
    if report["scenarios_pass"] != report["scenarios_total"]:
        import sys
        sys.exit(1)

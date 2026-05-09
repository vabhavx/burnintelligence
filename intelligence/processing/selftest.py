"""
Self-test that runs on pipeline startup. Quick sanity check: builds synthetic
ClusterFeatures for 4 key scenarios, runs the gate pipeline, and asserts
the correct gates fire. If it fails, the pipeline refuses to start.

Runs in < 10 ms. No DB, no network, no side effects.
"""
from __future__ import annotations

from intelligence.processing.gates import (
    ClusterFeatures,
    apply_falsification_gates,
)


def _make(**overrides) -> ClusterFeatures:
    d = dict(
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
    d.update(overrides)
    return ClusterFeatures(**d)


CHECKS = [
    {
        "name": "gdelt_batch_artifact → terminal zero",
        "features": _make(
            post_count=5, gdelt_batch_artifact=True,
            inter_arrival_mean=900.0, inter_arrival_std=0.0,
        ),
        "assert": lambda r: r.nvi_zero,  # terminal gate zeroes; nvi_cap stays at default
    },
    {
        "name": "single_source → cap 15",
        "features": _make(
            post_count=21, unique_domain_count=1, source_diversity=0.0,
        ),
        "assert": lambda r: (
            "single_source_cluster" in r.gates_applied and r.nvi_cap == 15.0
        ),
    },
    {
        "name": "wire_service → cap 25",
        "features": _make(
            post_count=30, wire_signal_known_syndicators=True,
            embedding_similarity_mean=0.92, mutation=0.05,
            dna_match_count=0,
        ),
        "assert": lambda r: (
            "wire_service" in r.gates_applied and r.nvi_cap == 25.0
        ),
    },
    {
        "name": "true coordination → no gates fire",
        "features": _make(
            post_count=25, unique_domain_count=12, source_diversity=0.70,
            burst=4.5, mutation=0.08, coordination=1.40,
            narrative_coherence=0.78, dna_match_count=4,
        ),
        "assert": lambda r: (
            len(r.gates_applied) == 0
            and not r.alert_suppressed
            and r.nvi_cap == 100.0
        ),
    },
    {
        "name": "insufficient_evidence → cap 40",
        "features": _make(post_count=6, coordination=1.02, burst=1.0),
        "assert": lambda r: (
            "insufficient_evidence" in r.gates_applied and r.nvi_cap <= 40.0
        ),
    },
    {
        "name": "combo: single_source beats wire_service at cap 15",
        "features": _make(
            post_count=15, unique_domain_count=1, source_diversity=0.0,
            wire_signal_hash_diversity=True, embedding_similarity_mean=0.86,
        ),
        "assert": lambda r: (
            "single_source_cluster" in r.gates_applied
            and r.nvi_cap == 15.0
        ),
    },
]


def run_selftest() -> dict:
    """Returns {"ok": True, "checks": N, "passed": N} or raises AssertionError."""
    passed = 0
    failures = []
    for check in CHECKS:
        try:
            result = apply_falsification_gates(check["features"])
            ok = check["assert"](result)
            if not ok:
                failures.append({
                    "name": check["name"],
                    "gates": result.gates_applied,
                    "cap": result.nvi_cap,
                    "nvi_zero": result.nvi_zero,
                    "suppressed": result.alert_suppressed,
                })
            else:
                passed += 1
        except Exception as e:
            failures.append({"name": check["name"], "error": str(e)})

    return {
        "ok": len(failures) == 0,
        "checks": len(CHECKS),
        "passed": passed,
        "failures": failures,
    }

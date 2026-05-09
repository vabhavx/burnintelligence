"""
Validation harness for the v5 falsification gate pipeline.

Loads `ground_truth_labels.json`, runs `compute_nvi()` against the live DB
on each fixture cluster, and compares the resulting `gates_applied` and
final NVI score to the expected values.

Output: a markdown table on stdout, plus a returned dict for programmatic
use (CI integration, regression bots).

This is NOT a unit test — it requires a populated `intelligence/intel.db`.
Use it as a regression check when modifying gate thresholds.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger("intelligence.validation")

DEFAULT_FIXTURES = Path(__file__).parent / "ground_truth_labels.json"


@dataclass
class FixtureResult:
    cluster_id: int
    expected_gates: set[str]
    expected_gates_exact: bool
    expected_nvi_max: float
    expected_alert_suppressed: Optional[bool]

    actual_gates: set[str]
    actual_nvi: float
    actual_alert_suppressed: bool

    cluster_exists: bool
    insufficient_data: bool

    notes: str = ""

    @property
    def gates_pass(self) -> bool:
        if not self.cluster_exists or self.insufficient_data:
            return False
        if self.expected_gates_exact:
            return self.actual_gates == self.expected_gates
        return self.expected_gates.issubset(self.actual_gates)

    @property
    def nvi_pass(self) -> bool:
        if not self.cluster_exists or self.insufficient_data:
            return False
        return self.actual_nvi <= self.expected_nvi_max

    @property
    def suppression_pass(self) -> bool:
        if self.expected_alert_suppressed is None:
            return True
        if not self.cluster_exists or self.insufficient_data:
            return False
        return self.actual_alert_suppressed == self.expected_alert_suppressed

    @property
    def overall_pass(self) -> bool:
        return self.gates_pass and self.nvi_pass and self.suppression_pass


def load_fixtures(path: Path = DEFAULT_FIXTURES) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["fixtures"]


def evaluate_fixture(db_conn, fixture: dict) -> FixtureResult:
    from intelligence.processing.nvi import compute_nvi

    cluster_id = fixture["cluster_id"]
    expected_gates = set(fixture.get("expected_gates", []))
    expected_gates_exact = bool(fixture.get("expected_gates_exact", False))
    expected_nvi_max = float(fixture["expected_nvi_max"])
    expected_alert_suppressed = fixture.get("expected_alert_suppressed")

    cluster_row = db_conn.execute(
        "SELECT id FROM narrative_clusters WHERE id = ?", (cluster_id,)
    ).fetchone()

    if not cluster_row:
        return FixtureResult(
            cluster_id=cluster_id,
            expected_gates=expected_gates,
            expected_gates_exact=expected_gates_exact,
            expected_nvi_max=expected_nvi_max,
            expected_alert_suppressed=expected_alert_suppressed,
            actual_gates=set(),
            actual_nvi=0.0,
            actual_alert_suppressed=False,
            cluster_exists=False,
            insufficient_data=False,
            notes=fixture.get("notes", ""),
        )

    result = compute_nvi(db_conn, cluster_id)

    if result.get("insufficient_data"):
        return FixtureResult(
            cluster_id=cluster_id,
            expected_gates=expected_gates,
            expected_gates_exact=expected_gates_exact,
            expected_nvi_max=expected_nvi_max,
            expected_alert_suppressed=expected_alert_suppressed,
            actual_gates=set(),
            actual_nvi=0.0,
            actual_alert_suppressed=True,
            cluster_exists=True,
            insufficient_data=True,
            notes=fixture.get("notes", ""),
        )

    return FixtureResult(
        cluster_id=cluster_id,
        expected_gates=expected_gates,
        expected_gates_exact=expected_gates_exact,
        expected_nvi_max=expected_nvi_max,
        expected_alert_suppressed=expected_alert_suppressed,
        actual_gates=set(result.get("gates_applied") or []),
        actual_nvi=float(result.get("nvi_score") or 0.0),
        actual_alert_suppressed=bool(result.get("alert_suppressed", False)),
        cluster_exists=True,
        insufficient_data=False,
        notes=fixture.get("notes", ""),
    )


def evaluate(db_path: Optional[str] = None,
             fixtures_path: Path = DEFAULT_FIXTURES) -> dict:
    """
    Run the harness against the live DB. Returns:
        {
          "fixtures": [{cluster_id, gates_pass, nvi_pass, ...}, ...],
          "summary": {total, passed, gate_recall_per_class, ...},
        }
    """
    from intelligence.db import get_connection, init_db

    if db_path:
        os.environ["INTEL_DB_PATH"] = db_path

    db = get_connection()
    init_db(db)

    fixtures = load_fixtures(fixtures_path)
    results = [evaluate_fixture(db, fx) for fx in fixtures]

    # Per-gate recall: for each gate that appears in any expected_gates set,
    # what fraction of fixtures that expected it actually saw it fire?
    expected_gate_universe: set[str] = set()
    for r in results:
        expected_gate_universe.update(r.expected_gates)

    gate_recall: dict[str, dict] = {}
    for gate_id in sorted(expected_gate_universe):
        expected_count = sum(1 for r in results if gate_id in r.expected_gates)
        fired_count = sum(
            1 for r in results
            if gate_id in r.expected_gates
            and r.cluster_exists and not r.insufficient_data
            and gate_id in r.actual_gates
        )
        gate_recall[gate_id] = {
            "expected": expected_count,
            "fired_when_expected": fired_count,
            "recall": (fired_count / expected_count) if expected_count else 0.0,
        }

    runnable = [r for r in results if r.cluster_exists and not r.insufficient_data]
    passed = sum(1 for r in runnable if r.overall_pass)

    summary = {
        "total_fixtures": len(results),
        "runnable": len(runnable),
        "missing_clusters": sum(1 for r in results if not r.cluster_exists),
        "insufficient_data": sum(1 for r in results if r.insufficient_data),
        "passed": passed,
        "failed": len(runnable) - passed,
        "gate_recall_per_class": gate_recall,
    }

    return {
        "fixtures": [
            {
                "cluster_id": r.cluster_id,
                "expected_gates": sorted(r.expected_gates),
                "actual_gates": sorted(r.actual_gates),
                "expected_nvi_max": r.expected_nvi_max,
                "actual_nvi": r.actual_nvi,
                "expected_alert_suppressed": r.expected_alert_suppressed,
                "actual_alert_suppressed": r.actual_alert_suppressed,
                "cluster_exists": r.cluster_exists,
                "insufficient_data": r.insufficient_data,
                "gates_pass": r.gates_pass,
                "nvi_pass": r.nvi_pass,
                "suppression_pass": r.suppression_pass,
                "overall_pass": r.overall_pass,
                "notes": r.notes,
            }
            for r in results
        ],
        "summary": summary,
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    summary = report["summary"]

    lines.append("# NVI Gate Pipeline Validation Report")
    lines.append("")
    lines.append(
        f"**{summary['passed']}/{summary['runnable']} runnable fixtures pass.** "
        f"({summary['missing_clusters']} clusters missing from DB; "
        f"{summary['insufficient_data']} returned insufficient_data.)"
    )
    lines.append("")
    lines.append("## Per-fixture results")
    lines.append("")
    lines.append("| Cluster | Gates pass | NVI pass | Suppression | Final NVI | Cap | Gates fired |")
    lines.append("|---|---|---|---|---|---|---|")
    for fx in report["fixtures"]:
        if not fx["cluster_exists"]:
            lines.append(
                f"| `{fx['cluster_id']}` | — | — | — | — | "
                f"{fx['expected_nvi_max']:.0f} | _cluster missing_ |"
            )
            continue
        if fx["insufficient_data"]:
            lines.append(
                f"| `{fx['cluster_id']}` | — | — | — | — | "
                f"{fx['expected_nvi_max']:.0f} | _insufficient_data_ |"
            )
            continue
        gates_str = ", ".join(fx["actual_gates"]) or "_none_"
        gp = "✓" if fx["gates_pass"] else "✗"
        np_ = "✓" if fx["nvi_pass"] else "✗"
        sp = "✓" if fx["suppression_pass"] else "✗"
        lines.append(
            f"| `{fx['cluster_id']}` | {gp} | {np_} | {sp} | "
            f"{fx['actual_nvi']:.1f} | {fx['expected_nvi_max']:.0f} | {gates_str} |"
        )

    lines.append("")
    lines.append("## Gate recall per class")
    lines.append("")
    lines.append("| Gate | Expected | Fired when expected | Recall |")
    lines.append("|---|---|---|---|")
    for gate_id, rec in summary["gate_recall_per_class"].items():
        lines.append(
            f"| `{gate_id}` | {rec['expected']} | {rec['fired_when_expected']} | "
            f"{rec['recall']:.2f} |"
        )

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    argv = argv if argv is not None else sys.argv[1:]

    fixtures_path = DEFAULT_FIXTURES
    if argv and argv[0]:
        fixtures_path = Path(argv[0])

    report = evaluate(fixtures_path=fixtures_path)
    print(render_markdown(report))

    summary = report["summary"]
    if summary["runnable"] == 0:
        return 2  # nothing ran — likely missing clusters
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

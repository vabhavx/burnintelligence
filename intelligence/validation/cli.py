"""
Validation CLI — fixture harness + regression baseline drift check.

Usage:
    python -m intelligence.validation.cli --mode fixtures
    python -m intelligence.validation.cli --mode regression
    python -m intelligence.validation.cli --mode all

Exit codes:
    0 = pass
    1 = at least one fixture / baseline cluster failed its expectations
    2 = nothing runnable (DB missing or no clusters available)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from intelligence.validation.evaluate import evaluate, render_markdown

logger = logging.getLogger("intelligence.validation.cli")

BASELINE_PATH = Path(__file__).parent / "regression_baseline.json"


def run_fixtures() -> tuple[int, str]:
    report = evaluate()
    md = render_markdown(report)
    summary = report["summary"]
    if summary["runnable"] == 0:
        return 2, md
    return (0 if summary["failed"] == 0 else 1), md


def run_regression() -> tuple[int, str]:
    """Re-score the top-100 baseline clusters and detect drift."""
    if not BASELINE_PATH.exists():
        return 2, f"# Regression check\n\n_No baseline at {BASELINE_PATH}._\n"

    baseline = json.loads(BASELINE_PATH.read_text())
    tolerance = float(baseline.get("tolerance_nvi", 5.0))
    clusters = baseline.get("clusters", [])
    if not clusters:
        return 2, "# Regression check\n\n_Baseline contains no clusters._\n"

    from intelligence.db import get_connection, init_db
    from intelligence.processing.nvi import compute_nvi

    db = get_connection()
    init_db(db)

    rows = []
    failures = 0
    runnable = 0
    for entry in clusters:
        cid = entry["cluster_id"]
        baseline_nvi = float(entry.get("nvi_score", 0.0))
        baseline_gates = set(entry.get("gates_applied") or [])

        cluster_row = db.execute(
            "SELECT id FROM narrative_clusters WHERE id = ?", (cid,)
        ).fetchone()
        if not cluster_row:
            rows.append((cid, baseline_nvi, None, baseline_gates, None, "missing"))
            continue

        try:
            result = compute_nvi(db, cid)
        except Exception as e:
            logger.warning("compute_nvi failed for %s: %s", cid, e)
            rows.append((cid, baseline_nvi, None, baseline_gates, None, f"error: {e}"))
            failures += 1
            continue

        if result.get("insufficient_data"):
            rows.append((cid, baseline_nvi, None, baseline_gates, None, "insufficient_data"))
            continue

        actual_nvi = float(result.get("nvi_score") or 0.0)
        actual_gates = set(result.get("gates_applied") or [])
        runnable += 1

        gates_drift = actual_gates != baseline_gates
        nvi_drift = abs(actual_nvi - baseline_nvi) > tolerance
        status = "ok"
        if gates_drift and nvi_drift:
            status = "drift: gates+nvi"
            failures += 1
        elif gates_drift:
            status = "drift: gates"
            failures += 1
        elif nvi_drift:
            status = f"drift: nvi (Δ={actual_nvi - baseline_nvi:+.1f})"
            failures += 1
        rows.append((cid, baseline_nvi, actual_nvi, baseline_gates, actual_gates, status))

    lines = [
        "# Regression Baseline Drift Report",
        "",
        f"Captured: {baseline.get('captured_at', '?')} • Tolerance: ±{tolerance} NVI points",
        f"**{runnable - failures}/{runnable} clusters within tolerance** "
        f"({failures} drifted; {len(clusters) - runnable - failures} not runnable).",
        "",
        "| Cluster | Baseline NVI | Actual NVI | Baseline gates | Actual gates | Status |",
        "|---|---|---|---|---|---|",
    ]
    for cid, b_nvi, a_nvi, b_gates, a_gates, status in rows:
        b_g = ", ".join(sorted(b_gates)) or "_none_"
        a_g = ", ".join(sorted(a_gates)) if a_gates is not None else "—"
        a_n = f"{a_nvi:.1f}" if a_nvi is not None else "—"
        lines.append(f"| `{cid}` | {b_nvi:.1f} | {a_n} | {b_g} | {a_g} | {status} |")

    md = "\n".join(lines) + "\n"
    if runnable == 0:
        return 2, md
    return (0 if failures == 0 else 1), md


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(prog="intelligence.validation.cli")
    parser.add_argument(
        "--mode",
        choices=("fixtures", "regression", "all"),
        default="fixtures",
        help="fixtures = ground_truth_labels harness; regression = baseline drift; all = both",
    )
    args = parser.parse_args(argv)

    if args.mode == "fixtures":
        code, md = run_fixtures()
        print(md)
        return code

    if args.mode == "regression":
        code, md = run_regression()
        print(md)
        return code

    # all
    fx_code, fx_md = run_fixtures()
    print(fx_md)
    print()
    rg_code, rg_md = run_regression()
    print(rg_md)
    if 1 in (fx_code, rg_code):
        return 1
    if fx_code == 2 and rg_code == 2:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
BurnTheLies Intelligence — Narrative Lifecycle Tracking

Classifies each narrative into a lifecycle phase based on NVI time series derivatives.
Phases: EMERGENCE → GROWTH → PEAK → SATURATION → DECAY → DORMANT

All deterministic. Computed from NVI snapshots, no ML.
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("intelligence.lifecycle")

# Phase definitions
PHASES = {
    "emergence": "Emerging",
    "growth": "Growing",
    "peak": "At Peak",
    "saturation": "Saturated",
    "decay": "Declining",
    "dormant": "Dormant",
}


def classify_lifecycle_phase(db_conn, cluster_id: int) -> dict:
    """
    Classify the current lifecycle phase of a narrative cluster.

    Returns:
        {
            "phase": str,           # emergence, growth, peak, saturation, decay, dormant
            "phase_label": str,     # Human-readable phase name
            "phase_since": str,     # ISO timestamp estimate of when this phase began
            "trajectory": str,      # accelerating, stable, decelerating
            "nvi_derivative": float, # rate of change per interval
            "projected_peak": str,  # estimated peak time if in growth phase
        }
    """
    from intelligence.db import update_lifecycle_phase

    # Get NVI time series
    snapshots = db_conn.execute("""
        SELECT timestamp, nvi_score, burst_zscore, spread_factor
        FROM nvi_snapshots
        WHERE cluster_id = ?
        ORDER BY timestamp ASC
    """, (cluster_id,)).fetchall()

    snapshots = [dict(s) for s in snapshots]

    # Get cluster metadata
    cluster = db_conn.execute("""
        SELECT first_seen, last_updated, post_count
        FROM narrative_clusters WHERE id = ?
    """, (cluster_id,)).fetchone()

    if not cluster:
        return _default_result("emergence")

    cluster = dict(cluster)
    post_count = cluster.get("post_count", 0)
    first_seen = cluster.get("first_seen", "")

    # Not enough data for phase classification
    if len(snapshots) < 2:
        result = _classify_from_metadata(first_seen, post_count)
        update_lifecycle_phase(db_conn, cluster_id, result["phase"], result)
        return result

    # Compute derivatives
    nvi_values = [s["nvi_score"] for s in snapshots]
    timestamps = []
    for s in snapshots:
        try:
            timestamps.append(
                datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
            )
        except (ValueError, TypeError):
            timestamps.append(datetime.now(timezone.utc))

    # First derivative: NVI change per interval
    derivatives = []
    for i in range(1, len(nvi_values)):
        dt = (timestamps[i] - timestamps[i-1]).total_seconds()
        if dt > 0:
            derivatives.append((nvi_values[i] - nvi_values[i-1]) / (dt / 3600))  # per hour
        else:
            derivatives.append(0.0)

    if not derivatives:
        result = _classify_from_metadata(first_seen, post_count)
        update_lifecycle_phase(db_conn, cluster_id, result["phase"], result)
        return result

    # Second derivative: acceleration
    second_derivatives = []
    for i in range(1, len(derivatives)):
        second_derivatives.append(derivatives[i] - derivatives[i-1])

    current_nvi = nvi_values[-1]
    peak_nvi = max(nvi_values)
    latest_derivative = derivatives[-1]
    recent_derivatives = derivatives[-3:] if len(derivatives) >= 3 else derivatives

    # Trajectory
    if len(second_derivatives) >= 1:
        avg_accel = sum(second_derivatives[-3:]) / len(second_derivatives[-3:])
        if avg_accel > 0.5:
            trajectory = "accelerating"
        elif avg_accel < -0.5:
            trajectory = "decelerating"
        else:
            trajectory = "stable"
    else:
        trajectory = "stable"

    # Phase classification
    now = datetime.now(timezone.utc)
    hours_since_first = 999
    try:
        first_ts = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
        hours_since_first = (now - first_ts).total_seconds() / 3600
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(
            f"first_seen parse failed cluster={cluster_id} "
            f"value={first_seen!r}: {e}"
        )

    # Check for recent activity
    hours_since_last_snapshot = 999
    if timestamps:
        hours_since_last_snapshot = (now - timestamps[-1]).total_seconds() / 3600

    # DORMANT: No new snapshots in 12+ hours OR NVI < 20
    if hours_since_last_snapshot > 12 or (current_nvi < 20 and hours_since_first > 12):
        phase = "dormant"
        phase_since = snapshots[-1]["timestamp"] if snapshots else first_seen

    # EMERGENCE: First 6 hours, small cluster
    elif hours_since_first < 6 and post_count < 15:
        phase = "emergence"
        phase_since = first_seen

    # GROWTH: NVI increasing over recent snapshots
    elif len(recent_derivatives) >= 2 and all(d > 0 for d in recent_derivatives):
        phase = "growth"
        # Find when growth started (last non-positive derivative)
        phase_since = first_seen
        for i in range(len(derivatives) - 1, -1, -1):
            if derivatives[i] <= 0 and i + 1 < len(snapshots):
                phase_since = snapshots[i + 1]["timestamp"]
                break

    # PEAK: NVI at or near maximum, derivative turning negative
    elif current_nvi >= peak_nvi * 0.9 and latest_derivative <= 0 and peak_nvi > 30:
        phase = "peak"
        # Peak started when NVI reached 90% of max
        phase_since = snapshots[-1]["timestamp"]
        for i, v in enumerate(nvi_values):
            if v >= peak_nvi * 0.9:
                phase_since = snapshots[i]["timestamp"]
                break

    # SATURATION: NVI stable (within 15% of peak for 3+ snapshots)
    elif (len(nvi_values) >= 3
          and all(abs(v - current_nvi) < peak_nvi * 0.15 for v in nvi_values[-3:])
          and peak_nvi > 30):
        phase = "saturation"
        phase_since = snapshots[-3]["timestamp"] if len(snapshots) >= 3 else first_seen

    # DECAY: NVI declining for recent snapshots
    elif len(recent_derivatives) >= 2 and all(d < 0 for d in recent_derivatives):
        phase = "decay"
        phase_since = snapshots[-1]["timestamp"]
        for i in range(len(derivatives) - 1, -1, -1):
            if derivatives[i] >= 0 and i + 1 < len(snapshots):
                phase_since = snapshots[i + 1]["timestamp"]
                break

    # Default: emergence
    else:
        phase = "emergence"
        phase_since = first_seen

    # Projected peak (if in growth phase)
    projected_peak = None
    if phase == "growth" and trajectory == "decelerating" and second_derivatives:
        avg_decel = abs(sum(second_derivatives[-3:]) / len(second_derivatives[-3:]))
        if avg_decel > 0 and latest_derivative > 0:
            hours_to_peak = latest_derivative / avg_decel
            projected_peak = (now + timedelta(hours=hours_to_peak)).isoformat()

    result = {
        "phase": phase,
        "phase_label": PHASES.get(phase, phase.title()),
        "phase_since": phase_since,
        "trajectory": trajectory,
        "nvi_derivative": round(latest_derivative, 4),
        "projected_peak": projected_peak,
        "current_nvi": round(current_nvi, 2),
        "peak_nvi": round(peak_nvi, 2),
        "snapshot_count": len(snapshots),
    }

    update_lifecycle_phase(db_conn, cluster_id, phase, result)
    return result


def classify_all_lifecycles(db_conn) -> list[dict]:
    """Classify lifecycle phases for all active clusters."""
    clusters = db_conn.execute("""
        SELECT id FROM narrative_clusters WHERE status = 'active'
    """).fetchall()

    results = []
    for c in clusters:
        result = classify_lifecycle_phase(db_conn, c["id"])
        result["cluster_id"] = c["id"]
        results.append(result)

    # Log summary
    phase_counts = {}
    for r in results:
        p = r["phase"]
        phase_counts[p] = phase_counts.get(p, 0) + 1

    logger.info(f"Lifecycle classification: {phase_counts}")
    return results


def _classify_from_metadata(first_seen: str, post_count: int) -> dict:
    """Fallback classification from cluster metadata only."""
    now = datetime.now(timezone.utc)
    hours = 999
    try:
        first_ts = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
        hours = (now - first_ts).total_seconds() / 3600
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"first_seen parse failed value={first_seen!r}: {e}")

    if hours < 6 and post_count < 15:
        phase = "emergence"
    elif post_count > 30:
        phase = "growth"
    else:
        phase = "emergence"

    return _default_result(phase, phase_since=first_seen)


def _default_result(phase: str = "emergence", phase_since: str = "") -> dict:
    return {
        "phase": phase,
        "phase_label": PHASES.get(phase, phase.title()),
        "phase_since": phase_since or datetime.now(timezone.utc).isoformat(),
        "trajectory": "stable",
        "nvi_derivative": 0.0,
        "projected_peak": None,
        "current_nvi": 0,
        "peak_nvi": 0,
        "snapshot_count": 0,
    }

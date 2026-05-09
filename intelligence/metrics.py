"""
Prometheus metrics for the Intelligence API and pipeline.

Exposes a /metrics endpoint via prometheus-fastapi-instrumentator and a small
set of custom counters/histograms that the pipeline calls into.

All helpers are no-ops if prometheus_client is unavailable, so import-time
failures cannot break the API.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("intelligence.metrics")

try:
    from prometheus_client import Counter, Histogram

    nvi_alerts_total = Counter(
        "nvi_alerts_total",
        "Total NVI alerts emitted, labelled by severity.",
        labelnames=("level",),
    )
    gates_fired_total = Counter(
        "gates_fired_total",
        "Total falsification gates that have fired.",
        labelnames=("gate",),
    )
    pipeline_cycle_duration_seconds = Histogram(
        "pipeline_cycle_duration_seconds",
        "Wall-clock duration of full pipeline cycles.",
        buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
    )
    pipeline_stage_failed_total = Counter(
        "pipeline_stage_failed_total",
        "Total pipeline stage failures, labelled by stage.",
        labelnames=("stage",),
    )

    _PROM_OK = True
except Exception as exc:  # pragma: no cover
    logger.warning("prometheus_client unavailable, metrics will be no-ops: %s", exc)
    nvi_alerts_total = None
    gates_fired_total = None
    pipeline_cycle_duration_seconds = None
    pipeline_stage_failed_total = None
    _PROM_OK = False


def record_alert(level: str) -> None:
    if _PROM_OK and nvi_alerts_total is not None:
        nvi_alerts_total.labels(level=level).inc()


def record_gate_fired(gate: str) -> None:
    if _PROM_OK and gates_fired_total is not None:
        gates_fired_total.labels(gate=gate).inc()


def record_cycle_duration(seconds: float) -> None:
    if _PROM_OK and pipeline_cycle_duration_seconds is not None:
        pipeline_cycle_duration_seconds.observe(seconds)


def record_stage_failure(stage: str) -> None:
    if _PROM_OK and pipeline_stage_failed_total is not None:
        pipeline_stage_failed_total.labels(stage=stage).inc()


def install(app) -> None:
    """Attach the Prometheus instrumentator and expose /metrics."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception as exc:  # pragma: no cover
        logger.warning("prometheus-fastapi-instrumentator unavailable: %s", exc)
        return

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

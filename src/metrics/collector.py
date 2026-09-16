"""Per-stage and overall metrics with immediate persistence."""

from __future__ import annotations

import math
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.utils.run_id import utc_now_iso


def peak_rss_bytes() -> int | None:
    """Peak resident set size of the process in bytes, when available."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if usage is None or usage <= 0:
        return None
    if usage < 10_000_000:
        return int(usage * 1024)
    return int(usage)


def current_rss_bytes() -> int | None:
    """Current resident set size in bytes using psutil when installed."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _summarize(values: list[float]) -> dict[str, float | int]:
    """Summary statistics for a numeric distribution."""
    if not values:
        return {"count": 0}
    count = len(values)
    total = float(sum(values))
    mean = total / count
    variance = float(sum((v - mean) ** 2 for v in values)) / count
    return {
        "count": count,
        "mean": round(mean, 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
        "total": round(total, 6),
        "stdev": round(math.sqrt(variance), 6),
    }


@dataclass
class StageMetrics:
    """Metrics accumulated for a single pipeline stage."""

    name: str
    start_iso: str = field(default_factory=utc_now_iso)
    end_iso: str | None = None
    runtime_s: float = 0.0
    counters: dict[str, float] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    distributions: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view with distributions summarised."""
        payload: dict[str, Any] = {
            "stage": self.name,
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "runtime_s": round(self.runtime_s, 4),
            "counters": {k: (int(v) if float(v).is_integer() else round(float(v), 6)) for k, v in self.counters.items()},
            "values": self.values,
        }
        if self.distributions:
            payload["distributions"] = {
                key: _summarize(list(values)) for key, values in self.distributions.items()
            }
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


class MetricsCollector:
    """Accumulates metrics and persists each stage as soon as it finishes.

    Persisting per stage means a crash mid-run still leaves diagnosable state,
    which the reliability requirements call for.
    """

    def __init__(self, store: Any | None = None, logger: Any | None = None) -> None:
        self.store = store
        self.logger = logger
        self.stages: dict[str, StageMetrics] = {}
        self.run_start_iso = utc_now_iso()
        self._run_start = time.perf_counter()
        self.metadata: dict[str, Any] = {}

    def _get(self, name: str) -> StageMetrics:
        stage = self.stages.get(name)
        if stage is None:
            stage = StageMetrics(name=name)
            self.stages[name] = stage
        return stage

    @contextmanager
    def timer(self, name: str) -> Iterator[StageMetrics]:
        """Time a stage and persist its metrics on exit."""
        stage = self._get(name)
        stage.start_iso = utc_now_iso()
        started = time.perf_counter()
        try:
            yield stage
        finally:
            stage.runtime_s = time.perf_counter() - started
            stage.end_iso = utc_now_iso()
            self.persist_stage(name)

    @property
    def stage(self) -> StageMetrics:
        """The most recently created stage, for use inside a timer."""
        if not self.stages:
            raise RuntimeError("no stage started yet")
        return list(self.stages.values())[-1]

    def increment(self, stage: str, key: str, amount: float = 1.0) -> None:
        """Add to a counter."""
        target = self._get(stage)
        target.counters[key] = target.counters.get(key, 0.0) + float(amount)

    def set(self, stage: str, key: str, value: Any) -> None:
        """Set a scalar value."""
        self._get(stage).values[key] = value

    def observe(self, stage: str, key: str, value: float) -> None:
        """Add a sample to a distribution."""
        if value is None:
            return
        numeric = float(value)
        if numeric != numeric:
            return
        self._get(stage).distributions.setdefault(key, []).append(numeric)

    def note(self, stage: str, message: str) -> None:
        """Attach a free-form note."""
        self._get(stage).notes.append(str(message))

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        """Merge a payload of counters/values into a stage."""
        target = self._get(stage)
        for key, value in payload.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    flattened = f"{key}.{inner_key}"
                    if isinstance(inner_value, (int, float)) and not isinstance(inner_value, bool):
                        target.distributions.setdefault(flattened, []).append(float(inner_value))
                    else:
                        target.values[flattened] = inner_value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                target.counters[key] = target.counters.get(key, 0.0) + float(value)
            else:
                target.values[key] = value

    def merge(self, stage: str, other: "MetricsCollector", other_stage: str | None = None) -> None:
        """Merge another collector's stage into this one."""
        source = other.stages.get(other_stage or stage)
        if source is None:
            return
        target = self._get(stage)
        for key, value in source.counters.items():
            target.counters[key] = target.counters.get(key, 0.0) + value
        target.values.update(source.values)
        for key, values in source.distributions.items():
            target.distributions.setdefault(key, []).extend(values)
        target.notes.extend(source.notes)

    def stage_dict(self, name: str) -> dict[str, Any]:
        """JSON-ready metrics for one stage."""
        return self._get(name).to_dict()

    def persist_stage(self, name: str) -> None:
        """Write a stage metrics file immediately."""
        if self.store is None:
            return
        try:
            self.store.write_json("metrics", f"{name}.json", self.stage_dict(name))
        except Exception:
            if self.logger is not None:
                self.logger.warning("failed to persist metrics for stage %s", name)

    def snapshot(self) -> dict[str, Any]:
        """Runtime counters for every stage so far."""
        return {
            "run_id": getattr(self.store, "run_id", None),
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
        }

    def persist_overall(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write ``metrics/overall.json``."""
        runtime = time.perf_counter() - self._run_start
        frames_processed = 0.0
        for stage in self.stages.values():
            value = stage.values.get("frames_processed") or stage.counters.get("frames_processed")
            if isinstance(value, (int, float)):
                frames_processed = max(frames_processed, float(value))
        payload: dict[str, Any] = {
            "run_id": getattr(self.store, "run_id", None),
            "run_start_iso": self.run_start_iso,
            "run_end_iso": utc_now_iso(),
            "total_runtime_s": round(runtime, 4),
            "throughput_fps": round(frames_processed / runtime, 6) if runtime > 0 and frames_processed else None,
            "peak_rss_bytes": peak_rss_bytes(),
            "current_rss_bytes": current_rss_bytes(),
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "metadata": self.metadata,
        }
        if extra:
            payload.update(extra)
        if self.store is not None:
            self.store.write_json("metrics", "overall.json", payload)
        return payload

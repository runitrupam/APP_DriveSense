"""Shared execution context passed to every pipeline stage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.metrics.collector import MetricsCollector
from src.output.writer import RunStore
from src.utils.config import Config

ProgressCallback = Callable[[str, int, int, str], None]


class ProgressReporter:
    """Throttled progress fan-out to the UI and the log."""

    def __init__(
        self,
        callback: ProgressCallback | None = None,
        logger: Any | None = None,
        min_interval_s: float = 0.25,
    ) -> None:
        self.callback = callback
        self.logger = logger
        self.min_interval_s = min_interval_s
        self._last_emit: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

    def report(self, stage: str, done: int, total: int, message: str = "") -> None:
        """Emit progress for a stage, throttled except on completion."""
        now = time.perf_counter()
        finished = total > 0 and done >= total
        last = self._last_emit.get(stage, 0.0)
        if not finished and (now - last) < self.min_interval_s:
            return
        self._last_emit[stage] = now
        payload = {"stage": stage, "done": int(done), "total": int(total), "message": message}
        self.events.append(payload)
        if self.callback is not None:
            try:
                self.callback(stage, int(done), int(total), message)
            except Exception:
                pass
        if self.logger is not None and (finished or done == 0):
            self.logger.info("stage=%s progress=%d/%d %s", stage, done, total, message)


@dataclass
class RunContext:
    """Everything a stage needs: config, storage, metrics, logging, progress."""

    config: Config
    store: RunStore
    metrics: MetricsCollector
    logger: Any
    progress: ProgressReporter = field(default_factory=ProgressReporter)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Identifier of the current execution."""
        return self.store.run_id

    @property
    def run_dir(self):
        """Filesystem location of the current execution."""
        return self.store.run_dir

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dotted config value."""
        return self.config.get(key, default)

    def flag(self, key: str, default: bool = False) -> bool:
        """Read a boolean config value, tolerating strings."""
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

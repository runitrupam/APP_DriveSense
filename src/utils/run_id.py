"""Run identifier and timestamp helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_run_id() -> str:
    """Build ``RUN_<UTCstamp>_<uuid8>`` for a new execution."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"RUN_{stamp}_{uuid.uuid4().hex[:8]}"


def timestamp_ms_for_frame(frame_id: int, fps: float) -> int:
    """Millisecond timestamp for a zero-based frame index.

    Uses rounded division on the nominal FPS so frame ids map to a stable,
    reproducible timeline instead of accumulating float drift.
    """
    if fps is None or fps <= 0:
        return int(frame_id)
    return int(round(frame_id * 1000.0 / float(fps)))


def ms_to_seconds(value: int | float | None) -> float | None:
    """Convert milliseconds to seconds."""
    if value is None:
        return None
    return round(float(value) / 1000.0, 3)

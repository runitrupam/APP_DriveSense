"""Rolling history plus smoothing that never erases a confirmed change.

Smoothing exists to suppress single-frame noise in lane and road state. Two
rules keep it honest:

* a value confirmed by the road analyzer's change tracker is always kept, even
  if it differs from the recent mode;
* a value recovered from history because the current frame had no evidence is
  flagged (``filled_from_history``) and carries reduced confidence, so consumers
  can see that the frame itself did not support it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.utils.context import RunContext


def mode_of(values: Iterable[Any]) -> Any:
    """Most common value, ignoring ``None``; ties resolve to the most recent."""
    counted = [value for value in values if value is not None]
    if not counted:
        return None
    counts = Counter(counted)
    best = max(counts.values())
    for value in reversed(counted):
        if counts[value] == best:
            return value
    return counted[-1]


@dataclass
class FrameHistoryEntry:
    """Rolling per-frame summary kept for temporal reasoning."""

    frame_id: int
    timestamp_ms: int
    object_ids: list[str] = field(default_factory=list)
    object_count: int = 0
    lane_count: int | None = None
    ego_lane: int | None = None
    road_type: str = "unknown"
    environment_counts: dict[str, int | None] = field(default_factory=dict)
    ego_motion: dict[str, Any] | None = None
    anomalies: list[str] = field(default_factory=list)


@dataclass
class SmoothedRoad:
    """Road state after temporal fusion."""

    lane_count: int | None
    ego_lane: int | None
    road_type: str
    reasons: dict[str, str] = field(default_factory=dict)
    filled_from_history: bool = False
    confidence_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary."""
        return {
            "lane_count": self.lane_count if self.lane_count is not None else "unknown",
            "ego_lane": self.ego_lane if self.ego_lane is not None else "unknown",
            "road_type": self.road_type,
            "reasons": dict(self.reasons),
            "filled_from_history": self.filled_from_history,
            "confidence_scale": round(float(self.confidence_scale), 4),
        }


class TemporalStateManager:
    """Maintains object, lane, road and environment history across a pass."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.smoothing_window = max(1, int(ctx.get("temporal.smoothing_window", 5)))
        self.preserve_changes = ctx.flag("temporal.preserve_confirmed_changes", True)
        self.disappearance_tolerance = int(ctx.get("temporal.object_disappearance_tolerance", 2))
        self.history: dict[int, FrameHistoryEntry] = {}
        self.confirmed_changes: dict[int, dict[str, Any]] = {}
        self.object_appearances: dict[str, list[int]] = {}
        self.object_disappearances: dict[str, int] = {}

    def reset(self) -> None:
        """Clear all history for a fresh pass."""
        self.history.clear()
        self.confirmed_changes.clear()
        self.object_appearances.clear()
        self.object_disappearances.clear()

    def record_frame(self, entry: FrameHistoryEntry) -> None:
        """Store a frame summary and update per-track appearance windows."""
        self.history[int(entry.frame_id)] = entry
        active = set(entry.object_ids)
        for track_id in active:
            self.object_appearances.setdefault(track_id, []).append(int(entry.frame_id))
        for track_id in list(self.object_disappearances):
            if track_id not in active:
                del self.object_disappearances[track_id]

    def register_events(self, frame_id: int, events: Iterable[Any]) -> None:
        """Record confirmed field changes so smoothing cannot undo them."""
        for event in events:
            if getattr(event, "type", "") == "lane_count_change":
                self.confirmed_changes.setdefault(int(frame_id), {})["lane_count"] = event.current
            elif getattr(event, "type", "") == "ego_lane_change":
                self.confirmed_changes.setdefault(int(frame_id), {})["ego_lane"] = event.current
            elif getattr(event, "type", "") == "road_type_change":
                self.confirmed_changes.setdefault(int(frame_id), {})["road_type"] = event.current
            elif getattr(event, "type", "") in {"merge", "split"}:
                self.confirmed_changes.setdefault(int(frame_id), {})["road_type"] = event.current

    def confirmed_value(self, field_name: str, frame_id: int) -> Any:
        """Latest confirmed value for a field at or before ``frame_id``."""
        best: Any = None
        for registered_frame in sorted(self.confirmed_changes):
            if registered_frame > frame_id:
                break
            if field_name in self.confirmed_changes[registered_frame]:
                best = self.confirmed_changes[registered_frame][field_name]
        return best

    def recent_values(self, field_name: str, frame_id: int, window: int | None = None) -> list[Any]:
        """Values of a field over the trailing window, oldest first."""
        size = self.smoothing_window if window is None else max(1, int(window))
        frames = [f for f in sorted(self.history) if f <= frame_id]
        selected = frames[-size:]
        return [getattr(self.history[f], field_name) for f in selected]

    def smooth_road(
        self,
        frame_id: int,
        lane_count: int | None,
        ego_lane: int | None,
        road_type: str | None,
    ) -> SmoothedRoad:
        """Fuse raw road values with history while preserving confirmed changes."""
        result = SmoothedRoad(lane_count=lane_count, ego_lane=ego_lane, road_type=road_type or "unknown")
        raw = {"lane_count": lane_count, "ego_lane": ego_lane, "road_type": road_type}
        for field_name, value in raw.items():
            confirmed = self.confirmed_value(field_name, frame_id)
            if self.preserve_changes and confirmed is not None:
                if value == confirmed or value is None:
                    result.reasons[field_name] = "confirmed_change_preserved"
                    if value is None:
                        self._assign(result, field_name, confirmed)
                        result.filled_from_history = True
                else:
                    self._assign(result, field_name, confirmed)
                    result.reasons[field_name] = "raw_disagrees_with_confirmed_change"
                    result.confidence_scale = min(result.confidence_scale, 0.8)
                continue

            if value is not None:
                history = self.recent_values(field_name, frame_id)
                if len(history) >= 2:
                    smoothed = mode_of(history + [value])
                    self._assign(result, field_name, smoothed)
                    result.reasons[field_name] = "smoothed_mode" if smoothed != value else "raw"
                    if smoothed != value:
                        result.confidence_scale = min(result.confidence_scale, 0.9)
                else:
                    result.reasons[field_name] = "raw"
                continue

            history = self.recent_values(field_name, frame_id)
            recovered = mode_of(history)
            if recovered is None:
                result.reasons[field_name] = "unknown_no_history"
            else:
                self._assign(result, field_name, recovered)
                result.reasons[field_name] = "filled_from_history"
                result.filled_from_history = True
                result.confidence_scale = min(result.confidence_scale, 0.7)
        return result

    @staticmethod
    def _assign(result: SmoothedRoad, field_name: str, value: Any) -> None:
        """Assign a field on the smoothed result."""
        if field_name == "lane_count":
            result.lane_count = value
        elif field_name == "ego_lane":
            result.ego_lane = value
        elif field_name == "road_type":
            result.road_type = value if value is not None else "unknown"

    def object_lifetime(self, track_id: str) -> tuple[int | None, int | None, int]:
        """First frame, last frame and observation count for a track."""
        frames = self.object_appearances.get(track_id) or []
        if not frames:
            return None, None, 0
        return int(min(frames)), int(max(frames)), len(frames)

    def object_gaps(self, track_id: str) -> list[dict[str, int]]:
        """Multi-frame gaps between observations of a track."""
        frames = sorted(self.object_appearances.get(track_id) or [])
        gaps: list[dict[str, int]] = []
        for previous, current in zip(frames, frames[1:]):
            if current - previous > 1:
                gaps.append(
                    {
                        "start_frame": int(previous + 1),
                        "end_frame": int(current - 1),
                        "length": int(current - previous - 1),
                    }
                )
        return gaps

    def statistics(self) -> dict[str, Any]:
        """Aggregate temporal metrics."""
        frame_ids = sorted(self.history)
        lane_counts = [
            entry.lane_count for entry in self.history.values() if entry.lane_count is not None
        ]
        return {
            "frames_recorded": len(self.history),
            "first_frame": frame_ids[0] if frame_ids else None,
            "last_frame": frame_ids[-1] if frame_ids else None,
            "distinct_tracks": len(self.object_appearances),
            "confirmed_changes": sum(len(changes) for changes in self.confirmed_changes.values()),
            "frames_with_lane_count": len(lane_counts),
            "mean_lane_count": round(sum(lane_counts) / len(lane_counts), 4) if lane_counts else None,
        }

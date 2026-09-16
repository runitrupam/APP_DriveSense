"""Stage 5b: road classification and exact lane-state change frames.

Lane-state changes are confirmed over ``temporal.lane_change_confirmation_frames``
to reject single-frame noise, but the recorded change frame is the first frame
that showed the new value. Confirmed changes are never smoothed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.output.schemas import AnomalyType
from src.road.lane_detector import LaneObservation
from src.utils.context import RunContext

ROAD_TYPES = (
    "straight",
    "curve_left",
    "curve_right",
    "intersection",
    "merge",
    "split",
    "roundabout",
    "unknown",
)

CURVE_TURN_THRESHOLD = 0.08
ROUNDABOUT_TURN_THRESHOLD = 0.30
INTERSECTION_HORIZONTAL_RATIO = 0.45


@dataclass
class LaneEvent:
    """A confirmed change in lane or road state."""

    type: str
    frame_id: int
    timestamp_ms: int
    field: str
    previous: Any
    current: Any
    confirmation_frames: int
    confidence: float | None = None
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "type": self.type,
            "frame_id": int(self.frame_id),
            "timestamp_ms": int(self.timestamp_ms),
            "field": self.field,
            "previous": self.previous,
            "current": self.current,
            "confirmation_frames": int(self.confirmation_frames),
            "confidence": self.confidence,
            "severity": self.severity,
        }


@dataclass
class RoadObservation:
    """Road state for a single frame, plus any change confirmed in that frame."""

    frame_id: int
    timestamp_ms: int
    lane_count: int | None = None
    ego_lane: int | None = None
    road_type: str = "unknown"
    confidence: float = 0.0
    curvature: float | None = None
    turn_ratio: float | None = None
    vanishing_point: tuple[float, float] | None = None
    horizontal_edge_ratio: float | None = None
    features: dict[str, Any] = field(default_factory=dict)
    events: list[LaneEvent] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def to_decision_summary(self) -> dict[str, Any]:
        """Compact summary used in the per-frame decision block."""
        return {
            "lane_count": self.lane_count if self.lane_count is not None else "unknown",
            "ego_lane": self.ego_lane if self.ego_lane is not None else "unknown",
            "road_type": self.road_type,
            "confidence": round(float(self.confidence), 4),
        }


class _ChangeTracker:
    """Confirms a value only after it persists for a configured number of frames."""

    def __init__(self, name: str, confirmation_frames: int) -> None:
        self.name = name
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.stable_value: Any = None
        self.candidate_value: Any = None
        self.candidate_first_frame: int | None = None
        self.candidate_count = 0
        self.unconfirmed_flips = 0

    def update(self, value: Any, frame_id: int) -> tuple[bool, Any, Any, int]:
        """Feed a raw value.

        Returns ``(changed, previous, current, confirmation_frames)`` where
        ``changed`` is true only once the new value has persisted.
        """
        if self.stable_value is None and value is not None:
            self.stable_value = value
            return False, None, value, 0

        if value is None:
            return False, self.stable_value, self.stable_value, 0

        if value == self.stable_value:
            if self.candidate_value is not None:
                self.unconfirmed_flips += 1
                self.candidate_value = None
                self.candidate_first_frame = None
                self.candidate_count = 0
            return False, self.stable_value, self.stable_value, 0

        if value == self.candidate_value:
            self.candidate_count += 1
        else:
            if self.candidate_value is not None:
                self.unconfirmed_flips += 1
            self.candidate_value = value
            self.candidate_first_frame = int(frame_id)
            self.candidate_count = 1

        if self.candidate_count >= self.confirmation_frames:
            previous = self.stable_value
            self.stable_value = self.candidate_value
            confirmed_frames = self.candidate_count
            self.candidate_value = None
            self.candidate_first_frame = None
            self.candidate_count = 0
            return True, previous, self.stable_value, confirmed_frames
        return False, self.stable_value, value, 0


class RoadAnalyzer:
    """Classifies road state and emits exact lane-change frames."""

    name = "geometric_road_analyzer"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        confirmation = int(ctx.get("temporal.lane_change_confirmation_frames", 3))
        self.confirmation_frames = max(1, confirmation)
        self._lane_count_tracker = _ChangeTracker("lane_count", confirmation)
        self._ego_lane_tracker = _ChangeTracker("ego_lane", confirmation)
        self._road_type_tracker = _ChangeTracker("road_type", confirmation)
        self._lane_count_history: list[int] = []
        self.lane_events: list[LaneEvent] = []
        self.metrics = ctx.metrics

    def reset(self) -> None:
        """Clear temporal state for a fresh pass."""
        confirmation = self.confirmation_frames
        self._lane_count_tracker = _ChangeTracker("lane_count", confirmation)
        self._ego_lane_tracker = _ChangeTracker("ego_lane", confirmation)
        self._road_type_tracker = _ChangeTracker("road_type", confirmation)
        self._lane_count_history = []
        self.lane_events = []

    def analyze(self, observation: LaneObservation) -> RoadObservation:
        """Classify road state for a frame and report confirmed changes."""
        frame_id = observation.frame_id
        timestamp_ms = observation.timestamp_ms
        turn_ratio = self._turn_ratio(observation)
        features = {
            "boundaries": len(observation.boundaries),
            "boundary_positions": [round(float(p), 2) for p in observation.boundary_positions],
            "lane_width_px": observation.lane_width_px,
            "horizontal_edge_ratio": observation.horizontal_edge_ratio,
            "lane_geometry_confidence": observation.confidence,
            "lane_detector_roi": list(observation.roi) if observation.roi else None,
        }
        if turn_ratio is not None:
            features["turn_ratio"] = round(float(turn_ratio), 4)

        road_type, type_confidence = self._classify(observation, turn_ratio)
        confidence = float(
            np.clip(0.5 * observation.confidence + 0.5 * type_confidence, 0.0, 1.0)
        )
        if observation.lane_count is None:
            confidence = min(confidence, 0.25)

        result = RoadObservation(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            lane_count=observation.lane_count,
            ego_lane=observation.ego_lane,
            road_type=road_type,
            confidence=round(confidence, 6),
            curvature=observation.curvature,
            turn_ratio=turn_ratio,
            vanishing_point=observation.vanishing_point,
            horizontal_edge_ratio=observation.horizontal_edge_ratio,
            features=features,
        )

        result.events.extend(self._detect_changes(observation, road_type))
        result.anomalies.extend(self._detect_anomalies(observation, road_type))
        self._record_metrics(result)
        return result

    def _turn_ratio(self, observation: LaneObservation) -> float | None:
        """Signed lateral drift of the lane boundaries across the ROI, over width.

        Positive means the road heads right into the distance, negative left.
        """
        if observation.height <= 0 or observation.width <= 0:
            return None
        deltas: list[float] = []
        for line in observation.boundaries:
            if len(line.points) < 2:
                continue
            ordered = sorted(line.points, key=lambda point: point[1])
            top = ordered[0]
            bottom = ordered[-1]
            deltas.append((float(top[0]) - float(bottom[0])) / float(observation.width))
        if not deltas:
            return None
        return float(np.mean(deltas))

    def _classify(
        self, observation: LaneObservation, turn_ratio: float | None
    ) -> tuple[str, float]:
        """Decide the road type with an explicit confidence."""
        horizontal_ratio = observation.horizontal_edge_ratio
        lane_count = observation.lane_count
        self._lane_count_history.append(lane_count if lane_count is not None else -1)
        recent = self._lane_count_history[-self.confirmation_frames * 2 :]

        if (
            horizontal_ratio is not None
            and horizontal_ratio >= INTERSECTION_HORIZONTAL_RATIO
            and lane_count is not None
        ):
            return "intersection", min(0.75, 0.35 + horizontal_ratio * 0.6)

        merge_or_split, direction_confidence = self._merge_or_split(recent)
        if merge_or_split is not None:
            return merge_or_split, direction_confidence

        if (
            turn_ratio is not None
            and abs(turn_ratio) >= ROUNDABOUT_TURN_THRESHOLD
            and observation.vanishing_point is None
            and (horizontal_ratio is None or horizontal_ratio < 0.35)
        ):
            return "roundabout", 0.35

        if turn_ratio is not None and abs(turn_ratio) >= CURVE_TURN_THRESHOLD:
            direction = "curve_right" if turn_ratio > 0 else "curve_left"
            magnitude = abs(turn_ratio)
            if magnitude < CURVE_TURN_THRESHOLD * 1.6:
                return direction, 0.45
            if magnitude < CURVE_TURN_THRESHOLD * 3.0:
                return direction, 0.65
            return direction, 0.8

        if observation.boundaries or lane_count is not None:
            return "straight", 0.55 if lane_count is not None else 0.35
        return "unknown", 0.0

    def _merge_or_split(self, recent: list[int]) -> tuple[str | None, float]:
        """Distinguish merge and split from a persistent lane-count change.

        A sustained increase in the visible lane count reads as a split, a
        sustained decrease as a merge. Requires at least one full confirmation
        window of the new count on both sides of the change.
        """
        window = self.confirmation_frames
        if len(recent) < window * 2:
            return None, 0.0
        usable = [value for value in recent if value > 0]
        if len(usable) < window * 2:
            return None, 0.0
        previous = usable[-window * 2 : -window]
        current = usable[-window:]
        if not previous or not current:
            return None, 0.0
        previous_median = float(np.median(previous))
        current_median = float(np.median(current))
        difference = current_median - previous_median
        if difference >= 1.0:
            return "split", min(0.7, 0.4 + 0.1 * difference)
        if difference <= -1.0:
            return "merge", min(0.7, 0.4 + 0.1 * abs(difference))
        return None, 0.0

    def _detect_changes(
        self, observation: LaneObservation, road_type: str
    ) -> list[LaneEvent]:
        """Emit confirmed lane count, ego lane and road type changes."""
        events: list[LaneEvent] = []
        events.extend(
            self._apply_tracker(
                self._lane_count_tracker,
                observation.lane_count,
                observation,
                "lane_count",
                "lane_count_jump",
            )
        )
        events.extend(
            self._apply_tracker(
                self._ego_lane_tracker,
                observation.ego_lane,
                observation,
                "ego_lane",
                "ego_lane_change",
            )
        )
        events.extend(
            self._apply_tracker(
                self._road_type_tracker,
                road_type,
                observation,
                "road_type",
                "road_type_change",
            )
        )
        self.lane_events.extend(events)
        return events

    def _apply_tracker(
        self,
        tracker: _ChangeTracker,
        value: Any,
        observation: LaneObservation,
        field_name: str,
        event_type: str,
    ) -> list[LaneEvent]:
        """Run one change tracker and convert a confirmation into an event."""
        changed, previous, current, confirmed_frames = tracker.update(value, observation.frame_id)
        if not changed:
            return []
        event_frame = observation.frame_id - max(0, confirmed_frames - 1)
        severity = "warning" if field_name == "lane_count" else "info"
        event = LaneEvent(
            type=event_type,
            frame_id=int(event_frame),
            timestamp_ms=int(observation.timestamp_ms),
            field=field_name,
            previous=previous,
            current=current,
            confirmation_frames=int(confirmed_frames),
            confidence=round(float(observation.confidence), 6),
            severity=severity,
        )
        return [event]

    def _detect_anomalies(self, observation: LaneObservation, road_type: str) -> list[str]:
        """Flag unstable lane estimates that are neither smooth nor confirmed."""
        anomalies: list[str] = []
        if (
            observation.lane_count is not None
            and observation.confidence < self.ctx.get("road.lane_detection_confidence_threshold", 0.35)
        ):
            anomalies.append(AnomalyType.UNSTABLE_LANE_ESTIMATE)
        if self._lane_count_tracker.unconfirmed_flips >= 3:
            anomalies.append(AnomalyType.UNSTABLE_LANE_ESTIMATE)
            self._lane_count_tracker.unconfirmed_flips = 0
        if road_type == "unknown":
            return anomalies
        return anomalies

    def _record_metrics(self, result: RoadObservation) -> None:
        """Persist per-frame road metrics."""
        metrics = self.metrics
        metrics.increment("road", "frames_processed")
        if result.road_type != "unknown":
            metrics.increment("road", f"road_type.{result.road_type}")
        else:
            metrics.increment("road", "road_type.unknown")
        metrics.observe("road", "confidence", result.confidence)
        if result.lane_count is not None:
            metrics.observe("road", "lane_count", float(result.lane_count))
        if result.turn_ratio is not None:
            metrics.observe("road", "turn_ratio", result.turn_ratio)
        for event in result.events:
            metrics.increment("road", f"events.{event.type}")
        for anomaly in result.anomalies:
            metrics.increment("road", f"anomalies.{anomaly}")

    def persist_events(self, store: Any) -> None:
        """Write ``temporal/lane_events.jsonl``."""
        writer = store.writer("temporal", "lane_events.jsonl")
        for event in self.lane_events:
            writer.write(event.to_dict())
        writer.close()

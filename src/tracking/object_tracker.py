"""Multi-object tracking with stable identities and track-level anomaly detection.

Matching is deterministic: candidate pairs are ranked by IoU and assigned in
order, so two runs over the same detections produce identical track ids. Tracks
use constant-velocity prediction, which keeps assignment robust across short
detection gaps without a heavy Kalman implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.detection.object_detector import Detection
from src.output.schemas import AnomalyType
from src.utils.context import RunContext
from src.utils.geometry import (
    bbox_center,
    bbox_to_xyxy,
    euclidean,
    iou,
    normalize_bbox,
    xyxy_to_bbox,
)

ID_PREFIXES = {
    "person": "pedestrian",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
}

VEHICLE_GROUP = frozenset({"car", "truck", "bus"})
TWO_WHEELER_GROUP = frozenset({"motorcycle", "bicycle"})


def _class_group(class_name: str) -> str:
    """Coarse group used to allow benign class flips between frames."""
    if class_name in VEHICLE_GROUP:
        return "vehicle"
    if class_name in TWO_WHEELER_GROUP:
        return "two_wheeler"
    return class_name


@dataclass
class TrackAnomaly:
    """An anomaly detected while maintaining a track."""

    type: str
    frame_id: int
    track_id: str
    detail: str = ""
    confidence: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return asdict(self)


@dataclass
class TrackObservation:
    """A track's state in one frame, as consumed by downstream stages."""

    track_id: str
    class_name: str
    bbox: dict[str, int]
    bbox_normalized: dict[str, float]
    center: dict[str, int]
    frame_id: int
    timestamp_ms: int
    detection_confidence: float
    tracking_confidence: float
    age: int
    first_frame: int
    last_frame: int
    velocity_px_per_frame: tuple[float, float] = (0.0, 0.0)
    matched_iou: float = 0.0
    source: str = "tracker"

    @property
    def is_new(self) -> bool:
        """True on the frame a track was first observed."""
        return self.age <= 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record for ``tracks_per_frame.jsonl``."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "track_id": self.track_id,
            "type": self.class_name,
            "bbox": self.bbox,
            "bbox_normalized": self.bbox_normalized,
            "center": self.center,
            "age": self.age,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "confidence": {
                "detection": round(float(self.detection_confidence), 6),
                "tracking": round(float(self.tracking_confidence), 6),
            },
            "velocity_px_per_frame": [
                round(float(self.velocity_px_per_frame[0]), 4),
                round(float(self.velocity_px_per_frame[1]), 4),
            ],
            "matched_iou": round(float(self.matched_iou), 6),
            "source": self.source,
        }


@dataclass
class Track:
    """Persistent identity state for one object."""

    track_id: str
    class_name: str
    first_frame: int
    last_frame: int
    bbox: dict[str, int]
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    matched_iou: float = 0.0
    confidences: list[float] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    lanes: dict[str, int | str] = field(default_factory=dict)
    anomalies: list[TrackAnomaly] = field(default_factory=list)
    class_history: list[str] = field(default_factory=list)
    ended: bool = False
    disappearance_reported: bool = False

    @property
    def num_observations(self) -> int:
        """Number of frames in which the track was actually observed."""
        return len(self.trajectory)

    @property
    def fragmented(self) -> bool:
        """True when the track has at least one multi-frame gap."""
        return len(self.gaps) > 0

    @property
    def max_gap(self) -> int:
        """Longest gap between consecutive observations."""
        return max((int(gap["length"]) for gap in self.gaps), default=0)

    @property
    def mean_confidence(self) -> float | None:
        """Mean detection confidence over the track's life."""
        if not self.confidences:
            return None
        return float(np.mean(self.confidences))

    @property
    def mean_velocity(self) -> tuple[float, float]:
        """Mean per-frame velocity in pixels."""
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        first = self.trajectory[0]
        last = self.trajectory[-1]
        span = max(1, int(last["frame_id"]) - int(first["frame_id"]))
        return ((last["cx"] - first["cx"]) / span, (last["cy"] - first["cy"]) / span)

    def to_dict(self) -> dict[str, Any]:
        """Full track record for ``tracks.jsonl``."""
        return {
            "track_id": self.track_id,
            "type": self.class_name,
            "class_history": list(self.class_history),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frame_span": int(self.last_frame) - int(self.first_frame) + 1,
            "num_observations": self.num_observations,
            "fragmented": self.fragmented,
            "gaps": list(self.gaps),
            "max_gap": self.max_gap,
            "mean_confidence": None if self.mean_confidence is None else round(self.mean_confidence, 6),
            "mean_velocity_px_per_frame": [
                round(float(self.mean_velocity[0]), 4),
                round(float(self.mean_velocity[1]), 4),
            ],
            "lanes_seen": sorted({str(value) for value in self.lanes.values()}),
            "trajectory": list(self.trajectory),
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
        }


class ObjectTracker:
    """Deterministic IoU plus constant-velocity multi-object tracker."""

    name = "iou_velocity"

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_match_threshold: float = 0.25,
        id_switch_iou_threshold: float = 0.1,
        teleport_distance_ratio: float = 0.35,
        impossible_speed_ratio: float = 0.60,
        disappearance_tolerance: int = 2,
    ) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_match_threshold = float(iou_match_threshold)
        self.id_switch_iou_threshold = float(id_switch_iou_threshold)
        self.teleport_distance_ratio = float(teleport_distance_ratio)
        self.impossible_speed_ratio = float(impossible_speed_ratio)
        self.disappearance_tolerance = int(disappearance_tolerance)
        self.tracks: dict[str, Track] = {}
        self._counters: dict[str, int] = {}
        self._size_by_frame: dict[int, tuple[int, int]] = {}
        self.frame_anomalies: dict[int, list[TrackAnomaly]] = {}
        self.finished: list[Track] = []
        self.frames_processed = 0
        self.total_observations = 0
        self._velocity_alpha = 0.6

    def reset(self) -> None:
        """Clear every track so the tracker can serve a fresh pass."""
        self.tracks.clear()
        self._counters.clear()
        self._size_by_frame.clear()
        self.frame_anomalies.clear()
        self.finished.clear()
        self.frames_processed = 0
        self.total_observations = 0

    def _new_track_id(self, class_name: str) -> str:
        """Allocate the next ``class_NNN`` identifier."""
        prefix = ID_PREFIXES.get(class_name, class_name)
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]:03d}"

    def _record_anomaly(
        self,
        anomaly_type: str,
        frame_id: int,
        track: Track,
        detail: str,
        confidence: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Attach an anomaly to a track and to the frame it happened in."""
        anomaly = TrackAnomaly(
            type=anomaly_type,
            frame_id=int(frame_id),
            track_id=track.track_id,
            detail=detail,
            confidence=confidence,
            data=data or {},
        )
        track.anomalies.append(anomaly)
        self.frame_anomalies.setdefault(int(frame_id), []).append(anomaly)

    def _predict(self, track: Track) -> dict[str, int]:
        """Translate the last bbox by the track velocity over its unobserved span."""
        if track.time_since_update <= 0:
            return dict(track.bbox)
        steps = float(track.time_since_update)
        dx = track.velocity[0] * steps
        dy = track.velocity[1] * steps
        bbox = track.bbox
        return xyxy_to_bbox(
            bbox["x"] + dx,
            bbox["y"] + dy,
            bbox["x"] + bbox["width"] + dx,
            bbox["y"] + bbox["height"] + dy,
        )

    def _match(
        self,
        detections: list[Detection],
        active: list[Track],
        predictions: dict[str, dict[str, int]],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """Greedy deterministic IoU matching between detections and active tracks."""
        if not active or not detections:
            return [], list(range(len(detections))), list(range(len(active)))

        candidates: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            det_xyxy = bbox_to_xyxy(detection.bbox)
            for track_index, track in enumerate(active):
                if _class_group(track.class_name) != _class_group(detection.class_name):
                    continue
                overlap = iou(det_xyxy, bbox_to_xyxy(predictions[track.track_id]))
                if overlap >= self.iou_match_threshold:
                    candidates.append((overlap, det_index, track_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        for overlap, det_index, track_index in candidates:
            if det_index in matched_dets or track_index in matched_tracks:
                continue
            matched_dets.add(det_index)
            matched_tracks.add(track_index)
            matches.append((det_index, track_index, overlap))

        unmatched_dets = [index for index in range(len(detections)) if index not in matched_dets]
        unmatched_tracks = [index for index in range(len(active)) if index not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> list[TrackObservation]:
        """Advance the tracker by one frame and return confirmed observations."""
        self.frames_processed += 1
        self._size_by_frame[int(frame_id)] = (int(frame_width), int(frame_height))
        diagonal = float(np.hypot(max(1, frame_width), max(1, frame_height)))

        active = [
            track
            for track in self.tracks.values()
            if not track.ended and track.time_since_update <= self.max_age
        ]
        predictions = {track.track_id: self._predict(track) for track in active}
        matches, unmatched_dets, unmatched_tracks = self._match(detections, active, predictions)

        for det_index, track_index, overlap in matches:
            self._apply_match(
                track=active[track_index],
                detection=detections[det_index],
                overlap=overlap,
                predicted_bbox=predictions[active[track_index].track_id],
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                diagonal=diagonal,
            )

        for track_index in unmatched_tracks:
            self._apply_miss(active[track_index], frame_id)

        for det_index in unmatched_dets:
            self._spawn(detections[det_index], frame_id, timestamp_ms)

        self._prune()

        observations = [
            self._observation(track, frame_id, timestamp_ms)
            for track in active
            if not track.ended and track.time_since_update == 0 and track.hits >= self.min_hits
        ]
        observations.sort(key=lambda observation: observation.track_id)
        self.total_observations += len(observations)
        return observations

    def _apply_match(
        self,
        track: Track,
        detection: Detection,
        overlap: float,
        predicted_bbox: dict[str, int],
        frame_id: int,
        timestamp_ms: int,
        diagonal: float,
    ) -> None:
        """Update a track from its matched detection."""
        previous_center = bbox_center(track.bbox)
        new_center = dict(detection.center)
        gap = track.time_since_update
        displacement = euclidean(
            (float(previous_center["x"]), float(previous_center["y"])),
            (float(new_center["x"]), float(new_center["y"])),
        )
        steps = float(max(1, gap + 1))
        step_displacement = displacement / steps

        if gap > 0:
            track.gaps.append(
                {
                    "start_frame": track.last_frame + 1,
                    "end_frame": frame_id - 1,
                    "length": gap,
                }
            )
            if track.disappearance_reported:
                self._record_anomaly(
                    AnomalyType.REAPPEARANCE,
                    frame_id,
                    track,
                    f"track reappeared after {gap} missing frame(s)",
                    confidence=0.5,
                    data={"gap": gap},
                )
            track.disappearance_reported = False

        if gap <= self.disappearance_tolerance and displacement > self.teleport_distance_ratio * diagonal:
            self._record_anomaly(
                AnomalyType.OBJECT_TELEPORTATION,
                frame_id,
                track,
                f"displacement {displacement:.0f}px exceeds "
                f"{self.teleport_distance_ratio:.0%} of the frame diagonal",
                confidence=0.6,
                data={"displacement_px": round(displacement, 2), "gap": gap},
            )
        if step_displacement > self.impossible_speed_ratio * diagonal:
            self._record_anomaly(
                AnomalyType.IMPOSSIBLE_MOTION,
                frame_id,
                track,
                f"per-frame speed {step_displacement:.0f}px exceeds the plausible limit",
                confidence=0.5,
                data={"step_displacement_px": round(step_displacement, 2)},
            )
        if _class_group(track.class_name) != _class_group(detection.class_name):
            self._record_anomaly(
                AnomalyType.TRACK_IDENTITY_SWITCH,
                frame_id,
                track,
                f"class changed from {track.class_name} to {detection.class_name}",
                confidence=0.4,
                data={"from": track.class_name, "to": detection.class_name},
            )
        prediction_overlap = iou(bbox_to_xyxy(predicted_bbox), bbox_to_xyxy(detection.bbox))
        if gap <= self.disappearance_tolerance and prediction_overlap < self.id_switch_iou_threshold:
            self._record_anomaly(
                AnomalyType.TRACK_IDENTITY_SWITCH,
                frame_id,
                track,
                "matched detection far from the predicted position",
                confidence=0.3,
                data={"predicted_iou": round(prediction_overlap, 4), "gap": gap},
            )

        delta_x = (new_center["x"] - previous_center["x"]) / steps
        delta_y = (new_center["y"] - previous_center["y"]) / steps
        alpha = self._velocity_alpha
        track.velocity = (
            alpha * track.velocity[0] + (1.0 - alpha) * delta_x,
            alpha * track.velocity[1] + (1.0 - alpha) * delta_y,
        )
        track.bbox = dict(detection.bbox)
        track.class_name = detection.class_name
        track.class_history.append(detection.class_name)
        track.confidences.append(float(detection.confidence))
        track.hits += 1
        track.age += 1
        track.time_since_update = 0
        track.last_frame = int(frame_id)
        track.matched_iou = float(overlap)
        track.trajectory.append(
            {
                "frame_id": int(frame_id),
                "timestamp_ms": int(timestamp_ms),
                "cx": int(new_center["x"]),
                "cy": int(new_center["y"]),
                "bbox": dict(detection.bbox),
            }
        )

    def _apply_miss(self, track: Track, frame_id: int) -> None:
        """Handle an unmatched track: age it and report disappearance once."""
        track.time_since_update += 1
        track.age += 1
        track.matched_iou = 0.0
        if (
            track.time_since_update == self.disappearance_tolerance
            and not track.disappearance_reported
        ):
            self._record_anomaly(
                AnomalyType.OBJECT_DISAPPEARANCE,
                frame_id,
                track,
                f"track not detected for {self.disappearance_tolerance} consecutive frame(s)",
                confidence=0.45,
                data={"missing_frames": track.time_since_update},
            )
            track.disappearance_reported = True

    def _spawn(self, detection: Detection, frame_id: int, timestamp_ms: int) -> None:
        """Create a new track from an unmatched detection."""
        track_id = self._new_track_id(detection.class_name)
        self.tracks[track_id] = Track(
            track_id=track_id,
            class_name=detection.class_name,
            first_frame=int(frame_id),
            last_frame=int(frame_id),
            bbox=dict(detection.bbox),
            velocity=(0.0, 0.0),
            hits=1,
            age=1,
            time_since_update=0,
            matched_iou=1.0,
            confidences=[float(detection.confidence)],
            class_history=[detection.class_name],
            trajectory=[
                {
                    "frame_id": int(frame_id),
                    "timestamp_ms": int(timestamp_ms),
                    "cx": int(detection.center["x"]),
                    "cy": int(detection.center["y"]),
                    "bbox": dict(detection.bbox),
                }
            ],
        )

    def _prune(self) -> None:
        """Retire tracks that have been missing longer than ``max_age``."""
        for track in self.tracks.values():
            if track.ended:
                continue
            if track.time_since_update > self.max_age:
                track.ended = True
                self.finished.append(track)

    def _observation(self, track: Track, frame_id: int, timestamp_ms: int) -> TrackObservation:
        """Build the observation record for a confirmed track."""
        width, height = self._size_by_frame.get(
            int(frame_id), self._size_by_frame.get(track.first_frame, (1, 1))
        )
        confidence = min(1.0, 0.35 + 0.05 * min(track.hits, 13))
        if track.matched_iou > 0:
            confidence *= 0.55 + 0.45 * min(1.0, track.matched_iou)
        if track.time_since_update > 0:
            confidence *= max(0.2, 1.0 - 0.2 * track.time_since_update)
        return TrackObservation(
            track_id=track.track_id,
            class_name=track.class_name,
            bbox=dict(track.bbox),
            bbox_normalized=normalize_bbox(track.bbox, width, height),
            center=bbox_center(track.bbox),
            frame_id=int(frame_id),
            timestamp_ms=int(timestamp_ms),
            detection_confidence=float(track.confidences[-1]) if track.confidences else 0.0,
            tracking_confidence=round(float(confidence), 6),
            age=int(track.age),
            first_frame=int(track.first_frame),
            last_frame=int(track.last_frame),
            velocity_px_per_frame=track.velocity,
            matched_iou=float(track.matched_iou),
        )

    def update_lane(self, track_id: str, frame_id: int, lane: int | str) -> None:
        """Attach the lane assignment for a track in a given frame."""
        track = self.tracks.get(track_id)
        if track is None:
            return
        track.lanes[str(int(frame_id))] = lane

    def anomalies_for_frame(self, frame_id: int) -> list[TrackAnomaly]:
        """Track anomalies that occurred in a frame."""
        return list(self.frame_anomalies.get(int(frame_id), []))

    def finalize(self) -> list[Track]:
        """Close every open track."""
        for track in self.tracks.values():
            if not track.ended:
                track.ended = True
                self.finished.append(track)
        self.finished.sort(key=lambda track: track.track_id)
        return list(self.finished)

    def statistics(self) -> dict[str, Any]:
        """Aggregate tracking metrics."""
        tracks = self.finished or list(self.tracks.values())
        by_class: dict[str, int] = {}
        for track in tracks:
            by_class[track.class_name] = by_class.get(track.class_name, 0) + 1
        lengths = [track.num_observations for track in tracks]

        def count(anomaly_type: str) -> int:
            return sum(
                1
                for track in tracks
                for anomaly in track.anomalies
                if anomaly.type == anomaly_type
            )

        return {
            "unique_tracks": len(tracks),
            "tracks_by_class": by_class,
            "fragmented_tracks": sum(1 for track in tracks if track.fragmented),
            "id_switch_candidates": count(AnomalyType.TRACK_IDENTITY_SWITCH),
            "teleportations": count(AnomalyType.OBJECT_TELEPORTATION),
            "disappearances": count(AnomalyType.OBJECT_DISAPPEARANCE),
            "reappearances": count(AnomalyType.REAPPEARANCE),
            "impossible_movements": count(AnomalyType.IMPOSSIBLE_MOTION),
            "total_observations": self.total_observations,
            "mean_track_length": round(float(np.mean(lengths)), 4) if lengths else 0.0,
            "min_track_length": int(min(lengths)) if lengths else 0,
            "max_track_length": int(max(lengths)) if lengths else 0,
            "frames_processed": self.frames_processed,
        }

    def persist(self, store: Any) -> None:
        """Write ``tracks.jsonl`` for every track."""
        writer = store.writer("tracking", "tracks.jsonl")
        for track in self.finished:
            writer.write(track.to_dict())
        writer.close()


def build_tracker(ctx: RunContext) -> ObjectTracker:
    """Instantiate the configured tracker."""
    engine = str(ctx.get("tracking.engine", "iou_velocity")).strip().lower()
    ctx.metrics.set("tracking", "engine", engine)
    return ObjectTracker(
        max_age=int(ctx.get("tracking.max_age", 30)),
        min_hits=int(ctx.get("tracking.min_hits", 3)),
        iou_match_threshold=float(ctx.get("tracking.iou_match_threshold", 0.25)),
        id_switch_iou_threshold=float(ctx.get("tracking.id_switch_iou_threshold", 0.1)),
        teleport_distance_ratio=float(ctx.get("tracking.teleport_distance_ratio", 0.35)),
        impossible_speed_ratio=float(ctx.get("tracking.impossible_speed_ratio", 0.60)),
        disappearance_tolerance=int(ctx.get("temporal.object_disappearance_tolerance", 2)),
    )

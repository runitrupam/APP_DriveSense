"""Deterministic forward/backward consistency checks.

The validator deliberately preserves disagreements instead of replacing the
chronological result.  Track identifiers from independent passes are not
compared directly; boxes and classes are matched by IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.output.schemas import AnomalyType
from src.utils.geometry import iou_bbox


@dataclass(frozen=True)
class FrameValidation:
    """Validation decision for one frame."""

    frame_id: int
    consistent: bool | str
    anomalies: tuple[str, ...] = ()
    details: tuple[dict[str, Any], ...] = ()
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation payload."""
        return {
            "frame_id": self.frame_id,
            "consistent": self.consistent,
            "anomalies": list(self.anomalies),
            "details": [dict(detail) for detail in self.details],
            "status": self.status,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Collection of per-frame decisions and aggregate counters."""

    frames: tuple[FrameValidation, ...] = ()

    @property
    def disagreements(self) -> int:
        """Number of frames where both passes disagree."""
        return sum(1 for frame in self.frames if frame.consistent is False)

    def by_frame(self) -> dict[int, FrameValidation]:
        """Index decisions by frame id without mutating the result."""
        return {frame.frame_id: frame for frame in self.frames}

    def summary(self) -> dict[str, Any]:
        """Aggregate validation metrics."""
        return {
            "frames_validated": len(self.frames),
            "frames_consistent": sum(1 for frame in self.frames if frame.consistent is True),
            "frames_disagreed": self.disagreements,
            "frames_unknown": sum(1 for frame in self.frames if frame.consistent == "unknown"),
            "anomalies": {
                anomaly: sum(anomaly in frame.anomalies for frame in self.frames)
                for anomaly in AnomalyType.ALL
                if any(anomaly in frame.anomalies for frame in self.frames)
            },
        }


def _box_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Compute IoU for schema or tracker bounding boxes."""
    try:
        return float(iou_bbox(dict(left), dict(right)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _objects_agree(forward: Iterable[Mapping[str, Any]], backward: Iterable[Mapping[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
    """Match objects by class and geometry, independent of generated track ids."""
    first = list(forward)
    second = list(backward)
    if not first and not second:
        return True, None
    if len(first) != len(second):
        return False, {"reason": "object_count", "forward": len(first), "backward": len(second)}

    available = set(range(len(second)))
    unmatched: list[dict[str, Any]] = []
    for item in first:
        candidates = [
            (index, _box_iou(item.get("bbox", {}), other.get("bbox", {})))
            for index, other in enumerate(second)
            if index in available and item.get("type") == other.get("type")
        ]
        if not candidates:
            unmatched.append({"reason": "object_class", "type": item.get("type")})
            continue
        index, overlap = max(candidates, key=lambda candidate: candidate[1])
        if overlap < 0.20:
            unmatched.append({"reason": "object_geometry", "type": item.get("type"), "iou": round(overlap, 4)})
            continue
        available.remove(index)
    if unmatched or available:
        return False, {"reason": "object_identity", "unmatched": unmatched, "remaining": len(available)}
    return True, None


def validate_frame(forward: Mapping[str, Any], backward: Mapping[str, Any] | None) -> FrameValidation:
    """Compare one forward summary with its same-frame backward summary."""
    frame_id = int(forward.get("frame_id", forward.get("frame", {}).get("frame_id", -1)))
    anomalies: list[str] = []
    details: list[dict[str, Any]] = []
    if backward is None:
        return FrameValidation(frame_id, "unknown", (), ({"reason": "missing_backward_summary"},), "anomalous")
    if not bool(forward.get("read_ok", True)) or not bool(backward.get("read_ok", True)):
        return FrameValidation(frame_id, "unknown", (), ({"reason": "decode_failure"},), "failed")

    first = forward.get("primary", forward)
    second = backward.get("primary", backward)
    compared = False
    for field in ("lane_count", "ego_lane", "road_type"):
        left = first.get(field)
        right = second.get(field)
        if left is None or right is None or left == "unknown" or right == "unknown":
            continue
        compared = True
        if left != right:
            anomalies.append(AnomalyType.FORWARD_BACKWARD_DISAGREEMENT)
            details.append({"field": field, "forward": left, "backward": right})
    objects_match, object_detail = _objects_agree(forward.get("objects", ()), backward.get("objects", ()))
    if not objects_match:
        compared = True
        anomalies.append(AnomalyType.FORWARD_BACKWARD_DISAGREEMENT)
        if object_detail:
            details.append(object_detail)
    if not compared:
        consistent: bool | str = "unknown"
    else:
        consistent = not anomalies
    status = "valid" if consistent is True else "anomalous" if consistent is False else "failed"
    return FrameValidation(frame_id, consistent, tuple(dict.fromkeys(anomalies)), tuple(details), status)


def validate_frames(
    forward: Iterable[Mapping[str, Any]], backward_by_frame: Mapping[int, Mapping[str, Any]]
) -> ValidationResult:
    """Validate ordered forward summaries against indexed backward summaries."""
    decisions = tuple(
        validate_frame(summary, backward_by_frame.get(int(summary.get("frame_id", -1))))
        for summary in forward
    )
    return ValidationResult(decisions)

"""Annotated frame rendering and the lane/object timeline plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from src.output.schemas import ObjectRecord
from src.utils.context import RunContext
from src.utils.geometry import bbox_to_xyxy

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 255),
    "truck": (0, 130, 255),
    "bus": (0, 80, 255),
    "motorcycle": (255, 120, 0),
    "bicycle": (255, 200, 0),
    "person": (0, 255, 0),
    "traffic_light": (0, 0, 255),
    "traffic_sign": (255, 0, 255),
}
DEFAULT_COLOR = (200, 200, 200)
ANOMALY_COLOR = (0, 0, 255)
LANE_COLOR = (255, 200, 0)
EGO_LANE_COLOR = (0, 255, 120)


class Annotator:
    """Draws detections, lanes and status onto frames."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.every_n = max(1, int(ctx.get("output.annotated_every_n_frames", 1)))
        self.enabled = ctx.flag("output.save_annotated_frames", True)
        self.saved = 0

    def should_save(self, frame_id: int) -> bool:
        """Whether this frame should be rendered to disk."""
        if not self.enabled:
            return False
        return int(frame_id) % self.every_n == 0

    def annotate(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        objects: Sequence[ObjectRecord],
        lane_observation: Any | None = None,
        road: Any | None = None,
        anomalies: Iterable[str] = (),
        ego_motion: Any | None = None,
        detections_count: int | None = None,
    ) -> np.ndarray:
        """Return a copy of the frame with overlays."""
        canvas = image.copy()
        if lane_observation is not None:
            self._draw_lanes(canvas, lane_observation, road)
        for record in objects:
            self._draw_object(canvas, record)
        self._draw_hud(
            canvas,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            objects=objects,
            lane_observation=lane_observation,
            road=road,
            anomalies=list(anomalies),
            ego_motion=ego_motion,
            detections_count=detections_count,
        )
        return canvas

    def _draw_lanes(self, canvas: np.ndarray, lane_observation: Any, road: Any | None) -> None:
        """Draw lane boundaries and highlight the ego lane."""
        ego_lane = getattr(road, "ego_lane", None) if road is not None else None
        for boundary in getattr(lane_observation, "boundaries", []):
            points = [tuple(int(v) for v in point) for point in boundary.points]
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, LANE_COLOR, 3)
        lanes = getattr(lane_observation, "lanes", []) or []
        height = canvas.shape[0]
        width = canvas.shape[1]
        for lane in lanes:
            left_x = int(lane.get("left_x", 0))
            right_x = int(lane.get("right_x", 0))
            top_y = int(0.55 * height)
            polygon = np.array(
                [(left_x, height - 1), (right_x, height - 1), (right_x, top_y), (left_x, top_y)],
                dtype=np.int32,
            )
            is_ego = ego_lane is not None and int(lane.get("lane_id", -1)) == int(ego_lane)
            if is_ego:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [polygon], EGO_LANE_COLOR)
                cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
            center_x = int((left_x + right_x) / 2)
            cv2.putText(
                canvas,
                f"L{int(lane.get('lane_id', 0))}",
                (max(0, center_x - 12), min(height - 8, int(0.75 * height))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                EGO_LANE_COLOR if is_ego else LANE_COLOR,
                2,
                cv2.LINE_AA,
            )
        _ = width

    def _draw_object(self, canvas: np.ndarray, record: ObjectRecord) -> None:
        """Draw one object with identity, state and confidence."""
        color = CLASS_COLORS.get(record.type, DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(v) for v in bbox_to_xyxy(record.bbox.model_dump()))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{record.track_id} {record.movement.state}"
        if record.movement.direction != "unknown":
            label += f" {record.movement.direction}"
        if record.lane != "unknown":
            label += f" lane={record.lane}"
        detection_confidence = record.confidence.detection
        if detection_confidence is not None:
            label += f" {detection_confidence:.2f}"
        scale = 0.45
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(text_height, y1 - baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        center = (int(record.center.x), int(record.center.y))
        cv2.circle(canvas, center, 3, color, -1)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        objects: Sequence[ObjectRecord],
        lane_observation: Any | None,
        road: Any | None,
        anomalies: list[str],
        ego_motion: Any | None,
        detections_count: int | None,
    ) -> None:
        """Status text block and anomaly banner."""
        lane_count = getattr(road, "lane_count", None) if road is not None else None
        ego_lane = getattr(road, "ego_lane", None) if road is not None else None
        road_type = getattr(road, "road_type", "unknown") if road is not None else "unknown"
        confidence = getattr(road, "confidence", None) if road is not None else None
        lines = [
            f"frame={frame_id} t={timestamp_ms}ms",
            f"road={road_type} lanes={lane_count if lane_count is not None else 'unknown'} "
            f"ego_lane={ego_lane if ego_lane is not None else 'unknown'}",
            f"objects={len(objects)} detections={detections_count if detections_count is not None else 'n/a'}",
        ]
        if confidence is not None:
            lines.append(f"road_confidence={confidence:.2f}")
        if lane_observation is not None:
            boundary_count = len(getattr(lane_observation, "boundaries", []))
            lines.append(f"lane_boundaries={boundary_count}")
        if ego_motion is not None:
            lines.append(
                f"ego_shift=({getattr(ego_motion, 'dx', 0.0):.1f},"
                f"{getattr(ego_motion, 'dy', 0.0):.1f}) resp={getattr(ego_motion, 'response', 0.0):.2f}"
            )
        if anomalies:
            lines.append("anomalies=" + ",".join(anomalies[:4]))

        y = 20
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(canvas, (6, y - text_height - 4), (12 + text_width, y + baseline), (0, 0, 0), -1)
            cv2.putText(
                canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            y += text_height + baseline + 8

    def save(self, image: np.ndarray, frame_id: int) -> Path:
        """Persist an annotated frame."""
        target = self.ctx.store.path("annotated", f"annotated_{int(frame_id):06d}.jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.saved += 1
        return target


class TimelinePlotter:
    """Renders the lane and object timeline for a run."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def plot(
        self,
        frame_summaries: Sequence[dict[str, Any]],
        events: Sequence[dict[str, Any]] = (),
        filename: str = "timeline.png",
    ) -> Path | None:
        """Plot lane count, ego lane, object count and lane-change markers."""
        if not frame_summaries:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.ctx.logger.warning("matplotlib unavailable; skipping timeline plot")
            return None

        frames = [int(entry["frame_id"]) for entry in frame_summaries]
        lane_counts = [
            entry.get("lane_count") if isinstance(entry.get("lane_count"), int) else None
            for entry in frame_summaries
        ]
        ego_lanes = [
            entry.get("ego_lane") if isinstance(entry.get("ego_lane"), int) else None
            for entry in frame_summaries
        ]
        object_counts = [int(entry.get("object_count", 0)) for entry in frame_summaries]

        figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].step(frames, lane_counts, where="mid", color="tab:blue")
        axes[0].set_ylabel("lane_count")
        axes[0].set_title("Lane and object timeline")
        axes[0].grid(alpha=0.3)

        axes[1].step(frames, ego_lanes, where="mid", color="tab:green")
        axes[1].set_ylabel("ego_lane")
        axes[1].grid(alpha=0.3)

        axes[2].plot(frames, object_counts, color="tab:orange")
        axes[2].set_ylabel("objects")
        axes[2].set_xlabel("frame_id")
        axes[2].grid(alpha=0.3)

        for event in events:
            event_frame = event.get("frame_id")
            if event_frame is None:
                continue
            for axis in axes:
                axis.axvline(int(event_frame), color="tab:red", alpha=0.35, linewidth=0.8)

        figure.tight_layout()
        target = self.ctx.store.path("annotated", filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=110)
        plt.close(figure)
        return target

"""Forward pass (frame 1 -> N) and the shared per-frame pass implementation.

:class:`FramePass` holds the whole per-frame pipeline. :class:`ForwardPass`
configures it for forward processing; the backward pass reuses it with reversed
iteration and validation-only persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.detection.object_detector import Detection
from src.frames.frame_extractor import FrameRecord
from src.ingestion.video_reader import FramePacket, VideoMetadata, VideoReader
from src.motion.movement import EgoMotion
from src.output.schemas import (
    BBox,
    BBoxNormalized,
    Confidences,
    Event,
    Movement,
    ObjectRecord,
    Point,
    Severity,
    AnomalyType,
    UNKNOWN,
)
from src.pipeline.stage_bundle import StageBundle
from src.road.lane_detector import LaneObservation
from src.road.road_analyzer import RoadObservation
from src.temporal.state_manager import FrameHistoryEntry, SmoothedRoad
from src.tracking.object_tracker import TrackObservation
from src.utils.context import RunContext
from src.utils.image import to_gray

ENVIRONMENT_ITEM_TYPES = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
}


def make_event(
    event_type: str,
    frame_id: int,
    timestamp_ms: int,
    description: str,
    severity: str = Severity.INFO,
    confidence: float | None = None,
    data: dict[str, Any] | None = None,
) -> Event:
    """Build an :class:`Event` record."""
    return Event(
        type=event_type,
        frame_id=int(frame_id),
        timestamp_ms=int(timestamp_ms),
        severity=severity,
        description=description,
        confidence=confidence,
        data=data or {},
    )


def assign_lane(lane_observation: LaneObservation | None, center_x: float) -> int | None:
    """Lane index containing ``center_x``, or ``None`` when undetermined."""
    if lane_observation is None:
        return None
    lanes = lane_observation.lanes or []
    for lane in lanes:
        left_x = float(lane.get("left_x", 0.0))
        right_x = float(lane.get("right_x", 0.0))
        if left_x <= center_x <= right_x:
            return int(lane["lane_id"])
    positions = lane_observation.boundary_positions or []
    if positions and lane_observation.lane_count:
        left_of_center = sum(1 for position in positions if position < center_x)
        if 1 <= left_of_center <= int(lane_observation.lane_count):
            return int(left_of_center)
    return None


@dataclass
class FrameOutcome:
    """Everything produced for one frame by one pass."""

    frame_id: int
    timestamp_ms: int
    direction: str = "forward"
    read_ok: bool = True
    width: int = 0
    height: int = 0
    frame_record: FrameRecord | None = None
    lane_observation: LaneObservation | None = None
    road: RoadObservation | None = None
    smoothed_road: SmoothedRoad | None = None
    objects: list[ObjectRecord] = field(default_factory=list)
    environment: Any | None = None
    ego_motion: EgoMotion | None = None
    events: list[Event] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    stage_errors: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    motion_records: list[dict[str, Any]] = field(default_factory=list)
    detection_count: int = 0
    track_ids: list[str] = field(default_factory=list)

    def add_anomaly(self, anomaly: str, description: str, severity: str = Severity.WARNING) -> None:
        """Record an anomaly once, with a matching event."""
        if anomaly not in self.anomalies:
            self.anomalies.append(anomaly)
        self.events.append(
            make_event(
                anomaly,
                self.frame_id,
                self.timestamp_ms,
                description,
                severity=severity,
            )
        )

    def primary_summary(self) -> dict[str, Any]:
        """Compact primary result preserved in the decision block."""
        road = self.road
        return {
            "lane_count": None if road is None or road.lane_count is None else int(road.lane_count),
            "ego_lane": None if road is None or road.ego_lane is None else int(road.ego_lane),
            "road_type": "unknown" if road is None else road.road_type,
            "object_count": len(self.objects),
            "track_ids": list(self.track_ids),
            "anomalies": list(self.anomalies),
        }

    def object_summaries(self) -> list[dict[str, Any]]:
        """Object geometry used for forward/backward identity matching."""
        return [
            {
                "track_id": record.track_id,
                "type": record.type,
                "bbox": record.bbox.model_dump(),
                "movement_state": record.movement.state,
                "lane": record.lane,
            }
            for record in self.objects
        ]


class FramePass:
    """Per-frame pipeline shared by the forward and backward passes."""

    def __init__(
        self,
        ctx: RunContext,
        bundle: StageBundle,
        direction: str = "forward",
        persist_artifacts: bool = True,
        frames_output: str | None = "forward_frames.jsonl",
    ) -> None:
        self.ctx = ctx
        self.bundle = bundle
        self.direction = direction
        self.persist_artifacts = persist_artifacts
        self.frames_output = frames_output
        self.low_detection_confidence = float(ctx.get("validation.min_detection_confidence", 0.35))
        self.low_tracking_confidence = float(ctx.get("validation.min_tracking_confidence", 0.30))
        self.low_confidence_ratio = float(ctx.get("validation.low_confidence_ratio_threshold", 0.6))
        self.motion_state_counts: dict[str, int] = {}
        self.direction_counts: dict[str, int] = {}
        self.frames_processed = 0
        self.failed_frames = 0

    def reset(self) -> None:
        """Reset pass-local counters and stage state."""
        self.bundle.reset_for_pass()
        self.motion_state_counts = {}
        self.direction_counts = {}
        self.frames_processed = 0
        self.failed_frames = 0

    def _stage(
        self,
        stage_name: str,
        func: Callable[[], Any],
        default: Any,
        outcome: FrameOutcome,
    ) -> Any:
        """Run a stage, degrading to ``default`` and recording the failure."""
        try:
            return func()
        except Exception as exc:
            message = f"{stage_name}: {type(exc).__name__}: {exc}"
            outcome.stage_errors.append(message)
            self.ctx.metrics.note(stage_name, message)
            self.ctx.metrics.increment(stage_name, "stage_failures")
            self.ctx.logger.exception("stage %s failed on frame %s", stage_name, outcome.frame_id)
            return default

    def process(self, packet: FramePacket) -> FrameOutcome:
        """Process one frame through every stage."""
        self.frames_processed += 1
        frame_id = int(packet.frame_id)
        timestamp_ms = int(packet.timestamp_ms)
        outcome = FrameOutcome(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            direction=self.direction,
        )

        frame_record = self.bundle.extractor.process_frame(packet)
        outcome.frame_record = frame_record

        if not packet.read_ok or packet.image is None:
            self.failed_frames += 1
            outcome.read_ok = False
            outcome.add_anomaly(
                AnomalyType.FRAME_DECODE_FAILURE,
                f"frame {frame_id} could not be decoded: {packet.error or 'unknown error'}",
                severity=Severity.ERROR,
            )
            outcome.stage_errors.append(packet.error or "frame_decode_failure")
            self._persist_frame(outcome)
            return outcome

        image = packet.image
        outcome.height, outcome.width = int(image.shape[0]), int(image.shape[1])
        self._record_frame_quality(outcome, frame_record)

        ego_motion = self.bundle.ego_motion.update(image)
        outcome.ego_motion = ego_motion
        self.bundle.movement.record_ego_motion(ego_motion)

        detections = self._detect(image, frame_id, timestamp_ms, outcome)
        outcome.detection_count = len(detections)

        lane_observation = self._stage(
            "lane",
            lambda: self.bundle.lane_detector.detect(image, frame_id, timestamp_ms),
            None,
            outcome,
        )
        outcome.lane_observation = lane_observation

        road = None
        if lane_observation is not None:
            road = self._stage(
                "road",
                lambda: self.bundle.road_analyzer.analyze(lane_observation),
                None,
                outcome,
            )
        outcome.road = road

        if road is not None:
            self.bundle.state.register_events(frame_id, road.events)
            for event in road.events:
                outcome.events.append(
                    make_event(
                        event.type,
                        event.frame_id,
                        event.timestamp_ms,
                        f"{event.field}: {event.previous} -> {event.current}",
                        severity=event.severity,
                        confidence=event.confidence,
                        data=event.to_dict(),
                    )
                )
            for anomaly in road.anomalies:
                outcome.add_anomaly(anomaly, f"lane estimate unstable on frame {frame_id}")

        smoothed = self.bundle.state.smooth_road(
            frame_id,
            None if road is None else road.lane_count,
            None if road is None else road.ego_lane,
            None if road is None else road.road_type,
        )
        outcome.smoothed_road = smoothed

        observations = self._track(detections, frame_id, timestamp_ms, outcome, image)
        outcome.track_ids = [observation.track_id for observation in observations]

        outcome.objects = self._build_objects(observations, road, outcome)
        self._record_object_anomalies(outcome)

        outcome.environment = self._environment(
            image, frame_id, timestamp_ms, outcome, road
        )

        self._annotate(image, outcome)
        self._record_frame_history(outcome)
        self._persist_frame(outcome)
        return outcome

    def _record_frame_quality(self, outcome: FrameOutcome, frame_record: FrameRecord) -> None:
        """Raise blur and duplicate anomalies from the frame index."""
        if frame_record.blurred:
            outcome.add_anomaly(
                AnomalyType.BLURRED_FRAME,
                f"frame {outcome.frame_id} blur score {frame_record.blur_score:.2f} below threshold",
            )
        if frame_record.duplicate_of is not None:
            outcome.add_anomaly(
                AnomalyType.DUPLICATE_FRAME,
                f"frame {outcome.frame_id} duplicates frame {frame_record.duplicate_of}",
            )

    def _detect(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        outcome: FrameOutcome,
    ) -> list[Detection]:
        """Run detection and persist detection artifacts."""
        detections = self._stage(
            "detection",
            lambda: self.bundle.detector.detect(image, frame_id, timestamp_ms),
            [],
            outcome,
        )
        detector = self.bundle.detector
        self.ctx.metrics.increment("detection", "frames_processed")
        self.ctx.metrics.increment("detection", "total_detections", len(detections))
        for detection in detections:
            self.ctx.metrics.increment("detection", f"class.{detection.class_name}")
            self.ctx.metrics.observe("detection", "confidence", detection.confidence)
        self.ctx.metrics.set("detection", "engine", detector.name)

        if self.persist_artifacts:
            writer = self.ctx.store.writer("detection", "detections.jsonl")
            for detection in detections:
                writer.write(detection.to_dict())
            frame_writer = self.ctx.store.writer("detection", "detections_per_frame.jsonl")
            frame_writer.write(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "engine": detector.name,
                    "count": len(detections),
                    "detections": [detection.to_dict() for detection in detections],
                }
            )
        return detections

    def _track(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp_ms: int,
        outcome: FrameOutcome,
        image: np.ndarray,
    ) -> list[TrackObservation]:
        """Update the tracker and translate tracker anomalies into events."""
        height, width = int(image.shape[0]), int(image.shape[1])
        observations = self._stage(
            "tracking",
            lambda: self.bundle.tracker.update(
                detections, frame_id, timestamp_ms, width, height
            ),
            [],
            outcome,
        )
        self.ctx.metrics.increment("tracking", "frames_processed")
        self.ctx.metrics.increment("tracking", "observations", len(observations))

        for anomaly in self.bundle.tracker.anomalies_for_frame(frame_id):
            outcome.add_anomaly(anomaly.type, anomaly.detail or anomaly.type)
            if self.persist_artifacts:
                outcome.events.append(
                    make_event(
                        anomaly.type,
                        frame_id,
                        timestamp_ms,
                        f"{anomaly.track_id}: {anomaly.detail}",
                        confidence=anomaly.confidence,
                        data={"track_id": anomaly.track_id, **anomaly.data},
                    )
                )
        return observations

    def _build_objects(
        self,
        observations: list[TrackObservation],
        road: RoadObservation | None,
        outcome: FrameOutcome,
    ) -> list[ObjectRecord]:
        """Build the per-frame object records with motion and direction."""
        vanishing_point = road.vanishing_point if road is not None else None
        objects: list[ObjectRecord] = []
        lane_observation = outcome.lane_observation
        for observation in observations:
            lane = assign_lane(lane_observation, float(observation.center["x"]))
            if lane is not None:
                self.bundle.tracker.update_lane(observation.track_id, observation.frame_id, lane)
            trajectory = self.bundle.trajectory_store.update(
                observation.track_id,
                observation.frame_id,
                observation.timestamp_ms,
                float(observation.center["x"]),
                float(observation.center["y"]),
                observation.bbox,
            )
            bbox_height = float(observation.bbox.get("height", 0) or 0)
            movement_result = self.bundle.movement.classify(
                trajectory, outcome.ego_motion or EgoMotion(), bbox_height
            )
            direction_result = self.bundle.direction.classify(
                movement_result.residual_vx,
                movement_result.residual_vy,
                observation.center,
                vanishing_point=vanishing_point,
                frame_size=(outcome.width, outcome.height),
                bbox_height=bbox_height,
            )
            movement_confidence = float(
                np.clip(0.5 * movement_result.confidence + 0.5 * direction_result.confidence, 0.0, 1.0)
            )
            record = ObjectRecord(
                track_id=observation.track_id,
                type=observation.class_name,
                bbox=BBox(**observation.bbox),
                bbox_normalized=BBoxNormalized(**observation.bbox_normalized),
                center=Point(**observation.center),
                lane=lane if lane is not None else UNKNOWN,
                movement=Movement(
                    state=movement_result.state,
                    direction=direction_result.relative,
                    image_direction=direction_result.image_direction,
                    longitudinal=direction_result.longitudinal,
                    lateral=direction_result.lateral,
                    speed_px_s=movement_result.observed_speed_px_s,
                    displacement_px=round(float(movement_result.displacement_px), 4),
                    confidence=round(movement_confidence, 6),
                ),
                confidence=Confidences(
                    detection=round(float(observation.detection_confidence), 6),
                    tracking=round(float(observation.tracking_confidence), 6),
                    movement=round(movement_confidence, 6),
                ),
                age=int(observation.age),
                first_frame=int(observation.first_frame),
                last_frame=int(observation.last_frame),
                source="detector",
            )
            objects.append(record)
            outcome.motion_records.append(
                {
                    "frame_id": observation.frame_id,
                    "timestamp_ms": observation.timestamp_ms,
                    "track_id": observation.track_id,
                    "type": observation.class_name,
                    "lane": record.lane,
                    "movement": movement_result.to_dict(),
                    "direction": direction_result.to_dict(),
                    "trajectory_length": len(trajectory),
                }
            )
            self._record_motion_metrics(movement_result, direction_result)
        return objects

    def _record_motion_metrics(self, movement_result: Any, direction_result: Any) -> None:
        """Aggregate movement and direction distributions."""
        self.ctx.metrics.increment("motion", "objects_processed")
        self.ctx.metrics.increment("motion", f"state.{movement_result.state}")
        self.ctx.metrics.increment("motion", f"direction.{direction_result.relative}")
        self.ctx.metrics.increment("motion", f"image_direction.{direction_result.image_direction}")
        self.ctx.metrics.observe("motion", "residual_speed_px", movement_result.residual_speed)
        self.ctx.metrics.observe("motion", "movement_confidence", movement_result.confidence)
        self.ctx.metrics.observe("motion", "direction_confidence", direction_result.confidence)
        self.motion_state_counts[movement_result.state] = (
            self.motion_state_counts.get(movement_result.state, 0) + 1
        )
        self.direction_counts[direction_result.relative] = (
            self.direction_counts.get(direction_result.relative, 0) + 1
        )

    def _record_object_anomalies(self, outcome: FrameOutcome) -> None:
        """Flag frames whose detection or tracking confidence is uniformly low."""
        if not outcome.objects:
            return
        detection_confidences = [
            record.confidence.detection
            for record in outcome.objects
            if record.confidence.detection is not None
        ]
        low_detection = [
            value for value in detection_confidences if value < self.low_detection_confidence
        ]
        if detection_confidences and (
            len(low_detection) / len(detection_confidences) >= self.low_confidence_ratio
        ):
            outcome.add_anomaly(
                AnomalyType.LOW_DETECTION_CONFIDENCE,
                f"{len(low_detection)}/{len(detection_confidences)} detections below "
                f"{self.low_detection_confidence}",
            )

        tracking_confidences = [
            record.confidence.tracking
            for record in outcome.objects
            if record.confidence.tracking is not None
        ]
        low_tracking = [value for value in tracking_confidences if value < self.low_tracking_confidence]
        if tracking_confidences and (
            len(low_tracking) / len(tracking_confidences) >= self.low_confidence_ratio
        ):
            outcome.add_anomaly(
                AnomalyType.LOW_TRACKING_CONFIDENCE,
                f"{len(low_tracking)}/{len(tracking_confidences)} tracks below "
                f"{self.low_tracking_confidence}",
            )

    def _environment(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        outcome: FrameOutcome,
        road: RoadObservation | None,
    ) -> Any:
        """Run the environment stage using detection-derived items."""
        from src.environment.environment_analyzer import EnvironmentItem

        items = []
        for record in outcome.objects:
            item_type = ENVIRONMENT_ITEM_TYPES.get(record.type)
            if item_type is None:
                continue
            items.append(
                EnvironmentItem(
                    type=item_type,
                    bbox=record.bbox.model_dump(),
                    confidence=record.confidence.detection,
                    source="detector",
                    track_id=record.track_id,
                    lane=record.lane,
                )
            )
        vanishing_point = road.vanishing_point if road is not None else None
        return self._stage(
            "environment",
            lambda: self.bundle.environment.analyze(
                image,
                frame_id,
                timestamp_ms,
                object_items=items,
                vanishing_point=vanishing_point,
            ),
            None,
            outcome,
        )

    def _annotate(self, image: np.ndarray, outcome: FrameOutcome) -> None:
        """Render and store an annotated frame when configured."""
        if not self.persist_artifacts or not self.bundle.annotator.should_save(outcome.frame_id):
            return
        try:
            annotated = self.bundle.annotator.annotate(
                image=image,
                frame_id=outcome.frame_id,
                timestamp_ms=outcome.timestamp_ms,
                objects=outcome.objects,
                lane_observation=outcome.lane_observation,
                road=outcome.road,
                anomalies=outcome.anomalies,
                ego_motion=outcome.ego_motion,
                detections_count=outcome.detection_count,
            )
            self.bundle.annotator.save(annotated, outcome.frame_id)
            self.ctx.metrics.increment("annotation", "frames_saved")
        except Exception as exc:
            outcome.stage_errors.append(f"annotation: {type(exc).__name__}: {exc}")
            self.ctx.logger.warning("annotation failed on frame %s: %s", outcome.frame_id, exc)

    def _record_frame_history(self, outcome: FrameOutcome) -> None:
        """Store a rolling history entry for temporal reasoning."""
        environment_counts: dict[str, int | None] = {}
        if outcome.environment is not None and getattr(outcome.environment, "counts", None):
            environment_counts = dict(outcome.environment.counts)
        self.bundle.state.record_frame(
            FrameHistoryEntry(
                frame_id=outcome.frame_id,
                timestamp_ms=outcome.timestamp_ms,
                object_ids=list(outcome.track_ids),
                object_count=len(outcome.objects),
                lane_count=None if outcome.road is None else outcome.road.lane_count,
                ego_lane=None if outcome.road is None else outcome.road.ego_lane,
                road_type="unknown" if outcome.road is None else outcome.road.road_type,
                environment_counts=environment_counts,
                ego_motion=None if outcome.ego_motion is None else outcome.ego_motion.to_dict(),
                anomalies=list(outcome.anomalies),
            )
        )

    def _persist_frame(self, outcome: FrameOutcome) -> None:
        """Write the pass's per-frame artifacts."""
        if not self.persist_artifacts:
            if self.frames_output:
                self.ctx.store.append_jsonl(
                    "temporal",
                    self.frames_output,
                    {
                        "frame_id": outcome.frame_id,
                        "timestamp_ms": outcome.timestamp_ms,
                        "direction": outcome.direction,
                        "read_ok": outcome.read_ok,
                        "primary": outcome.primary_summary(),
                        "objects": outcome.object_summaries(),
                        "anomalies": list(outcome.anomalies),
                        "stage_errors": list(outcome.stage_errors),
                    },
                )
            return

        for record in outcome.motion_records:
            self.ctx.store.append_jsonl("motion", "motion.jsonl", record)
        self.ctx.store.append_jsonl(
            "tracking",
            "tracks_per_frame.jsonl",
            {
                "frame_id": outcome.frame_id,
                "timestamp_ms": outcome.timestamp_ms,
                "track_ids": list(outcome.track_ids),
                "object_count": len(outcome.objects),
            },
        )
        if self.frames_output:
            self.ctx.store.append_jsonl(
                "temporal",
                self.frames_output,
                {
                    "frame_id": outcome.frame_id,
                    "timestamp_ms": outcome.timestamp_ms,
                    "read_ok": outcome.read_ok,
                    "primary": outcome.primary_summary(),
                    "smoothed": None if outcome.smoothed_road is None else outcome.smoothed_road.to_dict(),
                    "anomalies": list(outcome.anomalies),
                    "stage_errors": list(outcome.stage_errors),
                },
            )

    def finalize(self) -> dict[str, Any]:
        """Close the pass, persist aggregate artifacts and return statistics."""
        self.bundle.tracker.finalize()
        tracking_stats = self.bundle.tracker.statistics()
        self.ctx.metrics.record("tracking", tracking_stats)
        self.ctx.metrics.set("motion", "state_counts", dict(self.motion_state_counts))
        self.ctx.metrics.set("motion", "relative_direction_counts", dict(self.direction_counts))
        self.ctx.metrics.set("motion", "frames_processed", self.frames_processed)
        self.ctx.metrics.set("motion", "failed_frames", self.failed_frames)
        self.ctx.metrics.set("temporal", "state", self.bundle.state.statistics())
        self.ctx.metrics.set("annotation", "frames_saved", self.bundle.annotator.saved)

        if self.persist_artifacts:
            self.bundle.tracker.persist(self.ctx.store)
            self.bundle.road_analyzer.persist_events(self.ctx.store)
        return {
            "direction": self.direction,
            "frames_processed": self.frames_processed,
            "failed_frames": self.failed_frames,
            "tracking": tracking_stats,
            "motion_state_counts": dict(self.motion_state_counts),
            "direction_counts": dict(self.direction_counts),
            "temporal": self.bundle.state.statistics(),
        }


class ForwardPass(FramePass):
    """Frame 1 -> N pass producing the primary result."""

    def __init__(self, ctx: RunContext, bundle: StageBundle) -> None:
        super().__init__(
            ctx,
            bundle,
            direction="forward",
            persist_artifacts=True,
            frames_output="forward_frames.jsonl",
        )

    def iter_frames(self, video_path: str, metadata: VideoMetadata):
        """Yield every frame exactly once, forward."""
        with VideoReader(video_path, metadata=metadata) as reader:
            for packet in reader.iter_frames():
                yield packet

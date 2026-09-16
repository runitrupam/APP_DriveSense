"""Wiring of stage instances into a reusable bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.detection.object_detector import ObjectDetector, build_detector
from src.environment.environment_analyzer import EnvironmentAnalyzer
from src.frames.frame_extractor import FrameExtractor
from src.motion.direction import DirectionClassifier
from src.motion.movement import EgoMotionEstimator, MovementClassifier
from src.motion.trajectory import TrajectoryStore
from src.output.annotator import Annotator
from src.road.lane_detector import LaneDetector
from src.road.road_analyzer import RoadAnalyzer
from src.temporal.state_manager import TemporalStateManager
from src.tracking.object_tracker import ObjectTracker, build_tracker
from src.utils.context import RunContext


@dataclass
class StageBundle:
    """All stage instances needed to process a video."""

    detector: ObjectDetector
    tracker: ObjectTracker
    lane_detector: LaneDetector
    road_analyzer: RoadAnalyzer
    trajectory_store: TrajectoryStore
    ego_motion: EgoMotionEstimator
    movement: MovementClassifier
    direction: DirectionClassifier
    environment: EnvironmentAnalyzer
    extractor: FrameExtractor
    state: TemporalStateManager
    annotator: Annotator

    def reset_for_pass(self) -> None:
        """Reset every stateful stage so a new pass starts from scratch."""
        for component in (
            self.detector,
            self.tracker,
            self.road_analyzer,
            self.ego_motion,
            self.extractor,
            self.state,
        ):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        self.trajectory_store.clear()

    def describe(self) -> dict[str, Any]:
        """Provenance for every replaceable component."""
        return {
            "detector": self.detector.describe(),
            "tracker": {
                "engine": self.tracker.name,
                "max_age": self.tracker.max_age,
                "min_hits": self.tracker.min_hits,
                "iou_match_threshold": self.tracker.iou_match_threshold,
            },
            "lane_detector": {
                "engine": self.lane_detector.name,
                "roi_top_ratio": self.lane_detector.roi_top_ratio,
                "max_lanes": self.lane_detector.max_lanes,
            },
            "road_analyzer": {"engine": self.road_analyzer.name},
            "movement": self.movement.statistics(),
            "direction": self.direction.statistics(),
            "environment": {"engine": self.environment.name, "enabled": self.environment.enabled},
            "temporal": {
                "smoothing_window": self.state.smoothing_window,
                "preserve_confirmed_changes": self.state.preserve_changes,
            },
        }


def build_bundle(ctx: RunContext, fps: float) -> StageBundle:
    """Build every stage from configuration."""
    history_frames = int(ctx.get("motion.history_frames", 10))
    return StageBundle(
        detector=build_detector(ctx),
        tracker=build_tracker(ctx),
        lane_detector=LaneDetector(ctx),
        road_analyzer=RoadAnalyzer(ctx),
        trajectory_store=TrajectoryStore(history_frames=history_frames),
        ego_motion=EgoMotionEstimator(ctx.flag("motion.ego_motion_compensation", True)),
        movement=MovementClassifier(ctx, fps=fps),
        direction=DirectionClassifier(ctx, fps=fps),
        environment=EnvironmentAnalyzer(ctx),
        extractor=FrameExtractor(ctx),
        state=TemporalStateManager(ctx),
        annotator=Annotator(ctx),
    )

"""Stage 3: object detection."""

from __future__ import annotations

from src.detection.object_detector import (
    Detection,
    MotionObjectDetector,
    NullDetector,
    ObjectDetector,
    YoloObjectDetector,
    build_detector,
)

__all__ = [
    "Detection",
    "MotionObjectDetector",
    "NullDetector",
    "ObjectDetector",
    "YoloObjectDetector",
    "build_detector",
]

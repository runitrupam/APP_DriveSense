"""Output DTOs and the frozen per-frame JSON contract."""

from __future__ import annotations

from src.output.schemas import (
    SCHEMA_VERSION,
    UNKNOWN,
    AnomalyType,
    BBox,
    BBoxNormalized,
    Confidences,
    Decision,
    EgoVehicle,
    Environment,
    EnvironmentObject,
    Event,
    FrameAnalysis,
    FrameInfo,
    LaneBoundary,
    LaneEstimate,
    Movement,
    ObjectRecord,
    Point,
    Road,
    Validation,
)

__all__ = [
    "SCHEMA_VERSION",
    "UNKNOWN",
    "AnomalyType",
    "BBox",
    "BBoxNormalized",
    "Confidences",
    "Decision",
    "EgoVehicle",
    "Environment",
    "EnvironmentObject",
    "Event",
    "FrameAnalysis",
    "FrameInfo",
    "LaneBoundary",
    "LaneEstimate",
    "Movement",
    "ObjectRecord",
    "Point",
    "Road",
    "Validation",
]

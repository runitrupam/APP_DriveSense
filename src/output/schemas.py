"""The frozen per-frame output contract.

Downstream consumers depend on these field names. Changes are additive only, and
every enum-like value must be one of the documented strings so that ``unknown``
remains the single representation of insufficient confidence.
"""

from __future__ import annotations

import numbers
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
UNKNOWN = "unknown"

CountValue = int | str

MOVEMENT_STATES = frozenset({"moving", "stationary", UNKNOWN})
RELATIVE_DIRECTIONS = frozenset({"same_direction", "opposite_direction", "crossing", UNKNOWN})
IMAGE_DIRECTIONS = frozenset({"forward", "backward", "left", "right", "stationary", UNKNOWN})
LONGITUDINAL_DIRECTIONS = frozenset({"forward", "backward", "stationary", UNKNOWN})
LATERAL_DIRECTIONS = frozenset({"left", "right", "none", "stationary", UNKNOWN})

ENVIRONMENT_COUNT_FIELDS = (
    "pedestrians",
    "trees",
    "buildings",
    "traffic_lights",
    "traffic_signs",
    "street_lights",
    "vehicles",
    "barriers",
    "sidewalks",
)


class AnomalyType:
    """Anomaly identifiers emitted in ``Validation.anomalies`` and ``events``."""

    LANE_COUNT_JUMP = "lane_count_jump"
    TRACK_IDENTITY_SWITCH = "track_identity_switch"
    OBJECT_TELEPORTATION = "object_teleportation"
    OBJECT_DISAPPEARANCE = "object_disappearance"
    REAPPEARANCE = "reappearance"
    IMPOSSIBLE_MOTION = "impossible_motion"
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    LOW_TRACKING_CONFIDENCE = "low_tracking_confidence"
    BLURRED_FRAME = "blurred_frame"
    DUPLICATE_FRAME = "duplicate_frame"
    FRAME_DECODE_FAILURE = "frame_decode_failure"
    FORWARD_BACKWARD_DISAGREEMENT = "forward_backward_disagreement"
    UNSTABLE_LANE_ESTIMATE = "unstable_lane_estimate"

    ALL = (
        LANE_COUNT_JUMP,
        TRACK_IDENTITY_SWITCH,
        OBJECT_TELEPORTATION,
        OBJECT_DISAPPEARANCE,
        REAPPEARANCE,
        IMPOSSIBLE_MOTION,
        LOW_DETECTION_CONFIDENCE,
        LOW_TRACKING_CONFIDENCE,
        BLURRED_FRAME,
        DUPLICATE_FRAME,
        FRAME_DECODE_FAILURE,
        FORWARD_BACKWARD_DISAGREEMENT,
        UNSTABLE_LANE_ESTIMATE,
    )


class Severity:
    """Event severities."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _coerce_count(value: Any) -> CountValue:
    """Coerce a count to ``int`` or the literal ``unknown``."""
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
        return UNKNOWN
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if numeric != numeric:
            return UNKNOWN
        return int(round(numeric))
    return UNKNOWN


def _coerce_lane(value: Any) -> int | str:
    """Coerce a lane index (>= 0) or ``unknown``."""
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return UNKNOWN
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if numeric != numeric:
            return UNKNOWN
        return int(round(numeric))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
    return UNKNOWN


class BaseRecord(BaseModel):
    """Common configuration for all output records."""

    model_config = ConfigDict(extra="allow")


class BBox(BaseRecord):
    """Pixel bounding box."""

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


class BBoxNormalized(BaseRecord):
    """Bounding box normalised to 0..1 against frame width and height."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class Point(BaseRecord):
    """Integer pixel point."""

    x: int = 0
    y: int = 0


class Movement(BaseRecord):
    """Motion state and direction of an object.

    ``direction`` is relative to the ego vehicle (``same_direction`` /
    ``opposite_direction`` / ``crossing`` / ``unknown``), matching the reference
    record. ``image_direction`` is the raw image-space direction, and
    ``longitudinal`` / ``lateral`` decompose it. Anything unsupported stays
    ``unknown``.
    """

    state: str = UNKNOWN
    direction: str = UNKNOWN
    image_direction: str = UNKNOWN
    longitudinal: str = UNKNOWN
    lateral: str = UNKNOWN
    speed_px_s: float | None = None
    displacement_px: float | None = None
    confidence: float | None = None

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: str) -> str:
        return value if value in MOVEMENT_STATES else UNKNOWN

    @field_validator("direction")
    @classmethod
    def _check_direction(cls, value: str) -> str:
        return value if value in RELATIVE_DIRECTIONS else UNKNOWN

    @field_validator("image_direction")
    @classmethod
    def _check_image_direction(cls, value: str) -> str:
        return value if value in IMAGE_DIRECTIONS else UNKNOWN

    @field_validator("longitudinal")
    @classmethod
    def _check_longitudinal(cls, value: str) -> str:
        return value if value in LONGITUDINAL_DIRECTIONS else UNKNOWN

    @field_validator("lateral")
    @classmethod
    def _check_lateral(cls, value: str) -> str:
        return value if value in LATERAL_DIRECTIONS else UNKNOWN


class Confidences(BaseRecord):
    """Per-object confidence triple. ``None`` means not applicable."""

    detection: float | None = None
    tracking: float | None = None
    movement: float | None = None


class ObjectRecord(BaseRecord):
    """One tracked object in one frame."""

    track_id: str
    type: str = UNKNOWN
    bbox: BBox = Field(default_factory=BBox)
    bbox_normalized: BBoxNormalized = Field(default_factory=BBoxNormalized)
    center: Point = Field(default_factory=Point)
    lane: int | str = UNKNOWN
    movement: Movement = Field(default_factory=Movement)
    confidence: Confidences = Field(default_factory=Confidences)
    age: int | None = None
    first_frame: int | None = None
    last_frame: int | None = None
    source: str = "detector"

    @field_validator("lane", mode="before")
    @classmethod
    def _check_lane(cls, value: Any) -> int | str:
        return _coerce_lane(value)


class LaneBoundary(BaseRecord):
    """One lane boundary with pixel and normalised geometry."""

    lane_id: int
    side: str = UNKNOWN
    source: str = "hough"
    points: list[Point] = Field(default_factory=list)
    points_normalized: list[list[float]] = Field(default_factory=list)
    polynomial: list[float] | None = None
    confidence: float | None = None


class LaneEstimate(BaseRecord):
    """One lane region with its polygon and lateral placement."""

    lane_id: int
    polygon: list[Point] = Field(default_factory=list)
    polygon_normalized: list[list[float]] = Field(default_factory=list)
    center_offset_x: int | None = None
    confidence: float | None = None


class Road(BaseRecord):
    """Road and lane state for a frame."""

    lane_count: CountValue = UNKNOWN
    ego_lane: int | str = UNKNOWN
    road_type: str = UNKNOWN
    confidence: float | None = None
    curvature: float | None = None
    vanishing_point: Point | None = None
    lanes: list[LaneEstimate] = Field(default_factory=list)
    boundaries: list[LaneBoundary] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    lane_change_frame: int | None = None
    lane_change_type: str | None = None

    @field_validator("lane_count", mode="before")
    @classmethod
    def _check_lane_count(cls, value: Any) -> CountValue:
        return _coerce_count(value)

    @field_validator("ego_lane", mode="before")
    @classmethod
    def _check_ego_lane(cls, value: Any) -> int | str:
        return _coerce_lane(value)


class EgoVehicle(BaseRecord):
    """Ego vehicle state inferred from the camera stream."""

    heading: str = UNKNOWN
    lane: int | str = UNKNOWN
    speed_estimate_px_s: float | None = None
    source: str = "image_flow"

    @field_validator("lane", mode="before")
    @classmethod
    def _check_lane(cls, value: Any) -> int | str:
        return _coerce_lane(value)


class EnvironmentObject(BaseRecord):
    """A single environment element, detected or heuristically inferred."""

    type: str = UNKNOWN
    bbox: BBox = Field(default_factory=BBox)
    bbox_normalized: BBoxNormalized = Field(default_factory=BBoxNormalized)
    confidence: float | None = None
    source: str = "heuristic"
    track_id: str | None = None
    lane: int | str = UNKNOWN

    @field_validator("lane", mode="before")
    @classmethod
    def _check_lane(cls, value: Any) -> int | str:
        return _coerce_lane(value)


class Environment(BaseRecord):
    """Environment counts plus the individual records they were derived from.

    A count is ``unknown`` when the stage could not assess that class. The
    individual ``environment_objects`` records are authoritative; the counts are
    a derived convenience required by the output contract.
    """

    pedestrians: CountValue = 0
    trees: CountValue = 0
    buildings: CountValue = 0
    traffic_lights: CountValue = 0
    traffic_signs: CountValue = 0
    street_lights: CountValue = 0
    vehicles: CountValue = 0
    barriers: CountValue = 0
    sidewalks: CountValue = 0
    vegetation_ratio: float | None = None
    horizon_y: int | None = None
    confidence: float | None = None
    unknown_counts: list[str] = Field(default_factory=list)
    environment_objects: list[EnvironmentObject] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)

    @field_validator(*ENVIRONMENT_COUNT_FIELDS, mode="before")
    @classmethod
    def _check_counts(cls, value: Any) -> CountValue:
        return _coerce_count(value)


class Event(BaseRecord):
    """A notable per-frame occurrence: anomaly, transition or system message."""

    type: str
    frame_id: int
    timestamp_ms: int
    severity: str = Severity.INFO
    description: str = ""
    confidence: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Validation(BaseRecord):
    """Validation outcome for a frame."""

    status: Literal["valid", "anomalous", "failed"] = "valid"
    anomalies: list[str] = Field(default_factory=list)
    anomaly_details: list[Event] = Field(default_factory=list)
    forward_backward_consistent: bool | str = UNKNOWN
    corrected: bool = False
    unresolved: bool = False


class Decision(BaseRecord):
    """Preserved primary / validation / final results and the reason for the final one."""

    primary_result: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] = Field(default_factory=dict)
    final_result: str = "primary"
    decision_reason: str = "no_validation_disagreement"


class FrameInfo(BaseRecord):
    """Frame identity and decode quality."""

    frame_id: int
    timestamp_ms: int
    width: int = 0
    height: int = 0
    read_ok: bool = True
    blur_score: float | None = None
    duplicate_of: int | None = None
    error: str | None = None


class FrameAnalysis(BaseRecord):
    """The complete per-frame record. Exactly one is emitted per frame."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = UNKNOWN
    frame: FrameInfo
    ego_vehicle: EgoVehicle = Field(default_factory=EgoVehicle)
    road: Road = Field(default_factory=Road)
    objects: list[ObjectRecord] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    events: list[Event] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)
    decision: Decision = Field(default_factory=Decision)

    @model_validator(mode="after")
    def _ensure_anomalies_are_known(self) -> "FrameAnalysis":
        for anomaly in self.validation.anomalies:
            if anomaly not in AnomalyType.ALL:
                raise ValueError(f"unknown anomaly type in record: {anomaly}")
        return self


def count_value_or_unknown(value: Any) -> CountValue:
    """Public helper mirroring the count coercion used by the schema."""
    return _coerce_count(value)


def lane_value_or_unknown(value: Any) -> int | str:
    """Public helper mirroring the lane coercion used by the schema."""
    return _coerce_lane(value)

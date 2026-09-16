"""Object detection with a pluggable backend.

The primary backend is a road-scene YOLO model. A classical motion detector is
available as a transparent fallback so a run always completes and the manifest
records which engine actually produced the detections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

import cv2
import numpy as np

from src.utils.context import RunContext
from src.utils.geometry import bbox_area, bbox_center, bbox_to_xyxy, clip_xyxy, normalize_bbox, xyxy_to_bbox

REQUIRED_CLASSES = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "traffic_light",
    "traffic_sign",
)

CLASS_ALIASES = {
    "person": "person",
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "car": "car",
    "automobile": "car",
    "van": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "trafficlight": "traffic_light",
    "stop sign": "traffic_sign",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "trafficsign": "traffic_sign",
}

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})


def canonical_class(raw_name: str) -> str | None:
    """Map a backend class name to a required class, or ``None`` if not needed."""
    key = str(raw_name).strip().lower().replace("_", " ")
    if not key:
        return None
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    collapsed = key.replace(" ", "_")
    if collapsed in CLASS_ALIASES:
        return CLASS_ALIASES[collapsed]
    return None


@dataclass
class Detection:
    """A single detection with pixel and normalised geometry."""

    class_name: str
    confidence: float
    bbox: dict[str, int]
    bbox_normalized: dict[str, float]
    center: dict[str, int]
    frame_id: int
    timestamp_ms: int
    raw_class: str = ""
    source: str = "detector"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        """Pixel area of the box."""
        return bbox_area(self.bbox)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 6)
        payload["area"] = round(float(self.area), 3)
        return payload


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface every detection backend implements."""

    name: str

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> list[Detection]:
        """Return detections for one BGR frame."""

    def describe(self) -> dict[str, Any]:
        """Provenance for the manifest."""


def _finalize_detection(
    class_name: str,
    raw_class: str,
    confidence: float,
    xyxy: Iterable[float],
    frame_shape: tuple[int, ...],
    frame_id: int,
    timestamp_ms: int,
    source: str,
) -> Detection:
    """Build a :class:`Detection` with clipped, normalised geometry."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    clipped = clip_xyxy(tuple(xyxy), width, height)
    bbox = xyxy_to_bbox(*clipped)
    return Detection(
        class_name=class_name,
        confidence=float(confidence),
        bbox=bbox,
        bbox_normalized=normalize_bbox(bbox, width, height),
        center=bbox_center(bbox),
        frame_id=int(frame_id),
        timestamp_ms=int(timestamp_ms),
        raw_class=raw_class,
        source=source,
    )


class NullDetector:
    """Detector that returns nothing; used when detection is disabled."""

    name = "none"

    def __init__(self, **_: Any) -> None:
        self.device = "none"

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> list[Detection]:
        """Always empty."""
        return []

    def describe(self) -> dict[str, Any]:
        """Provenance block."""
        return {"engine": "none", "model": None, "device": None, "weights_loaded": False}


class YoloObjectDetector:
    """Ultralytics YOLO backend restricted to the required road-scene classes."""

    name = "yolo"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        input_size: int = 640,
        max_detections: int = 300,
        class_confidence: dict[str, float] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.requested_device = device
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)
        self.max_detections = int(max_detections)
        self.class_confidence = {str(k): float(v) for k, v in (class_confidence or {}).items()}
        self.model = YOLO(model_path)
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> str:
        """Pick cuda/mps/cpu according to availability."""
        import torch

        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> list[Detection]:
        """Run inference and map boxes to required classes."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index in range(len(boxes)):
                raw_id = int(boxes.cls[index].item())
                raw_name = str(self.names.get(raw_id, raw_id))
                class_name = canonical_class(raw_name)
                if class_name is None:
                    continue
                confidence = float(boxes.conf[index].item())
                threshold = self.class_confidence.get(
                    class_name, self.class_confidence.get("default", self.confidence_threshold)
                )
                if confidence < threshold:
                    continue
                xyxy = [float(v) for v in boxes.xyxy[index].tolist()]
                detections.append(
                    _finalize_detection(
                        class_name=class_name,
                        raw_class=raw_name,
                        confidence=confidence,
                        xyxy=xyxy,
                        frame_shape=frame.shape,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        source=self.name,
                    )
                )
        return detections

    def describe(self) -> dict[str, Any]:
        """Provenance block."""
        return {
            "engine": self.name,
            "model": self.model_path,
            "device": self.device,
            "requested_device": self.requested_device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "input_size": self.input_size,
            "weights_loaded": True,
        }


class MotionObjectDetector:
    """Classical fallback: background subtraction plus shape heuristics.

    Detects only objects that move relative to the background, and classifies
    them with low confidence, so downstream stages treat its output as weak
    evidence. It exists so the pipeline degrades gracefully instead of failing.
    """

    name = "motion"

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        min_area: int = 400,
        history: int = 200,
        var_threshold: float = 24.0,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.min_area = int(min_area)
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.device = "cpu"
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=False
        )
        self._kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> list[Detection]:
        """Return motion blobs classified as vehicles or people."""
        if frame is None or frame.size == 0:
            return []
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self._subtractor.apply(blurred)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = int(frame.shape[0]), int(frame.shape[1])
        frame_area = float(max(1, height * width))
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height <= 1:
                continue
            aspect = box_height / float(max(1, box_width))
            if aspect >= 1.6 and box_height >= 0.06 * height:
                class_name = "person"
            elif box_width >= box_height:
                class_name = "car"
            elif aspect >= 1.2:
                class_name = "motorcycle"
            else:
                class_name = "truck"
            area_ratio = area / frame_area
            confidence = float(min(0.55, 0.20 + area_ratio * 4.0))
            if confidence < self.confidence_threshold and class_name in {"car", "truck", "bus"}:
                confidence = min(self.confidence_threshold, 0.40)
            detections.append(
                _finalize_detection(
                    class_name=class_name,
                    raw_class=f"motion_blob:{class_name}",
                    confidence=confidence,
                    xyxy=(x, y, x + box_width, y + box_height),
                    frame_shape=frame.shape,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    source=self.name,
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def describe(self) -> dict[str, Any]:
        """Provenance block."""
        return {
            "engine": self.name,
            "model": "MOG2+shape-heuristics",
            "device": "cpu",
            "confidence_threshold": self.confidence_threshold,
            "min_area": self.min_area,
            "weights_loaded": False,
            "limitations": "detects only motion relative to the background; low-confidence classes",
        }


def build_detector(ctx: RunContext) -> ObjectDetector:
    """Instantiate the configured detector, falling back when unavailable.

    ``engine: auto`` tries YOLO first and silently degrades to the classical
    motion detector. The chosen engine is recorded in the returned object's
    ``describe()`` block, which the manifest stores.
    """
    engine = str(ctx.get("detection.engine", "auto")).strip().lower()
    confidence = float(ctx.get("detection.confidence_threshold", 0.40))
    class_confidence = ctx.config.section("detection").get("class_confidence", {}) or {}

    if engine in {"none", "null", "disabled"}:
        return NullDetector()

    if engine in {"auto", "yolo", "ultralytics"}:
        try:
            detector = YoloObjectDetector(
                model_path=str(ctx.get("detection.model", "yolov8n.pt")),
                device=str(ctx.get("detection.device", "auto")),
                confidence_threshold=confidence,
                iou_threshold=float(ctx.get("detection.iou_threshold", 0.50)),
                input_size=int(ctx.get("detection.input_size", 640)),
                max_detections=int(ctx.get("detection.max_detections", 300)),
                class_confidence=class_confidence,
            )
            ctx.logger.info("detector=yolo model=%s device=%s", detector.model_path, detector.device)
            return detector
        except Exception as exc:
            if engine == "yolo":
                ctx.logger.warning("yolo detector unavailable (%s); falling back to motion detector", exc)
            else:
                ctx.logger.info("yolo detector unavailable (%s); using motion detector", exc)
            ctx.metrics.note("detection", f"yolo_unavailable: {type(exc).__name__}")

    return MotionObjectDetector(confidence_threshold=confidence)

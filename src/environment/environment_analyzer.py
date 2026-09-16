"""Deterministic environment analysis.

Vehicles, pedestrians, traffic lights and traffic signs come from detections.
Trees, buildings, street lights, sidewalks and barriers have no trained detector
in V1, so they are inferred from colour and edge structure. Every item records
its ``source`` and a confidence, and a class the heuristics cannot assess is
reported as ``unknown`` rather than zero, so a future model can replace this
stage without changing the output contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from src.utils.context import RunContext
from src.utils.geometry import xyxy_to_bbox
from src.utils.image import (
    connected_components,
    excess_green_index,
    hsv_channels,
    resize_to_width,
)

WORKING_WIDTH = 640
VEGETATION_EXG_THRESHOLD = 140
SKY_VALUE_THRESHOLD = 140.0
SKY_SATURATION_THRESHOLD = 85.0
BRIGHT_VALUE_THRESHOLD = 205
BRIGHT_SATURATION_THRESHOLD = 70
BUILDING_CELL = 32
BUILDING_MIN_CELLS = 3


@dataclass
class EnvironmentItem:
    """One environment element with its provenance."""

    type: str
    bbox: dict[str, int]
    confidence: float | None = None
    source: str = "heuristic"
    track_id: str | None = None
    lane: int | str = "unknown"
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        payload = asdict(self)
        payload.pop("features", None)
        return payload


@dataclass
class EnvironmentObservation:
    """Environment summary for one frame."""

    frame_id: int
    timestamp_ms: int
    items: list[EnvironmentItem] = field(default_factory=list)
    counts: dict[str, int | None] = field(default_factory=dict)
    vegetation_ratio: float | None = None
    horizon_y: int | None = None
    confidence: float | None = None
    features: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def items_of(self, item_type: str) -> list[EnvironmentItem]:
        """Items of one type."""
        return [item for item in self.items if item.type == item_type]

    def to_decision_summary(self) -> dict[str, Any]:
        """Compact summary for the per-frame decision block."""
        return {
            "horizon_y": self.horizon_y,
            "vegetation_ratio": None
            if self.vegetation_ratio is None
            else round(float(self.vegetation_ratio), 4),
            "item_count": len(self.items),
        }


class EnvironmentAnalyzer:
    """Colour and edge based environment inference."""

    name = "heuristic_environment"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.enabled = ctx.flag("environment.enabled", True)
        self.min_vegetation_area = int(ctx.get("environment.min_vegetation_component_px", 900))
        self.horizon_search_ratio = float(ctx.get("environment.horizon_search_ratio", 0.35))
        self.building_edge_threshold = float(
            ctx.get("environment.building_edge_density_threshold", 0.12)
        )

    def analyze(
        self,
        frame: np.ndarray | None,
        frame_id: int,
        timestamp_ms: int,
        object_items: list[EnvironmentItem] | None = None,
        vanishing_point: tuple[float, float] | None = None,
    ) -> EnvironmentObservation:
        """Analyse one frame, merging detection-derived and heuristic items."""
        observation = EnvironmentObservation(
            frame_id=int(frame_id),
            timestamp_ms=int(timestamp_ms),
            enabled=self.enabled,
        )
        detection_items = list(object_items or [])
        observation.items.extend(detection_items)

        if not self.enabled:
            observation.counts = self._counts_for(observation.items)
            return observation

        if frame is None or frame.size == 0:
            observation.counts = self._counts_for(observation.items)
            observation.confidence = 0.0
            return observation

        height, width = int(frame.shape[0]), int(frame.shape[1])
        working = resize_to_width(frame, WORKING_WIDTH)
        scale_x = width / float(working.shape[1])
        scale_y = height / float(working.shape[0])
        work_height, work_width = int(working.shape[0]), int(working.shape[1])

        horizon = self._estimate_horizon(working, vanishing_point, scale_y)
        observation.horizon_y = int(horizon)

        vegetation, vegetation_ratio = self._vegetation(working)
        observation.vegetation_ratio = round(float(vegetation_ratio), 6)

        observation.items.extend(
            self._trees(vegetation, work_width, horizon, scale_x, scale_y)
        )
        observation.items.extend(self._buildings(working, vegetation, horizon, scale_x, scale_y))
        observation.items.extend(self._street_lights(working, horizon, scale_x, scale_y))
        observation.items.extend(self._sidewalks(working, vegetation, scale_x, scale_y))
        observation.items.extend(self._barriers(working, scale_x, scale_y))

        observation.counts = self._counts_for(observation.items)
        observation.confidence = self._confidence(observation)
        observation.features = {
            "working_resolution": [work_width, work_height],
            "horizon_ratio": round(float(horizon) / max(1, work_height), 4),
            "vegetation_ratio": round(float(vegetation_ratio), 4),
            "items_by_type": {
                item_type: len(observation.items_of(item_type))
                for item_type in sorted({item.type for item in observation.items})
            },
        }
        self._record_metrics(observation)
        return observation

    def _estimate_horizon(
        self,
        working: np.ndarray,
        vanishing_point: tuple[float, float] | None,
        scale_y: float,
    ) -> float:
        """Horizon row in working coordinates from sky detection or the vanishing point."""
        height = int(working.shape[0])
        _, saturation, value = hsv_channels(working)
        row_value = value.mean(axis=1)
        row_saturation = saturation.mean(axis=1)
        sky_rows = np.where(
            (row_value > SKY_VALUE_THRESHOLD) & (row_saturation < SKY_SATURATION_THRESHOLD)
        )[0]
        horizon: float | None = None
        if sky_rows.size > 0:
            prefix_end = 0
            for index in range(height):
                if index in sky_rows:
                    prefix_end = index
                else:
                    break
            if prefix_end > 0:
                horizon = float(prefix_end) + 0.02 * height
        if horizon is None and vanishing_point is not None:
            candidate = float(vanishing_point[1]) / max(1e-6, scale_y)
            if 0.1 * height < candidate < 0.9 * height:
                horizon = candidate
        if horizon is None:
            horizon = 0.45 * height
        lower = self.horizon_search_ratio * height
        upper = 0.75 * height
        return float(np.clip(horizon, lower, upper))

    def _vegetation(self, working: np.ndarray) -> tuple[np.ndarray, float]:
        """Excess-green vegetation mask and its pixel ratio."""
        exg = excess_green_index(working)
        mask = (exg > VEGETATION_EXG_THRESHOLD).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
        return mask, ratio

    def _trees(
        self,
        vegetation: np.ndarray,
        work_width: int,
        horizon: float,
        scale_x: float,
        scale_y: float,
    ) -> list[EnvironmentItem]:
        """Large vegetation components read as tree canopies."""
        items: list[EnvironmentItem] = []
        for x, y, width, height, area in connected_components(
            vegetation, float(self.min_vegetation_area)
        ):
            if width < 12 or height < 12:
                continue
            coverage = area / float(max(1, width * height))
            confidence = float(np.clip(0.25 + 0.4 * min(1.0, area / 12000.0) + 0.2 * coverage, 0.0, 0.85))
            items.append(
                EnvironmentItem(
                    type="tree",
                    bbox=self._scale_bbox(x, y, width, height, scale_x, scale_y),
                    confidence=confidence,
                    source="heuristic_exg",
                    features={"area_working_px": round(area, 1), "fill_ratio": round(coverage, 4)},
                )
            )
        return items

    def _buildings(
        self,
        working: np.ndarray,
        vegetation: np.ndarray,
        horizon: float,
        scale_x: float,
        scale_y: float,
    ) -> list[EnvironmentItem]:
        """Clusters of high vertical-edge density above the horizon read as buildings."""
        height, width = int(working.shape[0]), int(working.shape[1])
        top = 0
        bottom = int(max(1, min(height, horizon)))
        if bottom - top < BUILDING_CELL * 2:
            return []
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        strong = (grad_x > 80.0).astype(np.float32)

        rows = (bottom - top) // BUILDING_CELL
        cols = width // BUILDING_CELL
        if rows < 1 or cols < 2:
            return []
        cell_mask = np.zeros((rows, cols), dtype=np.uint8)
        vegetation_small = cv2.resize(
            vegetation, (cols, rows), interpolation=cv2.INTER_AREA
        )
        for row in range(rows):
            for col in range(cols):
                y0 = top + row * BUILDING_CELL
                y1 = y0 + BUILDING_CELL
                x0 = col * BUILDING_CELL
                x1 = x0 + BUILDING_CELL
                density = float(strong[y0:y1, x0:x1].mean())
                vegetation_ratio = float(vegetation_small[row, col]) / 255.0
                if density > self.building_edge_threshold and vegetation_ratio < 0.25:
                    cell_mask[row, col] = 255

        components: list[tuple[int, int, int, int, float]] = []
        count, _, stats, _ = cv2.connectedComponentsWithStats(cell_mask, connectivity=8)
        for index in range(1, count):
            x, y, cell_width, cell_height, area = (int(v) for v in stats[index])
            if area >= BUILDING_MIN_CELLS:
                components.append((x, y, cell_width, cell_height, float(area)))

        items: list[EnvironmentItem] = []
        for x, y, cell_width, cell_height, area in components:
            bbox = self._scale_bbox(
                x * BUILDING_CELL,
                y * BUILDING_CELL,
                cell_width * BUILDING_CELL,
                cell_height * BUILDING_CELL,
                scale_x,
                scale_y,
            )
            confidence = float(np.clip(0.25 + 0.05 * area, 0.0, 0.7))
            items.append(
                EnvironmentItem(
                    type="building",
                    bbox=bbox,
                    confidence=confidence,
                    source="heuristic_vertical_edges",
                    features={"cells": int(area)},
                )
            )
        return items

    def _street_lights(
        self,
        working: np.ndarray,
        horizon: float,
        scale_x: float,
        scale_y: float,
    ) -> list[EnvironmentItem]:
        """Bright small blobs supported by a vertical pole read as street lights."""
        height, width = int(working.shape[0]), int(working.shape[1])
        _, saturation, value = hsv_channels(working)
        bright = ((value > BRIGHT_VALUE_THRESHOLD) & (saturation < BRIGHT_SATURATION_THRESHOLD)).astype(
            np.uint8
        ) * 255
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

        items: list[EnvironmentItem] = []
        for x, y, blob_width, blob_height, area in connected_components(bright, 6.0, max_components=60):
            if area > 900 or blob_height < 2:
                continue
            if y + blob_height > horizon * 1.35:
                continue
            pole_top = min(height - 1, y + blob_height)
            pole_bottom = min(height - 1, pole_top + int(3.2 * blob_height) + 6)
            if pole_bottom - pole_top < 5:
                continue
            center_x = x + blob_width // 2
            half_width = max(2, blob_width // 2)
            x0 = max(0, center_x - half_width)
            x1 = min(width, center_x + half_width)
            pole = grad_x[pole_top:pole_bottom, x0:x1]
            if pole.size == 0:
                continue
            if float((pole > 60.0).mean()) < 0.08:
                continue
            bbox = self._scale_bbox(x, y, blob_width, blob_height, scale_x, scale_y)
            items.append(
                EnvironmentItem(
                    type="street_light",
                    bbox=bbox,
                    confidence=0.30,
                    source="heuristic_bright_blob_with_pole",
                    features={"area_working_px": round(area, 1)},
                )
            )
        return items

    def _sidewalks(
        self,
        working: np.ndarray,
        vegetation: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> list[EnvironmentItem]:
        """Low-saturation, low-vegetation flanks of the bottom band read as sidewalks."""
        height, width = int(working.shape[0]), int(working.shape[1])
        top = int(0.72 * height)
        flank = max(4, int(0.14 * width))
        _, saturation, _ = hsv_channels(working)
        items: list[EnvironmentItem] = []
        for side, x0, x1 in (("left", 0, flank), ("right", width - flank, width)):
            band = slice(top, height)
            saturation_mean = float(saturation[band, x0:x1].mean())
            vegetation_ratio = float(np.count_nonzero(vegetation[band, x0:x1])) / float(
                max(1, vegetation[band, x0:x1].size)
            )
            if saturation_mean > 95.0 or vegetation_ratio > 0.15:
                continue
            items.append(
                EnvironmentItem(
                    type="sidewalk",
                    bbox=self._scale_bbox(x0, top, x1 - x0, height - top, scale_x, scale_y),
                    confidence=0.25,
                    source="heuristic_low_saturation_flank",
                    features={
                        "side": side,
                        "saturation_mean": round(saturation_mean, 2),
                        "vegetation_ratio": round(vegetation_ratio, 4),
                    },
                )
            )
        return items

    def _barriers(
        self,
        working: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> list[EnvironmentItem]:
        """Runs of rows dominated by horizontal edges read as barriers or guardrails."""
        height, width = int(working.shape[0]), int(working.shape[1])
        top = int(0.35 * height)
        bottom = int(0.92 * height)
        if bottom - top < 8:
            return []
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        strong = grad_y > 70.0
        row_ratio = strong[top:bottom].mean(axis=1)
        run_start: int | None = None
        items: list[EnvironmentItem] = []
        for offset, ratio in enumerate(row_ratio):
            if ratio > 0.28:
                if run_start is None:
                    run_start = offset
            else:
                if run_start is not None and offset - run_start >= 4:
                    items.append(self._barrier_item(strong, top + run_start, top + offset, width, scale_x, scale_y))
                run_start = None
        if run_start is not None and len(row_ratio) - run_start >= 4:
            items.append(
                self._barrier_item(strong, top + run_start, bottom, width, scale_x, scale_y)
            )
        return items

    def _barrier_item(
        self,
        strong: np.ndarray,
        y0: int,
        y1: int,
        width: int,
        scale_x: float,
        scale_y: float,
    ) -> EnvironmentItem:
        """Bounding box of a horizontal-edge run, trimmed to the occupied columns."""
        band = strong[y0:y1, :]
        column_ratio = band.mean(axis=0)
        columns = np.where(column_ratio > 0.2)[0]
        x0 = int(columns.min()) if columns.size else 0
        x1 = int(columns.max()) + 1 if columns.size else width
        return EnvironmentItem(
            type="barrier",
            bbox=self._scale_bbox(x0, y0, max(1, x1 - x0), max(1, y1 - y0), scale_x, scale_y),
            confidence=0.30,
            source="heuristic_horizontal_edges",
            features={"row_span": int(y1 - y0), "column_span": int(x1 - x0)},
        )

    @staticmethod
    def _scale_bbox(
        x: int,
        y: int,
        width: int,
        height: int,
        scale_x: float,
        scale_y: float,
    ) -> dict[str, int]:
        """Scale a working-resolution box back to frame coordinates."""
        return xyxy_to_bbox(x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y)

    def _counts_for(self, items: list[EnvironmentItem]) -> dict[str, int | None]:
        """Derive the required counts from the individual items."""
        counts: dict[str, int | None] = {
            "pedestrians": 0,
            "trees": 0,
            "buildings": 0,
            "traffic_lights": 0,
            "traffic_signs": 0,
            "street_lights": 0,
            "vehicles": 0,
            "barriers": 0,
            "sidewalks": 0,
        }
        for item in items:
            if item.type == "person":
                counts["pedestrians"] = (counts["pedestrians"] or 0) + 1
            elif item.type in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                counts["vehicles"] = (counts["vehicles"] or 0) + 1
            elif item.type == "traffic_light":
                counts["traffic_lights"] = (counts["traffic_lights"] or 0) + 1
            elif item.type == "traffic_sign":
                counts["traffic_signs"] = (counts["traffic_signs"] or 0) + 1
            elif item.type == "tree":
                counts["trees"] = (counts["trees"] or 0) + 1
            elif item.type == "building":
                counts["buildings"] = (counts["buildings"] or 0) + 1
            elif item.type == "street_light":
                counts["street_lights"] = (counts["street_lights"] or 0) + 1
            elif item.type == "barrier":
                counts["barriers"] = (counts["barriers"] or 0) + 1
            elif item.type == "sidewalk":
                counts["sidewalks"] = (counts["sidewalks"] or 0) + 1
        return counts

    def _confidence(self, observation: EnvironmentObservation) -> float:
        """Frame confidence from evidence volume and heuristic agreement."""
        if not observation.items:
            return 0.15
        confidence = 0.30
        if observation.horizon_y is not None:
            confidence += 0.15
        if observation.vegetation_ratio is not None and observation.vegetation_ratio > 0:
            confidence += 0.10
        heuristic_weights = [item.confidence or 0.0 for item in observation.items]
        confidence += 0.25 * float(np.mean(heuristic_weights)) if heuristic_weights else 0.0
        return float(np.clip(confidence, 0.0, 0.8))

    def _record_metrics(self, observation: EnvironmentObservation) -> None:
        """Persist per-frame environment metrics."""
        metrics = self.ctx.metrics
        metrics.increment("environment", "frames_processed")
        metrics.increment("environment", "items_total", len(observation.items))
        for item_type, count in observation.counts.items():
            if count is None:
                continue
            metrics.increment("environment", f"count.{item_type}", float(count))
        if observation.vegetation_ratio is not None:
            metrics.observe("environment", "vegetation_ratio", observation.vegetation_ratio)
        if observation.horizon_y is not None:
            metrics.observe("environment", "horizon_y", float(observation.horizon_y))
        if observation.confidence is not None:
            metrics.observe("environment", "confidence", observation.confidence)

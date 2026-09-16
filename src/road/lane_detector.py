"""Classical geometric lane detection.

The detector reports geometry it can actually support and returns ``None`` for
lane count and ego lane when the evidence is weak, so callers emit ``unknown``
instead of guessing. Confidence combines how many scanlines agree on each
boundary with how regular the resulting lane spacing is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.utils.context import RunContext
from src.utils.geometry import polyval_x_of_y

SCANLINE_RATIOS = (0.55, 0.62, 0.70, 0.78, 0.86, 0.94)
REFERENCE_Y_RATIO = 0.90


@dataclass
class LaneLine:
    """One fitted lane boundary."""

    side: str
    points: list[tuple[int, int]] = field(default_factory=list)
    polynomial: list[float] | None = None
    slope: float | None = None
    intercept: float | None = None
    confidence: float = 0.0
    source: str = "hough"

    def x_at(self, y: float) -> float | None:
        """Lateral position of the boundary at a given image row."""
        if self.polynomial:
            return float(polyval_x_of_y(self.polynomial, y))
        if self.slope is not None and self.intercept is not None and abs(self.slope) > 1e-6:
            return float((y - self.intercept) / self.slope)
        return None

    def to_dict(self, width: int, height: int) -> dict[str, Any]:
        """JSON-ready geometry with pixel and normalised points."""
        normalized = [
            [round(x / max(1, width), 6), round(y / max(1, height), 6)] for x, y in self.points
        ]
        return {
            "side": self.side,
            "source": self.source,
            "points": [{"x": int(x), "y": int(y)} for x, y in self.points],
            "points_normalized": normalized,
            "polynomial": self.polynomial,
            "slope": None if self.slope is None else round(float(self.slope), 6),
            "confidence": round(float(self.confidence), 6),
        }


@dataclass
class LaneObservation:
    """Lane geometry for one frame."""

    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    lane_count: int | None = None
    ego_lane: int | None = None
    lane_count_confidence: float = 0.0
    boundaries: list[LaneLine] = field(default_factory=list)
    boundary_positions: list[float] = field(default_factory=list)
    lanes: list[dict[str, Any]] = field(default_factory=list)
    vanishing_point: tuple[float, float] | None = None
    curvature: float | None = None
    lane_width_px: float | None = None
    horizontal_edge_ratio: float | None = None
    roi: tuple[int, int, int, int] | None = None
    success: bool = False
    reason: str = ""

    @property
    def confidence(self) -> float:
        """Frame-level lane confidence."""
        return round(float(self.lane_count_confidence), 6)

    def geometry_summary(self) -> dict[str, Any]:
        """Compact geometry for the forward/backward decision records."""
        return {
            "lane_count": self.lane_count,
            "ego_lane": self.ego_lane,
            "boundaries": len(self.boundaries),
            "confidence": self.confidence,
        }


class LaneDetector:
    """Detects lane boundaries, lane count and the ego lane."""

    name = "classical_hough"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.roi_top_ratio = float(ctx.get("road.roi_top_ratio", 0.55))
        self.canny_low = int(ctx.get("road.canny_low", 60))
        self.canny_high = int(ctx.get("road.canny_high", 180))
        self.kernel_size = int(ctx.get("road.kernel_size", 5))
        self.hough_threshold = int(ctx.get("road.hough_threshold", 40))
        self.hough_min_length_ratio = float(ctx.get("road.hough_min_line_length_ratio", 0.08))
        self.hough_max_gap = int(ctx.get("road.hough_max_line_gap", 60))
        self.min_lane_width_px = int(ctx.get("road.min_lane_width_px", 40))
        self.max_lanes = int(ctx.get("road.max_lanes", 8))
        self.confidence_threshold = float(ctx.get("road.lane_detection_confidence_threshold", 0.35))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> LaneObservation:
        """Detect lane geometry for one BGR frame."""
        if frame is None or frame.size == 0:
            return LaneObservation(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                width=0,
                height=0,
                success=False,
                reason="empty_frame",
            )

        height, width = int(frame.shape[0]), int(frame.shape[1])
        observation = LaneObservation(
            frame_id=int(frame_id),
            timestamp_ms=int(timestamp_ms),
            width=width,
            height=height,
        )
        roi_top = int(min(max(self.roi_top_ratio, 0.2), 0.9) * height)
        roi = (0, roi_top, width, height - roi_top)
        observation.roi = roi

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        enhanced = self._clahe.apply(blurred)
        roi_mask = np.zeros_like(enhanced)
        cv2.fillPoly(
            roi_mask,
            [self._roi_polygon(width, height, roi_top)],
            color=255,
        )
        masked = cv2.bitwise_and(enhanced, roi_mask)
        edges = cv2.Canny(masked, self.canny_low, self.canny_high)
        observation.horizontal_edge_ratio = self._horizontal_edge_ratio(masked, edges)

        segments = self._hough_segments(edges, roi_top, height)
        left_line, right_line = self._fit_sides(segments, roi_top, height, width)

        peaks = self._scanline_boundaries(gray, roi_top, height, width)
        hough_positions = self._hough_boundary_positions(left_line, right_line, height)
        positions = self._merge_positions(peaks, hough_positions, width)

        observation.boundaries = [line for line in (left_line, right_line) if line is not None]
        observation.boundary_positions = positions
        observation.lane_width_px = self._estimate_lane_width(positions)

        lane_count, count_confidence, lanes = self._estimate_lanes(positions, width)
        observation.lanes = lanes
        observation.lane_count = lane_count
        observation.lane_count_confidence = count_confidence
        observation.ego_lane = self._estimate_ego_lane(lanes, positions, width)

        observation.vanishing_point = self._vanishing_point(left_line, right_line)
        observation.curvature = self._curvature(left_line, right_line)
        geometry_confidence = self._combine_confidence(left_line, right_line, count_confidence)
        observation.lane_count_confidence = round(geometry_confidence, 6)
        observation.success = lane_count is not None or bool(observation.boundaries)
        if not observation.success:
            observation.reason = "no_lane_evidence"

        self._record_metrics(observation)
        return observation

    def _horizontal_edge_ratio(self, enhanced: np.ndarray, edges: np.ndarray) -> float | None:
        """Fraction of edge pixels whose gradient is near-horizontal.

        Crosswalk bars, stop lines and intersection markings push this up, which
        the road analyzer uses as weak evidence for intersections.
        """
        if enhanced is None or edges is None or edges.size == 0:
            return None
        mask = edges > 0
        count = int(np.count_nonzero(mask))
        if count < 50:
            return None
        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        angle = np.abs(np.degrees(np.arctan2(grad_y[mask], grad_x[mask])))
        horizontal = np.count_nonzero((angle < 30.0) | (angle > 150.0))
        return float(horizontal) / float(count)

    def _roi_polygon(self, width: int, height: int, roi_top: int) -> np.ndarray:
        """Trapezoidal region of interest covering the road ahead."""
        return np.array(
            [
                [(int(0.02 * width), height - 1), (int(0.98 * width), height - 1)],
                [(int(0.65 * width), roi_top), (int(0.35 * width), roi_top)],
            ],
            dtype=np.int32,
        )

    def _hough_segments(
        self, edges: np.ndarray, roi_top: int, height: int
    ) -> list[tuple[int, int, int, int]]:
        """Probabilistic Hough segments with a plausible lane orientation."""
        min_length = max(12, int(self.hough_min_length_ratio * max(1, height - roi_top)))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=min_length,
            maxLineGap=self.hough_max_gap,
        )
        segments: list[tuple[int, int, int, int]] = []
        if lines is None:
            return segments
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = (int(v) for v in line)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6:
                continue
            if 1.0 * abs(dy) < 0.35 * abs(dx):
                continue
            segments.append((x1, y1, x2, y2))
        return segments

    def _fit_sides(
        self,
        segments: Sequence[tuple[int, int, int, int]],
        roi_top: int,
        height: int,
        width: int,
    ) -> tuple[LaneLine | None, LaneLine | None]:
        """Fit a left and a right lane boundary from Hough segments."""
        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        center_x = width / 2.0
        for x1, y1, x2, y2 in segments:
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.15:
                continue
            midpoint_x = (x1 + x2) / 2.0
            points = [(x1, y1), (x2, y2)]
            if slope < 0 and midpoint_x <= center_x * 1.05:
                left_points.extend(points)
            elif slope > 0 and midpoint_x >= center_x * 0.95:
                right_points.extend(points)

        left = self._build_line("left", left_points, roi_top, height)
        right = self._build_line("right", right_points, roi_top, height)
        return left, right

    def _build_line(
        self,
        side: str,
        points: list[tuple[int, int]],
        roi_top: int,
        height: int,
    ) -> LaneLine | None:
        """Fit and sample a boundary polyline from collected edge points."""
        if len(points) < 4:
            return None
        points = self._dedupe(points)
        if len(points) < 4:
            return None
        ys = np.array([p[1] for p in points], dtype=np.float64)
        xs = np.array([p[0] for p in points], dtype=np.float64)
        degree = 2 if len(points) >= 12 and float(np.ptp(ys)) > 20 else 1
        try:
            coeffs = [float(c) for c in np.polyfit(ys, xs, degree)]
        except Exception:
            return None

        sample_ys = np.linspace(roi_top, height - 1, num=8)
        sampled: list[tuple[int, int]] = []
        for y in sample_ys:
            x = polyval_x_of_y(coeffs, float(y))
            if not np.isfinite(x):
                continue
            sampled.append((int(np.clip(x, 0, 4096)), int(y)))
        if len(sampled) < 3:
            return None

        if degree == 1:
            slope = 1.0 / coeffs[0] if abs(coeffs[0]) > 1e-6 else None
            intercept = -coeffs[1] * slope if slope is not None else None
        else:
            slope, intercept = self._linearise(coeffs, height)

        support = min(1.0, len(points) / 30.0)
        spread = min(1.0, float(np.ptp(ys)) / max(1.0, float(height - roi_top)))
        confidence = float(np.clip(0.35 + 0.4 * support + 0.25 * spread, 0.0, 1.0))
        return LaneLine(
            side=side,
            points=sampled,
            polynomial=coeffs,
            slope=slope,
            intercept=intercept,
            confidence=confidence,
            source="hough_polyfit",
        )

    @staticmethod
    def _linearise(coeffs: list[float], height: int) -> tuple[float | None, float | None]:
        """Local linear slope/intercept near the bottom of the frame."""
        if not coeffs:
            return None, None
        derivative = 2.0 * coeffs[0] * float(height) + (coeffs[1] if len(coeffs) > 1 else 0.0)
        dx_dy = derivative
        if abs(dx_dy) < 1e-6:
            return None, None
        slope = 1.0 / dx_dy
        x_bottom = polyval_x_of_y(coeffs, float(height))
        intercept = float(height) - slope * x_bottom
        return float(slope), float(intercept)

    @staticmethod
    def _dedupe(points: list[tuple[int, int]], bucket: int = 8) -> list[tuple[int, int]]:
        """Collapse near-duplicate points from overlapping Hough segments."""
        seen: dict[tuple[int, int], tuple[int, int]] = {}
        for x, y in points:
            key = (int(x) // bucket, int(y) // bucket)
            seen.setdefault(key, (x, y))
        return list(seen.values())

    def _scanline_boundaries(
        self, gray: np.ndarray, roi_top: int, height: int, width: int
    ) -> list[float]:
        """Cluster strong lateral gradient peaks across several scanlines."""
        clusters: list[dict[str, float]] = []
        scanlines = [int(r * height) for r in SCANLINE_RATIOS if int(r * height) > roi_top]
        if not scanlines:
            return []
        tolerance = max(10.0, 0.05 * width)
        for y in scanlines:
            row = gray[max(0, y - 1) : min(height, y + 2), : width].astype(np.float32).mean(axis=0)
            gradient = np.abs(np.diff(row))
            if gradient.size < 3:
                continue
            peak_height = float(gradient.max())
            if peak_height < 6.0:
                continue
            threshold = max(0.45 * peak_height, 8.0)
            for index in range(1, gradient.size - 1):
                value = float(gradient[index])
                if value < threshold:
                    continue
                if value < gradient[index - 1] or value < gradient[index + 1]:
                    continue
                x = float(index)
                best = None
                best_distance = tolerance
                for cluster in clusters:
                    distance = abs(cluster["mean_x"] - x)
                    if distance < best_distance:
                        best = cluster
                        best_distance = distance
                if best is None:
                    clusters.append({"mean_x": x, "count": 1.0, "strength": value, "sum_x": x})
                else:
                    best["sum_x"] += x
                    best["count"] += 1.0
                    best["strength"] += value
                    best["mean_x"] = best["sum_x"] / best["count"]

        minimum_support = max(2.0, 0.34 * len(scanlines))
        accepted = [c for c in clusters if c["count"] >= minimum_support]
        accepted.sort(key=lambda cluster: cluster["mean_x"])
        self._scanline_support = {round(c["mean_x"], 3): c["count"] for c in accepted}
        return [float(c["mean_x"]) for c in accepted]

    def _hough_boundary_positions(
        self,
        left: LaneLine | None,
        right: LaneLine | None,
        height: int,
    ) -> list[float]:
        """Boundary x positions from Hough fits at a reference scanline."""
        reference_y = REFERENCE_Y_RATIO * height
        positions: list[float] = []
        for line in (left, right):
            if line is None:
                continue
            x = line.x_at(reference_y)
            if x is not None and np.isfinite(x):
                positions.append(float(x))
        return positions

    def _merge_positions(
        self, peaks: list[float], hough_positions: list[float], width: int
    ) -> list[float]:
        """Merge independent boundary estimates, preferring scanline clusters."""
        tolerance = max(12.0, 0.04 * width)
        merged = list(peaks)
        for position in hough_positions:
            if all(abs(position - existing) > tolerance for existing in merged):
                merged.append(float(position))
        merged.sort()
        return [float(np.clip(p, 0, width - 1)) for p in merged]

    def _estimate_lane_width(self, positions: Sequence[float]) -> float | None:
        """Median gap between adjacent boundaries."""
        if len(positions) < 2:
            return None
        gaps = [b - a for a, b in zip(positions, positions[1:]) if b - a > 0]
        if not gaps:
            return None
        return float(np.median(gaps))

    def _estimate_lanes(
        self, positions: Sequence[float], width: int
    ) -> tuple[int | None, float, list[dict[str, Any]]]:
        """Derive lane count, confidence and lane regions from boundary positions."""
        if len(positions) < 2:
            return None, 0.0, []

        gaps = [b - a for a, b in zip(positions, positions[1:])]
        usable = [gap for gap in gaps if gap >= self.min_lane_width_px]
        if not usable:
            return None, 0.0, []
        median_gap = float(np.median(usable))
        consistent = [gap for gap in gaps if 0.5 * median_gap <= gap <= 1.8 * median_gap]
        lane_count = len(consistent)
        if lane_count < 1 or lane_count > self.max_lanes:
            return None, 0.0, []

        regularity = 1.0 - min(1.0, float(np.std(consistent)) / max(1.0, median_gap))
        coverage = min(1.0, len(usable) / max(1, lane_count))
        confidence = float(np.clip(0.45 + 0.35 * regularity + 0.20 * coverage, 0.0, 1.0))
        if lane_count == 1:
            confidence *= 0.7

        lanes: list[dict[str, Any]] = []
        for index in range(lane_count):
            left_x = positions[index]
            right_x = positions[index + 1]
            lanes.append(
                {
                    "lane_id": index + 1,
                    "left_x": float(left_x),
                    "right_x": float(right_x),
                    "center_x": float((left_x + right_x) / 2.0),
                    "width_px": float(right_x - left_x),
                }
            )
        return lane_count, confidence, lanes

    def _estimate_ego_lane(
        self,
        lanes: Sequence[dict[str, Any]],
        positions: Sequence[float],
        width: int,
    ) -> int | None:
        """Lane containing the bottom-centre of the frame."""
        if not lanes:
            return None
        center_x = width / 2.0
        for lane in lanes:
            if float(lane["left_x"]) <= center_x <= float(lane["right_x"]):
                return int(lane["lane_id"])
        if positions:
            left_of_center = sum(1 for position in positions if position < center_x)
            candidate = left_of_center
            if 1 <= candidate <= len(lanes):
                return int(candidate)
        return None

    def _vanishing_point(
        self, left: LaneLine | None, right: LaneLine | None
    ) -> tuple[float, float] | None:
        """Intersection of the two boundary fits, when stable."""
        if left is None or right is None:
            return None
        if left.slope is None or right.slope is None:
            return None
        if left.intercept is None or right.intercept is None:
            return None
        denominator = float(left.slope) - float(right.slope)
        if abs(denominator) < 1e-3:
            return None
        x = (float(right.intercept) - float(left.intercept)) / denominator
        y = float(left.slope) * x + float(left.intercept)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return (float(x), float(y))

    def _curvature(self, left: LaneLine | None, right: LaneLine | None) -> float | None:
        """Mean quadratic coefficient of the boundary fits."""
        coefficients = [
            line.polynomial[0]
            for line in (left, right)
            if line is not None and line.polynomial and len(line.polynomial) >= 3
        ]
        if not coefficients:
            return None
        return float(np.mean(coefficients))

    def _combine_confidence(
        self,
        left: LaneLine | None,
        right: LaneLine | None,
        count_confidence: float,
    ) -> float:
        """Frame confidence from boundary support and lane estimate quality."""
        line_confidences = [line.confidence for line in (left, right) if line is not None]
        if not line_confidences:
            return round(count_confidence * 0.3, 6)
        line_score = float(np.mean(line_confidences))
        pair_bonus = 0.15 if len(line_confidences) == 2 else 0.0
        return float(np.clip(0.5 * count_confidence + 0.5 * line_score + pair_bonus, 0.0, 1.0))

    def _record_metrics(self, observation: LaneObservation) -> None:
        """Persist per-frame lane metrics."""
        metrics = self.ctx.metrics
        metrics.increment("lane", "frames_processed")
        if observation.success:
            metrics.increment("lane", "frames_with_lane_evidence")
        if observation.lane_count is not None:
            metrics.increment("lane", "frames_with_lane_count")
            metrics.observe("lane", "lane_count", float(observation.lane_count))
        if observation.ego_lane is not None:
            metrics.increment("lane", "frames_with_ego_lane")
        metrics.observe("lane", "lane_confidence", observation.confidence)
        if observation.vanishing_point is not None:
            metrics.increment("lane", "frames_with_vanishing_point")

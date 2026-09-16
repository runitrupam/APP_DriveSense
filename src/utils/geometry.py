"""Bounding-box, polygon and line geometry with no OpenCV dependency."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

BBoxDict = dict[str, int]
Point = tuple[float, float]


def xyxy_to_bbox(x1: float, y1: float, x2: float, y2: float) -> BBoxDict:
    """Convert corner coordinates to ``{x, y, width, height}`` integer pixels."""
    left = int(math.floor(min(x1, x2)))
    top = int(math.floor(min(y1, y2)))
    right = int(math.ceil(max(x1, x2)))
    bottom = int(math.ceil(max(y1, y2)))
    return {"x": left, "y": top, "width": max(0, right - left), "height": max(0, bottom - top)}


def bbox_to_xyxy(bbox: Mapping[str, float]) -> tuple[float, float, float, float]:
    """Convert ``{x, y, width, height}`` to ``(x1, y1, x2, y2)``."""
    x = float(bbox["x"])
    y = float(bbox["y"])
    return x, y, x + float(bbox["width"]), y + float(bbox["height"])


def bbox_center(bbox: Mapping[str, float]) -> dict[str, int]:
    """Integer center point of a bbox."""
    return {
        "x": int(round(float(bbox["x"]) + float(bbox["width"]) / 2.0)),
        "y": int(round(float(bbox["y"]) + float(bbox["height"]) / 2.0)),
    }


def bbox_area(bbox: Mapping[str, float]) -> float:
    """Area of a bbox, never negative."""
    return max(0.0, float(bbox["width"])) * max(0.0, float(bbox["height"]))


def clip_xyxy(xyxy: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Clamp corner coordinates into the frame."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_bbox(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """IoU for two bbox dicts."""
    return iou(bbox_to_xyxy(a), bbox_to_xyxy(b))


def normalize_bbox(bbox: Mapping[str, float], width: int, height: int) -> dict[str, float]:
    """Normalise a bbox to 0..1 against frame size, rounded to 6 decimals."""
    if width <= 0 or height <= 0:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": round(min(max(float(bbox["x"]) / width, 0.0), 1.0), 6),
        "y": round(min(max(float(bbox["y"]) / height, 0.0), 1.0), 6),
        "width": round(min(max(float(bbox["width"]) / width, 0.0), 1.0), 6),
        "height": round(min(max(float(bbox["height"]) / height, 0.0), 1.0), 6),
    }


def denormalize_bbox(bbox: Mapping[str, float], width: int, height: int) -> BBoxDict:
    """Inverse of :func:`normalize_bbox`."""
    return xyxy_to_bbox(
        float(bbox["x"]) * width,
        float(bbox["y"]) * height,
        (float(bbox["x"]) + float(bbox["width"])) * width,
        (float(bbox["y"]) + float(bbox["height"])) * height,
    )


def scale_bbox(bbox: Mapping[str, float], factor: float) -> BBoxDict:
    """Scale a bbox around its center."""
    center = bbox_center(bbox)
    half_w = float(bbox["width"]) * factor / 2.0
    half_h = float(bbox["height"]) * factor / 2.0
    return xyxy_to_bbox(
        center["x"] - half_w,
        center["y"] - half_h,
        center["x"] + half_w,
        center["y"] + half_h,
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = (yi > y) != (yj > y)
        if intersects:
            denominator = yj - yi if (yj - yi) != 0 else 1e-9
            x_cross = (xj - xi) * (y - yi) / denominator + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def fit_line(points: Iterable[Point]) -> tuple[float, float] | None:
    """Least-squares ``y = slope * x + intercept``; ``None`` if degenerate."""
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 2:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x <= 1e-9:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / variance_x
    return slope, mean_y - slope * mean_x


def line_y_at(slope: float, intercept: float, x: float) -> float:
    """Evaluate a line at ``x``."""
    return slope * float(x) + intercept


def line_x_at(slope: float, intercept: float, y: float) -> float:
    """Evaluate a line at ``y``; returns ``nan`` for horizontal lines."""
    if abs(slope) <= 1e-9:
        return float("nan")
    return (float(y) - intercept) / slope


def fit_polynomial_y_of_x(points: Sequence[Point], degree: int = 2) -> list[float] | None:
    """Fit ``x = f(y)`` in the lane-detection convention (x as function of y)."""
    if len(points) < degree + 1:
        return None
    ys = [float(p[1]) for p in points]
    xs = [float(p[0]) for p in points]
    try:
        coeffs = [float(c) for c in _polyfit(ys, xs, degree)]
    except Exception:
        return None
    return coeffs


def _polyfit(x: Sequence[float], y: Sequence[float], degree: int) -> list[float]:
    """Minimal numpy-backed polynomial fit returning highest power first."""
    import numpy as np

    coeffs = np.polyfit(list(x), list(y), degree)
    return [float(c) for c in coeffs]


def polyval_x_of_y(coeffs: Sequence[float], y: float) -> float:
    """Evaluate highest-power-first coefficients at ``y``."""
    result = 0.0
    for coefficient in coeffs:
        result = result * float(y) + float(coefficient)
    return result


def curvature_from_coeffs(coeffs: Sequence[float]) -> float:
    """Real-world-ish curvature proxy from ``x = a*y^2 + b*y + c`` coefficients."""
    if len(coeffs) == 0:
        return 0.0
    if len(coeffs) >= 3:
        return float(coeffs[-3])
    return 0.0


def count_peaks(values: Sequence[float], min_height: float, min_distance: int = 8) -> list[int]:
    """Indices of local maxima above ``min_height`` separated by ``min_distance``."""
    peaks: list[int] = []
    n = len(values)
    for index in range(1, n - 1):
        if values[index] < min_height:
            continue
        if values[index] < values[index - 1] or values[index] < values[index + 1]:
            continue
        if peaks and index - peaks[-1] < min_distance:
            if values[index] > values[peaks[-1]]:
                peaks[-1] = index
            continue
        peaks.append(index)
    return peaks


def euclidean(a: Point, b: Point) -> float:
    """Distance between two points."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def displacement_sequence(points: Sequence[Point]) -> list[Point]:
    """Successive ``(dx, dy)`` differences for a point sequence."""
    out: list[Point] = []
    for previous, current in zip(points, points[1:]):
        out.append((current[0] - previous[0], current[1] - previous[1]))
    return out


def angle_degrees(dx: float, dy: float) -> float:
    """Angle of a vector in degrees, measured from the positive x-axis."""
    return math.degrees(math.atan2(float(dy), float(dx)))


def direction_from_offset(dx: float, dy: float, dead_zone: float) -> str:
    """Cardinal image direction for an offset, honouring a dead zone."""
    if abs(dx) < dead_zone and abs(dy) < dead_zone:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "backward" if dy > 0 else "forward"

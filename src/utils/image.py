"""Image helpers built on OpenCV and NumPy."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

MIN_BLUR_THRESHOLD = 0.0


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 grayscale image."""
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian; lower means blurrier."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-pixel difference between two images."""
    if a is None or b is None:
        return float("inf")
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(cv2.absdiff(a, b)))


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Difference hash as a big integer; robust to small exposure changes."""
    gray = to_gray(image)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bool(bit))
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Bit distance between two integer hashes."""
    return int(bin(int(a) ^ int(b)).count("1"))


def estimate_global_shift(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate dominant translation between consecutive frames.

    Returns ``(dx, dy, response)`` in pixels. The response is the phase
    correlation peak, used as a confidence proxy.
    """
    if previous_gray is None or current_gray is None or previous_gray.size == 0 or current_gray.size == 0:
        return 0.0, 0.0, 0.0
    height = min(previous_gray.shape[0], current_gray.shape[0])
    width = min(previous_gray.shape[1], current_gray.shape[1])
    if height < 16 or width < 16:
        return 0.0, 0.0, 0.0
    previous = previous_gray[:height, :width].astype(np.float32)
    current = current_gray[:height, :width].astype(np.float32)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(previous, current, window)
    except cv2.error:
        return 0.0, 0.0, 0.0
    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(response):
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(response)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize preserving aspect ratio."""
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def hsv_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue, saturation and value channels of a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def connected_components(
    mask: np.ndarray,
    min_area: float,
    max_components: int = 200,
) -> list[tuple[int, int, int, int, float]]:
    """Filtered connected components as ``(x, y, w, h, area)`` tuples."""
    if mask is None or mask.size == 0:
        return []
    binary = (mask > 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[int, int, int, int, float]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area >= min_area:
            components.append((x, y, width, height, float(area)))
    components.sort(key=lambda item: item[4], reverse=True)
    return components[:max_components]


def excess_green_index(image: np.ndarray) -> np.ndarray:
    """ExG vegetation index in [-1, 1] scaled to uint8 0..255."""
    bgr = image.astype(np.float32)
    total = bgr[:, :, 0] + bgr[:, :, 1] + bgr[:, :, 2] + 1e-6
    exg = (2.0 * bgr[:, :, 1] - bgr[:, :, 2] - bgr[:, :, 0]) / total
    normalized = np.clip((exg + 0.5) * 255.0, 0, 255)
    return normalized.astype(np.uint8)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude as float32."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def vertical_edge_density(gray: np.ndarray, region: Sequence[int] | None = None) -> float:
    """Fraction of pixels with a strong vertical edge inside a region."""
    if region is not None:
        x, y, width, height = (int(v) for v in region)
        gray = gray[max(0, y) : max(0, y + height), max(0, x) : max(0, x + width)]
    if gray.size == 0:
        return 0.0
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    return float(np.mean(grad_x > 60.0))


def draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.5,
) -> np.ndarray:
    """Draw text with a filled background box for readability."""
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(image, (x, y - height - 4), (x + width + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return image


def clamp_box_to_frame(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Integer, in-frame corner coordinates."""
    left = int(max(0, min(width - 1, round(x1))))
    top = int(max(0, min(height - 1, round(y1))))
    right = int(max(left + 1, min(width - 1, round(x2))))
    bottom = int(max(top + 1, min(height - 1, round(y2))))
    return left, top, right, bottom

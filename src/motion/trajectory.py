"""Trajectory buffering, smoothing and velocity estimation.

All motion estimates use a configurable history window rather than a single
frame difference, which the requirements call out explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class TrajectoryPoint:
    """One observed position of an object."""

    frame_id: int
    timestamp_ms: int
    x: float
    y: float
    bbox_height: float = 0.0
    bbox_width: float = 0.0
    bbox: dict[str, int] = field(default_factory=dict)


class Trajectory:
    """Ordered history of an object's positions with derived motion."""

    def __init__(self, track_id: str, max_points: int = 60) -> None:
        self.track_id = track_id
        self.max_points = max(2, int(max_points))
        self.points: list[TrajectoryPoint] = []

    def append(self, point: TrajectoryPoint) -> None:
        """Append an observation, trimming to ``max_points``."""
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points :]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def last(self) -> TrajectoryPoint | None:
        """Most recent point."""
        return self.points[-1] if self.points else None

    @property
    def first(self) -> TrajectoryPoint | None:
        """Oldest retained point."""
        return self.points[0] if self.points else None

    def window(self, size: int) -> list[TrajectoryPoint]:
        """Most recent ``size`` points."""
        if size <= 0:
            return []
        return self.points[-size:]

    def positions(self, size: int | None = None) -> np.ndarray:
        """``(N, 2)`` array of positions for a window."""
        points = self.window(size) if size else list(self.points)
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([[p.x, p.y] for p in points], dtype=np.float64)

    def smoothed_positions(self, size: int) -> np.ndarray:
        """Moving-average smoothed positions over the window.

        The average is centred and padded so the smoothed series has the same
        length as the window and never shifts the timeline.
        """
        positions = self.positions(size)
        if positions.shape[0] < 3:
            return positions
        window = max(1, min(5, positions.shape[0] // 2))
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.empty_like(positions)
        for column in range(positions.shape[1]):
            smoothed[:, column] = np.convolve(positions[:, column], kernel, mode="same")
        return smoothed

    def displacement(self, size: int, smoothed: bool = True) -> tuple[float, float, float]:
        """Total ``(dx, dy, distance)`` over the window."""
        positions = self.smoothed_positions(size) if smoothed else self.positions(size)
        if positions.shape[0] < 2:
            return 0.0, 0.0, 0.0
        dx = float(positions[-1, 0] - positions[0, 0])
        dy = float(positions[-1, 1] - positions[0, 1])
        return dx, dy, float(np.hypot(dx, dy))

    def frame_span(self, size: int) -> int:
        """Number of frames spanned by the window."""
        points = self.window(size)
        if len(points) < 2:
            return 0
        return max(1, int(points[-1].frame_id) - int(points[0].frame_id))

    def velocity_px_per_frame(self, size: int) -> tuple[float, float, float]:
        """``(vx, vy, speed)`` in pixels per frame."""
        dx, dy, distance = self.displacement(size)
        span = self.frame_span(size)
        if span <= 0:
            return 0.0, 0.0, 0.0
        return dx / span, dy / span, distance / span

    def velocity_px_per_s(self, size: int, fps: float) -> tuple[float, float, float]:
        """``(vx, vy, speed)`` in pixels per second."""
        vx, vy, speed = self.velocity_px_per_frame(size)
        scale = float(fps) if fps and fps > 0 else 1.0
        return vx * scale, vy * scale, speed * scale

    def mean_bbox_height(self, size: int) -> float:
        """Mean bbox height over the window, a cheap depth proxy."""
        points = self.window(size)
        heights = [p.bbox_height for p in points if p.bbox_height > 0]
        if not heights:
            return 0.0
        return float(np.mean(heights))

    def to_list(self) -> list[dict[str, float | int]]:
        """Serialisable trajectory."""
        return [
            {
                "frame_id": int(point.frame_id),
                "timestamp_ms": int(point.timestamp_ms),
                "x": round(float(point.x), 3),
                "y": round(float(point.y), 3),
            }
            for point in self.points
        ]


class TrajectoryStore:
    """Maintains a :class:`Trajectory` per track id."""

    def __init__(self, history_frames: int = 10, max_points: int = 90) -> None:
        self.history_frames = max(2, int(history_frames))
        self.max_points = max(2, int(max_points))
        self.trajectories: dict[str, Trajectory] = {}

    def update(
        self,
        track_id: str,
        frame_id: int,
        timestamp_ms: int,
        center_x: float,
        center_y: float,
        bbox: dict[str, int] | None = None,
    ) -> Trajectory:
        """Record a position and return the updated trajectory."""
        trajectory = self.trajectories.get(track_id)
        if trajectory is None:
            trajectory = Trajectory(track_id, max_points=self.max_points)
            self.trajectories[track_id] = trajectory
        bbox = bbox or {}
        trajectory.append(
            TrajectoryPoint(
                frame_id=int(frame_id),
                timestamp_ms=int(timestamp_ms),
                x=float(center_x),
                y=float(center_y),
                bbox_height=float(bbox.get("height", 0) or 0),
                bbox_width=float(bbox.get("width", 0) or 0),
                bbox=dict(bbox),
            )
        )
        return trajectory

    def get(self, track_id: str) -> Trajectory | None:
        """Trajectory for a track, if any."""
        return self.trajectories.get(track_id)

    def prune(self, active_track_ids: Iterable[str]) -> None:
        """Drop trajectories for tracks that no longer exist."""
        active = set(active_track_ids)
        for track_id in list(self.trajectories):
            if track_id not in active:
                del self.trajectories[track_id]

    def clear(self) -> None:
        """Remove every trajectory."""
        self.trajectories.clear()

    def statistics(self) -> dict[str, float]:
        """Distribution of trajectory lengths."""
        lengths = [len(trajectory) for trajectory in self.trajectories.values()]
        if not lengths:
            return {"trajectories": 0, "mean_length": 0.0, "max_length": 0}
        return {
            "trajectories": len(lengths),
            "mean_length": round(float(np.mean(lengths)), 4),
            "max_length": int(max(lengths)),
        }

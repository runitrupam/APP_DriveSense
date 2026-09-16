"""Ego-motion estimation and moving/stationary classification.

Image-space motion alone cannot decide whether an object is moving: with a
moving camera, every static object translates in the image. The classifier
therefore subtracts the dominant background translation estimated by phase
correlation, and only calls an object moving when its *residual* motion is
significant. When the residual cannot be trusted the state is ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.motion.trajectory import Trajectory
from src.utils.context import RunContext
from src.utils.image import estimate_global_shift, to_gray


@dataclass
class EgoMotion:
    """Estimated dominant background translation between two frames."""

    dx: float = 0.0
    dy: float = 0.0
    response: float = 0.0
    available: bool = False
    magnitude: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "dx": round(float(self.dx), 4),
            "dy": round(float(self.dy), 4),
            "response": round(float(self.response), 4),
            "available": bool(self.available),
            "magnitude": round(float(self.magnitude), 4),
        }


class EgoMotionEstimator:
    """Phase-correlation ego-motion estimator with a validity gate."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.history: list[EgoMotion] = []
        self._previous_gray: np.ndarray | None = None
        self.min_response = 0.05

    def reset(self) -> None:
        """Clear frame-to-frame state."""
        self.history.clear()
        self._previous_gray = None

    def update(self, frame: np.ndarray | None) -> EgoMotion:
        """Estimate ego-motion for the current frame given the previous frame."""
        if not self.enabled or frame is None or frame.size == 0:
            self._previous_gray = None if frame is None else to_gray(frame)
            return EgoMotion(available=False)

        gray = to_gray(frame)
        if self._previous_gray is None:
            self._previous_gray = gray
            motion = EgoMotion(available=False)
        else:
            dx, dy, response = estimate_global_shift(self._previous_gray, gray)
            motion = EgoMotion(
                dx=dx,
                dy=dy,
                response=response,
                available=response >= self.min_response,
                magnitude=float(np.hypot(dx, dy)),
            )
            self._previous_gray = gray
        self.history.append(motion)
        if len(self.history) > 120:
            self.history = self.history[-120:]
        return motion

    def mean_shift(self, window: int = 5) -> tuple[float, float, float]:
        """Mean shift and mean response over recent frames."""
        recent = [motion for motion in self.history[-window:] if motion.available]
        if not recent:
            return 0.0, 0.0, 0.0
        dx = float(np.mean([motion.dx for motion in recent]))
        dy = float(np.mean([motion.dy for motion in recent]))
        response = float(np.mean([motion.response for motion in recent]))
        return dx, dy, response

    def statistics(self) -> dict[str, Any]:
        """Distribution of estimated ego-motion magnitudes."""
        magnitudes = [motion.magnitude for motion in self.history if motion.available]
        return {
            "samples": len(self.history),
            "valid_samples": len(magnitudes),
            "mean_magnitude": round(float(np.mean(magnitudes)), 4) if magnitudes else 0.0,
            "max_magnitude": round(float(max(magnitudes)), 4) if magnitudes else 0.0,
        }


@dataclass
class MovementResult:
    """Movement classification for one object in one frame."""

    state: str = "unknown"
    confidence: float = 0.0
    residual_vx: float = 0.0
    residual_vy: float = 0.0
    residual_speed: float = 0.0
    observed_speed_px_s: float | None = None
    displacement_px: float = 0.0
    threshold_px: float = 0.0
    reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "state": self.state,
            "confidence": round(float(self.confidence), 6),
            "residual_vx": round(float(self.residual_vx), 4),
            "residual_vy": round(float(self.residual_vy), 4),
            "residual_speed_px_per_frame": round(float(self.residual_speed), 4),
            "observed_speed_px_s": self.observed_speed_px_s,
            "displacement_px": round(float(self.displacement_px), 4),
            "threshold_px": round(float(self.threshold_px), 4),
            "reason": self.reason,
            "features": self.features,
        }


class MovementClassifier:
    """Decides moving, stationary or unknown from residual motion."""

    name = "residual_motion"

    def __init__(self, ctx: RunContext, fps: float = 30.0) -> None:
        self.ctx = ctx
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.history_frames = max(2, int(ctx.get("motion.history_frames", 10)))
        self.stationary_threshold = float(ctx.get("motion.stationary_threshold", 5))
        self.stationary_ratio = float(ctx.get("motion.stationary_threshold_ratio", 0.02))
        self.compensation = ctx.flag("motion.ego_motion_compensation", True)

    def classify(
        self,
        trajectory: Trajectory | None,
        ego_motion: EgoMotion,
        bbox_height: float = 0.0,
    ) -> MovementResult:
        """Classify movement using a multi-frame window and ego-motion."""
        if trajectory is None or len(trajectory) < 2:
            return MovementResult(state="unknown", confidence=0.0, reason="insufficient_history")

        history = min(self.history_frames, len(trajectory))
        vx, vy, speed = trajectory.velocity_px_per_frame(history)
        _, _, distance = trajectory.displacement(history)
        depth_scale = max(1.0, float(bbox_height)) if bbox_height > 0 else 0.0
        threshold = max(self.stationary_threshold, self.stationary_ratio * depth_scale)

        residual_vx, residual_vy = vx, vy
        compensated = False
        if self.compensation and ego_motion.available:
            global_dx, global_dy, _ = self._global_reference(ego_motion)
            residual_vx = vx - global_dx
            residual_vy = vy - global_dy
            compensated = True
        residual_speed = float(np.hypot(residual_vx, residual_vy))

        history_quality = min(1.0, len(trajectory) / float(self.history_frames))
        confidence = 0.25 + 0.45 * history_quality

        if compensated:
            confidence += 0.20 * min(1.0, ego_motion.response / 0.4)
        else:
            confidence -= 0.15 if self.compensation else 0.0

        if residual_speed >= threshold * 2.0:
            state = "moving"
            confidence += 0.15
        elif residual_speed >= threshold:
            state = "moving"
            confidence -= 0.05
        elif residual_speed < threshold * 0.5:
            state = "stationary"
        else:
            state = "stationary"
            confidence -= 0.10

        if (
            not compensated
            and self.compensation
            and not ego_motion.available
            and residual_speed < threshold * 2.0
        ):
            state = "unknown"
            confidence = min(confidence, 0.25)
            reason = "ego_motion_unavailable"
        elif compensated:
            reason = "residual_after_ego_motion_compensation"
        else:
            reason = "raw_image_motion"

        confidence = float(np.clip(confidence, 0.0, 1.0))
        observed_speed_px_s = round(float(speed) * self.fps, 4)
        return MovementResult(
            state=state,
            confidence=confidence,
            residual_vx=float(residual_vx),
            residual_vy=float(residual_vy),
            residual_speed=residual_speed,
            observed_speed_px_s=observed_speed_px_s,
            displacement_px=float(distance),
            threshold_px=float(threshold),
            reason=reason,
            features={
                "history_used": int(history),
                "ego_compensated": compensated,
                "ego_response": round(float(ego_motion.response), 4),
                "raw_speed_px_per_frame": round(float(speed), 4),
            },
        )

    def _global_reference(self, ego_motion: EgoMotion) -> tuple[float, float, float]:
        """Background shift to subtract, using the recent mean when available."""
        recent = [motion for motion in getattr(self, "_recent", []) if motion.available]
        if recent:
            return (
                float(np.mean([motion.dx for motion in recent])),
                float(np.mean([motion.dy for motion in recent])),
                float(np.mean([motion.response for motion in recent])),
            )
        return ego_motion.dx, ego_motion.dy, ego_motion.response

    def record_ego_motion(self, ego_motion: EgoMotion) -> None:
        """Track recent ego-motion samples for a smoother background estimate."""
        recent = getattr(self, "_recent", None)
        if recent is None:
            self._recent = []
            recent = self._recent
        recent.append(ego_motion)
        if len(recent) > 10:
            del recent[:-10]

    def statistics(self) -> dict[str, Any]:
        """Configuration actually used for classification."""
        return {
            "history_frames": self.history_frames,
            "stationary_threshold_px": self.stationary_threshold,
            "stationary_threshold_ratio": self.stationary_ratio,
            "ego_motion_compensation": self.compensation,
            "fps": self.fps,
        }

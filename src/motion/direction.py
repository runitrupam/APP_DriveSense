"""Image-space and ego-relative direction classification.

Image-space motion is not real-world direction. The relative judgement here is
anchored on the vanishing point: movement toward it means the object is receding
ahead of the ego vehicle (same direction), movement away from it means the object
is approaching the ego vehicle (opposite direction). When no vanishing point is
available the fallback is the longitudinal image direction, reported with lower
confidence, and below the confidence floor the result is ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.utils.context import RunContext

CROSSING_LATERAL_RATIO = 2.5
CROSSING_MIN_FACTOR = 3.0


@dataclass
class DirectionResult:
    """Direction classification for one object in one frame."""

    image_direction: str = "unknown"
    longitudinal: str = "unknown"
    lateral: str = "unknown"
    relative: str = "unknown"
    confidence: float = 0.0
    reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "image_direction": self.image_direction,
            "longitudinal": self.longitudinal,
            "lateral": self.lateral,
            "relative": self.relative,
            "confidence": round(float(self.confidence), 6),
            "reason": self.reason,
            "features": self.features,
        }


class DirectionClassifier:
    """Classifies image-space and ego-relative direction."""

    name = "vanishing_point_direction"

    def __init__(self, ctx: RunContext, fps: float = 30.0) -> None:
        self.ctx = ctx
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.stationary_threshold = float(ctx.get("motion.stationary_threshold", 5))
        self.confidence_threshold = float(ctx.get("motion.direction_confidence_threshold", 0.30))
        self.dead_zone = max(0.5, 0.5 * self.stationary_threshold)

    def classify(
        self,
        residual_vx: float,
        residual_vy: float,
        center: Mapping[str, float] | tuple[float, float],
        vanishing_point: tuple[float, float] | None = None,
        frame_size: tuple[int, int] | None = None,
        bbox_height: float = 0.0,
    ) -> DirectionResult:
        """Classify direction from residual motion and scene geometry."""
        center_x, center_y = self._center(center)
        speed = float(np.hypot(residual_vx, residual_vy))
        dead_zone = self.dead_zone
        if bbox_height > 0:
            dead_zone = max(dead_zone, 0.02 * float(bbox_height))

        if speed < dead_zone:
            return DirectionResult(
                image_direction="stationary",
                longitudinal="stationary",
                lateral="none",
                relative="unknown",
                confidence=float(min(1.0, 0.3 + 0.3 * (1.0 - speed / max(dead_zone, 1e-6)))),
                reason="below_direction_dead_zone",
                features={"speed_px_per_frame": round(speed, 4), "dead_zone": round(dead_zone, 4)},
            )

        longitudinal = self._longitudinal(residual_vy, dead_zone)
        lateral = self._lateral(residual_vx, dead_zone)
        image_direction = self._image_direction(residual_vx, residual_vy, dead_zone)

        magnitude_score = min(1.0, speed / (dead_zone * 4.0))
        confidence = 0.30 + 0.45 * magnitude_score
        reason = "image_motion_only"

        relative = "unknown"
        if abs(residual_vx) > CROSSING_LATERAL_RATIO * abs(residual_vy) and abs(residual_vx) > (
            CROSSING_MIN_FACTOR * dead_zone
        ):
            relative = "crossing"
            confidence += 0.15
            reason = "dominant_lateral_motion"
        elif vanishing_point is not None:
            toward = self._toward_vanishing_point(
                residual_vx, residual_vy, (center_x, center_y), vanishing_point
            )
            if toward is None:
                relative = "unknown"
                reason = "vanishing_point_degenerate"
            else:
                relative = "same_direction" if toward else "opposite_direction"
                confidence += 0.20
                reason = "vanishing_point_convergence"
        elif longitudinal == "forward":
            relative = "same_direction"
            confidence -= 0.10
            reason = "longitudinal_fallback"
        elif longitudinal == "backward":
            relative = "opposite_direction"
            confidence -= 0.10
            reason = "longitudinal_fallback"

        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence < self.confidence_threshold:
            relative = "unknown"
            reason = f"confidence_below_threshold:{reason}"

        return DirectionResult(
            image_direction=image_direction,
            longitudinal=longitudinal,
            lateral=lateral,
            relative=relative,
            confidence=confidence,
            reason=reason,
            features={
                "speed_px_per_frame": round(speed, 4),
                "dead_zone": round(dead_zone, 4),
                "vanishing_point": [round(float(v), 2) for v in vanishing_point]
                if vanishing_point is not None
                else None,
                "residual_vx": round(float(residual_vx), 4),
                "residual_vy": round(float(residual_vy), 4),
            },
        )

    @staticmethod
    def _center(center: Mapping[str, float] | tuple[float, float]) -> tuple[float, float]:
        """Accept either a mapping or a tuple for the object center."""
        if isinstance(center, Mapping):
            return float(center["x"]), float(center["y"])
        return float(center[0]), float(center[1])

    @staticmethod
    def _longitudinal(residual_vy: float, dead_zone: float) -> str:
        """Vertical image motion: up is forward (away), down is backward (toward)."""
        if residual_vy <= -dead_zone:
            return "forward"
        if residual_vy >= dead_zone:
            return "backward"
        return "stationary"

    @staticmethod
    def _lateral(residual_vx: float, dead_zone: float) -> str:
        """Horizontal image motion."""
        if residual_vx <= -dead_zone:
            return "left"
        if residual_vx >= dead_zone:
            return "right"
        return "none"

    @staticmethod
    def _image_direction(residual_vx: float, residual_vy: float, dead_zone: float) -> str:
        """Dominant-axis image direction."""
        if abs(residual_vy) >= abs(residual_vx):
            if residual_vy <= -dead_zone:
                return "forward"
            if residual_vy >= dead_zone:
                return "backward"
        else:
            if residual_vx <= -dead_zone:
                return "left"
            if residual_vx >= dead_zone:
                return "right"
        return "stationary"

    @staticmethod
    def _toward_vanishing_point(
        residual_vx: float,
        residual_vy: float,
        center: tuple[float, float],
        vanishing_point: tuple[float, float],
    ) -> bool | None:
        """True when motion points at the vanishing point, False when away.

        Returns ``None`` when the object sits on the vanishing point and the
        comparison carries no information.
        """
        to_vp_x = float(vanishing_point[0]) - center[0]
        to_vp_y = float(vanishing_point[1]) - center[1]
        norm = float(np.hypot(to_vp_x, to_vp_y))
        if norm < 1e-3:
            return None
        dot = (residual_vx * to_vp_x + residual_vy * to_vp_y) / norm
        if dot > 0:
            return True
        if dot < 0:
            return False
        return None

    def statistics(self) -> dict[str, Any]:
        """Configuration actually used for direction classification."""
        return {
            "dead_zone_px_per_frame": self.dead_zone,
            "confidence_threshold": self.confidence_threshold,
            "stationary_threshold_px": self.stationary_threshold,
        }

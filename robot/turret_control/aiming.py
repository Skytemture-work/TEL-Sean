"""Camera-target to turret-angle conversion.

This module contains no camera, CAN or vendor-specific code, so it can be
developed and tested on a laptop.
"""

from dataclasses import dataclass
import math
from typing import Optional

from .config import TurretConfig


@dataclass(frozen=True)
class TargetPixel:
    """A selected YOLO target in image coordinates."""

    u_px: float
    v_px: float
    confidence: float = 1.0


@dataclass(frozen=True)
class AimCommand:
    """Command produced by the aiming layer."""

    has_target: bool
    target_yaw_rad: Optional[float]
    target_pitch_rad: Optional[float]
    yaw_error_rad: float = 0.0
    pitch_error_rad: float = 0.0
    stable: bool = False
    fire_allowed: bool = False


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AimingController:
    """Convert a target pixel into safe horizontal/vertical motor positions."""

    def __init__(self, config: TurretConfig):
        config.validate()
        self.config = config
        self._stable_frames = 0

    def pixel_to_error(self, target: TargetPixel) -> tuple[float, float]:
        """Return yaw and pitch error in radians.

        Positive yaw means target is to the right. Positive pitch means target
        is above the image center. The signs can be changed in TurretConfig to
        match the installed motor directions.
        """

        yaw_error = math.atan2(
            target.u_px - self.config.camera_center_x_px,
            self.config.focal_length_x_px,
        )
        pitch_error = math.atan2(
            self.config.camera_center_y_px - target.v_px,
            self.config.focal_length_y_px,
        )
        return (
            self.config.yaw_sign * yaw_error,
            self.config.pitch_sign * pitch_error,
        )

    def update(self, target: Optional[TargetPixel]) -> AimCommand:
        """Create one command from the newest target detection.

        No target, low confidence, or a target outside the configured limits
        never enables firing.  The caller should stop/hold the motors when
        ``has_target`` is false.
        """

        if target is None or target.confidence < self.config.minimum_confidence:
            self._stable_frames = 0
            return AimCommand(
                has_target=False,
                target_yaw_rad=None,
                target_pitch_rad=None,
            )

        yaw_error, pitch_error = self.pixel_to_error(target)
        yaw_target = clamp(
            self.config.yaw_zero_rad + yaw_error,
            self.config.yaw_min_rad,
            self.config.yaw_max_rad,
        )
        pitch_target = clamp(
            self.config.pitch_zero_rad + pitch_error,
            self.config.pitch_min_rad,
            self.config.pitch_max_rad,
        )

        within_tolerance = (
            abs(yaw_error) <= self.config.aim_tolerance_rad
            and abs(pitch_error) <= self.config.aim_tolerance_rad
        )
        self._stable_frames = self._stable_frames + 1 if within_tolerance else 0
        stable = self._stable_frames >= self.config.stable_frames_required

        return AimCommand(
            has_target=True,
            target_yaw_rad=yaw_target,
            target_pitch_rad=pitch_target,
            yaw_error_rad=yaw_error,
            pitch_error_rad=pitch_error,
            stable=stable,
            # The launcher should still have a separate physical safety gate.
            fire_allowed=stable,
        )

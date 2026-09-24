"""Configuration for the turret controller.

All angles in this package are radians.  Motor IDs, zero positions and limits
must be measured on the real mechanism before using the CAN backend.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TurretConfig:
    # Camera intrinsics.  Replace these with the calibrated camera values.
    image_width_px: int = 1280
    image_height_px: int = 720
    focal_length_x_px: float = 700.0
    focal_length_y_px: float = 700.0

    # Mechanical zero: direction of the launcher when the motor is at zero.
    yaw_zero_rad: float = 0.0
    pitch_zero_rad: float = 0.0

    # Change these to -1 if the installed motor direction is reversed.
    yaw_sign: int = 1
    pitch_sign: int = 1

    # Mechanical safety limits.  These are deliberately conservative defaults.
    yaw_min_rad: float = math.radians(-80.0)
    yaw_max_rad: float = math.radians(80.0)
    pitch_min_rad: float = math.radians(-20.0)
    pitch_max_rad: float = math.radians(35.0)

    # Position-mode command limit.  This is not the baseball launch speed.
    max_motor_speed_rad_s: float = 1.0

    # Aiming and firing conditions.
    aim_tolerance_rad: float = math.radians(1.5)
    stable_frames_required: int = 5
    minimum_confidence: float = 0.40

    # Motor IDs are placeholders until the existing AGX motor test is checked.
    yaw_motor_id: int = 1
    pitch_motor_id: int = 2

    @property
    def camera_center_x_px(self) -> float:
        return self.image_width_px / 2.0

    @property
    def camera_center_y_px(self) -> float:
        return self.image_height_px / 2.0

    def validate(self) -> None:
        if self.focal_length_x_px <= 0 or self.focal_length_y_px <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.yaw_sign not in (-1, 1) or self.pitch_sign not in (-1, 1):
            raise ValueError("yaw_sign and pitch_sign must be either 1 or -1")
        if self.yaw_min_rad >= self.yaw_max_rad:
            raise ValueError("yaw limits are invalid")
        if self.pitch_min_rad >= self.pitch_max_rad:
            raise ValueError("pitch limits are invalid")
        if self.max_motor_speed_rad_s <= 0:
            raise ValueError("max_motor_speed_rad_s must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.stable_frames_required < 1:
            raise ValueError("stable_frames_required must be at least 1")

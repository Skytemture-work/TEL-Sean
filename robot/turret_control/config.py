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

    # Where the camera is bolted.  This changes the control law, not just a sign.
    #
    # False (this robot): the camera is fixed to the base and does not move when
    #   the turret turns, so the pixel error is an absolute bearing and the
    #   target angle is yaw_zero + error.  The pixel error never shrinks as the
    #   turret moves, so "on target" must be judged from motor feedback.
    # True: the camera rides on the turret, the pixel error is relative to where
    #   the turret already points, so the target angle is current + error and
    #   the pixel error going to zero is what "on target" means.
    camera_on_turret: bool = False

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

    # --- Target board -------------------------------------------------------
    # 3 columns x 4 rows of holes.  Top row is the small ones, the three rows
    # below are large.  Known diameters are what make single-camera ranging
    # possible: distance = focal_px * diameter_m / apparent_px.
    small_hole_diameter_m: float = 0.20
    large_hole_diameter_m: float = 0.40

    # Engagement order: every large hole first, then the small ones; columns
    # left to right; within a column bottom to top.  Column-at-a-time keeps the
    # horizontal motor to one move per column.
    engage_large_first: bool = True
    engage_left_to_right: bool = True
    engage_bottom_to_top: bool = True

    # --- Parallax -----------------------------------------------------------
    # Where the muzzle sits in the camera frame, in metres: +x right, +y up,
    # +z forward along the optical axis.  With a base-fixed camera the muzzle
    # is never exactly at the camera, so the bearing the camera measures is not
    # the bearing the ball needs.  The error grows as the target gets closer.
    # MEASURE THESE on the built mechanism; zeros mean "camera is the muzzle".
    muzzle_offset_x_m: float = 0.0
    muzzle_offset_y_m: float = 0.0
    muzzle_offset_z_m: float = 0.0
    compensate_parallax: bool = True

    # --- Ballistics ---------------------------------------------------------
    # The ball is assumed to leave at full speed every shot.  MEASURE this;
    # the drop correction is only as good as the speed number.
    muzzle_speed_m_s: float = 20.0
    gravity_m_s2: float = 9.81
    compensate_gravity: bool = True

    # Both corrections need the range to the target, which a single YOLO box
    # does not provide.  When a correction is enabled but no distance is given,
    # refuse to fire rather than shooting on an assumption.
    require_distance_to_fire: bool = True

    # Aiming and firing conditions.
    aim_tolerance_rad: float = math.radians(1.5)
    stable_frames_required: int = 5
    minimum_confidence: float = 0.40

    # Measured on the real motors, 2026-09-24.  Motor 2 drives the horizontal
    # axis, motor 1 the vertical one.  Their MST_ID differs, which the CAN
    # backend reads off the motor rather than assuming.
    #   motor 2 "horizontal": ESC_ID 0x03, MST_ID 0x13
    #   motor 1 "vertical"  : ESC_ID 0x01, MST_ID 0x00
    yaw_motor_id: int = 0x03
    pitch_motor_id: int = 0x01

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
        if self.compensate_gravity and self.muzzle_speed_m_s <= 0:
            raise ValueError("muzzle_speed_m_s must be positive to compensate gravity")
        if self.gravity_m_s2 < 0:
            raise ValueError("gravity_m_s2 must not be negative")
        if self.small_hole_diameter_m <= 0 or self.large_hole_diameter_m <= 0:
            raise ValueError("hole diameters must be positive")
        if self.small_hole_diameter_m >= self.large_hole_diameter_m:
            raise ValueError("small_hole_diameter_m must be smaller than large")

    @property
    def needs_distance(self) -> bool:
        """True when a correction cannot be applied without the target range."""
        if self.compensate_gravity:
            return True
        return self.compensate_parallax and self.muzzle_offset != (0.0, 0.0, 0.0)

    @property
    def muzzle_offset(self) -> tuple:
        return (self.muzzle_offset_x_m, self.muzzle_offset_y_m, self.muzzle_offset_z_m)

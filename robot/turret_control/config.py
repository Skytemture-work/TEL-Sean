"""Configuration for the turret controller.

All angles in this package are radians.  Motor IDs, zero positions and limits
must be measured on the real mechanism before using the CAN backend.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

from .limits import DEFAULT_LIMITS_PATH, load_axis_limits
from .pitch import load_pitch_calibration, load_yaw_calibration


_DEFAULT_STARTUP_POSE = Path(__file__).with_name("startup_pose.json")
_DEFAULT_YAW_CALIBRATION = load_yaw_calibration(_DEFAULT_STARTUP_POSE)
_DEFAULT_PITCH_CALIBRATION = load_pitch_calibration(_DEFAULT_STARTUP_POSE)
try:
    _DEFAULT_AXIS_LIMITS = load_axis_limits(DEFAULT_LIMITS_PATH)
except FileNotFoundError:
    # Keep the old conservative behavior if a checkout predates the limits
    # file.  New checkouts include the file and the manual capture tool replaces
    # it with measured motor-space endpoints.
    _DEFAULT_AXIS_LIMITS = {
        "yaw": SimpleNamespace(
            min_rad=min(
                _DEFAULT_YAW_CALIBRATION.motor_from_muzzle(math.radians(-35.0)),
                _DEFAULT_YAW_CALIBRATION.motor_from_muzzle(math.radians(35.0)),
            ),
            max_rad=max(
                _DEFAULT_YAW_CALIBRATION.motor_from_muzzle(math.radians(-35.0)),
                _DEFAULT_YAW_CALIBRATION.motor_from_muzzle(math.radians(35.0)),
            ),
            motor_id=0x03,
        ),
        "pitch": SimpleNamespace(
            min_rad=min(
                _DEFAULT_PITCH_CALIBRATION.motor_from_muzzle(0.0),
                _DEFAULT_PITCH_CALIBRATION.motor_from_muzzle(math.radians(45.0)),
            ),
            max_rad=max(
                _DEFAULT_PITCH_CALIBRATION.motor_from_muzzle(0.0),
                _DEFAULT_PITCH_CALIBRATION.motor_from_muzzle(math.radians(45.0)),
            ),
            motor_id=0x01,
        ),
    }
_DEFAULT_YAW_LIMITS = (
    _DEFAULT_AXIS_LIMITS["yaw"].min_rad,
    _DEFAULT_AXIS_LIMITS["yaw"].max_rad,
)
_DEFAULT_PITCH_LIMITS = (
    _DEFAULT_AXIS_LIMITS["pitch"].min_rad,
    _DEFAULT_AXIS_LIMITS["pitch"].max_rad,
)


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

    # Use the actual camera -> yaw-axis -> pitch-axis geometry for a
    # base-fixed camera.  The virtual screen test turns this off because a
    # displayed target has no real-world range.
    use_axis_geometry: bool = True

    # Camera optical axis direction when both motor axes are at their
    # calibrated zero.  Positive yaw points the camera view to the right of
    # the yaw-zero forward direction; positive pitch points it upward.
    camera_to_yaw_yaw_rad: float = 0.0
    camera_to_yaw_pitch_rad: float = 0.0

    # Position of the horizontal rotation axis, measured from the camera
    # origin and expressed in camera coordinates (+x right, +y up, +z forward).
    # Measure these on the vehicle after the camera is mounted.
    yaw_axis_offset_x_m: float = 0.0
    yaw_axis_offset_y_m: float = 0.0
    yaw_axis_offset_z_m: float = 0.0

    # Position of the vertical rotation axis from the horizontal axis at yaw
    # zero, expressed in yaw-zero coordinates.  Most mounts have x ~= 0.
    pitch_axis_offset_x_m: float = 0.0
    pitch_axis_offset_y_m: float = 0.0
    pitch_axis_offset_z_m: float = 0.0

    # Mechanical zero: direction of the launcher when the motor is at zero.
    # Defaults are read from startup_pose.json so a manual origin capture also
    # applies to the autonomous controller on its next launch.
    yaw_zero_rad: float = _DEFAULT_YAW_CALIBRATION.motor_zero_rad
    pitch_zero_rad: float = _DEFAULT_PITCH_CALIBRATION.motor_zero_rad

    # Change these to -1 if the installed motor direction is reversed.
    # Motor feedback decreases when the turret turns physically right.
    yaw_sign: int = _DEFAULT_YAW_CALIBRATION.motor_sign
    # Motor feedback increases while this mechanism lowers the muzzle.
    pitch_sign: int = _DEFAULT_PITCH_CALIBRATION.motor_sign

    # Mechanical safety limits.  These are deliberately conservative defaults.
    # Numeric encoder limits: physical +35 deg right is the lower coordinate,
    # physical -35 deg left is the upper coordinate because yaw_sign=-1.
    yaw_min_rad: float = _DEFAULT_YAW_LIMITS[0]
    yaw_max_rad: float = _DEFAULT_YAW_LIMITS[1]
    # Numeric motor-space limits corresponding to physical muzzle 45..0 deg.
    # (pitch_sign=-1, so the upper muzzle pose has the lower motor coordinate.)
    pitch_min_rad: float = _DEFAULT_PITCH_LIMITS[0]
    pitch_max_rad: float = _DEFAULT_PITCH_LIMITS[1]

    # Position-mode command limit.  This is not the baseball launch speed.
    max_motor_speed_rad_s: float = 1.0

    # --- Target board -------------------------------------------------------
    # 3 columns x 4 rows of holes.  Top row is the small ones, the three rows
    # below are large.  Known diameters are what make single-camera ranging
    # possible: distance = focal_px * diameter_m / apparent_px.
    small_hole_diameter_m: float = 0.20
    large_hole_diameter_m: float = 0.40
    # Allowed physical centre-position error when validating a detected grid.
    # The official drawing and field measurement allow roughly +/-2 cm.
    board_position_tolerance_cm: float = 2.0
    # Dark-board ROI detector.  Tune value threshold if field illumination
    # makes the black surface appear brighter or the background darker.
    board_dark_value_threshold: int = 105
    board_min_area_fraction: float = 0.04

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
        if self.board_position_tolerance_cm <= 0:
            raise ValueError("board_position_tolerance_cm must be positive")
        if not 0 <= self.board_dark_value_threshold <= 255:
            raise ValueError("board_dark_value_threshold must be between 0 and 255")
        if not 0.0 < self.board_min_area_fraction < 1.0:
            raise ValueError("board_min_area_fraction must be between 0 and 1")
        geometry_values = (
            self.camera_to_yaw_yaw_rad,
            self.camera_to_yaw_pitch_rad,
            self.yaw_axis_offset_x_m,
            self.yaw_axis_offset_y_m,
            self.yaw_axis_offset_z_m,
            self.pitch_axis_offset_x_m,
            self.pitch_axis_offset_y_m,
            self.pitch_axis_offset_z_m,
        )
        if not all(math.isfinite(value) for value in geometry_values):
            raise ValueError("camera and axis geometry values must be finite")

    @property
    def needs_distance(self) -> bool:
        """True when a correction cannot be applied without the target range."""
        if self.use_axis_geometry and not self.camera_on_turret:
            return True
        if self.compensate_gravity:
            return True
        return self.compensate_parallax and self.muzzle_offset != (0.0, 0.0, 0.0)

    @property
    def muzzle_offset(self) -> tuple:
        return (self.muzzle_offset_x_m, self.muzzle_offset_y_m, self.muzzle_offset_z_m)

    @property
    def motor_limits(self) -> Dict[int, Tuple[float, float]]:
        """Absolute motor-position limits for the final backend safety check."""
        return {
            self.yaw_motor_id: (self.yaw_min_rad, self.yaw_max_rad),
            self.pitch_motor_id: (self.pitch_min_rad, self.pitch_max_rad),
        }

    @property
    def yaw_axis_offset(self) -> tuple:
        return (
            self.yaw_axis_offset_x_m,
            self.yaw_axis_offset_y_m,
            self.yaw_axis_offset_z_m,
        )

    @property
    def pitch_axis_offset(self) -> tuple:
        return (
            self.pitch_axis_offset_x_m,
            self.pitch_axis_offset_y_m,
            self.pitch_axis_offset_z_m,
        )

"""Camera-target to turret-angle conversion.

This module contains no camera, CAN or vendor-specific code, so it can be
developed and tested on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from . import ballistics
from .config import TurretConfig
from .geometry import aim_angles_from_target, normalized_image_coordinates


@dataclass(frozen=True)
class TargetPixel:
    """A selected YOLO target in image coordinates."""

    u_px: float
    v_px: float
    confidence: float = 1.0
    # Optical-axis depth to the target in metres.  None means "unknown", which
    # disables the axis geometry and gravity corrections -- and blocks firing when the config
    # says those corrections matter.
    distance_m: Optional[float] = None
    # Empirical elevation correction for this target, physical radians,
    # positive = raise the muzzle.  Applied before limit/workspace checks.
    pitch_trim_rad: float = 0.0
    # Same for azimuth, physical radians, positive = turn the muzzle right.
    yaw_trim_rad: float = 0.0
    # Per-target "arrived" tolerance on the servo error; None uses
    # config.aim_tolerance_rad.  Only ever tightens what may fire.
    aim_tolerance_rad: Optional[float] = None


@dataclass(frozen=True)
class AimCommand:
    """Command produced by the aiming layer."""

    has_target: bool
    target_yaw_rad: Optional[float]
    target_pitch_rad: Optional[float]
    # Bearing of the target measured from the camera axis.
    yaw_error_rad: float = 0.0
    pitch_error_rad: float = 0.0
    # How far the motors still are from the commanded angle.  None when the
    # caller passed no feedback.
    yaw_servo_error_rad: Optional[float] = None
    pitch_servo_error_rad: Optional[float] = None
    # False when a mechanical limit clipped the target, when the projectile
    # cannot reach it at this muzzle speed, or when a needed range is missing.
    target_reachable: bool = True
    # Why the shot is not available, for logging.  None when it is fine.
    blocked_reason: Optional[str] = None
    distance_m: Optional[float] = None
    # Distances after transforming the target into the two joint frames.
    yaw_axis_distance_m: Optional[float] = None
    pitch_axis_distance_m: Optional[float] = None
    # Physical barrel elevation that points the barrel line straight at the
    # target (no gravity, no trim), and the ballistic correction added on top.
    geometric_pitch_rad: Optional[float] = None
    ballistic_correction_rad: Optional[float] = None
    stable: bool = False
    fire_allowed: bool = False


# Float slack when checking whether clamp() actually clipped the target.
_REACHABLE_EPS_RAD = 1e-9


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AimingController:
    """Convert a target pixel into safe horizontal/vertical motor positions."""

    def __init__(self, config: TurretConfig):
        config.validate()
        self.config = config
        self._stable_frames = 0

    def reset_stability(self) -> None:
        """Require fresh settled observations after an external motion gate."""
        self._stable_frames = 0

    def pixel_to_error(self, target: TargetPixel) -> tuple[float, float]:
        """Return yaw and pitch error in radians.

        Positive yaw means target is to the right. Positive pitch means target
        is above the image center. The signs can be changed in TurretConfig to
        match the installed motor directions.
        """

        normalized_x, normalized_y = normalized_image_coordinates(
            target.u_px,
            target.v_px,
            self.config,
        )
        yaw_error = math.atan2(normalized_x, 1.0)
        pitch_error = math.atan2(normalized_y, 1.0)
        return (
            self.config.yaw_sign * yaw_error,
            self.config.pitch_sign * pitch_error,
        )

    def update(
        self,
        target: Optional[TargetPixel],
        current_yaw_rad: Optional[float] = None,
        current_pitch_rad: Optional[float] = None,
    ) -> AimCommand:
        """Create one command from the newest target detection.

        ``current_yaw_rad``/``current_pitch_rad`` are the measured motor angles.
        With a base-fixed camera they are required to decide whether the turret
        actually reached the aim point, because the camera does not move with
        the turret and so the pixel error never shrinks.  Without them firing
        stays disabled rather than guessing.

        No target, low confidence, a target the mechanical limits cannot reach,
        or missing feedback all keep ``fire_allowed`` false.  The caller must
        keep *holding* the motors when ``has_target`` is false -- never stop
        sending frames, or the comms-loss watchdog disables the motors and the
        unbraked vertical axis drops.
        """

        if target is None or target.confidence < self.config.minimum_confidence:
            self._stable_frames = 0
            return AimCommand(
                has_target=False,
                target_yaw_rad=None,
                target_pitch_rad=None,
            )

        blocked_reason = None
        geometry_result = None
        geometric_pitch = None
        ballistic_correction = None
        if self.config.use_axis_geometry and not self.config.camera_on_turret:
            # A base-fixed camera must use the target's 3-D position.  The
            # virtual screen test disables this branch because its apparent
            # hole size is not a physical range measurement.
            if target.distance_m is None and self.config.require_distance_to_fire:
                blocked_reason = "no range to target; axis geometry is unavailable"
                yaw_error, pitch_error = self.pixel_to_error(target)
            elif target.distance_m is None:
                # Explicit diagnostic fallback.  Normal field configuration
                # keeps require_distance_to_fire=True.
                yaw_error, pitch_error = self.pixel_to_error(target)
            else:
                geometry_config = self.config
                if not self.config.compensate_parallax:
                    # Keep camera alignment, but intentionally remove the
                    # camera-to-axis translation from the calculation.
                    from dataclasses import replace

                    geometry_config = replace(
                        self.config,
                        yaw_axis_offset_x_m=0.0,
                        yaw_axis_offset_y_m=0.0,
                        yaw_axis_offset_z_m=0.0,
                        pitch_axis_offset_x_m=0.0,
                        pitch_axis_offset_y_m=0.0,
                        pitch_axis_offset_z_m=0.0,
                        pitch_exit_offset_y_m=0.0,
                        pitch_exit_offset_z_m=0.0,
                    )
                try:
                    wanted_yaw, wanted_pitch, geometry_result = aim_angles_from_target(
                        target.u_px,
                        target.v_px,
                        target.distance_m,
                        geometry_config,
                    )
                    yaw_error = wanted_yaw - self.config.yaw_zero_rad
                    pitch_error = wanted_pitch - self.config.pitch_zero_rad

                    geometric_pitch = geometry_result.pitch_rad
                    if self.config.compensate_gravity:
                        horizontal = math.hypot(
                            geometry_result.target_from_pitch_axis.x,
                            geometry_result.target_from_pitch_axis.z,
                        )
                        pitch_for_projectile = ballistics.rotating_exit_gravity_pitch(
                            horizontal,
                            geometry_result.target_from_pitch_axis.y,
                            self.config.muzzle_speed_m_s,
                            self.config.gravity_m_s2,
                            geometry_config.pitch_exit_offset_y_m,
                            geometry_config.pitch_exit_offset_z_m,
                            initial_pitch_rad=geometric_pitch,
                            launch_angle_ratio=self.config.launch_angle_ratio,
                        )
                        ballistic_correction = pitch_for_projectile - geometric_pitch
                        pitch_error = self.config.pitch_sign * pitch_for_projectile
                except (ValueError, ballistics.OutOfRange) as exc:
                    blocked_reason = str(exc)
                    yaw_error, pitch_error = self.pixel_to_error(target)
        else:
            yaw_error, pitch_error = self.pixel_to_error(target)
            # Legacy/camera-on-turret path: re-aim from the muzzle and add the
            # drop compensation.  Both need the range.
            missing_range = self.config.needs_distance and target.distance_m is None
            if missing_range and self.config.require_distance_to_fire:
                blocked_reason = "no range to target; parallax/drop cannot be applied"
            else:
                # solve_aim returns the bearing unchanged when distance is
                # None, so this is the deliberate straight-line fallback.
                try:
                    yaw_error, pitch_error = ballistics.solve_aim(
                        yaw_error,
                        pitch_error,
                        target.distance_m,
                        muzzle_offset=self.config.muzzle_offset,
                        muzzle_speed_m_s=self.config.muzzle_speed_m_s,
                        gravity_m_s2=self.config.gravity_m_s2,
                        compensate_parallax=self.config.compensate_parallax,
                        compensate_gravity=self.config.compensate_gravity,
                    )
                except ballistics.OutOfRange as exc:
                    blocked_reason = str(exc)

        if target.pitch_trim_rad and not self.config.camera_on_turret:
            # Base-fixed camera: pitch_error is a motor offset from zero, and
            # motor = zero + pitch_sign * physical elevation.
            pitch_error += self.config.pitch_sign * target.pitch_trim_rad
        if target.yaw_trim_rad and not self.config.camera_on_turret:
            yaw_error += self.config.yaw_sign * target.yaw_trim_rad

        if self.config.camera_on_turret:
            # The error is relative to where the turret already points.
            yaw_base = current_yaw_rad if current_yaw_rad is not None else self.config.yaw_zero_rad
            pitch_base = (
                current_pitch_rad if current_pitch_rad is not None else self.config.pitch_zero_rad
            )
        else:
            yaw_base = self.config.yaw_zero_rad
            pitch_base = self.config.pitch_zero_rad

        yaw_wanted = yaw_base + yaw_error
        pitch_wanted = pitch_base + pitch_error
        yaw_target = clamp(yaw_wanted, self.config.yaw_min_rad, self.config.yaw_max_rad)
        pitch_target = clamp(pitch_wanted, self.config.pitch_min_rad, self.config.pitch_max_rad)

        # A clipped target means the turret cannot physically point at it, so
        # "stable" would otherwise latch on while aiming at the end stop.
        reachable = (
            abs(yaw_target - yaw_wanted) <= _REACHABLE_EPS_RAD
            and abs(pitch_target - pitch_wanted) <= _REACHABLE_EPS_RAD
        )
        if not reachable and blocked_reason is None:
            blocked_reason = "target is outside the mechanical limits"
        if blocked_reason is not None:
            reachable = False

        yaw_servo_error = None
        pitch_servo_error = None
        if current_yaw_rad is not None:
            yaw_servo_error = yaw_target - current_yaw_rad
        if current_pitch_rad is not None:
            pitch_servo_error = pitch_target - current_pitch_rad

        tolerance = self.config.aim_tolerance_rad
        if target.aim_tolerance_rad is not None:
            tolerance = min(tolerance, target.aim_tolerance_rad)
        if self.config.camera_on_turret:
            # Driving the target to image centre is exactly the goal.
            within_tolerance = (
                abs(yaw_error) <= tolerance
                and abs(pitch_error) <= tolerance
            )
        elif yaw_servo_error is None or pitch_servo_error is None:
            # Base-fixed camera with no feedback: there is no way to know the
            # turret arrived.  Fail closed.
            within_tolerance = False
        else:
            within_tolerance = (
                abs(yaw_servo_error) <= tolerance
                and abs(pitch_servo_error) <= tolerance
            )

        within_tolerance = within_tolerance and reachable
        self._stable_frames = self._stable_frames + 1 if within_tolerance else 0
        stable = self._stable_frames >= self.config.stable_frames_required

        return AimCommand(
            has_target=True,
            target_yaw_rad=yaw_target,
            target_pitch_rad=pitch_target,
            yaw_error_rad=yaw_error,
            pitch_error_rad=pitch_error,
            yaw_servo_error_rad=yaw_servo_error,
            pitch_servo_error_rad=pitch_servo_error,
            target_reachable=reachable,
            blocked_reason=blocked_reason,
            distance_m=target.distance_m,
            yaw_axis_distance_m=(
                geometry_result.yaw_axis_distance_m if geometry_result is not None else None
            ),
            pitch_axis_distance_m=(
                geometry_result.pitch_axis_distance_m if geometry_result is not None else None
            ),
            geometric_pitch_rad=geometric_pitch,
            ballistic_correction_rad=ballistic_correction,
            stable=stable,
            # The launcher should still have a separate physical safety gate.
            fire_allowed=stable,
        )

"""3-D camera-to-turret geometry for a base-fixed camera.

Coordinate convention
---------------------
The camera frame is ``+x right, +y up, +z forward``.  The target distance used
here is the optical-axis depth ``z``.  This is the quantity returned by the
pinhole relation ``focal_px * real_diameter / apparent_diameter`` used by the
board detector; converting it to ``x/y/z`` before subtracting the axis offsets
avoids treating an off-centre target as if it were on the optical axis.

The camera is fixed to the vehicle.  Therefore the target bearing is not an
increment to the turret's current motor position: it is an absolute bearing
from the horizontal axis zero.  The horizontal-axis and vertical-axis offsets
are included so close targets are corrected for parallax.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from .config import TurretConfig


@dataclass(frozen=True)
class Point3D:
    """A point/vector in one of the documented right-handed frames."""

    x: float
    y: float
    z: float

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class AxisAim:
    """Angles and useful distances calculated from one target point."""

    yaw_rad: float
    pitch_rad: float
    camera_point: Point3D
    target_from_yaw_axis: Point3D
    target_from_pitch_axis: Point3D
    yaw_axis_distance_m: float
    pitch_axis_distance_m: float


def _rotate_y(point: Point3D, angle_rad: float) -> Point3D:
    """Rotate around +y; positive angle turns +z toward +x."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return Point3D(
        c * point.x + s * point.z,
        point.y,
        -s * point.x + c * point.z,
    )


def _rotate_x(point: Point3D, angle_rad: float) -> Point3D:
    """Rotate around +x in a frame whose +y points upward."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return Point3D(
        point.x,
        c * point.y - s * point.z,
        s * point.y + c * point.z,
    )


def camera_to_yaw_zero(point: Point3D, config: TurretConfig) -> Point3D:
    """Transform a vector from camera axes to the yaw-zero axes.

    ``camera_to_yaw_yaw_rad`` and ``camera_to_yaw_pitch_rad`` describe where
    the camera optical axis points when the turret is at mechanical zero.
    They are normally small installation/calibration angles.
    """
    # R_y(yaw) * R_x(-pitch) makes camera +z point at the configured yaw and
    # pitch in the turret frame.  Applying the two elementary rotations in
    # this order also keeps the convention easy to verify on the bench.
    return _rotate_y(
        _rotate_x(point, -config.camera_to_yaw_pitch_rad),
        config.camera_to_yaw_yaw_rad,
    )


def normalized_image_coordinates(
    u_px: float,
    v_px: float,
    config: TurretConfig,
) -> Tuple[float, float]:
    """Return an undistorted camera ray as ``(x right, y up)`` at z=1.

    This is the inverse of OpenCV's 4/5/8-coefficient Brown--Conrady camera
    model.  Keeping it here lets both the simple bearing path and the full
    camera-to-axis geometry consume the same calibrated ray.
    """

    distorted_x = (
        u_px - config.camera_center_x_px
    ) / config.focal_length_x_px
    distorted_y = (
        v_px - config.camera_center_y_px
    ) / config.focal_length_y_px
    coefficients = list(config.camera_distortion_coeffs) + [0.0] * 8
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients[:8]
    if not any(config.camera_distortion_coeffs):
        return distorted_x, -distorted_y

    x = distorted_x
    y = distorted_y
    for _ in range(8):
        radius2 = x * x + y * y
        radius4 = radius2 * radius2
        radius6 = radius4 * radius2
        numerator = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
        denominator = 1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6
        if abs(denominator) < 1e-12:
            raise ValueError("camera distortion denominator is zero")
        radial = numerator / denominator
        if abs(radial) < 1e-12:
            raise ValueError("camera distortion radial scale is zero")
        tangential_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        tangential_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - tangential_x) / radial
        y = (distorted_y - tangential_y) / radial
    return x, -y


def pixel_to_camera_point(
    u_px: float,
    v_px: float,
    distance_m: float,
    config: TurretConfig,
) -> Point3D:
    """Convert an image point and optical-axis depth into a camera point."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    z = distance_m
    normalized_x, normalized_y = normalized_image_coordinates(u_px, v_px, config)
    return Point3D(
        normalized_x * z,
        normalized_y * z,
        z,
    )


def target_to_axis_aim(
    u_px: float,
    v_px: float,
    distance_m: float,
    config: TurretConfig,
) -> AxisAim:
    """Calculate absolute yaw/pitch angles from a camera detection.

    The horizontal-axis position is measured in the camera frame.  The
    vertical-axis position is measured in the yaw-zero frame, from the
    horizontal axis to the pitch axis.  The small Newton solve accounts for a
    non-zero horizontal offset of the pitch axis; with the common ``x=0``
    mount it reduces to the usual ``atan2(x, z)`` calculation.
    """
    camera_point = pixel_to_camera_point(u_px, v_px, distance_m, config)
    yaw_axis_in_camera = Point3D(
        config.yaw_axis_offset_x_m,
        config.yaw_axis_offset_y_m,
        config.yaw_axis_offset_z_m,
    )
    target_from_yaw = camera_to_yaw_zero(camera_point - yaw_axis_in_camera, config)
    yaw_axis_distance = target_from_yaw.norm()
    if yaw_axis_distance <= 0:
        raise ValueError("target coincides with the horizontal axis")

    pitch_axis_offset = Point3D(
        config.pitch_axis_offset_x_m,
        config.pitch_axis_offset_y_m,
        config.pitch_axis_offset_z_m,
    )

    # Solve for the yaw that puts the target in the vertical-axis plane.
    yaw_rad = math.atan2(target_from_yaw.x, target_from_yaw.z)
    for _ in range(8):
        in_yaw_frame = _rotate_y(target_from_yaw, -yaw_rad)
        error_x = in_yaw_frame.x - pitch_axis_offset.x
        derivative = -in_yaw_frame.z
        if abs(derivative) < 1e-9:
            break
        correction = error_x / derivative
        yaw_rad -= correction
        if abs(correction) < 1e-10:
            break

    target_from_pitch = _rotate_y(target_from_yaw, -yaw_rad) - pitch_axis_offset
    pitch_axis_distance = target_from_pitch.norm()
    if pitch_axis_distance <= 0:
        raise ValueError("target coincides with the vertical axis")

    pitch_rad = math.atan2(
        target_from_pitch.y,
        math.hypot(target_from_pitch.x, target_from_pitch.z),
    )
    return AxisAim(
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        camera_point=camera_point,
        target_from_yaw_axis=target_from_yaw,
        target_from_pitch_axis=target_from_pitch,
        yaw_axis_distance_m=yaw_axis_distance,
        pitch_axis_distance_m=pitch_axis_distance,
    )


def aim_angles_from_target(
    u_px: float,
    v_px: float,
    distance_m: float,
    config: TurretConfig,
) -> Tuple[float, float, AxisAim]:
    """Return motor-space target angles and the underlying geometry result."""
    result = target_to_axis_aim(u_px, v_px, distance_m, config)
    return (
        config.yaw_zero_rad + config.yaw_sign * result.yaw_rad,
        config.pitch_zero_rad + config.pitch_sign * result.pitch_rad,
        result,
    )

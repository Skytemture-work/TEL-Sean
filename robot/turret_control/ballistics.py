"""Parallax and gravity corrections.

Pure maths, no camera / CAN / vendor code, so every branch here is unit
testable on any machine.

Frame convention (camera frame): +x right, +y up, +z forward along the optical
axis.  Bearings follow the same convention the aiming layer uses:

    yaw   = atan2(x, z)                 positive to the right
    pitch = atan2(y, hypot(x, z))       positive upwards

Both corrections need the *range* to the target.  A single YOLO box gives a
bearing, not a range, so the caller must supply a distance (stereo depth, a
known target size, or a measured tape) or accept the straight-line assumption.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class OutOfRange(ValueError):
    """The projectile cannot reach the target at this muzzle speed."""


def bearing_to_unit_vector(yaw_rad: float, pitch_rad: float) -> Tuple[float, float, float]:
    """Unit vector pointing along the given bearing."""
    cos_pitch = math.cos(pitch_rad)
    return (
        cos_pitch * math.sin(yaw_rad),
        math.sin(pitch_rad),
        cos_pitch * math.cos(yaw_rad),
    )


def unit_vector_to_bearing(x: float, y: float, z: float) -> Tuple[float, float]:
    """Inverse of :func:`bearing_to_unit_vector`."""
    return math.atan2(x, z), math.atan2(y, math.hypot(x, z))


def apply_parallax(
    yaw_rad: float,
    pitch_rad: float,
    distance_m: float,
    muzzle_offset: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Re-aim a camera bearing so it comes from the muzzle instead.

    Returns ``(yaw, pitch, muzzle_to_target_distance)``.

    The camera and the launcher are not at the same point, so pointing the
    launcher along the bearing the *camera* measured misses sideways.  The
    error is roughly ``offset / distance`` radians, i.e. negligible far away
    and large up close: a 10 cm offset at 2 m is already 2.9 degrees.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")

    ux, uy, uz = bearing_to_unit_vector(yaw_rad, pitch_rad)
    # Target position in camera frame.
    tx, ty, tz = ux * distance_m, uy * distance_m, uz * distance_m
    # Same point, measured from the muzzle.
    mx, my, mz = tx - muzzle_offset[0], ty - muzzle_offset[1], tz - muzzle_offset[2]

    range_m = math.sqrt(mx * mx + my * my + mz * mz)
    if range_m <= 0:
        raise ValueError("target coincides with the muzzle")

    yaw, pitch = unit_vector_to_bearing(mx, my, mz)
    return yaw, pitch, range_m


def gravity_pitch(
    horizontal_distance_m: float,
    height_m: float,
    muzzle_speed_m_s: float,
    gravity_m_s2: float = 9.81,
) -> float:
    """Launch pitch that makes a projectile pass through (distance, height).

    Solves ``h = d*tan(t) - g*d^2 / (2*v^2*cos^2(t))`` for ``t`` and returns the
    flatter of the two solutions -- the flat shot spends less time in the air,
    so it is less sensitive to speed error and to a moving target.

    Raises :class:`OutOfRange` when no solution exists.
    """
    d = horizontal_distance_m
    h = height_m
    v = muzzle_speed_m_s
    g = gravity_m_s2

    if v <= 0:
        raise ValueError("muzzle_speed_m_s must be positive")
    if g == 0:
        return math.atan2(h, d)
    if d <= 0:
        # Straight up (or the target is on top of the muzzle); no ballistic arc
        # is meaningful, so just point at it.
        return math.atan2(h, d) if (d or h) else 0.0

    v2 = v * v
    discriminant = v2 * v2 - g * (g * d * d + 2.0 * h * v2)
    if discriminant < 0:
        raise OutOfRange(
            f"cannot reach {d:.2f} m out and {h:.2f} m up at {v:.1f} m/s"
        )

    root = math.sqrt(discriminant)
    # The minus root is the flat trajectory, the plus root the lobbed one.
    return math.atan2(v2 - root, g * d)


def solve_aim(
    yaw_rad: float,
    pitch_rad: float,
    distance_m: Optional[float],
    muzzle_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    muzzle_speed_m_s: float = 0.0,
    gravity_m_s2: float = 9.81,
    compensate_parallax: bool = False,
    compensate_gravity: bool = False,
) -> Tuple[float, float]:
    """Turn a camera bearing into the bearing the launcher should hold.

    With both corrections off, or with no distance available, this returns the
    input unchanged -- the straight-line assumption.
    """
    yaw, pitch = yaw_rad, pitch_rad

    if distance_m is None or distance_m <= 0:
        return yaw, pitch

    range_m = distance_m
    if compensate_parallax and muzzle_offset != (0.0, 0.0, 0.0):
        yaw, pitch, range_m = apply_parallax(yaw, pitch, distance_m, muzzle_offset)

    if compensate_gravity:
        horizontal = range_m * math.cos(pitch)
        height = range_m * math.sin(pitch)
        pitch = gravity_pitch(horizontal, height, muzzle_speed_m_s, gravity_m_s2)

    return yaw, pitch

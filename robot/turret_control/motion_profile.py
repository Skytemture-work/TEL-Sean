"""Shared motion limits: manual mode and automatic aiming move equally fast.

Both main.py manual control (manual.py, MIT + smoothstep trajectories) and the
board auto-aim pipeline (posvel + trapezoidal trajectories) read their DM
limits here, and both use the same MKS pusher constants from manual.py, so
neither mode is slower than the other.
"""

from __future__ import annotations

import math

# DM yaw/pitch.  Previous values: manual 0.40 rad/s (acceleration unbounded,
# ~2.8 rad/s^2 peak on adjacent-hole moves), auto 0.35 rad/s / 0.28 rad/s^2.
DM_SPEED_RAD_S = 0.60
DM_ACCELERATION_RAD_S2 = 3.0

SMOOTHSTEP_MIN_DURATION_S = 0.25


def smoothstep_duration(
    distance_rad: float,
    speed_rad_s: float = DM_SPEED_RAD_S,
    acceleration_rad_s2: float = DM_ACCELERATION_RAD_S2,
    minimum_s: float = SMOOTHSTEP_MIN_DURATION_S,
) -> float:
    """Shortest smoothstep duration within both the speed and acceleration limit.

    smoothstep peaks at 1.5 * d / T velocity and 6 * d / T^2 acceleration.
    """
    distance = abs(float(distance_rad))
    if speed_rad_s <= 0 or acceleration_rad_s2 <= 0:
        raise ValueError("speed and acceleration must be positive")
    return max(
        1.5 * distance / speed_rad_s,
        math.sqrt(6.0 * distance / acceleration_rad_s2),
        minimum_s,
    )

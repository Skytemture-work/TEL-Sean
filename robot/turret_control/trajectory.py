"""Small, deterministic motion profiles for motor commands."""

from __future__ import annotations

import math
from typing import Tuple


def smoothstep_sample(
    start_rad: float,
    target_rad: float,
    elapsed_s: float,
    duration_s: float,
) -> Tuple[float, float]:
    """Return a position and velocity on a zero-velocity smoothstep profile.

    The cubic smoothstep has zero velocity at both ends, so the command does
    not jump from a stopped pose to a finite speed or stop abruptly.  The
    returned velocity is intended for MIT ``dq`` feed-forward.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    if elapsed_s <= 0.0:
        return start_rad, 0.0
    if elapsed_s >= duration_s:
        return target_rad, 0.0

    u = elapsed_s / duration_s
    blend = u * u * (3.0 - 2.0 * u)
    blend_rate = 6.0 * u * (1.0 - u) / duration_s
    delta = target_rad - start_rad
    return start_rad + delta * blend, delta * blend_rate


def acceleration_limited_step(
    position_rad: float,
    velocity_rad_s: float,
    target_rad: float,
    max_speed_rad_s: float,
    max_acceleration_rad_s2: float,
    dt_s: float,
) -> Tuple[float, float]:
    """Advance a command with bounded speed and acceleration.

    The next velocity is chosen so that a fixed target remains reachable with
    the available braking acceleration.  Integrating the average of the old
    and new velocity avoids a position step at startup.  A target changed too
    close to a moving command may be passed while the command brakes, because
    preserving the acceleration limit is safer than snapping the velocity to
    zero; it will then return to the new target continuously.
    """
    if max_speed_rad_s <= 0.0:
        raise ValueError("max_speed_rad_s must be positive")
    if max_acceleration_rad_s2 <= 0.0:
        raise ValueError("max_acceleration_rad_s2 must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    remaining = target_rad - position_rad
    max_delta_velocity = max_acceleration_rad_s2 * dt_s

    # Once both errors fit inside one acceleration quantum, settling exactly
    # avoids a tiny discrete-time limit cycle.  This final correction still
    # respects the per-tick velocity limit, and its position correction is at
    # most a * dt^2 (0.000115 degrees with the panel defaults).
    settle_position_tolerance = max(1e-12, max_delta_velocity * dt_s)
    if (
        abs(remaining) <= settle_position_tolerance
        and abs(velocity_rad_s) <= max_delta_velocity
    ):
        return target_rad, 0.0

    direction = 1.0 if remaining >= 0.0 else -1.0
    distance = abs(remaining)
    toward_velocity = direction * velocity_rad_s

    # For trapezoidal integration, require the distance left after this tick
    # to be at least the continuous stopping distance of the new velocity:
    #
    #   distance - (v + v_next) * dt / 2 >= v_next**2 / (2*a)
    #
    # The positive root is the greatest viable next velocity.  If a target is
    # changed so late that the current state is already outside this envelope,
    # the acceleration lower bound wins and brakes at exactly the configured
    # limit instead of introducing a command discontinuity.
    discriminant = (
        max_delta_velocity * max_delta_velocity
        - 4.0 * max_delta_velocity * toward_velocity
        + 8.0 * max_acceleration_rad_s2 * distance
    )
    viable_velocity = 0.5 * (
        -max_delta_velocity + math.sqrt(max(0.0, discriminant))
    )
    new_toward_velocity = min(
        max_speed_rad_s,
        toward_velocity + max_delta_velocity,
        viable_velocity,
    )
    new_toward_velocity = max(
        -max_speed_rad_s,
        toward_velocity - max_delta_velocity,
        new_toward_velocity,
    )
    new_velocity = direction * new_toward_velocity
    new_position = position_rad + 0.5 * (velocity_rad_s + new_velocity) * dt_s
    return new_position, new_velocity

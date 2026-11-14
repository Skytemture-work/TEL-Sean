"""Pre-shot check (and optional fine correction) with the muzzle camera.

Once the chassis-camera aim has settled on a hole, the muzzle camera sees
where that hole really is relative to the barrel.  When the barrel is aimed
correctly, the hole appears at

* the barrel point for its range (camera/barrel parallax, see
  ``muzzle_camera.barrel_aim_pixel``),
* lowered by the barrel's deliberate elevation above the line of sight
  (ballistic correction + any configured per-hole pitch trim),
* shifted by the measured camera-vs-barrel misalignment (boresight offsets:
  where the barrel points in the image relative to the parallel-axis model,
  + = right / up; equal to the mean measured error of shots that hit).

The difference between that expected pixel and the detected hole centre is
the remaining aim error.  Modes:

``measure``  record the error for every shot, never move or delay the gun
             (collects the data to calibrate the boresight offsets);
``apply``    hold the shot, add the error to that hole's aim, re-settle,
             re-measure, then release the shot.

Any doubt (hole not found, implausibly large error, timeout) releases the
shot on the unchanged chassis-camera aim, which is already accurate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Optional, Tuple

from .muzzle_camera import MuzzleIntrinsics, barrel_aim_pixel

MODES = ("off", "measure", "apply")


@dataclass(frozen=True)
class MuzzleCorrectionSettings:
    offset_x_m: float
    offset_y_m: float
    boresight_yaw_rad: float = 0.0
    boresight_pitch_rad: float = 0.0
    # Wait after the aim reports stable before trusting a frame (vibration).
    settle_delay_s: float = 0.12
    # Errors below this are within measurement noise: no correction.
    deadband_rad: float = math.radians(0.12)
    # A larger error is treated as a wrong match, never as a correction.
    max_error_rad: float = math.radians(1.0)
    # Total correction per hole is capped as well.
    max_total_rad: float = math.radians(1.0)
    max_corrections: int = 2
    # Give up waiting for a usable measurement after this long.
    timeout_s: float = 1.5


def expected_hole_pixel(intrinsics: MuzzleIntrinsics, settings: MuzzleCorrectionSettings,
                        distance_m: float, elevation_rad: float) -> Tuple[float, float]:
    """Where the hole should appear when the barrel is aimed correctly.

    ``elevation_rad`` is how far the barrel is deliberately raised above the
    hole (ballistic correction + pitch trim); the hole then appears lower.
    """
    u, v = barrel_aim_pixel(intrinsics, distance_m, settings.offset_x_m, settings.offset_y_m)
    yaw = math.atan2(u - intrinsics.cx, intrinsics.fx) + settings.boresight_yaw_rad
    down = math.atan2(v - intrinsics.cy, intrinsics.fy) + elevation_rad - settings.boresight_pitch_rad
    return (intrinsics.cx + intrinsics.fx * math.tan(yaw),
            intrinsics.cy + intrinsics.fy * math.tan(down))


def aim_error_rad(intrinsics: MuzzleIntrinsics, expected: Tuple[float, float],
                  measured: Tuple[float, float]) -> Tuple[float, float]:
    """Physical (yaw, pitch) still needed: + yaw = turn right, + pitch = raise."""
    yaw = (math.atan2(measured[0] - intrinsics.cx, intrinsics.fx)
           - math.atan2(expected[0] - intrinsics.cx, intrinsics.fx))
    pitch = (math.atan2(expected[1] - intrinsics.cy, intrinsics.fy)
             - math.atan2(measured[1] - intrinsics.cy, intrinsics.fy))
    return yaw, pitch


def locate_hole(plan, detections, hole_id: str) -> Optional[Tuple[float, float]]:
    """Centre of the *detected* hole that BoardVision's plan labels ``hole_id``.

    The plan supplies identities (it may be inferred from the board outline);
    the measurement must come from a real detection, uniquely nearest to the
    labelled position, of a plausible size.
    """
    if plan is None:
        return None
    targets = list(getattr(plan, "targets", plan))
    labelled = [t for t in targets if t.hole_id == hole_id]
    if len(labelled) != 1:
        return None
    reference = labelled[0]
    spacing = min((math.hypot(t.u_px - reference.u_px, t.v_px - reference.v_px)
                   for t in targets if t is not reference), default=None)
    if not spacing:
        return None
    gate = 0.35 * spacing
    candidates = []
    for detection in detections:
        distance = math.hypot(detection.u_px - reference.u_px, detection.v_px - reference.v_px)
        if (distance <= gate and reference.size_px > 0
                and 0.6 * reference.size_px <= detection.size_px <= 1.6 * reference.size_px):
            candidates.append((distance, detection))
    if len(candidates) != 1:
        return None  # nothing, or ambiguous
    detection = candidates[0][1]
    return float(detection.u_px), float(detection.v_px)


@dataclass
class MuzzleShotRecord:
    """What the muzzle camera saw for one shot; goes into the shot log."""

    hole_id: Optional[str]
    verdict: str = "pending"
    error_yaw_rad: Optional[float] = None
    error_pitch_rad: Optional[float] = None
    expected_px: Optional[Tuple[float, float]] = None
    measured_px: Optional[Tuple[float, float]] = None
    corrections: int = 0
    total_yaw_rad: float = 0.0
    total_pitch_rad: float = 0.0
    measurements: int = 0

    def as_log(self) -> Dict[str, object]:
        def deg(value):
            return None if value is None else round(math.degrees(value), 3)
        return {
            "muzzle_verdict": self.verdict,
            "muzzle_error_yaw_deg": deg(self.error_yaw_rad),
            "muzzle_error_pitch_deg": deg(self.error_pitch_rad),
            "muzzle_correction_yaw_deg": deg(self.total_yaw_rad),
            "muzzle_correction_pitch_deg": deg(self.total_pitch_rad),
            "muzzle_corrections": self.corrections,
        }


# measure(hole_id, captured_after) -> None while no new enough frame exists,
# or (found, measured_px) once a frame was analysed.
Measure = Callable[[str, float], Optional[Tuple[bool, Optional[Tuple[float, float]]]]]


@dataclass
class _ShotState:
    key: Tuple
    record: MuzzleShotRecord
    started_at: float
    settled_since: Optional[float] = None
    done: bool = False


class MuzzleShotCorrector:
    """Per-shot state machine; call ``step`` once per control tick."""

    def __init__(self, mode: str, intrinsics: MuzzleIntrinsics,
                 settings: MuzzleCorrectionSettings, measure: Measure,
                 apply_correction: Callable[[str, float, float], None]):
        if mode not in MODES:
            raise ValueError(f"unknown muzzle correction mode: {mode}")
        self.mode = mode
        self.intrinsics = intrinsics
        self.settings = settings
        self._measure = measure
        self._apply = apply_correction
        self._state: Optional[_ShotState] = None
        # Accumulated correction per hole: the error is systematic for a
        # stationary robot, so the next ball at the same hole starts from it.
        self.totals: Dict[str, Tuple[float, float]] = {}
        self.last: Optional[MuzzleShotRecord] = None

    def reset(self) -> None:
        """Forget all corrections (plan change or chassis moved)."""
        for hole_id in list(self.totals):
            self._apply(hole_id, 0.0, 0.0)
        self.totals.clear()
        self._state = None

    def record_for(self, key) -> Optional[MuzzleShotRecord]:
        state = self._state
        return state.record if state is not None and state.key == key else None

    def step(self, now: float, key, hole_id: Optional[str], stable: bool,
             distance_m: Optional[float], elevation_rad: Optional[float]) -> Optional[str]:
        """Return a reason to hold the shot, or None to let it go.

        ``key`` identifies the shot (plan revision, index, hole).  ``stable``,
        ``distance_m`` and ``elevation_rad`` come from the latest aim command.
        """
        if self.mode == "off" or hole_id is None:
            return None
        state = self._state
        if state is None or state.key != key:
            yaw, pitch = self.totals.get(hole_id, (0.0, 0.0))
            state = self._state = _ShotState(
                key, MuzzleShotRecord(hole_id, total_yaw_rad=yaw, total_pitch_rad=pitch), now)
            self.last = state.record
        if state.done:
            return None
        hold = "砲口相機確認中" if self.mode == "apply" else None
        if now - state.started_at > self.settings.timeout_s and state.settled_since is not None:
            return self._finish(state, "timeout")
        if not stable or distance_m is None or elevation_rad is None:
            state.settled_since = None
            if not stable:
                state.started_at = now  # time spent aiming is not a muzzle timeout
            return hold
        if state.settled_since is None:
            state.settled_since = now
        if now - state.settled_since < self.settings.settle_delay_s:
            return hold
        observation = self._measure(hole_id, state.settled_since + self.settings.settle_delay_s)
        if observation is None:
            return hold  # no frame captured after settling yet
        found, measured = observation
        record = state.record
        record.measurements += 1
        if not found or measured is None:
            return self._finish(state, "not_found")
        expected = expected_hole_pixel(self.intrinsics, self.settings, distance_m, elevation_rad)
        yaw, pitch = aim_error_rad(self.intrinsics, expected, measured)
        record.expected_px, record.measured_px = expected, measured
        record.error_yaw_rad, record.error_pitch_rad = yaw, pitch
        size = math.hypot(yaw, pitch)
        if size > self.settings.max_error_rad:
            return self._finish(state, "rejected")
        if size <= self.settings.deadband_rad:
            return self._finish(state, "verified" if record.corrections == 0 else "corrected")
        if self.mode != "apply":
            return self._finish(state, "measured")
        if record.corrections >= self.settings.max_corrections:
            return self._finish(state, "corrected")
        total_yaw = record.total_yaw_rad + yaw
        total_pitch = record.total_pitch_rad + pitch
        if math.hypot(total_yaw, total_pitch) > self.settings.max_total_rad:
            return self._finish(state, "rejected")
        record.total_yaw_rad, record.total_pitch_rad = total_yaw, total_pitch
        record.corrections += 1
        self.totals[hole_id] = (total_yaw, total_pitch)
        self._apply(hole_id, total_yaw, total_pitch)
        state.settled_since = None  # re-settle on the corrected aim, then re-measure
        state.started_at = now
        return "砲口相機修正後重新到位"

    def _finish(self, state: _ShotState, verdict: str) -> None:
        state.done = True
        state.record.verdict = verdict
        return None

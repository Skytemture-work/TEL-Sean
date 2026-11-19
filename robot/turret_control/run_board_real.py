"""Real-camera, real-motor target-board aiming run.

This controls the two DAMIAO motors and can build the upper/lower-half
maximum-score plans from the 2026-09-03 rules. ``--panel-fire`` starts the dual
VESC flywheels at speed 100 and requires a fresh panel click for every MKS
stage. The legacy CLI defaults to an operator-confirmed ``f`` key.
An integrating process can instead pass a real
``fire_callback`` and receive structured state through ``telemetry_callback``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from contextlib import ExitStack
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import cv2
import numpy as np

if "bool" not in np.__dict__:
    np.bool = bool  # type: ignore[attr-defined]

from .board import BoardTarget, EngagementPlan
from .board_vision import BoardVision, DEFAULT_MODEL, DEFAULT_CONFIDENCE
from .board_depth_plane import apply_board_depth_plane, fit_board_depth_plane
from .board_pose import (
    BoardPoseLock,
    BoardPoseTracker,
    board_lean_deg,
    camera_matrix,
    distortion_coefficients,
    prefer_observed_depth,
    project_targets_from_pose,
    update_plan_geometry,
)
from .config import (
    TurretConfig,
    with_recorded_mechanism_geometry,
    with_zed2_hd720_uvc_calibration,
)
from .competition_geometry import with_competition_geometry
from .controller import TurretController
from .dashboard import (
    DashboardState,
    build_dashboard_snapshot,
    start_dashboard_server,
)
from .engagement import BoardEngagement, PartialPoseShotGate
from .fixed_board import calibrate_layout, load_layout
from .shared_motor_backend import (
    DEFAULT_AIM_ACCELERATION_RAD_S2,
    DEFAULT_AIM_SPEED_RAD_S,
    MAX_AIM_ACCELERATION_RAD_S2,
    MAX_AIM_SPEED_RAD_S,
    SharedMotorBackend,
)
from .run_yolo_demo import _inference_frame
from .official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM
from .live_preview import LatestFrameGrabber, LivePreviewPublisher
from .match_sequencer import MatchSequencer
from .scoring_light import ScoreRoundController, draw_score_light_overlay
from .scoring_strategy import (
    MATCH_PHASES,
    UPPER_BONUS_BALLS,
    append_reallocated_grid_shots,
    build_max_score_strategy,
)
from .zed_calibration import load_zed2_uvc_calibration
from .zed_camera import ZedDepthCamera

FireCallback = Callable[[BoardTarget], bool]
TelemetryCallback = Callable[[Dict[str, Any]], None]


_SMALL_HOLE_IDS = ("H10", "H11", "H12")


def _fire_tolerance_map(large_deg, small_deg):
    """Per-hole arrival tolerance in radians; holes left out keep the default."""
    result = {}
    for number in range(1, 13):
        hole = f"H{number:02d}"
        deg = small_deg if hole in _SMALL_HOLE_IDS else large_deg
        if deg is not None:
            result[hole] = math.radians(deg)
    return result


_MAX_PITCH_TRIM_DEG = 3.0
# 2026-11-16 (robot ~5.4 m from the board, 35 m/s model): H06 landed 15 cm
# high on average over 10 shots; -1.5 deg on H06, H08, H01 and H11 was much
# more accurate, so every hole starts from it.  --pitch-trim-all changes it,
# --pitch-trim HOLE=DEG replaces it for one hole.
DEFAULT_PITCH_TRIM_ALL_DEG = -1.5


# Ballistic model measured on 2026-11-13 (board 1:1, pitch axis 5.49 m from the
# board and 31.5 cm above the floor, flywheels speed=100 / 4000 RPM):
# * the PnP board lean averaged +0.81 deg on a vertical board, so the real
#   camera mount pitch is ~20 - 0.8 = 19.2 deg (CAD value 20 deg);
# * the four row misses (8/11/15/20 cm low) at the inclinometer-verified barrel
#   angles fit a drag-free arc of 32 m/s launched at 0.942 x barrel elevation
#   within +-0.7 cm.  Re-fit after changing the flywheel speed.
# * 2026-11-14: with backspin (top 3800 / bottom 4200 RPM) the lift made 32 m/s
#   land 10-15 cm high on the whole grid at ~6 m and 43 m/s 10-15 cm low;
#   35 m/s tested most accurate.  The parameter only scales the modelled
#   drop (an effective speed), the flywheels are unchanged.
DEFAULT_CAMERA_PITCH_DEG = 19.2
EQUAL_SPEED_MUZZLE_SPEED_M_S = 32.0  # 2026-11-13 fit, both wheels 4000 RPM
DEFAULT_MUZZLE_SPEED_M_S = 35.0
DEFAULT_LAUNCH_ANGLE_RATIO = 0.942
# Pre-aim on release: H11 muzzle targets were 31.6-31.9 deg in the
# 2026-11-16 runs (robot stopped ~5.2-5.5 m from the board).
DEFAULT_PREAIM_PITCH_DEG = 31.7
# Lower half starts on H01 (bottom of column 1): 12.7-13.5 deg on 2026-11-16.
DEFAULT_PREAIM_LOWER_PITCH_DEG = 13.0


def _parse_shot_order(text):
    """Parse ``"H07,H08,H05,H05"`` (repeats allowed) into hole IDs."""
    if text is None:
        return None
    order = []
    for item in str(text).replace(" ", "").split(","):
        if not item:
            continue
        hole = item.upper()
        if not (hole.startswith("H") and hole[1:].isdigit() and 1 <= int(hole[1:]) <= 12):
            raise ValueError(f"--shot-order 孔位必須是 H01..H12：{item!r}")
        order.append(f"H{int(hole[1:]):02d}")
    if not 1 <= len(order) <= 36:
        raise ValueError("--shot-order 需要 1～36 個孔位")
    return order


def _shot_record(shot, shot_no, previous_started_at, command, config, *, pitch_trim_deg,
                 range_pair, lean_deg):
    """One JSONL row describing a launcher stage at the moment it started."""
    def deg(value):
        return None if value is None else round(math.degrees(value), 3)

    def physical(target, error, zero, sign):
        if target is None:
            return None, None
        current = target - error if error is not None else None
        return (deg((target - zero) * sign),
                deg((current - zero) * sign) if current is not None else None)

    yaw_target, yaw_current = physical(command.target_yaw_rad, command.yaw_servo_error_rad,
                                       config.yaw_zero_rad, config.yaw_sign)
    pitch_target, pitch_current = physical(command.target_pitch_rad, command.pitch_servo_error_rad,
                                           config.pitch_zero_rad, config.pitch_sign)
    rpm = shot.get("rpm")
    started = shot["started_at"]
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "shot": shot_no,
        "hole_id": shot["hole_id"],
        "plan_index": shot["plan_index"],
        "slot": shot["slot"],
        "stage": shot["stage"],
        "rpm_left": round(rpm[0], 1) if rpm else None,
        "rpm_right": round(rpm[1], 1) if rpm else None,
        "rpm_age_s": round(started - rpm[2], 3) if rpm else None,
        "since_prev_shot_s": (round(started - previous_started_at, 2)
                              if previous_started_at is not None else None),
        # Previous stage: mechanism time (start -> ball released) and the time
        # from that release until this stage started (aiming / settle wait).
        "prev_mechanism_s": shot.get("prev_mechanism_s"),
        "wait_after_release_s": shot.get("wait_after_release_s"),
        "muzzle_yaw_target_deg": yaw_target,
        "muzzle_yaw_current_deg": yaw_current,
        "muzzle_pitch_target_deg": pitch_target,
        "muzzle_pitch_current_deg": pitch_current,
        "geometric_pitch_deg": deg(command.geometric_pitch_rad),
        "ballistic_correction_deg": deg(command.ballistic_correction_rad),
        "pitch_trim_deg": pitch_trim_deg,
        "pnp_distance_m": round(range_pair[0], 3) if range_pair and range_pair[0] else None,
        "zed_distance_m": round(range_pair[1], 3) if range_pair and range_pair[1] else None,
        "pitch_axis_distance_m": (round(command.pitch_axis_distance_m, 3)
                                  if command.pitch_axis_distance_m is not None else None),
        "board_lean_deg": round(lean_deg, 2) if lean_deg is not None else None,
    }


def _with_default_pitch_trim(all_deg, per_hole):
    """Every hole gets ``all_deg``; a hole given in ``per_hole`` replaces it."""
    if not all_deg:
        return dict(per_hole)
    return {**{f"H{n:02d}": all_deg for n in range(1, 13)}, **per_hole}


def _parse_pitch_trims(items):
    """Parse ``["H10=0.8", ...]`` into ``{"H10": 0.8}`` (degrees, + = muzzle up)."""
    trims = {}
    for item in items:
        hole, sep, value = str(item).partition("=")
        hole = hole.strip().upper()
        if not sep or not hole:
            raise ValueError(f"--pitch-trim 格式應為 HOLE=DEG，例如 H10=0.8：{item!r}")
        if hole.startswith("H") and hole[1:].isdigit():
            number = int(hole[1:])
            hole = f"H{number:02d}"
        else:
            number = 0
        if not 1 <= number <= 12:
            raise ValueError(f"--pitch-trim 孔位必須是 H01..H12：{item!r}")
        try:
            deg = float(value)
        except ValueError:
            raise ValueError(f"--pitch-trim 角度必須是數字：{item!r}") from None
        if not math.isfinite(deg) or abs(deg) > _MAX_PITCH_TRIM_DEG:
            raise ValueError(
                f"--pitch-trim 角度必須在 ±{_MAX_PITCH_TRIM_DEG:g} 度內：{item!r}")
        if hole in trims:
            raise ValueError(f"--pitch-trim 重複指定 {hole}")
        trims[hole] = deg
    return trims


def _stereo_depth_needed(use_zed_sdk, range_source, zed_depth_compare=False):
    """Whether the control loop must wait for a stereo depth map.

    Aiming ranges come from the board PnP pose in the default ``pnp`` mode, so
    waiting for (and computing) a fresh ZED ULTRA depth frame every loop only
    feeds the dashboard's PnP/ZED comparison.  Request depth only when the
    range source is ``zed`` or the comparison was asked for explicitly.
    """
    return bool(use_zed_sdk and (range_source == "zed" or zed_depth_compare))


def _yolo_ms(vision_result):
    """Ultralytics preprocess+inference+postprocess time, excluding board analysis."""
    speed = getattr(getattr(vision_result, "result", None), "speed", None) or {}
    try:
        return float(sum(v for v in speed.values() if v is not None))
    except (TypeError, ValueError):
        return None


def _apply_manual_aim_action(plan, action, confirmed_holes):
    """Apply one validated navigation request on the control thread."""
    if plan is None or action is None:
        return plan, False
    operation, expected_index, expected_hole = action
    current = plan.current()
    if plan.index != expected_index or getattr(current, "hole_id", None) != expected_hole:
        return plan, False
    if operation == "confirm" and current is not None:
        confirmed_holes.append(current.hole_id)
        plan.skip()
        return plan, True
    if operation == "reset":
        confirmed_holes.clear()
        return EngagementPlan(plan.targets), True
    return plan, False


def _restore_completed_aim_plan(plan, confirmed_holes):
    """Restore a known completed 12-hole validation after a restart."""
    if plan is None or len(plan) != 12 or {t.hole_id for t in plan.targets} != {
        f"H{number:02d}" for number in range(1, 13)
    }:
        return False
    confirmed_holes[:] = [t.hole_id for t in plan.targets]
    while not plan.finished:
        plan.skip()
    return True


def _all_holes_confirmed(plan, confirmed_holes):
    return (plan is not None and plan.finished and len(confirmed_holes) == 12
            and set(confirmed_holes) == {f"H{n:02d}" for n in range(1, 13)})


UNREACHABLE_SKIP_S = 3.0
# Headless: draw the dashboard overlay at most this often (~5 Hz); it cost
# ~60 ms on every frame of a ~375 ms control loop.
PREVIEW_RENDER_S = 0.2
# E-stop during a reload: how often to retry the DM until the 24 V is back.
POWER_RETRY_S = 3.0


class UnreachableHoleSkipper:
    """Match mode: give up on a hole that stays outside the turret's reach.

    The per-target timeout is off in panel/match firing, so a hole the turret
    cannot reach (e.g. H11 outside the high-elevation workspace after the
    chassis stopped off-centre) used to stall the whole match.  Once the
    current hole has been unreachable for ``hold_s`` -- with a stable pose and
    an idle launcher -- every consecutive plan entry for that hole is skipped.
    """

    def __init__(self, hold_s=UNREACHABLE_SKIP_S, *, clock=time.monotonic):
        self.hold_s = float(hold_s)
        self._clock = clock
        self._hole = None
        self._since = None

    def update(self, plan, status, pose_ready, launcher_idle):
        """Return ``(hole_id, skipped_count)`` when a hole was skipped, else None."""
        target = plan.current() if plan is not None else None
        unreachable = bool(
            target is not None and pose_ready and launcher_idle
            and status is not None and status.target is not None
            and status.target.hole_id == target.hole_id
            and not status.command.target_reachable
        )
        if not unreachable:
            self._hole = self._since = None
            return None
        now = self._clock()
        if target.hole_id != self._hole:
            self._hole, self._since = target.hole_id, now
            return None
        if now - self._since < self.hold_s:
            return None
        hole, count = self._hole, 0
        while plan.current() is not None and plan.current().hole_id == hole:
            plan.skip()
            count += 1
        self._hole = self._since = None
        return hole, count


def _auto_fire_aim_ready(launcher, plan, pose_ready, previous_status):
    """The aim part: the previous tick had a fire-allowed aim on this hole.

    Starts the auto-fire settle timer, also while the pusher still finishes
    the previous stage."""
    current = plan.current() if plan is not None else None
    return bool(
        current is not None
        and pose_ready
        and previous_status is not None
        and not previous_status.fired
        and previous_status.target is not None
        and previous_status.target.hole_id == current.hole_id
        and previous_status.command.fire_allowed
        and launcher.flywheel_ready
    )


def _auto_fire_mechanism_ready(driver, launcher):
    """The mechanism part: nothing pending or moving, cooldown over."""
    return bool(not launcher.pending and not launcher.mechanism_busy
                and driver.fire_cooldown_ready)


def _auto_fire_ready(driver, launcher, plan, pose_ready, previous_status):
    """True when the previous tick already had a fire-allowed aim on this hole."""
    return (_auto_fire_aim_ready(launcher, plan, pose_ready, previous_status)
            and _auto_fire_mechanism_ready(driver, launcher))


def _update_panel_engagement(driver, launcher, dashboard, plan, pose_ready, blocked_reason,
                             auto_fire=None, previous_status=None, fire_hold=None,
                             hold_reason=None):
    """Validate a click on this frame; never retain an unexecuted authorization.

    With ``auto_fire`` the click comes from ``AutoFireTrigger`` instead of the
    web button (web clicks are consumed and ignored); the click is then checked
    by exactly the same launcher and engagement gates.
    """
    launcher.check_health()
    # A pose lost during a push (someone walks past the camera, the chassis
    # is bumped, a few holes missed for a frame) must not end the match.  The
    # push finishes on its own and is counted on release without a pose; the
    # turret already holds still while the launcher is busy, and no new shot
    # starts until the pose is valid again (firing is gated on pose_ready).
    # hold_reason (e.g. waiting for a reload) keeps the turret still and
    # blocks firing; fire_hold only blocks firing.
    driver.set_aiming_enabled(
        pose_ready and not launcher.busy and hold_reason is None,
        "發射階段執行中，砲口保持位置" if launcher.busy else (hold_reason or blocked_reason),
    )
    driver.set_firing_enabled(pose_ready and fire_hold is None and hold_reason is None,
                              blocked_reason if not pose_ready else (hold_reason or fire_hold))
    action = dashboard.take_panel_fire()
    if auto_fire is not None:
        action = auto_fire.take(
            plan, driver.plan_revision,
            _auto_fire_aim_ready(launcher, plan, pose_ready, previous_status),
            can_fire=_auto_fire_mechanism_ready(driver, launcher),
        )
    launcher.arm(action, plan, driver.plan_revision)
    try:
        return driver.update()
    finally:
        launcher.finish_tick()


def _make_muzzle_corrector(mode, model, config, muzzle_preview, driver, args):
    """Pre-shot muzzle check sharing the chassis YOLO model (one call per shot)."""
    from .muzzle_correction import MuzzleCorrectionSettings, MuzzleShotCorrector, locate_hole

    intrinsics = muzzle_preview.intrinsics
    # Same BoardVision entry point, configured for the upright muzzle image.
    muzzle_config = replace(
        config, image_width_px=intrinsics.width_px, image_height_px=intrinsics.height_px,
        focal_length_x_px=intrinsics.fx, focal_length_y_px=intrinsics.fy,
        camera_principal_x_px=intrinsics.cx, camera_principal_y_px=intrinsics.cy,
        camera_distortion_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))
    muzzle_vision = BoardVision(model, muzzle_config, confidence=args.conf, imgsz=args.imgsz,
                                device=args.device)

    def measure(hole_id, captured_after):
        frame, captured_at = muzzle_preview.latest_frame()
        if frame is None or captured_at < captured_after:
            return None
        result = muzzle_vision.process(frame, now=captured_at)
        measured = locate_hole(result.plan, result.targets, hole_id)
        return measured is not None, measured

    settings = MuzzleCorrectionSettings(
        offset_x_m=config.muzzle_barrel_offset_x_m,
        offset_y_m=config.muzzle_barrel_offset_y_m,
        boresight_yaw_rad=math.radians(config.muzzle_boresight_yaw_deg),
        boresight_pitch_rad=math.radians(config.muzzle_boresight_pitch_deg),
    )
    print(f"砲口相機發射前確認：{mode}"
          + ("（只量測記錄，不改變瞄準與發射時機）" if mode == "measure"
             else "（誤差超過 0.12° 時修正瞄準並重新到位，最多 2 次、上限 1°）"), flush=True)
    return MuzzleShotCorrector(mode, intrinsics, settings, measure, driver.set_muzzle_correction)


def _step_muzzle_corrector(corrector, muzzle_preview, driver, plan, previous_status,
                           pitch_trim_deg, now):
    """Advance the muzzle check for the current shot; return a fire hold or None."""
    current = plan.current() if plan is not None else None
    hole_id = current.hole_id if current is not None else None
    command = previous_status.command if previous_status is not None else None
    same_target = (previous_status is not None and previous_status.target is not None
                   and previous_status.target.hole_id == hole_id)
    stable = bool(same_target and command is not None and command.stable)
    distance = command.distance_m if same_target and command is not None else None
    elevation = None
    if same_target and command is not None:
        elevation = ((command.ballistic_correction_rad or 0.0)
                     + math.radians(pitch_trim_deg.get(hole_id or "", 0.0)))
    key = (driver.plan_revision, plan.index if plan is not None else None, hole_id)
    hold = corrector.step(now, key, hole_id, stable, distance, elevation)
    record = corrector.record_for(key)
    if record is not None and record.measurements:
        if record.error_yaw_rad is not None:
            text = (f"{hole_id} {record.verdict}  err yaw {math.degrees(record.error_yaw_rad):+.2f} "
                    f"pitch {math.degrees(record.error_pitch_rad):+.2f} deg")
        else:
            text = f"{hole_id} {record.verdict}"
        muzzle_preview.set_measurement((record.expected_px, record.measured_px, text))
    return hold


def _take_external_reload(reload_event, waiting):
    """Consume an external reload confirmation, only while a reload is awaited.

    Outside the wait the event is cleared, so an early or accidental press is
    never carried over into the next reload.
    """
    if reload_event is None:
        return False
    if not waiting:
        reload_event.clear()
        return False
    if reload_event.is_set():
        reload_event.clear()
        return True
    return False


def _home_turret_and_wait(motors, timeout_s=20.0):
    """Send both DM axes home along the recorded safe path and wait.

    Shared by the normal shutdown and by an early exit (before START), so a
    Ctrl+C at any point leaves the turret at home.  Never raises; the caller
    disables the motors afterwards.
    """
    try:
        snapshot = motors.worker.snapshot()
        if snapshot.closed or not snapshot.enabled:
            print("馬達控制器已關閉或未使能，不執行回 home。", flush=True)
            return
        try:
            motors.request_home()
        except (ValueError, RuntimeError) as exc:
            print(f"Ctrl+C 回 home 未執行：{exc}", flush=True)
            return
        print("Ctrl+C：雙軸沿安全路徑返回 home，等待到位。", flush=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state, message = motors.home_status()
            if state == "arrived":
                print("Ctrl+C：已回 home，準備失能。", flush=True)
                return
            if state == "error":
                print(f"Ctrl+C 回 home 失敗：{message}", flush=True)
                return
            time.sleep(0.02)
        print("Ctrl+C 回 home 逾時，現在停止並失能。", flush=True)
    except Exception as exc:
        print(f"Ctrl+C 回 home 流程發生錯誤：{exc}", flush=True)


def _shutdown_on_exit_signals():
    """Turn SIGTERM / SIGHUP (kill, SSH hang-up) into a KeyboardInterrupt.

    The normal shutdown (flywheels off, pusher and turret home, both ZED
    cameras released) then runs instead of the process dying with the
    hardware still claimed.  On a hang-up the terminal is gone, so output is
    redirected to /dev/null first: a failing print must not abort shutdown.
    Returns the previous handlers for ``_restore_signals``; a no-op outside
    the main thread (signal handlers can only be set there).
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    watched = [signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])

    def handler(signum, _frame):
        for sig in watched:  # one shutdown only; let it finish
            signal.signal(sig, signal.SIG_IGN)
        if signum == getattr(signal, "SIGHUP", None):
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
        raise KeyboardInterrupt(f"received signal {signum}")

    return {sig: signal.signal(sig, handler) for sig in watched}


def _restore_signals(previous):
    for sig, old in previous.items():
        signal.signal(sig, old)


def _start_muzzle_preview(config, dashboard, args, camera=None):
    """Open the muzzle ZED for the dashboard; any failure only disables it.

    ``camera`` is a muzzle ZED already opened during the prewarm phase.
    """
    from .muzzle_camera import MuzzlePreview, MuzzleZedCamera

    try:
        if camera is None:
            camera = MuzzleZedCamera(config.muzzle_zed_serial)
        preview = MuzzlePreview(
            camera, dashboard.update_muzzle_frame,
            offset_x_m=config.muzzle_barrel_offset_x_m,
            offset_y_m=config.muzzle_barrel_offset_y_m,
            fallback_distance_m=config.muzzle_fallback_distance_m,
            fps=min(args.preview_fps, 10.0), scale=args.preview_scale,
            jpeg_quality=args.preview_quality,
        )
    except Exception as exc:
        if camera is not None:
            camera.release()
        print(f"砲口相機未啟用（不影響主流程）：{exc}", flush=True)
        return None, None
    print(f"砲口相機 S/N {config.muzzle_zed_serial}：畫面已轉正，黃圈為砲管指向點"
          f"（假設與砲管平行，偏移 x={config.muzzle_barrel_offset_x_m * 1000:+.1f} mm、"
          f"y={config.muzzle_barrel_offset_y_m * 1000:+.1f} mm）。", flush=True)
    return camera, preview


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    fire_callback: Optional[FireCallback] = None,
    telemetry_callback: Optional[TelemetryCallback] = None,
    stop_event: Optional[threading.Event] = None,
    reload_event: Optional[threading.Event] = None,
    next_handoff_command: Optional[Callable[[Optional[float]], Optional[str]]] = None,
    positioning_callback: Optional[Callable[[float, int], None]] = None,
    event_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Run the real board pipeline with cleanup for early motor handoff."""
    with ExitStack() as startup_cleanup:
        _run_board_real(
            argv, fire_callback=fire_callback, telemetry_callback=telemetry_callback,
            stop_event=stop_event, reload_event=reload_event,
            next_handoff_command=next_handoff_command, positioning_callback=positioning_callback,
            event_callback=event_callback, startup_cleanup=startup_cleanup,
        )


def _run_board_real(
    argv, *, fire_callback, telemetry_callback, stop_event, reload_event,
    next_handoff_command, positioning_callback, startup_cleanup, event_callback=None,
):
    """Run the real board pipeline.

    ``fire_callback`` is an opt-in automatic launcher hook.  It is called only
    after the vision, pose, reachability and motor-settling gates all pass.  It
    must return True only after the requested ball was actually sent; False
    keeps the same target active.

    ``stop_event`` lets a host program (e.g. ``main.py`` via
    ``auto_session.AutoAimSession``) end the run: once set, the loop exits and
    the normal shutdown runs -- stop the launcher, return along the recorded
    safe path to home, disable the motors, release the camera.

    ``reload_event`` is an external "magazine refilled" confirmation (e.g. the
    remote's RELOAD button through ``AutoAimSession.confirm_reload``), the same
    as the dashboard button.  It only counts while ``--match-fire`` is waiting
    for a reload; a press at any other time is discarded, never stored.

    The optional stdin adapter returns an empty string on timeout and None
    on EOF. Positioning samples are sent before START via positioning_callback.

    ``telemetry_callback`` receives the same dictionary exposed by
    ``GET /api/status`` once per processed camera frame.  Both callbacks run on
    this control thread and therefore must return quickly.
    """
    parser = argparse.ArgumentParser(description="Real camera + real motors, 12-hole board aiming")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV / UVC 相機編號")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--port", default=None, help="DM USB-to-CAN 序列埠；預設依裝置識別碼自動偵測")
    parser.add_argument(
        "--aim-speed", type=float, default=DEFAULT_AIM_SPEED_RAD_S,
        help=f"雙軸自動瞄準的平滑速度上限 rad/s（預設 {DEFAULT_AIM_SPEED_RAD_S}，"
             f"上限 {MAX_AIM_SPEED_RAD_S}）",
    )
    parser.add_argument(
        "--aim-acceleration", type=float, default=DEFAULT_AIM_ACCELERATION_RAD_S2,
        help=f"雙軸自動瞄準的平滑加速度上限 rad/s²（預設 {DEFAULT_AIM_ACCELERATION_RAD_S2}，"
             f"上限 {MAX_AIM_ACCELERATION_RAD_S2}）；相鄰孔移動主要受加速度限制",
    )
    camera_mode = parser.add_mutually_exclusive_group()
    camera_mode.add_argument("--zed-uvc", action="store_true", help="使用 ZED UVC 原始影像，不提供立體深度")
    camera_mode.add_argument("--opencv-camera", action="store_true", help="使用一般 OpenCV 相機，不提供立體深度")
    parser.add_argument(
        "--zed-calibration", type=Path,
        help="ZED 2 原廠 SN*.conf 標定檔（配合 --zed-uvc 的 HD720 未校正左眼影像）",
    )
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=6.0,
        help="等待目前孔位成功發射的秒數；0 表示不自動略過；面板射擊模式固定不自動略過",
    )
    parser.add_argument("--headless", action="store_true", help="不開 OpenCV 視窗")
    launcher_mode = parser.add_mutually_exclusive_group()
    launcher_mode.add_argument(
        "--aim-only",
        action="store_true",
        help="實機驗證模式：允許安全範圍內追瞄，但完全停用發射與 F 操作",
    )
    launcher_mode.add_argument(
        "--panel-fire", action="store_true",
        help="雙 VESC 預設 speed 100，自動瞄準；每次實際發射僅接受網頁射擊按鈕",
    )
    launcher_mode.add_argument(
        "--auto-fire", action="store_true",
        help="全自動：與 --panel-fire 相同的瞄準與發射流程，瞄準到位並穩定 "
             "--auto-fire-settle 秒後自動發射，不需點擊；Ctrl+C 停止並回 home",
    )
    launcher_mode.add_argument(
        "--match-fire", action="store_true",
        help="比賽全自動：計分模式＋自動發射。啟動後保持不動，偵測到指定得分格亮燈即開賽；"
             "依規則配球，比賽時間到或彈藥打完即停止發射",
    )
    parser.add_argument(
        "--light-calibrate", type=int, nargs="?", const=18, default=None, metavar="N",
        help="紅燈校正：啟動後等辨識到亮紅燈的小孔，就對那一孔連射 N 顆（不加 N 預設 18，範圍 1～18）；"
             "不排九宮格、燈熄也不換孔。未指定 --aim-only/--panel-fire 時自動發射",
    )
    parser.add_argument(
        "--balls-loaded", type=int, default=18,
        help="比賽模式：彈藥倉已裝球數（預設 18＝6 格 × 3 顆；範圍 1～18）",
    )
    parser.add_argument(
        "--match-duration", type=float, default=240.0,
        help="比賽模式：發射時限秒數（預設 240，刻意比 3 分鐘賽制長，避免程式提早停機；"
             "賽後以急停結束。由 main.py 執行時從 VRA 發車起算，單獨執行時從辨識到靶板起算；範圍 30～300）",
    )
    parser.add_argument(
        "--fire-tolerance-deg", type=float, default=None,
        help="大孔（九宮格）發射前砲口到位容許誤差，度（發射模式預設 0.5；僅瞄準模式 2.0）",
    )
    parser.add_argument(
        "--small-fire-tolerance-deg", type=float, default=None,
        help="上方小孔發射前砲口到位容許誤差，度（發射模式預設 0.25；僅瞄準模式 2.0）",
    )
    parser.add_argument(
        "--push-rpm", type=int, default=None,
        help="推球馬達 ID1 推球速度 RPM（預設沿用 manual.py 的 SHOOTING_SPEED_RPM=200；範圍 20～600）",
    )
    parser.add_argument(
        "--push-accel", type=int, default=None,
        help="推球馬達 ID1 推球加速度參數（MKS 0～255，數值越大加速越快；"
             "預設沿用 manual.py 的 SHOOTING_ACCELERATION=200；範圍 1～255）",
    )
    parser.add_argument(
        "--push-home-rpm", type=int, default=None,
        help="推球馬達 ID1 回原點速度 RPM（預設沿用 HOME_SPEED_RPM=40；範圍 20～120；調高可能誤判到底）",
    )
    parser.add_argument(
        "--auto-fire-settle", type=float, default=0.15,
        help="全自動模式：同一孔位需連續符合發射條件的秒數（預設 0.15，範圍 0.1～5；原 0.3）",
    )
    parser.add_argument(
        "--reload-first-settle", type=float, default=1.0,
        help="裝填後重新定位完成的第一發，瞄準穩定秒數（預設 1.0；其他發用 --auto-fire-settle）",
    )
    parser.add_argument(
        "--stable-frames", type=int, default=2,
        help="砲口需連續幾個辨識畫面都在容許誤差內才算到位（預設 2，範圍 1～10；原 3）",
    )
    parser.add_argument("--restore-completed-aim", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        help="即時影像、孔位與砲塔資料儀表板的 port（預設 8080；0 停用）",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=15.0,
        help="儀表板即時畫面更新率；相機取像與預覽改由背景執行緒處理，"
             "辨識結果以最新一次 YOLO 疊圖顯示（預設 15；0 = 舊流程，每次辨識才更新畫面）",
    )
    parser.add_argument(
        "--preview-scale", type=float, default=0.5,
        help="儀表板即時畫面縮放倍率（預設 0.5 = 640x360，降低 Wi-Fi 負載與延遲；1 = 原尺寸）",
    )
    parser.add_argument(
        "--preview-quality", type=int, default=70,
        help="儀表板即時畫面 JPEG 品質（預設 70，範圍 30～95）",
    )
    parser.add_argument(
        "--pitch-trim",
        action="append",
        default=[],
        metavar="HOLE=DEG",
        help="依孔位微調仰角（度，正值 = 砲口抬高），可重複指定，例如 "
             "--pitch-trim H10=0.8 --pitch-trim H11=0.8；指定的孔取代 --pitch-trim-all 的值"
             "（不是相加）；微調後仍經過限位與工作空間檢查",
    )
    parser.add_argument(
        "--pitch-trim-all", type=float, default=DEFAULT_PITCH_TRIM_ALL_DEG, metavar="DEG",
        help=f"所有孔的預設仰角微調（度，正值 = 砲口抬高；預設 {DEFAULT_PITCH_TRIM_ALL_DEG:g}，"
             f"2026-11-16 實測；0 = 不微調；範圍 ±{_MAX_PITCH_TRIM_DEG:g}）",
    )
    parser.add_argument(
        "--shot-order", default=None, metavar="H07,H08,...",
        help="測試用：指定射擊／瞄準順序（逗號分隔，可重複同一孔，最多 36 個；僅限 --no-score-mode）",
    )
    parser.add_argument(
        "--shot-log", type=Path, default=None,
        help="每發記錄 JSONL 檔（預設 shot_logs/shots_<時間>.jsonl，發射模式自動開啟）",
    )
    parser.add_argument(
        "--camera-pitch-deg", type=float, default=DEFAULT_CAMERA_PITCH_DEG,
        help=f"相機安裝仰角（度；預設 {DEFAULT_CAMERA_PITCH_DEG}，由垂直靶面的傾斜量校正；"
             "儀表板 board_lean_avg_deg 應接近 0）",
    )
    parser.add_argument(
        "--preaim-pitch-deg", type=float, default=DEFAULT_PREAIM_PITCH_DEG,
        help=f"放行後（車還在跑）砲口先轉到的仰角，約為 H11（預設 {DEFAULT_PREAIM_PITCH_DEG}；0 = 不預瞄）",
    )
    parser.add_argument(
        "--preaim-lower-pitch-deg", type=float, default=DEFAULT_PREAIM_LOWER_PITCH_DEG,
        help=f"下半場預瞄仰角，約為 H01（預設 {DEFAULT_PREAIM_LOWER_PITCH_DEG}；0 = 不預瞄）",
    )
    parser.add_argument(
        "--preaim-yaw-deg", type=float, default=0.0,
        help="預瞄的水平角（度，相對中心；預設 0 = 正前方）",
    )
    parser.add_argument(
        "--no-muzzle-camera", action="store_true",
        help="不開啟砲口 ZED 相機（儀表板只顯示車體相機）",
    )
    parser.add_argument(
        "--muzzle-correction", choices=("off", "measure", "apply"), default="measure",
        help="砲口相機發射前確認：measure＝只量測並記錄誤差（預設，不影響瞄準與發射）；"
             "apply＝量到誤差就修正瞄準、重新到位後才發射；off＝不使用",
    )
    parser.add_argument(
        "--no-pose-lock", action="store_true",
        help="停用遮蔽姿態鎖定：靶板被擋住時立即停止瞄準／發射（舊行為）",
    )
    parser.add_argument(
        "--range-source", choices=("pnp", "zed"), default="pnp",
        help="孔的距離來源：pnp = 整片靶的 PnP 位置（預設）；zed = ZED 深度（舊行為，孔內深度會偏遠）",
    )
    parser.add_argument(
        "--zed-depth-compare", action="store_true",
        help="pnp 模式下仍每圈計算 ZED 深度，只供儀表板「距離 PnP / ZED」對照（會拖慢迴圈；預設關閉）",
    )
    parser.add_argument(
        "--muzzle-speed", type=float, default=DEFAULT_MUZZLE_SPEED_M_S,
        help=f"彈道模型出球速度 m/s（預設 {DEFAULT_MUZZLE_SPEED_M_S}，飛輪 speed=100 實測擬合）",
    )
    parser.add_argument(
        "--launch-angle-ratio", type=float, default=DEFAULT_LAUNCH_ANGLE_RATIO,
        help=f"出球仰角 / 砲管仰角（預設 {DEFAULT_LAUNCH_ANGLE_RATIO}，實測擬合）",
    )
    parser.add_argument(
        "--no-ballistics", action="store_true",
        help="關閉重力／出球角補償，砲管直線對準孔心（相機角度與距離來源修正仍套用）",
    )
    parser.add_argument(
        "--virtual-board",
        action="store_true",
        help="螢幕虛擬靶測試：關閉距離、視差與重力補正",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        help="固定相機版面 JSON；提供後不再使用每幀候選孔建立計畫",
    )
    parser.add_argument(
        "--calibrate-layout",
        type=Path,
        help="第一次完整偵測到 12 孔時儲存固定版面 JSON",
    )
    score_mode = parser.add_mutually_exclusive_group()
    score_mode.add_argument(
        "--score-mode", dest="score_mode", action="store_true", default=True,
        help="競賽計分模式（預設啟用）：按 G 鎖定亮燈孔並建立射擊計畫",
    )
    score_mode.add_argument(
        "--no-score-mode", dest="score_mode", action="store_false",
        help="切換一般模式",
    )
    parser.add_argument(
        # 0 (2026-11-16): the bonus light is on from the start, so do not wait
        # with the turret idle.  Aim at H11 at once; the light is confirmed in
        # ~3 frames, usually before the first push, and the remaining bonus
        # balls then switch to the lit hole.  In every test run the 5 s wait
        # ended without a light anyway.
        "--light-wait", type=float, default=0.0,
        help="上半場：開賽後等待確認亮燈的秒數，逾時先射 H11（之後確認亮燈即改射亮燈孔）；預設 0（燈開賽即亮，不等），範圍 0～55",
    )
    parser.add_argument(
        "--match-phase",
        choices=MATCH_PHASES,
        default="upper",
        help="比賽半場；決定 27/18 球上限與最大得分射擊順序",
    )
    args = parser.parse_args(argv)
    if (not math.isfinite(args.aim_speed) or args.aim_speed <= 0
            or args.aim_speed > MAX_AIM_SPEED_RAD_S):
        parser.error(f"--aim-speed 必須在 0～{MAX_AIM_SPEED_RAD_S} rad/s 之間")
    if (not math.isfinite(args.aim_acceleration) or args.aim_acceleration <= 0
            or args.aim_acceleration > MAX_AIM_ACCELERATION_RAD_S2):
        parser.error(f"--aim-acceleration 必須在 0～{MAX_AIM_ACCELERATION_RAD_S2} rad/s² 之間")
    if args.cooldown < 0:
        parser.error("--cooldown 不可小於 0")
    if args.target_timeout < 0:
        parser.error("--target-timeout 不可小於 0")
    if not math.isfinite(args.preview_fps) or args.preview_fps < 0:
        parser.error("--preview-fps 必須是 >= 0 的有限數值")
    if not math.isfinite(args.preview_scale) or not 0.25 <= args.preview_scale <= 1.0:
        parser.error("--preview-scale 必須在 0.25～1 之間")
    if not 30 <= args.preview_quality <= 95:
        parser.error("--preview-quality 必須在 30～95 之間")
    try:
        pitch_trim_deg = _parse_pitch_trims(args.pitch_trim)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.pitch_trim_all) or abs(args.pitch_trim_all) > _MAX_PITCH_TRIM_DEG:
        parser.error(f"--pitch-trim-all 必須在 ±{_MAX_PITCH_TRIM_DEG:g} 度內")
    pitch_trim_deg = _with_default_pitch_trim(args.pitch_trim_all, pitch_trim_deg)
    try:
        shot_order = _parse_shot_order(args.shot_order)
    except ValueError as exc:
        parser.error(str(exc))
    if shot_order is not None and (args.score_mode or args.restore_completed_aim):
        parser.error("--shot-order 只能搭配 --no-score-mode，且不能搭配 --restore-completed-aim")
    if not math.isfinite(args.camera_pitch_deg) or not 10.0 <= args.camera_pitch_deg <= 30.0:
        parser.error("--camera-pitch-deg 必須在 10～30 度之間")
    from .limits import PITCH_MAX_DEG, YAW_DESIGN_LIMIT_DEG
    if not math.isfinite(args.preaim_pitch_deg) or not 0.0 <= args.preaim_pitch_deg <= PITCH_MAX_DEG:
        parser.error(f"--preaim-pitch-deg 必須在 0～{PITCH_MAX_DEG:g} 度之間（0 = 不預瞄）")
    if (not math.isfinite(args.preaim_lower_pitch_deg)
            or not 0.0 <= args.preaim_lower_pitch_deg <= PITCH_MAX_DEG):
        parser.error(f"--preaim-lower-pitch-deg 必須在 0～{PITCH_MAX_DEG:g} 度之間（0 = 不預瞄）")
    if not math.isfinite(args.preaim_yaw_deg) or abs(args.preaim_yaw_deg) > YAW_DESIGN_LIMIT_DEG:
        parser.error(f"--preaim-yaw-deg 必須在 ±{YAW_DESIGN_LIMIT_DEG:g} 度之內")
    if not math.isfinite(args.muzzle_speed) or not 10.0 <= args.muzzle_speed <= 60.0:
        parser.error("--muzzle-speed 必須在 10～60 m/s 之間")
    if not math.isfinite(args.launch_angle_ratio) or not 0.8 <= args.launch_angle_ratio <= 1.1:
        parser.error("--launch-angle-ratio 必須在 0.8～1.1 之間")
    if args.zed_calibration is not None and not args.zed_uvc:
        parser.error("--zed-calibration 必須配合 --zed-uvc 使用")
    if args.restore_completed_aim and (not args.aim_only or args.score_mode):
        parser.error("--restore-completed-aim 需要 --aim-only --no-score-mode")
    if not math.isfinite(args.reload_first_settle) or not 0.0 <= args.reload_first_settle <= 5.0:
        parser.error("--reload-first-settle 必須在 0～5 秒之間")
    if not math.isfinite(args.auto_fire_settle) or not 0.1 <= args.auto_fire_settle <= 5.0:
        parser.error("--auto-fire-settle 必須在 0.1～5 秒之間")
    for flag, value, low, high in (("--push-rpm", args.push_rpm, 20, 600),
                                   ("--push-accel", args.push_accel, 1, 255),
                                   ("--push-home-rpm", args.push_home_rpm, 20, 120)):
        if value is not None and not low <= value <= high:
            parser.error(f"{flag} 必須在 {low}～{high} 之間")
    if not 1 <= args.balls_loaded <= 18:
        parser.error("--balls-loaded 必須在 1～18 之間")
    if not math.isfinite(args.light_wait) or not 0.0 <= args.light_wait <= 55.0:
        parser.error("--light-wait 必須在 0～55 秒之間")
    if not math.isfinite(args.match_duration) or not 30.0 <= args.match_duration <= 300.0:
        parser.error("--match-duration 必須在 30～300 秒之間")
    for flag, value in (("--fire-tolerance-deg", args.fire_tolerance_deg),
                        ("--small-fire-tolerance-deg", args.small_fire_tolerance_deg)):
        if value is not None and (not math.isfinite(value) or not 0.05 <= value <= 2.0):
            parser.error(f"{flag} 必須在 0.05～2.0 度之間")
    if not 1 <= args.stable_frames <= 10:
        parser.error("--stable-frames 必須在 1～10 之間")
    if args.match_fire and not args.score_mode:
        parser.error("--match-fire 使用計分模式，不能搭配 --no-score-mode")
    if args.light_calibrate is not None:
        if not 1 <= args.light_calibrate <= 18:
            parser.error("--light-calibrate 必須在 1～18 顆之間")
        if args.match_fire or not args.score_mode:
            parser.error("--light-calibrate 不能搭配 --match-fire 或 --no-score-mode")
        if not (args.aim_only or args.panel_fire or args.auto_fire):
            args.auto_fire = True
    if args.match_fire or args.panel_fire or args.auto_fire:
        # Shots: the servo settles to ~0.05 deg, so a tight gate costs no time.
        if args.fire_tolerance_deg is None:
            args.fire_tolerance_deg = 0.5
        if args.small_fire_tolerance_deg is None:
            args.small_fire_tolerance_deg = 0.25
    if args.auto_fire or args.match_fire:
        # Same pipeline as --panel-fire; only the click source differs.
        args.panel_fire = True
    if not args.panel_fire and any(
            v is not None for v in (args.push_rpm, args.push_accel, args.push_home_rpm)):
        parser.error("--push-rpm/--push-accel/--push-home-rpm 只用於 --panel-fire、--auto-fire 或 --match-fire")
    if args.panel_fire and (not args.web_port or fire_callback is not None or args.virtual_board):
        parser.error("--panel-fire 需要網頁面板，不能搭配 fire_callback 或 --virtual-board")

    config = with_competition_geometry(TurretConfig()) if not args.virtual_board else TurretConfig()
    if args.zed_calibration is not None:
        try:
            config = load_zed2_uvc_calibration(config, args.zed_calibration)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.zed_uvc and not args.virtual_board:
        print(
            "警告：尚未載入此台 ZED 2 的原廠左眼標定，"
            "目前焦距、主點與畸變是暫值；請指定 --zed-calibration SNxxxx.conf。",
            flush=True,
        )

    if args.zed_calibration is not None:
        print(f"已載入 ZED 2 HD720 左眼原廠標定：{args.zed_calibration}", flush=True)
    elif args.zed_uvc:
        config = with_zed2_hd720_uvc_calibration(config)
        print(
            "相機校正：ZED 2 SN 21947470 HD720 原始 UVC 左圖（含鏡頭畸變）。",
            flush=True,
        )
    if args.virtual_board:
        config = replace(
            config,
            use_axis_geometry=False,
            compensate_parallax=False,
            compensate_gravity=False,
            require_distance_to_fire=False,
        )
        print("虛擬靶模式：只依畫面位置瞄準，已關閉距離／視差／重力補正。", flush=True)
    if args.aim_only or args.panel_fire:
        config = with_recorded_mechanism_geometry(config)
    if not args.virtual_board:
        config = replace(
            config,
            stable_frames_required=args.stable_frames,
            camera_to_yaw_pitch_rad=math.radians(args.camera_pitch_deg),
            compensate_gravity=not args.no_ballistics,
            muzzle_speed_m_s=args.muzzle_speed,
            launch_angle_ratio=args.launch_angle_ratio,
        )
        print(
            f"相機仰角 {args.camera_pitch_deg:g}°；距離來源 {args.range_source}；"
            + ("彈道補償關閉（砲管直線對準）。" if args.no_ballistics else
               f"彈道補償：出球 {args.muzzle_speed:g} m/s、出球角 = {args.launch_angle_ratio:g} × 砲管仰角。"),
            flush=True,
        )
    if args.aim_only:
        print(
            "實機瞄準驗證模式：已套用量測機構幾何，只允許安全範圍內調整砲口，"
            "不會發射。",
            flush=True,
        )
    if not args.virtual_board and (
        config.camera_principal_x_px is None
        or config.camera_principal_y_px is None
    ):
        print(
            "提醒：相機主點仍使用畫面中心暫值；正式自動瞄準前請填入相機標定的 "
            "fx/fy/cx/cy 與畸變參數。",
            flush=True,
        )
    if not args.virtual_board and config.use_axis_geometry:
        if config.yaw_axis_offset == (0.0, 0.0, 0.0) and (
            config.pitch_axis_offset_x_m,
            config.pitch_axis_offset_y_m,
            config.pitch_axis_offset_z_m,
        ) == (0.0, 0.0, 0.0):
            print(
                "提醒：相機到水平／垂直關節的實測偏移仍是 0；"
                "目前可測試流程，但尚未套用機構視差校正。",
                flush=True,
            )
        if not config.mechanism_geometry_verified:
            raise SystemExit(
                "安全停止：機構幾何尚未核准；不會建立實機瞄準流程或使能馬達。"
                "請先填入帶正負號的相機／兩軸座標，並完成實機核對。"
            )
    # Import ultralytics (~4 s on the AGX), load the TensorRT engine (~4 s)
    # and run one warm-up inference in the background while the camera,
    # dashboard, motors and launcher start.  Joined before the control loop.
    # Ultralytics also re-runs a pip install of numpy==1.23.5 on every Jetson
    # TensorRT load (~4 s, and it never takes effect): disable that.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    model_started_at = time.monotonic()
    model_box: Dict[str, Any] = {}

    def _load_model():
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            model_box["error"] = SystemExit("請先安裝 ultralytics：python3 -m pip install ultralytics")
            return
        try:
            loaded = YOLO(str(args.model), task="segment")
        except Exception as exc:
            model_box["error"] = exc
            return
        model_box["model"] = loaded
        try:
            loaded.predict(
                np.zeros((config.image_height_px, config.image_width_px, 3), dtype=np.uint8),
                imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        except Exception as exc:  # the first real frame will raise it properly
            model_box["warmup_error"] = exc

    model_thread = threading.Thread(target=_load_model, name="yolo-load", daemon=True)
    model_thread.start()

    # Prepare the separate muzzle camera while the main camera is starting.
    # If it fails, retry when the firing preview is started.
    muzzle_box: Dict[str, Any] = {"claimed": False}
    muzzle_thread = None
    if (next_handoff_command is not None and args.web_port
            and not (args.zed_uvc or args.opencv_camera)
            and not args.no_muzzle_camera and not args.virtual_board):
        def _open_muzzle_early():
            try:
                from .muzzle_camera import MuzzleZedCamera

                muzzle_box["camera"] = MuzzleZedCamera(config.muzzle_zed_serial)
                print("[Aim] 砲口 ZED 已在預熱階段開啟", flush=True)
            except Exception as exc:
                print(f"[Aim] 砲口 ZED 預熱開啟失敗，稍後再試：{exc}", flush=True)

        def _release_unclaimed_muzzle():
            muzzle_thread.join()
            if not muzzle_box["claimed"] and "camera" in muzzle_box:
                muzzle_box["camera"].release()

        muzzle_thread = threading.Thread(target=_open_muzzle_early, name="muzzle-open", daemon=True)
        muzzle_thread.start()
        startup_cleanup.callback(_release_unclaimed_muzzle)

    def take_early_muzzle_camera():
        if muzzle_thread is None:
            return None
        muzzle_thread.join()
        muzzle_box["claimed"] = True
        return muzzle_box.get("camera")

    motors = None
    launcher = None
    previous_signals = None

    def create_motors():
        return SharedMotorBackend(
            args.port,
            speed_rad_s=args.aim_speed,
            acceleration_rad_s2=args.aim_acceleration,
        )

    def create_launcher(spin_up=True):
        from .panel_launcher import PanelLauncher

        return PanelLauncher(
            motors,
            push_rpm=args.push_rpm,
            push_acceleration=args.push_accel,
            push_home_rpm=args.push_home_rpm,
            spin_up=spin_up,
        )

    # ---- emergency stop at any time after PREPARE_MOTORS -------------------
    # The motor 24 V drops (DM worker trips, MKS drivers reboot, VESC USB
    # drops); the Xavier, the cameras and this process stay up.  The match
    # state is kept; the team pulls the robot back to the start zone, repairs
    # it, and the remote's reload switch (0 -> 1) is the new departure.
    start_repositioning = False  # START arrived during a recovery before START
    motor_users = []  # objects that follow a DM replacement (.motors)

    def _swap_dm_after_power_loss():
        nonlocal motors
        try:
            motors.close()
        except Exception as exc:
            print(f"[E-STOP] 關閉斷電前的雙軸控制：{exc}", flush=True)
        new_motors = create_motors()
        new_motors.enable()  # closes itself if the motors do not answer
        motors = new_motors
        for user in motor_users:
            user.motors = new_motors
        monitor = getattr(launcher.controller, "dm_controller", None)
        if monitor is not None and hasattr(monitor, "motors"):
            monitor.motors = new_motors

    def _power_recovery_try(state):
        """One attempt to bring the mechanism back; True once DM, MKS and VESC answer.

        The magazine and the flywheels stay still (hands are near the
        launcher) until the reload switch confirms the departure.
        """
        controller = launcher.controller
        if controller.busy:
            return False  # the interrupted stage is still timing out
        try:
            if not state["dm"]:
                try:
                    controller.dm_controller.check_health()
                except Exception:
                    _swap_dm_after_power_loss()
                    print("[E-STOP] 偵測到復電：DM 已重新使能（砲口鎖住）。", flush=True)
                state["dm"] = True
            if not state["mks"]:
                controller.reconfigure_mks_after_power_loss()
                state["note"] = launcher.recover_after_power_loss()
                state["mks"] = True
            if not state["vesc"]:
                launcher.reconnect_flywheels_stopped()
                state["vesc"] = True
        except Exception as exc:
            if state["dm"]:  # power is back, a later step failed: say why
                print(f"[E-STOP] 復電後準備未完成（{exc}），{POWER_RETRY_S:g} 秒後重試。", flush=True)
            return False
        print("[E-STOP] 機構已就緒（DM、MKS、VESC），飛輪不轉、推球與彈藥倉不動；"
              "車拉回出發區、修好後，遙控器 RELOAD 撥 0 再撥 1 出發。", flush=True)
        if state["note"]:
            print(f"[E-STOP] 注意：{state['note']}", flush=True)
        return True

    def _enter_power_loss(cause):
        print(f"[E-STOP] 馬達斷電或機構異常（{cause}）：保留比賽進度與剩餘球數，飛輪停止。"
              "復電後自動準備機構；車拉回出發區、修好後，遙控器 RELOAD 撥 0 再撥 1，"
              "就從出發區重新出發，接著射下一孔。", flush=True)
        if reload_event is not None:
            reload_event.clear()  # only a press after the stop counts
        launcher.release_flywheels_after_power_loss()
        if event_callback is not None:
            event_callback("POWER_LOST")  # main.py stops the chassis
        now_s = time.monotonic()
        return {"dm": False, "mks": False, "vesc": False, "note": None, "ready": False,
                "confirmed": False, "next_try": now_s, "next_note": now_s + 5.0}

    def _step_power_loss(state, extra_confirm=False):
        """Non-blocking tick of a recovery; True once ready and confirmed."""
        now_s = time.monotonic()
        # A press before the mechanism is ready is kept, not lost.
        state["confirmed"] = bool(state["confirmed"] or extra_confirm
                                  or _take_external_reload(reload_event, True))
        if not state["ready"] and now_s >= state["next_try"]:
            state["next_try"] = now_s + POWER_RETRY_S
            state["ready"] = _power_recovery_try(state)
            return False
        if state["ready"] and state["confirmed"]:
            return True
        if now_s >= state["next_note"]:
            state["next_note"] = now_s + 5.0
            if state["ready"]:
                print("[E-STOP] 機構就緒，等待 RELOAD（撥 0 再撥 1）…"
                      + (f"（{state['note']}）" if state["note"] else ""), flush=True)
            else:
                print("[E-STOP] 等待復電…" + ("（已收到 RELOAD，復電後立即出發）"
                                             if state["confirmed"] else ""), flush=True)
        return False

    def _resume_mechanism_after_power_loss():
        """Departure after a recovery outside the reload: wheels up, ID1 home."""
        launcher.start_flywheels()
        if not launcher.start_resume_after_power_loss():
            print("[E-STOP] 推球回原點未能開始（機構忙碌或故障）。", flush=True)

    use_zed_sdk = not (args.zed_uvc or args.opencv_camera)
    depth_needed = _stereo_depth_needed(use_zed_sdk, args.range_source, args.zed_depth_compare)
    if use_zed_sdk and not depth_needed:
        print("ZED 立體深度：pnp 模式不需要，每圈不再等待深度圖（--zed-depth-compare 可恢復對照）。",
              flush=True)
    if use_zed_sdk:
        try:
            camera = ZedDepthCamera(config)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        config = camera.config
        print("ZED SDK 深度模式：使用校正後左眼影像及同步雙眼深度。", flush=True)
    else:
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise SystemExit(f"無法開啟 /dev/video{args.camera}")
        requested_width = config.image_width_px * (2 if args.zed_uvc else 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_height_px)
        if not args.virtual_board and config.camera_principal_x_px is None:
            print("提醒：相機內參仍為暫值，正式瞄準前請完成標定。", flush=True)

    startup_cleanup.callback(camera.release)

    def create_vision():
        return BoardVision(model_box["model"], config, confidence=args.conf, imgsz=args.imgsz,
                           device=args.device, virtual_board=args.virtual_board)

    if next_handoff_command is not None:
        import config as drive_config

        previous_signals = _shutdown_on_exit_signals()
        startup_cleanup.callback(_restore_signals, previous_signals)
        print("[Aim] 預熱中（模型背景載入），等待 SWA 放行", flush=True)
        # Commands are served from the start: the release (PREPARE_MOTORS)
        # must not wait for the model.  Positioning frames are processed only
        # once the model is ready.
        model_ready = False
        positioning = True
        positioning_vision = None
        positioning_pose = None
        reference_plan = None
        board_center = np.array([[BOARD_WIDTH_CM / 200.0, BOARD_HEIGHT_CM / 200.0, 0.0]])
        intrinsics = camera_matrix(config)
        distortion = distortion_coefficients(config)

        preaim_phase = None  # "upper" / "lower" from main.py ("PHASE ..."), if known

        def _send_preaim(phase):
            """Turn toward the half's first target while the chassis drives.

            Upper: H11 (top centre); lower: the grid starts at H01 (bottom of
            column 1), so only the elevation is pre-set there (13 deg; H01
            measured 12.7-13.5 deg) and the ~10 deg yaw is left for the stop.
            """
            lower = phase == "lower"
            pitch_deg = args.preaim_lower_pitch_deg if lower else args.preaim_pitch_deg
            if pitch_deg <= 0 or motors is None:
                return
            try:
                motors.send_targets({
                    0x03: (motors.yaw_calibration.motor_from_muzzle(
                        math.radians(args.preaim_yaw_deg)), 0.0, 0.0, 0.0),
                    0x01: (motors.pitch_calibration.motor_from_muzzle(
                        math.radians(pitch_deg)), 0.0, 0.0, 0.0),
                })
                print(f"[Aim] 預瞄（{'下' if lower else '上'}半場）：砲口先轉到 yaw "
                      f"{args.preaim_yaw_deg:+.1f}°、仰角 {pitch_deg:.1f}°"
                      f"（約 {'H01' if lower else 'H11'}）", flush=True)
            except (ValueError, RuntimeError) as exc:
                print(f"[Aim] 預瞄未執行：{exc}", flush=True)

        def reset_positioning():
            nonlocal positioning_vision, positioning_pose, reference_plan
            positioning_vision = create_vision() if model_ready else None
            positioning_pose = BoardPoseTracker(config)
            reference_plan = None

        power_lost = None  # recovery state after an emergency stop before START
        deferred_start = None  # START received during that recovery
        next_health_at = 0.0

        while stop_event is None or not stop_event.is_set():
            if launcher is not None:
                # Emergency stop while the chassis is still on its way.
                now_s = time.monotonic()
                if power_lost is None and now_s >= next_health_at:
                    next_health_at = now_s + 0.1
                    try:
                        launcher.check_health()
                    except Exception as exc:
                        power_lost = _enter_power_loss(exc)
                if power_lost is not None and _step_power_loss(power_lost):
                    power_lost = None
                    _resume_mechanism_after_power_loss()
                    if event_callback is not None:
                        event_callback("REPOSITION")  # main.py: depart again
                    print("[E-STOP] 已收到 RELOAD：從出發區重新出發；飛輪起轉、推球回原點。",
                          flush=True)
                    if deferred_start is not None:
                        # main.py already sent START: shoot once it reports POSITIONED.
                        args.match_phase = deferred_start
                        start_repositioning = True
                        break
            if not model_ready and not model_thread.is_alive():
                model_thread.join()
                if "model" not in model_box:
                    raise model_box.get("error") or RuntimeError("YOLO 模型載入失敗")
                if "warmup_error" in model_box:
                    raise RuntimeError("YOLO 預熱失敗") from model_box["warmup_error"]
                model_ready = True
                print(f"[Aim] PREWARM_READY（{time.monotonic() - model_started_at:.1f} s）", flush=True)
                if event_callback is not None:
                    event_callback("MODEL_READY")  # main.py turns the magazine as a sign
                if positioning:
                    reset_positioning()
            if not model_ready:
                timeout = 0.1  # keep polling the model while serving commands
            elif positioning:
                timeout = 0.005  # a frame per loop; camera.read paces it
            else:
                timeout = 0.2 if stop_event is not None else None
            command = next_handoff_command(timeout)
            if command is None:
                return
            if command == "PREPARE_MOTORS" and launcher is None:
                motors = create_motors()
                # Lambdas: an emergency stop may replace ``motors`` before START.
                startup_cleanup.callback(lambda: motors.close())
                motors.enable()
                # Pre-aim while the chassis is still driving: head for the usual
                # first target (H11, top centre) now, so after the stop only a
                # small correction is left.  Sent before the launcher init so the
                # turret moves while the MKS / VESC start up.
                _send_preaim(preaim_phase)
                # Early exit (Ctrl+C / main.py stopping us before START): the
                # same shutdown as after a match -- flywheels off and pusher
                # home (launcher.close), then the turret home, then disable.
                # ExitStack is LIFO, so register in reverse order.
                startup_cleanup.callback(lambda: _home_turret_and_wait(motors))
                launcher = create_launcher(spin_up=True)
                startup_cleanup.callback(launcher.close, home_pusher=True)
                print("[Aim] DM 已使能，飛輪預轉中，等待定位完成"
                      f"（子程序啟動後 {time.monotonic() - model_started_at:.1f} s）", flush=True)
            elif command in ("PHASE upper", "PHASE lower"):
                # main.py reads the half switch on release, before positioning.
                preaim_phase = command.split()[1]
                if launcher is not None:  # already pre-aimed for the other half
                    _send_preaim(preaim_phase)
            elif command == "VISION_ON" and not positioning:
                positioning = True
                reset_positioning()
            elif command == "VISION_OFF":
                positioning = False
            elif command.startswith("DEPART "):
                # VRA release in main.py: the match clock (time.monotonic()).
                try:
                    args.match_clock_origin = float(command.split()[1])
                except (IndexError, ValueError):
                    print(f"[Aim] 無法解析發車時間：{command!r}", flush=True)
            elif command in ("START upper", "START lower"):
                if launcher is None:
                    raise RuntimeError("未完成 SWA 機構交接，拒絕開始投球")
                if power_lost is not None:
                    deferred_start = command.split()[1]
                    print("[Aim] 斷電復原中收到 START：復電並收到 RELOAD 後再開始。", flush=True)
                    continue
                args.match_phase = command.split()[1]
                break
            elif command == "" and positioning and positioning_vision is not None:
                captured_at = time.monotonic()
                if use_zed_sdk:
                    ok, frame, _ = camera.read(want_depth=False)
                else:
                    ok, frame = camera.read()
                if not ok:
                    continue
                frame = _inference_frame(frame, args.zed_uvc)
                result = positioning_vision.process(frame, now=captured_at)
                plan = result.plan
                if plan is not None and len(plan) == 12 and all(t.hole_id is not None for t in plan.targets):
                    reference_plan = plan
                if reference_plan is None:
                    continue
                pose = positioning_pose.update(result.targets, reference_plan, result.quad, now=captured_at)
                if pose is None or not pose.stable or not positioning_pose.is_fresh(captured_at):
                    continue
                projected, _ = cv2.projectPoints(board_center, pose.rvec, pose.tvec, intrinsics, distortion)
                error_px = float(projected[0, 0, 0] - config.camera_center_x_px)
                if positioning_callback is not None and math.isfinite(error_px):
                    vy = drive_config.AUTO_STRAFE_SIGN * int(np.clip(error_px / 4, -15, 15))
                    positioning_callback(captured_at, vy)
        else:
            return

    # The preview reader starts after positioning so it never competes for
    # frames with the positioning loop and only sees post-START frames.
    dashboard = DashboardState() if args.web_port else None
    if dashboard is not None and args.aim_only and not args.score_mode:
        dashboard.enable_manual_aim()
    if dashboard is not None and args.panel_fire:
        dashboard.enable_panel_fire()
    web_server = None
    grabber = None
    preview = None
    muzzle_camera = None
    muzzle_preview = None

    def _stop_early_services():
        for stop in ((muzzle_preview.stop if muzzle_preview is not None else None),
                     (muzzle_camera.release if muzzle_camera is not None else None),
                     (preview.stop if preview is not None else None),
                     (grabber.stop if grabber is not None else None),
                     (web_server.shutdown if web_server is not None else None),
                     (web_server.server_close if web_server is not None else None)):
            if stop is not None:
                try:
                    stop()
                except Exception as exc:
                    print(f"關閉儀表板／預覽時發生錯誤：{exc}", flush=True)

    startup_cleanup.callback(_stop_early_services)
    if dashboard is not None:
        web_server = start_dashboard_server(dashboard, args.web_port)
        print(f"即時儀表板：http://<robot-ip>:{args.web_port}", flush=True)
        if args.preview_fps > 0:
            # Camera capture and dashboard preview run in background
            # threads; the control loop makes one decision per inference
            # on the newest frame.
            grabber = LatestFrameGrabber(
                camera, has_depth=use_zed_sdk, depth_on_demand=use_zed_sdk)
            preview = LivePreviewPublisher(
                grabber,
                dashboard.update_frame,
                fps=args.preview_fps,
                frame_transform=lambda f: _inference_frame(f, args.zed_uvc),
                scale=args.preview_scale,
                jpeg_quality=args.preview_quality,
            )
            print(f"即時預覽：{args.preview_fps:g} FPS、{args.preview_scale:g} 倍尺寸、"
                  f"JPEG 品質 {args.preview_quality}（辨識結果以最新一次 YOLO 疊圖顯示）。",
                  flush=True)
        # The muzzle camera needs the ZED SDK; UVC/OpenCV runs (and tests
        # with a mocked camera) never touch it.
        if use_zed_sdk and not args.no_muzzle_camera and not args.virtual_board:
            muzzle_camera, muzzle_preview = _start_muzzle_preview(
                config, dashboard, args, camera=take_early_muzzle_camera())

    if args.match_fire:
        print(
            f"比賽全自動模式（{args.match_phase}）：辨識到靶板即開賽並開始射擊"
            + (f"，先射亮燈加分孔 {UPPER_BONUS_BALLS} 顆（{args.light_wait:g} 秒內未確認亮燈先射 H11；"
               "限開賽後 60 秒內），再照蛇形順序射九宮格（每個大孔最多 3 顆），射完 18 顆後等待裝填" if args.match_phase == "upper"
               else "，不射加分孔")
            + f"；已裝 {args.balls_loaded} 顆，時限 {args.match_duration:g} 秒"
            + ("（自 VRA 發車起算）" if getattr(args, "match_clock_origin", None) is not None
               else "（自建立計畫起算）") + "；"
            f"到位容許誤差 大孔 {args.fire_tolerance_deg:g}°／小孔 {args.small_fire_tolerance_deg:g}°。",
            flush=True,
        )
    elif args.light_calibrate is not None:
        print(
            f"紅燈校正模式：辨識到亮紅燈的小孔即鎖定，對該孔連射 {args.light_calibrate} 顆"
            "（燈熄也不換孔、不打九宮格）。",
            flush=True,
        )
    elif args.score_mode:
        print(
            f"特殊得分燈模式（{args.match_phase}）：比賽哨聲時按 G，"
            "程式會鎖定紅燈孔 60 秒。",
            flush=True,
        )
    if motors is None:
        motors = create_motors()
    print(
        f"自動瞄準軌跡速度上限 {args.aim_speed:.3f} rad/s，"
        f"加速度上限 {args.aim_acceleration:.3f} rad/s²。",
        flush=True,
    )
    turret = TurretController(config, motors)
    motor_users.append(turret)
    driver = BoardEngagement(
        config,
        turret,
        shot_cooldown_s=args.cooldown,
        target_timeout_s=(
            None if args.panel_fire or args.target_timeout == 0 else args.target_timeout
        ),
        pitch_trim_rad={hole: math.radians(deg) for hole, deg in pitch_trim_deg.items()},
        fire_tolerance_rad=_fire_tolerance_map(
            args.fire_tolerance_deg, args.small_fire_tolerance_deg),
    )
    if pitch_trim_deg:
        print("仰角微調（度，正值抬高）："
              + "，".join(f"{hole} {deg:+.2f}" for hole, deg in sorted(pitch_trim_deg.items())),
              flush=True)
    pose_tracker = BoardPoseTracker(config)
    pose_lock = None if args.no_pose_lock or args.virtual_board else BoardPoseLock(config)
    pose_locked_previous = False
    muzzle_corrector = None
    muzzle_plan_revision = None
    muzzle_release_seen = None
    vision = None
    plan = None
    board_plan = None
    score_round = ScoreRoundController(
        duration_s=config.score_light_duration_s,
        confirm_frames=config.score_light_confirm_frames,
        min_score=config.score_light_min_score,
        min_margin=config.score_light_min_margin,
        inner_radius_cm=(
            config.score_light_hole_radius_cm
            + config.score_light_edge_gap_cm
        ),
        strip_width_cm=config.score_light_strip_width_cm,
        search_tolerance_cm=config.score_light_search_tolerance_cm,
        min_saturation=config.score_light_min_saturation,
        min_value=config.score_light_min_value,
    )
    if args.match_fire or args.light_calibrate is not None:
        # The remote switches to auto mode at the whistle, so firing starts as
        # soon as the board is seen.  The light is still watched: the upper
        # half inserts its bonus balls once the lit hole is confirmed.
        # Light calibration likewise locks the first confirmed light itself.
        score_round.arm_auto_start()
    strategy_round_id = None
    expiry_processed_round_id = None
    last_bonus_hole_id = None
    fire_requested = False

    def operator_fire(_target) -> bool:
        nonlocal fire_requested
        if not fire_requested:
            return False
        fire_requested = False
        print("F：確認此孔，請在這裡接入實際發射器命令。")
        return True

    external_fire = fire_callback is not None
    driver.fire = fire_callback if external_fire else operator_fire
    fire_label = "AUTO FIRE" if args.auto_fire else "PANEL button only" if args.panel_fire else "AIM ONLY / no fire" if args.aim_only else (
        "external fire callback" if external_fire else "F confirm"
    )
    window = f"REAL BOARD AIM - {fire_label} / S skip / G score start / Q quit"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    shot_log = None
    logged_shots = 0
    last_shot_started_at = None
    wait_track: Dict[str, Any] = {}
    last_latency: Optional[Dict[str, Any]] = None
    manual_confirmed_holes = []
    home_active = False
    home_error = None
    restore_completed = args.restore_completed_aim
    auto_fire = None
    previous_status = None
    score_probe_state = None
    last_render_at = float("-inf")
    # Match firing never times a hole out, so skip holes the turret cannot reach.
    unreachable_skipper = UnreachableHoleSkipper() if args.match_fire else None
    match = (MatchSequencer(
        args.match_phase, args.match_duration, args.balls_loaded,
        lambda new_plan: driver.set_plan(new_plan), light_wait_s=args.light_wait,
        clock_origin=getattr(args, "match_clock_origin", None),
        start_reload_rotation=lambda: bool(launcher is not None and launcher.start_reload_rotation()),
        stop_flywheels=lambda: launcher.stop_flywheels() if launcher is not None else None,
        start_flywheels=lambda: launcher.start_flywheels() if launcher is not None else None,
        on_magazine_empty=lambda: _on_magazine_empty())
        if args.match_fire else None)

    def begin_home():
        nonlocal home_active, home_error
        home_active = True
        try:
            motors.request_home()
            home_error = None
            print("12 孔已確認，依記錄的安全路徑返回 home。", flush=True)
        except (ValueError, RuntimeError) as exc:
            home_error = str(exc)
            print(f"回 home 未執行：{home_error}", flush=True)


    def shutdown_to_home():
        """Stop launching, return through the recorded safe home path, then close."""
        if launcher is not None:
            try:
                launcher.close()
                print("已停止飛輪與發射機構，準備回 home。", flush=True)
            except Exception as exc:
                print(f"停止發射器時發生錯誤：{exc}", flush=True)
        try:
            _home_turret_and_wait(motors)
        finally:
            try:
                turret.close()
            except Exception as exc:
                print(f"關閉雙軸馬達時發生錯誤：{exc}", flush=True)

    last_report = 0.0
    last_target_id = None
    partial_shot_gate = PartialPoseShotGate()
    board_lean_samples = deque(maxlen=150)
    if args.layout:
        board_plan = load_layout(args.layout)
        if not args.score_mode:
            plan = board_plan
            if restore_completed:
                if not _restore_completed_aim_plan(plan, manual_confirmed_holes):
                    raise ValueError("無法還原已完成的 12 孔計畫")
                restore_completed = False
            driver.set_plan(plan)
        print(f"已載入固定靶板版面：{args.layout}", flush=True)
    if args.score_mode:
        # Hold position until the whistle and red-light selection establish a
        # legal competition firing plan.
        plan = EngagementPlan([])
        driver.set_plan(plan)

    if previous_signals is None:
        previous_signals = _shutdown_on_exit_signals()
    # The normal runtime shutdown now owns the prepared hardware.
    startup_cleanup.pop_all()
    try:
        if next_handoff_command is None:
            motors.enable()
        if args.panel_fire:
            if launcher is None:
                launcher = create_launcher()
            else:
                launcher.start_flywheels()
            print(
                "推球馬達 ID1："
                f"推球 {launcher.controller.shooting_speed_rpm} RPM、"
                f"加速度 {launcher.controller.shooting_acceleration}、"
                f"回零 {launcher.controller.home_speed_rpm} RPM。",
                flush=True,
            )
            driver.fire = launcher.fire
            driver.fire_ready = launcher.fire_ready
            driver.fire_released = launcher.shot_released
            shot_log_path = args.shot_log or (
                Path(__file__).resolve().parent.parent / "shot_logs"
                / f"shots_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl")
            shot_log_path.parent.mkdir(parents=True, exist_ok=True)
            ignore_file = shot_log_path.parent / ".gitignore"
            if not ignore_file.exists():
                ignore_file.write_text("*\n", encoding="utf-8")
            shot_log = shot_log_path.open("a", encoding="utf-8")
            print(f"每發記錄：{shot_log_path}", flush=True)
            if args.auto_fire or args.match_fire:
                from .panel_launcher import AutoFireTrigger

                auto_fire = AutoFireTrigger(args.auto_fire_settle)
            if args.match_fire:
                print(f"比賽全自動：雙 VESC speed=100，目標 {launcher.target_rpm:g} RPM；"
                      f"開賽後瞄準到位並穩定 {args.auto_fire_settle:g} 秒即自動發射。"
                      "按 Ctrl+C 停止發射並回 home。", flush=True)
            elif args.auto_fire:
                print(f"全自動射擊模式：雙 VESC speed=100，目標 {launcher.target_rpm:g} RPM；"
                      f"瞄準到位並穩定 {args.auto_fire_settle:g} 秒後自動發射，不需點擊。"
                      "按 Ctrl+C 停止發射並回 home。", flush=True)
            else:
                print(f"面板射擊模式：雙 VESC speed=100，目標 {launcher.target_rpm:g} RPM；"
                      "僅點擊面板「射擊」才會執行 FIRE。", flush=True)
        if args.restore_completed_aim:
            begin_home()
        model_thread.join()
        if "model" not in model_box:
            raise model_box.get("error") or RuntimeError("YOLO 模型載入失敗")
        vision = create_vision()
        if muzzle_preview is not None and args.muzzle_correction != "off" and launcher is not None:
            muzzle_corrector = _make_muzzle_corrector(
                args.muzzle_correction, model_box["model"], config, muzzle_preview, driver, args)
        print(
            f"YOLO 模型載入與預熱完成（啟動後 {time.monotonic() - model_started_at:.1f} s）"
            + (f"；預熱失敗：{model_box['warmup_error']}" if "warmup_error" in model_box else "。"),
            flush=True,
        )
        # Emergency stop during a reload (required by the referees): the motor
        # 24 V drops, the DM worker trips and ends, the MKS drivers reboot.
        # Keep the match state, wait for the operator's reload confirmation,
        # then rebuild the DM control, re-enable the MKS and carry on.
        forced_reload_confirm = False

        def _wait_out_power_loss(cause):
            """Emergency stop (or a mechanism fault) at any time during the match.

            Keep the match state and the balls left.  Once the 24 V is back the
            DM is re-enabled (turret held), the MKS set up again and the VESC
            reconnected with the wheels released; the reload switch (or the
            dashboard's reload button) is the departure from the start zone.
            During the reload wait that confirmation is also the reload.
            Returns True to continue, False on stop.
            """
            state = _enter_power_loss(cause)
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                if dashboard is not None:
                    # Keep the dashboard's reload button accepted (3 s freshness).
                    snapshot = dashboard.telemetry_snapshot()
                    snapshot["generated_at_ms"] = int(time.time() * 1000)
                    snapshot["match"] = match.telemetry(time.monotonic())
                    snapshot["power_lost"] = ("已復電，機構就緒，修好後按 RELOAD 出發" if state["ready"]
                                              else "急停斷電，等待復電")
                    dashboard.update_telemetry(snapshot)
                if _step_power_loss(state, bool(dashboard is not None
                                                and dashboard.take_reload_done())):
                    break
                time.sleep(0.1)
            if match.reload_wait:
                # The confirmation is the reload: match.step turns the
                # magazine (ID1 homes first) and starts the flywheels.
                print("[E-STOP] 已收到裝填完成：出發；彈藥倉轉格（ID1 先回原點）與飛輪起轉"
                      "同時進行。", flush=True)
            else:
                _resume_mechanism_after_power_loss()
                print("[E-STOP] 已收到 RELOAD：從出發區重新出發；飛輪起轉、推球回原點，"
                      f"到定點後接著射 {' '.join(match.remaining_ids()[:1]) or '下一孔'}"
                      f"（剩 {match.balls_remaining} 顆）。", flush=True)
            _begin_reposition(force=True)
            return True

        def _recover_from_fault(exc):
            """Handle a launcher/DM failure: True = recovered, False = stop."""
            if match is None or launcher is None or next_handoff_command is None:
                raise exc
            return _wait_out_power_loss(exc)

        # Reload = robot pulled back to the start zone by hand.  The reload
        # confirmation is the new departure: main.py drives the positioning
        # again (vy from here), the magazine turns and the flywheels spin up
        # meanwhile, and shooting resumes once main.py reports POSITIONED.
        # An emergency stop before START already sent REPOSITION: hold the
        # first shot until main.py reports POSITIONED.
        repositioning = start_repositioning
        reposition_vision = start_repositioning
        reload_was_waiting = False
        reload_estop_ready = False
        # The turret goes home during the reload: an e-stop there leaves the
        # pitch in place (it may drop at a low elevation, the axis has no brake).
        reload_home_pending = False
        reload_home_ok = False
        reload_home_since = 0.0
        RELOAD_HOME_TIMEOUT_S = 10.0

        def _on_magazine_empty():
            """Last ball of the magazine out: turret home + chassis back, now.

            Called by the match before it waits for the flywheels to stop, so
            the turret, the chassis, the pusher homing and the flywheel stop
            all run at the same time.
            """
            nonlocal reload_home_pending, reload_home_ok, reload_home_since
            reload_home_ok = False
            try:
                if not hasattr(motors, "request_home"):
                    raise RuntimeError("此馬達後端不支援回 home")
                motors.request_home()
                reload_home_pending = True
                reload_home_since = time.monotonic()
                print("[Aim] 彈匣射完：砲台同時回 home（急停斷電時仰角不會掉）。", flush=True)
            except (ValueError, RuntimeError) as exc:
                print(f"[Aim] 砲台回 home 未執行（{exc}）：停在原位。", flush=True)
            if event_callback is not None:
                event_callback("RETURN_START")
        board_center_m =np.array([[BOARD_WIDTH_CM / 200.0, BOARD_HEIGHT_CM / 200.0, 0.0]])

        def _begin_reposition(force=False):
            """Reload confirmed = departure from the start zone (once).

            ``force``: an emergency stop during a re-positioning starts it over.
            """
            nonlocal repositioning, reposition_vision, pose_tracker
            if (repositioning and not force) or event_callback is None:
                return
            repositioning = True
            reposition_vision = True
            if pose_lock is not None:
                pose_lock.release("裝填：車已移回出發區")
            pose_tracker = BoardPoseTracker(config)
            event_callback("REPOSITION")
            print("[Aim] 裝填完成：通知底盤重新出發定位；砲口邊走邊瞄準下一孔，"
                  "彈藥倉轉格與飛輪起轉同時進行，到定點前不發射。", flush=True)

        def _drain_reposition_commands():
            nonlocal repositioning, reposition_vision
            while next_handoff_command is not None:
                command = next_handoff_command(0)
                if not command:
                    return
                if command == "VISION_ON":
                    reposition_vision = True
                elif command == "VISION_OFF":
                    reposition_vision = False
                elif command == "POSITIONED" and repositioning:
                    repositioning = False
                    reposition_vision = False
                    if auto_fire is not None:
                        # The turret just came a long way (from home) on a
                        # robot that just stopped: let the first shot settle.
                        auto_fire.settle_next_shot(args.reload_first_settle)
                        print(f"[Aim] 裝填後第一發：瞄準穩定 {args.reload_first_settle:g} 秒才發射。",
                              flush=True)
                    if pose_lock is not None:
                        pose_lock.release("裝填後重新定位完成")
                    print("[Aim] 重新定位完成：重新取得靶面姿態後繼續射擊。", flush=True)

        frame_seq = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                print("收到停止要求：結束自動瞄準，停止發射器並回 home。", flush=True)
                break
            if repositioning:
                _drain_reposition_commands()
            if launcher is not None:
                try:
                    launcher.check_health()
                except RuntimeError as exc:
                    if not _recover_from_fault(exc):
                        break
                    forced_reload_confirm = match.reload_wait
                    continue
            loop_started_at = time.perf_counter()
            capture_started_at = loop_started_at
            if grabber is not None:
                ok, frame, depth_data, frame_seq, frame_ready_at = grabber.wait_next(
                    frame_seq, timeout=1.0, need_depth=depth_needed)
                if not ok:
                    if grabber.failed:
                        break
                    continue  # no frame yet; re-check stop_event and health
            else:
                if use_zed_sdk:
                    ok, frame, depth_data = camera.read(want_depth=depth_needed)
                else:
                    ok, frame = camera.read()
                    depth_data = None
                if not ok:
                    break
                frame_ready_at = time.perf_counter()
            frame_waited_at = time.perf_counter()
            vision_frame = _inference_frame(frame, args.zed_uvc)
            frame_size_ok = (
                vision_frame.shape[1] == config.image_width_px
                and vision_frame.shape[0] == config.image_height_px
            )
            now = time.monotonic()
            inference_started_at = time.perf_counter()
            vision_result = vision.process(vision_frame, now=now, depth_data=depth_data)
            inference_finished_at = time.perf_counter()
            raw_detected = vision_result.raw_targets
            detected = vision_result.targets
            board_surface = vision_result.surface
            trusted_quad = vision_result.quad
            candidate_plan = vision_result.plan
            inferred_plan_used = vision_result.inferred

            if (
                candidate_plan is not None
                and len(candidate_plan) == 12
                and all(t.hole_id is not None for t in candidate_plan.targets)
            ):
                board_plan = candidate_plan
                if not args.score_mode and plan is None:
                    plan = candidate_plan
                    if shot_order is not None:
                        by_id = {t.hole_id: t for t in candidate_plan.targets}
                        plan = EngagementPlan([by_id[hole] for hole in shot_order])
                    if restore_completed:
                        if not _restore_completed_aim_plan(plan, manual_confirmed_holes):
                            plan = None
                            continue
                        restore_completed = False
                    driver.set_plan(plan)
                    plan_source = "推算" if inferred_plan_used else "實測"
                    print(
                        f"{plan_source}靶板計畫建立："
                        + " ".join(t.hole_id or "??" for t in plan.targets),
                        flush=True,
                    )
                    if args.calibrate_layout and not inferred_plan_used:
                        calibrate_layout(detected, config, args.calibrate_layout)
                        print(f"已保存固定靶板版面：{args.calibrate_layout}", flush=True)

            pose = None
            pose_locked = False
            depth_plane = None
            range_compare = {}
            pose_ready = args.virtual_board and frame_size_ok
            pose_blocked_reason = None
            if not frame_size_ok:
                pose_blocked_reason = (
                    "camera frame size does not match calibrated intrinsics"
                )
            elif args.virtual_board:
                pass
            else:
                pose_reference = board_plan or plan
                pose = pose_tracker.update(
                    detected,
                    pose_reference,
                    trusted_quad,
                    now=now,
                )
                if pose is None and args.score_mode:
                    # Bridge a short occlusion using matched visible holes.
                    pose = pose_tracker.update_partial(
                        detected if board_surface is not None else raw_detected,
                        pose_reference,
                        now=now,
                    )
                if pose is not None and pose_reference is not None:
                    live_targets = project_targets_from_pose(
                        pose_reference,
                        pose,
                        config,
                    )
                    pnp_targets = live_targets
                    zed_targets = (prefer_observed_depth(pnp_targets, detected)
                                   if depth_data is not None else pnp_targets)
                    # Both ranges per hole, for the dashboard comparison only.
                    range_compare = {
                        hole_id: (pnp_targets[hole_id].distance_m,
                                  zed_targets[hole_id].distance_m
                                  if zed_targets[hole_id].distance_source == "zed_depth" else None)
                        for hole_id in pnp_targets
                    }
                    if depth_data is not None and args.range_source == "zed":
                        if board_surface is not None:
                            depth_plane = fit_board_depth_plane(
                                depth_data, board_surface.mask, pose, live_targets,
                                detected, config,
                            )
                            if depth_plane is not None:
                                live_targets = apply_board_depth_plane(live_targets, depth_plane, config)
                        if depth_plane is None:
                            live_targets = zed_targets
                    update_plan_geometry(board_plan, live_targets)
                    if plan is not board_plan:
                        update_plan_geometry(plan, live_targets)
                pose_ready = (
                    pose is not None
                    and pose.stable
                    and pose_tracker.is_fresh(now)
                )
                if not pose_ready:
                    pose_blocked_reason = "board pose moving, stale or unstable"
                elif not partial_shot_gate.allows(pose):
                    pose_ready = False
                    pose_blocked_reason = "partial pose shot used; waiting for stable PnP"
                if pose_lock is not None:
                    lock_decision = pose_lock.update(
                        pose, pose_ready, now, getattr(camera, "imu_orientation", None))
                    if not pose_ready and lock_decision.usable:
                        # Vision blocked or settling, chassis unmoved: keep the
                        # geometry from the last stable PnP and stay ready.
                        pose = replace(pose_lock.pose, source="locked")
                        pose_ready = True
                        pose_blocked_reason = None
                    elif not pose_ready and lock_decision.reason:
                        pose_blocked_reason = f"{pose_blocked_reason}; {lock_decision.reason}"
                pose_locked = pose is not None and pose.source == "locked"
                if pose_locked != pose_locked_previous:
                    print("視覺遮蔽：沿用最後穩定靶板姿態繼續瞄準／發射。" if pose_locked
                          else "靶板姿態鎖定結束："
                          + (pose_lock.release_reason or "已恢復即時 PnP。"), flush=True)
                    pose_locked_previous = pose_locked

            if (repositioning and reposition_vision and positioning_callback is not None
                    and pose_ready and pose is not None and pose.source == "pnp"):
                # Same strafe signal as the pre-START positioning loop.
                import config as drive_config

                projected, _ = cv2.projectPoints(
                    board_center_m, pose.rvec, pose.tvec,
                    camera_matrix(config), distortion_coefficients(config))
                error_px = float(projected[0, 0, 0] - config.camera_center_x_px)
                if math.isfinite(error_px):
                    positioning_callback(
                        now, drive_config.AUTO_STRAFE_SIGN * int(np.clip(error_px / 4, -15, 15)))

            # An operator confirmation belongs to the exact stable pose in
            # which it was given.  Never keep it latched across robot motion or
            # a lost visual pose and fire later without a fresh confirmation.
            if not pose_ready:
                fire_requested = False
                driver.set_aiming_enabled(False, pose_blocked_reason)
            else:
                driver.set_aiming_enabled(True)

            if not pose_ready:
                driver.set_firing_enabled(False, pose_blocked_reason)
            elif args.aim_only:
                driver.set_firing_enabled(
                    False,
                    "實機驗證模式：只調整砲口，不發射",
                )
            elif args.panel_fire or external_fire or fire_requested:
                driver.set_firing_enabled(True)
            else:
                # In CLI/manual mode, waiting for F is not a failed launcher
                # attempt and must not consume the target timeout window.
                driver.set_firing_enabled(
                    False,
                    "waiting for operator F confirmation",
                )

            pose_done_at = time.perf_counter()
            if now - last_report >= 2.0:
                pose_report = "virtual"
                if not args.virtual_board:
                    if pose is None:
                        pose_report = "invalid"
                    else:
                        pose_report = (
                            f"{'stable' if pose_ready else 'settling'} "
                            f"n={pose.correspondence_count} "
                            f"rmse={pose.reprojection_error_px:.1f}px"
                        )
                print(
                    f"YOLO 候選孔：靶面內 {len(detected)} / 原始 {len(raw_detected)}；"
                    f"PnP={pose_report}；"
                    f"frame={vision_frame.shape[1]}x{vision_frame.shape[0]}"
                    f"/{config.image_width_px}x{config.image_height_px}",
                    flush=True,
                )
                if launcher is not None and previous_status is not None:
                    # Why no shot yet: every auto-fire gate on one line.
                    command = previous_status.command

                    def _deg(value):
                        return "-" if value is None else f"{math.degrees(value):+.2f}°"

                    print(
                        "發射狀態："
                        f"{auto_fire.state if auto_fire is not None else '手動'}｜"
                        f"孔={previous_status.target.hole_id if previous_status.target else '-'}｜"
                        f"到位={command.fire_allowed}｜"
                        f"伺服誤差 yaw={_deg(command.yaw_servo_error_rad)} "
                        f"pitch={_deg(command.pitch_servo_error_rad)}｜"
                        f"可達={command.target_reachable}｜"
                        f"阻擋={command.blocked_reason or '-'}｜"
                        f"機構忙={launcher.mechanism_busy}｜待發={launcher.pending}｜"
                        f"飛輪到速={launcher.flywheel_ready}｜"
                        f"冷卻完成={driver.fire_cooldown_ready}",
                        flush=True,
                    )
                last_report = now

            reference_plan = board_plan or plan
            if score_probe_state is None or match is None or not match.bonus_closed:
                # ~30 ms per frame; once the match's bonus window is closed (or
                # in the lower half) the light no longer matters.
                score_probe_state = score_round.update(
                    vision_frame,
                    reference_plan,
                    trusted_quad,
                )
            score_done_at = time.perf_counter()
            score_light_scores = (
                score_probe_state.scores
                if reference_plan is not None or trusted_quad is not None
                else None
            )
            score_state = score_probe_state if args.score_mode else None
            if match is not None:
                try:
                    match.step(
                        now, board_plan,
                        score_state.active_hole_id if score_state.phase == "active" else None,
                        launcher_idle=not (launcher is not None and launcher.pending),
                        mechanism_busy=bool(launcher is not None and launcher.mechanism_busy),
                        reload_confirmed=(
                            bool(dashboard is not None and dashboard.take_reload_done())
                            | _take_external_reload(reload_event, match.reload_wait)
                            | forced_reload_confirm),
                    )
                except RuntimeError as exc:
                    # e.g. the flywheel stop before the reload lost the VESC.
                    if not _recover_from_fault(exc):
                        break
                    forced_reload_confirm = match.reload_wait
                    continue
                forced_reload_confirm = False
                if reload_was_waiting and not match.reload_wait and match.stopped is None:
                    _begin_reposition()  # no-op if the e-stop recovery already did it
                if reload_home_pending:
                    home_state, home_message = motors.home_status()
                    if home_state == "arrived":
                        reload_home_pending = False
                        reload_home_ok = True
                        print("[Aim] 砲台已回 home。", flush=True)
                    elif home_state == "error":
                        reload_home_pending = False
                        print(f"[Aim] 砲台回 home 失敗（{home_message}）：停在原位。", flush=True)
                    elif time.monotonic() - reload_home_since > RELOAD_HOME_TIMEOUT_S:
                        reload_home_pending = False
                        print(f"[Aim] 砲台 {RELOAD_HOME_TIMEOUT_S:g} 秒內未確認回到 home。", flush=True)
                if (match.reload_wait and not reload_estop_ready and not reload_home_pending
                        and not (launcher is not None and launcher.mechanism_busy)):
                    # The flywheels already stopped inside match.step and the
                    # pusher / magazine are at rest: only now is an e-stop safe
                    # (a magazine stopped between two slots would be lost).
                    reload_estop_ready = True
                    if event_callback is not None:
                        # main.py now drives the chassis back to the start zone.
                        print("[Aim] 飛輪已停，推球與彈藥倉已停妥，" + ("砲台在 home" if reload_home_ok else "砲台未回 home") + "；底盤自動退回出發區，"
                              "看到 [Return] 已退回出發區 後再按急停裝填，裝填好按遙控器 RELOAD 出發。",
                              flush=True)
                    else:
                        print("[Aim] 可以按急停：飛輪已停，推球與彈藥倉已停妥，" + ("砲台在 home" if reload_home_ok else "砲台未回 home") + "；拉回出發區裝填後，"
                              "按遙控器 RELOAD 出發。", flush=True)
                    if event_callback is not None:
                        # From the reload until this program ends main.py keeps
                        # auto mode (the e-stop reboots the ESP32, which may say M).
                        event_callback("RELOAD_WAIT")
                if not match.reload_wait:
                    reload_estop_ready = False
                    reload_home_pending = False
                reload_was_waiting = match.reload_wait
                plan = match.plan if match.plan is not None else plan
                if match.bonus_hole_id is not None:
                    last_bonus_hole_id = match.bonus_hole_id
                if (match.stopped is not None and not (
                        launcher is not None and (launcher.pending or launcher.mechanism_busy))):
                    # Match over and the mechanism is at rest: leave the loop so
                    # the normal shutdown stops the flywheels, homes the pusher
                    # and the turret, and releases both ZED cameras.
                    print(f"比賽流程結束（{match.stopped}）：停止飛輪、推球回原點、砲台回 home、"
                          "釋放相機。", flush=True)
                    break
            if args.light_calibrate is not None:
                # Calibration: every ball into the first confirmed lit hole.
                # The 60 s light window and grid reallocation do not apply.
                if (
                    score_state.phase == "active"
                    and score_state.active_hole_id is not None
                    and strategy_round_id != score_round.round_id
                    and reference_plan is not None
                    and not (launcher is not None and launcher.pending)
                ):
                    lit = next((t for t in reference_plan.targets
                                if t.hole_id == score_state.active_hole_id), None)
                    if lit is not None:
                        plan = EngagementPlan([lit] * args.light_calibrate)
                        driver.set_plan(plan)
                        strategy_round_id = score_round.round_id
                        last_bonus_hole_id = lit.hole_id
                        print(f"紅燈校正：鎖定 {lit.hole_id}，連射 {args.light_calibrate} 顆。",
                              flush=True)
            elif args.score_mode and not args.match_fire:
                if (
                    score_state.phase == "active"
                    and score_state.active_hole_id is not None
                    and strategy_round_id != score_round.round_id
                    and reference_plan is not None
                    and not (launcher is not None and launcher.pending)
                ):
                    try:
                        strategy = build_max_score_strategy(
                            reference_plan,
                            score_state.active_hole_id,
                            args.match_phase,
                        )
                    except ValueError as exc:
                        print(f"無法建立最大得分射擊計畫：{exc}", flush=True)
                    else:
                        plan = strategy.plan
                        driver.set_plan(plan)
                        strategy_round_id = score_round.round_id
                        expiry_processed_round_id = None
                        last_bonus_hole_id = score_state.active_hole_id
                        print(
                            (f"指定得分孔鎖定：{last_bonus_hole_id}；"
                             if strategy.bonus_balls else
                             f"亮燈 {last_bonus_hole_id} 開賽（不射加分孔）；")
                            + f"{strategy.total_balls} 球計畫，"
                            f"理論最高 {strategy.theoretical_max_points} 分。",
                            flush=True,
                        )
                        print("射擊順序：" + " ".join(strategy.hole_ids), flush=True)

                if (
                    score_state.phase == "expired"
                    and expiry_processed_round_id != score_round.round_id
                    and not (launcher is not None and launcher.pending)
                ):
                    skipped_bonus = 0
                    while (
                        plan is not None
                        and plan.current() is not None
                        and plan.current().hole_id in ("H10", "H11", "H12")
                    ):
                        plan.skip()
                        skipped_bonus += 1
                    if skipped_bonus:
                        # The 60-second LED window has closed.  Keep all
                        # remaining 2-point grid shots, but never send a late
                        # ball to the expired bonus hole.  The skipped balls
                        # are still physically available, so reallocate them.
                        try:
                            replacements = append_reallocated_grid_shots(
                                plan,
                                board_plan,
                                args.match_phase,
                                skipped_bonus,
                            )
                        except ValueError as exc:
                            replacements = ()
                            print(f"加分球重新配置失敗：{exc}", flush=True)
                        driver.set_plan(plan)
                        print(
                            f"指定得分燈已熄滅，略過 {skipped_bonus} 顆未完成加分球，"
                            "改投九宮格："
                            + (" ".join(replacements) if replacements else "無可用格"),
                            flush=True,
                        )
                    expiry_processed_round_id = score_round.round_id

            if dashboard is not None and args.aim_only and not args.score_mode:
                action = dashboard.take_manual_action()
                home_state, _ = motors.home_status() if home_active and home_error is None else ("error" if home_error else "idle", home_error)
                if home_active and home_state != "arrived":
                    action = None
                plan, applied = _apply_manual_aim_action(plan, action, manual_confirmed_holes)
                if applied:
                    driver.set_plan(plan)
                    print("僅瞄準模式：已更新人工確認順序，不發射。", flush=True)
                    if action[0] == "reset":
                        home_active = False
                        home_error = None
                    elif _all_holes_confirmed(plan, manual_confirmed_holes):
                        begin_home()
            muzzle_hold = None
            if muzzle_corrector is not None:
                if driver.plan_revision != muzzle_plan_revision or (
                        pose_lock is not None and pose_lock.release_reason
                        and pose_lock.release_reason != muzzle_release_seen):
                    # New plan or the chassis moved: per-hole corrections are stale.
                    muzzle_corrector.reset()
                    muzzle_plan_revision = driver.plan_revision
                    muzzle_release_seen = pose_lock.release_reason if pose_lock is not None else None
                muzzle_hold = _step_muzzle_corrector(
                    muzzle_corrector, muzzle_preview, driver, plan, previous_status,
                    pitch_trim_deg, time.monotonic())
            if launcher is not None:
                try:
                    status = _update_panel_engagement(
                        driver, launcher, dashboard, plan, pose_ready, pose_blocked_reason,
                        auto_fire=auto_fire, previous_status=previous_status,
                        # While driving back to the shooting spot the turret
                        # already tracks the next hole; only firing waits.
                        fire_hold=(muzzle_hold or (
                            "裝填後重新定位中，到定點前不發射" if repositioning else None)),
                        hold_reason=match.hold_reason if match is not None else None)
                except RuntimeError as exc:
                    if not _recover_from_fault(exc):
                        break
                    forced_reload_confirm = match.reload_wait
                    continue
            else:
                status = driver.update()
            if unreachable_skipper is not None and repositioning:
                # Seen from the start zone a hole may be out of reach for a
                # while; never skip it on the way.  None resets the timer.
                unreachable_skipper.update(None, None, False, False)
            elif unreachable_skipper is not None and launcher is not None:
                skipped = unreachable_skipper.update(
                    driver.plan, status, pose_ready,
                    not launcher.pending and not launcher.mechanism_busy)
                if skipped is not None:
                    hole, count = skipped
                    upcoming = driver.plan.current() if driver.plan is not None else None
                    reason = status.command.blocked_reason or "unreachable"
                    print(f"{hole} 連續 {UNREACHABLE_SKIP_S:g} 秒打不到（{reason}）：跳過 {count} 發，"
                          f"改射 {upcoming.hole_id if upcoming is not None else '（無剩餘孔位）'}。",
                          flush=True)
            if launcher is not None:
                # Where the time between "ball released" and "next push" goes.
                released_at = launcher.last_released_at
                if released_at is not None and wait_track.get("released_at") != released_at:
                    wait_track = {"released_at": released_at, "frames": 0, "pose_unready": 0,
                                  "allowed_at": None, "click_at": None}
                if released_at is not None and not launcher.pending:
                    tick_at = time.monotonic()
                    wait_track["frames"] += 1
                    if not pose_ready:
                        wait_track["pose_unready"] += 1
                    if status.command.fire_allowed and wait_track["allowed_at"] is None:
                        wait_track["allowed_at"] = tick_at
            if (launcher is not None and shot_log is not None
                    and launcher.shots_started != logged_shots and launcher.last_shot):
                logged_shots = launcher.shots_started
                shot = launcher.last_shot
                hole = shot["hole_id"] or ""
                record = _shot_record(
                    shot, logged_shots, last_shot_started_at, status.command, config,
                    pitch_trim_deg=pitch_trim_deg.get(hole, 0.0),
                    range_pair=range_compare.get(hole),
                    lean_deg=board_lean_deg(pose, config.camera_to_yaw_pitch_rad)
                    if pose is not None and pose.source == "pnp" else None,
                )
                record["pose_locked"] = pose_locked
                muzzle_record = muzzle_corrector.last if muzzle_corrector is not None else None
                if muzzle_record is not None and muzzle_record.hole_id == hole:
                    record.update(muzzle_record.as_log())
                if last_latency:
                    record["vision_ms"] = {key: round(value, 1) for key, value in last_latency.items()
                                           if key.endswith("_ms") and isinstance(value, (int, float))}
                if wait_track.get("released_at") is not None:
                    span = shot["started_at"] - wait_track["released_at"]
                    record["wait_frames"] = wait_track["frames"]
                    record["wait_pose_unready_frames"] = wait_track["pose_unready"]
                    record["wait_until_fire_allowed_s"] = (
                        round(wait_track["allowed_at"] - wait_track["released_at"], 2)
                        if wait_track["allowed_at"] is not None else None)
                    record["vision_hz_during_wait"] = (
                        round(wait_track["frames"] / span, 1) if span > 0 else None)
                last_shot_started_at = shot["started_at"]
                shot_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                shot_log.flush()
                print(
                    f"[SHOT {logged_shots}] {record['hole_id']} 彈倉第 {record['slot']} 格第 "
                    f"{record['stage']} 顆；飛輪 L/R {record['rpm_left']}/{record['rpm_right']} RPM；"
                    f"仰角 目標 {record['muzzle_pitch_target_deg']}° 實際 {record['muzzle_pitch_current_deg']}°"
                    + (f"；砲口相機 {record['muzzle_verdict']} 誤差 yaw "
                       f"{record['muzzle_error_yaw_deg']}° pitch {record['muzzle_error_pitch_deg']}°"
                       if "muzzle_verdict" in record else ""),
                    flush=True,
                )
            previous_status = status
            if match is not None and status.fired:
                match.record_fired()
            partial_shot_gate.record(pose, status.fired)
            control_done_at = time.perf_counter()
            yaw_motor_state = motors.get_state(config.yaw_motor_id)
            pitch_motor_state = motors.get_state(config.pitch_motor_id)
            if (
                args.score_mode
                and args.light_calibrate is None
                and status.skipped
                and status.target is not None
                and status.target.hole_id in ("H10", "H11", "H12")
                and plan is not None
                and board_plan is not None
            ):
                try:
                    replacements = append_reallocated_grid_shots(
                        plan,
                        board_plan,
                        args.match_phase,
                        1,
                    )
                except ValueError as exc:
                    replacements = ()
                    print(f"逾時加分球重新配置失敗：{exc}", flush=True)
                if replacements:
                    print(
                        f"{status.target.hole_id} 逾時未發射，保留該球並改投 "
                        f"{replacements[0]}。",
                        flush=True,
                    )
            target_id = status.target.hole_id if status.target is not None else None
            if target_id != last_target_id:
                if status.target is None:
                    print("瞄準流程完成或尚未建立計畫。", flush=True)
                else:
                    yaw_text = (
                        f"{status.command.target_yaw_rad:+.3f} rad"
                        if status.command.target_yaw_rad is not None
                        else "—"
                    )
                    pitch_text = (
                        f"{status.command.target_pitch_rad:+.3f} rad"
                        if status.command.target_pitch_rad is not None
                        else "—"
                    )
                    print(
                        f"目前瞄準 {target_id}："
                        f"yaw={yaw_text}，"
                        f"pitch={pitch_text}，"
                        + (
                            f"水平軸距離={status.command.yaw_axis_distance_m:.2f} m，"
                            if status.command.yaw_axis_distance_m is not None
                            else ""
                        )
                        + f"remaining={status.remaining}",
                        flush=True,
                    )
                last_target_id = target_id
            render_due = (not args.headless
                          or (dashboard is not None
                              and time.monotonic() - last_render_at >= PREVIEW_RENDER_S))
            display = None
            if render_due:
                last_render_at = time.monotonic()
                display = vision_result.render(vision_frame)
                if score_light_scores is not None:
                    draw_score_light_overlay(
                        display,
                        score_light_scores,
                        reference_plan,
                        trusted_quad,
                        active_hole_id=(
                            score_state.active_hole_id
                            if score_state is not None
                            else None
                        ),
                        min_score=config.score_light_min_score,
                    )
                if args.score_mode and status.target is not None:
                    current_point = (
                        round(status.target.u_px),
                        round(status.target.v_px),
                    )
                    cv2.circle(display, current_point, 20, (255, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(
                        display,
                        f"NEXT {status.target.hole_id}",
                        (current_point[0] + 10, current_point[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if args.score_mode and score_state is not None:
                    score_label = (
                        score_state.active_hole_id
                        or last_bonus_hole_id
                        or "未鎖定"
                    )
                    text = (
                        f"SCORE {score_state.phase} {score_label} "
                        f"{score_state.remaining_s:.1f}s | remaining={status.remaining}"
                    )
                else:
                    text = f"12-hole {'READY' if plan else 'WAIT'} | remaining={status.remaining}"
                cv2.putText(display, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
                if score_light_scores is not None:
                    score_text = " ".join(
                        f"{hole_id}:{score_light_scores.get(hole_id, 0.0):.2f}"
                        for hole_id in ("H10", "H11", "H12")
                    )
                    cv2.putText(
                        display,
                        f"RED {score_text}",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
                if not frame_size_ok:
                    pose_text = "CAMERA SIZE MISMATCH - FIRE BLOCKED"
                    pose_color = (0, 0, 255)
                elif args.virtual_board:
                    pose_text = "POSE virtual bypass"
                    pose_color = (0, 220, 0)
                elif pose is None:
                    pose_text = "POSE invalid - FIRE BLOCKED"
                    pose_color = (0, 165, 255)
                else:
                    pose_text = (
                        f"POSE {pose.source.upper()} {'READY' if pose_ready else 'SETTLING'} "
                        f"n={pose.correspondence_count} "
                        f"err={pose.reprojection_error_px:.1f}px "
                        f"v={pose.translation_speed_m_s:.2f}m/s "
                        f"w={pose.rotation_speed_rad_s:.2f}rad/s"
                    )
                    pose_color = (0, 220, 0) if pose_ready else (0, 165, 255)
                cv2.putText(
                    display,
                    pose_text,
                    (12, 82 if args.score_mode or score_light_scores is not None else 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    pose_color,
                    2,
                    cv2.LINE_AA,
                )
            render_done_at = time.perf_counter()
            jpeg_encode_ms = None
            preview_stats = {}
            if muzzle_preview is not None:
                # Parallax of the barrel point depends on the target range.
                muzzle_preview.set_distance(
                    status.target.distance_m if status.target is not None else None)
            if preview is not None and display is not None:
                preview.set_overlay(display, vision_frame)
                preview_stats = preview.stats()
                jpeg_encode_ms = preview_stats.get("preview_jpeg_encode_ms")
            elif dashboard is not None and display is not None:
                encode_started_at = time.perf_counter()
                ok_jpg, encoded = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                jpeg_encode_ms = (time.perf_counter() - encode_started_at) * 1000.0
                if ok_jpg:
                    dashboard.update_frame(encoded.tobytes())
            latency = {
                "capture_ms": (frame_waited_at - capture_started_at) * 1000.0,
                "inference_ms": (inference_finished_at - inference_started_at) * 1000.0,
                "yolo_ms": _yolo_ms(vision_result),
                "pose_ms": (pose_done_at - inference_finished_at) * 1000.0,
                "score_probe_ms": (score_done_at - pose_done_at) * 1000.0,
                "control_ms": (control_done_at - score_done_at) * 1000.0,
                "render_ms": (render_done_at - control_done_at) * 1000.0,
                "frame_age_ms": (time.perf_counter() - frame_ready_at) * 1000.0,
                "loop_ms": (time.perf_counter() - loop_started_at) * 1000.0,
                "jpeg_encode_ms": jpeg_encode_ms,
                "preview_fps": preview_stats.get("preview_fps"),
                "vision_hz": 1.0 / max(time.perf_counter() - loop_started_at, 1e-6),
            }
            last_latency = latency
            if dashboard is not None or telemetry_callback is not None:
                telemetry = build_dashboard_snapshot(
                    plan=plan,
                    board_plan=board_plan,
                    engagement_status=status,
                    config=config,
                    pose=pose,
                    pose_ready=pose_ready,
                    depth_plane=depth_plane,
                    virtual_board=args.virtual_board,
                    frame_size_ok=frame_size_ok,
                    raw_detection_count=len(raw_detected),
                    board_detection_count=len(detected),
                    board_surface_found=board_surface is not None,
                    score_mode=args.score_mode,
                    match_phase=args.match_phase,
                    score_state=score_state,
                    score_light_scores=score_light_scores,
                    latency=latency,
                    bonus_hole_id=last_bonus_hole_id,
                    operator_fire_pending=(launcher.pending if launcher is not None else fire_requested),
                    aim_only=args.aim_only,
                    confirmed_holes=manual_confirmed_holes,
                    current_yaw_motor_rad=(
                        yaw_motor_state.position_rad
                        if yaw_motor_state is not None
                        else None
                    ),
                    current_pitch_motor_rad=(
                        pitch_motor_state.position_rad
                        if pitch_motor_state is not None
                        else None
                    ),
                )
                if launcher is not None:
                    current = plan.current() if plan is not None else None
                    telemetry["system"]["panel_fire"] = True
                    telemetry["system"]["auto_fire"] = auto_fire is not None
                    if args.match_fire:
                        telemetry["match"] = match.telemetry(
                            now, score_state.phase if score_state is not None else None)
                    telemetry["launcher"] = launcher.snapshot()
                    telemetry["panel_control"] = {
                        "enabled": True,
                        "ready": bool(current is not None and pose_ready and not launcher.pending
                                      and not launcher.mechanism_busy and launcher.flywheel_ready
                                      and not status.fired and status.command.fire_allowed
                                      and driver.fire_cooldown_ready),
                        "plan_index": plan.index if plan is not None else None,
                        "plan_revision": driver.plan_revision,
                        "hole_id": getattr(current, "hole_id", None),
                        "auto_fire": auto_fire.snapshot() if auto_fire is not None else None,
                        "pitch_trim_deg": pitch_trim_deg.get(getattr(current, "hole_id", None) or "", 0.0),
                        "nonce": dashboard.fire_nonce,
                    }
                if args.aim_only and not args.score_mode:
                    home_state, home_message = motors.home_status() if home_active and home_error is None else ("error" if home_error else "idle", home_error)
                    worker_state = motors.worker.snapshot() if home_active else None
                    telemetry["home"] = {"state": home_state, "message": home_message,
                                         "phase": getattr(worker_state, "yaw_sequence_phase", None),
                                         "motor_status": getattr(worker_state, "status", None)}
                    telemetry["manual_control"] = {
                        "enabled": (plan is not None and len(plan) > 0
                                    and (not home_active or home_state == "arrived")),
                        "mode": "aim_only",
                        "current_hole_id": getattr(status.target, "hole_id", None),
                        "plan_index": plan.index if plan is not None else None,
                    }
                target_telemetry = telemetry.get("target")
                if isinstance(target_telemetry, dict):
                    pair = range_compare.get(target_telemetry.get("hole_id") or "")
                    target_telemetry["pnp_distance_m"] = (
                        round(pair[0], 3) if pair and pair[0] is not None else None)
                    target_telemetry["zed_distance_m"] = (
                        round(pair[1], 3) if pair and pair[1] is not None else None)
                lean = None
                if pose is not None and pose.source == "pnp" and pose_ready:
                    lean = board_lean_deg(pose, config.camera_to_yaw_pitch_rad)
                    if lean is not None:
                        board_lean_samples.append(lean)
                telemetry.setdefault("pose", {})["board_lean_deg"] = (
                    round(lean, 2) if lean is not None else None)
                telemetry["pose"]["board_lean_avg_deg"] = (
                    round(float(np.mean(board_lean_samples)), 2)
                    if len(board_lean_samples) >= 30 else None)
                telemetry["pose"]["board_lean_samples"] = len(board_lean_samples)
                if telemetry_callback is not None:
                    telemetry_callback(telemetry)
                if dashboard is not None:
                    dashboard.update_telemetry(telemetry)
            key = -1
            if not args.headless:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
            if launcher is not None and launcher.pending and key in (ord("g"), ord("s")):
                print("發射階段尚未完成，忽略切換孔位／計畫操作。", flush=True)
                key = -1
            if key == ord("f"):
                if args.panel_fire:
                    print("此模式僅接受面板「射擊」按鈕，F 不會發射。", flush=True)
                elif args.aim_only:
                    fire_requested = False
                    print("實機驗證模式已停用發射。", flush=True)
                elif external_fire:
                    print(
                        "目前由外部 fire_callback 控制發射，F 不會另送一次命令。",
                        flush=True,
                    )
                elif pose_ready:
                    fire_requested = True
                else:
                    fire_requested = False
                    print("姿態尚未穩定，已拒絕 F 發射確認。", flush=True)
            elif key == ord("g") and args.score_mode and not args.match_fire:
                score_round.start_round()
                strategy_round_id = None
                expiry_processed_round_id = None
                last_bonus_hole_id = None
                plan = EngagementPlan([])
                driver.set_plan(plan)
                fire_requested = False
                print("已開始特殊得分回合，等待三個紅燈中的亮燈孔。", flush=True)
            elif key == ord("s") and plan is not None:
                plan.skip()
            elif key in (ord("q"), 27):
                break
    finally:
        # Run every cleanup even if one device fails to close. Stop new panel
        # requests and unwind the launcher before disabling the aiming worker.
        with ExitStack() as cleanup:
            # LIFO: registered first so the handlers are restored only after
            # every device has been shut down.
            cleanup.callback(_restore_signals, previous_signals)
            if web_server is not None:
                cleanup.callback(web_server.server_close)
                cleanup.callback(web_server.shutdown)
            if not args.headless:
                cleanup.callback(cv2.destroyAllWindows)
            cleanup.callback(camera.release)
            # Stop background readers before the camera is released (LIFO).
            if grabber is not None:
                cleanup.callback(grabber.stop)
            if preview is not None:
                cleanup.callback(preview.stop)
            if muzzle_camera is not None:
                cleanup.callback(muzzle_camera.release)
            if muzzle_preview is not None:
                cleanup.callback(muzzle_preview.stop)
            cleanup.callback(driver.close)
            cleanup.callback(shutdown_to_home)
            if shot_log is not None:
                cleanup.callback(shot_log.close)
            if dashboard is not None:
                cleanup.callback(dashboard.stop)


if __name__ == "__main__":
    main()

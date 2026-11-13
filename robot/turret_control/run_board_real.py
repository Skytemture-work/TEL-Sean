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
    BoardPoseTracker,
    board_lean_deg,
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
from .live_preview import LatestFrameGrabber, LivePreviewPublisher
from .scoring_light import ScoreRoundController, draw_score_light_overlay
from .scoring_strategy import (
    MATCH_PHASES,
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


# Ballistic model measured on 2026-11-13 (board 1:1, pitch axis 5.49 m from the
# board and 31.5 cm above the floor, flywheels speed=100 / 4000 RPM):
# * the PnP board lean averaged +0.81 deg on a vertical board, so the real
#   camera mount pitch is ~20 - 0.8 = 19.2 deg (CAD value 20 deg);
# * the four row misses (8/11/15/20 cm low) at the inclinometer-verified barrel
#   angles fit a drag-free arc of 32 m/s launched at 0.942 x barrel elevation
#   within +-0.7 cm.  Re-fit after changing the flywheel speed.
DEFAULT_CAMERA_PITCH_DEG = 19.2
DEFAULT_MUZZLE_SPEED_M_S = 32.0
DEFAULT_LAUNCH_ANGLE_RATIO = 0.942


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


def _auto_fire_ready(driver, launcher, plan, pose_ready, previous_status):
    """True when the previous tick already had a fire-allowed aim on this hole."""
    current = plan.current() if plan is not None else None
    return bool(
        current is not None
        and pose_ready
        and previous_status is not None
        and not previous_status.fired
        and previous_status.target is not None
        and previous_status.target.hole_id == current.hole_id
        and previous_status.command.fire_allowed
        and not launcher.pending
        and not launcher.mechanism_busy
        and launcher.flywheel_ready
        and driver.fire_cooldown_ready
    )


def _update_panel_engagement(driver, launcher, dashboard, plan, pose_ready, blocked_reason,
                             auto_fire=None, previous_status=None):
    """Validate a click on this frame; never retain an unexecuted authorization.

    With ``auto_fire`` the click comes from ``AutoFireTrigger`` instead of the
    web button (web clicks are consumed and ignored); the click is then checked
    by exactly the same launcher and engagement gates.
    """
    launcher.check_health()
    if launcher.busy and not pose_ready:
        raise RuntimeError("發射期間靶板姿態失效，停止發射控制")
    driver.set_aiming_enabled(
        pose_ready and not launcher.busy,
        "發射階段執行中，砲口保持位置" if launcher.busy else blocked_reason,
    )
    driver.set_firing_enabled(pose_ready, blocked_reason)
    action = dashboard.take_panel_fire()
    if auto_fire is not None:
        action = auto_fire.take(
            plan, driver.plan_revision,
            _auto_fire_ready(driver, launcher, plan, pose_ready, previous_status),
        )
    launcher.arm(action, plan, driver.plan_revision)
    try:
        return driver.update()
    finally:
        launcher.finish_tick()


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    fire_callback: Optional[FireCallback] = None,
    telemetry_callback: Optional[TelemetryCallback] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the real board pipeline.

    ``fire_callback`` is an opt-in automatic launcher hook.  It is called only
    after the vision, pose, reachability and motor-settling gates all pass.  It
    must return True only after the requested ball was actually sent; False
    keeps the same target active.

    ``stop_event`` lets a host program (e.g. ``main.py`` via
    ``auto_session.AutoAimSession``) end the run: once set, the loop exits and
    the normal shutdown runs -- stop the launcher, return along the recorded
    safe path to home, disable the motors, release the camera.

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
        "--balls-loaded", type=int, default=18,
        help="比賽模式：彈藥倉已裝球數（預設 18＝6 格 × 3 顆；範圍 1～18）",
    )
    parser.add_argument(
        "--match-duration", type=float, default=180.0,
        help="比賽模式：從開賽起算的發射時限秒數（預設 180；範圍 30～300）",
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
        "--auto-fire-settle", type=float, default=0.3,
        help="全自動模式：同一孔位需連續符合發射條件的秒數（預設 0.3，範圍 0.2～5）",
    )
    parser.add_argument(
        "--stable-frames", type=int, default=3,
        help="砲口需連續幾個辨識畫面都在容許誤差內才算到位（預設 3，範圍 1～10；原本 5）",
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
             "--pitch-trim H10=0.8 --pitch-trim H11=0.8；微調後仍經過限位與工作空間檢查",
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
        "--range-source", choices=("pnp", "zed"), default="pnp",
        help="孔的距離來源：pnp = 整片靶的 PnP 位置（預設）；zed = ZED 深度（舊行為，孔內深度會偏遠）",
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
    try:
        shot_order = _parse_shot_order(args.shot_order)
    except ValueError as exc:
        parser.error(str(exc))
    if shot_order is not None and (args.score_mode or args.restore_completed_aim):
        parser.error("--shot-order 只能搭配 --no-score-mode，且不能搭配 --restore-completed-aim")
    if not math.isfinite(args.camera_pitch_deg) or not 10.0 <= args.camera_pitch_deg <= 30.0:
        parser.error("--camera-pitch-deg 必須在 10～30 度之間")
    if not math.isfinite(args.muzzle_speed) or not 10.0 <= args.muzzle_speed <= 60.0:
        parser.error("--muzzle-speed 必須在 10～60 m/s 之間")
    if not math.isfinite(args.launch_angle_ratio) or not 0.8 <= args.launch_angle_ratio <= 1.1:
        parser.error("--launch-angle-ratio 必須在 0.8～1.1 之間")
    if args.zed_calibration is not None and not args.zed_uvc:
        parser.error("--zed-calibration 必須配合 --zed-uvc 使用")
    if args.restore_completed_aim and (not args.aim_only or args.score_mode):
        parser.error("--restore-completed-aim 需要 --aim-only --no-score-mode")
    if not math.isfinite(args.auto_fire_settle) or not 0.2 <= args.auto_fire_settle <= 5.0:
        parser.error("--auto-fire-settle 必須在 0.2～5 秒之間")
    for flag, value, low, high in (("--push-rpm", args.push_rpm, 20, 600),
                                   ("--push-accel", args.push_accel, 1, 255),
                                   ("--push-home-rpm", args.push_home_rpm, 20, 120)):
        if value is not None and not low <= value <= high:
            parser.error(f"{flag} 必須在 {low}～{high} 之間")
    if not 1 <= args.balls_loaded <= 18:
        parser.error("--balls-loaded 必須在 1～18 之間")
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
    if args.match_fire:
        print(
            f"比賽全自動模式（{args.match_phase}）：保持不動等待指定得分格亮燈即開賽；"
            f"已裝 {args.balls_loaded} 顆，時限 {args.match_duration:g} 秒；"
            f"到位容許誤差 大孔 {args.fire_tolerance_deg:g}°／小孔 {args.small_fire_tolerance_deg:g}°。"
            "無法偵測燈號時，可在 OpenCV 視窗（不加 --headless）按 G 手動開賽。",
            flush=True,
        )
    elif args.score_mode:
        print(
            f"特殊得分燈模式（{args.match_phase}）：比賽哨聲時按 G，"
            "程式會鎖定紅燈孔 60 秒。",
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
    use_zed_sdk = not (args.zed_uvc or args.opencv_camera)
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

    # Dashboard and live camera preview come up right after the camera, so
    # the operator sees the image while the model and motors are starting.
    dashboard = DashboardState() if args.web_port else None
    if dashboard is not None and args.aim_only and not args.score_mode:
        dashboard.enable_manual_aim()
    if dashboard is not None and args.panel_fire:
        dashboard.enable_panel_fire()
    web_server = None
    grabber = None
    preview = None

    def _stop_early_services():
        for stop in ((preview.stop if preview is not None else None),
                     (grabber.stop if grabber is not None else None),
                     (web_server.shutdown if web_server is not None else None),
                     (web_server.server_close if web_server is not None else None)):
            if stop is not None:
                try:
                    stop()
                except Exception as exc:
                    print(f"關閉儀表板／預覽時發生錯誤：{exc}", flush=True)

    try:
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
    except BaseException:
        _stop_early_services()
        camera.release()
        raise

    try:
        motors = SharedMotorBackend(
            args.port,
            speed_rad_s=args.aim_speed,
            acceleration_rad_s2=args.aim_acceleration,
        )
        print(
            f"自動瞄準軌跡速度上限 {args.aim_speed:.3f} rad/s，"
            f"加速度上限 {args.aim_acceleration:.3f} rad/s²。",
            flush=True,
        )
    except BaseException:
        _stop_early_services()
        camera.release()
        raise
    turret = TurretController(config, motors)
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
    if args.match_fire:
        # Stay still until the scoring light (= whistle) is confirmed.
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
    launcher = None
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
    match_started_at = None
    match_stop_reason = None
    balls_remaining = args.balls_loaded

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
            deadline = time.monotonic() + 20.0
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

    try:
        motors.enable()
        if args.panel_fire:
            from .panel_launcher import PanelLauncher

            launcher = PanelLauncher(
                motors,
                push_rpm=args.push_rpm,
                push_acceleration=args.push_accel,
                push_home_rpm=args.push_home_rpm,
            )
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
        vision = BoardVision(model_box["model"], config, confidence=args.conf, imgsz=args.imgsz,
                             device=args.device, virtual_board=args.virtual_board)
        print(
            f"YOLO 模型載入與預熱完成（啟動後 {time.monotonic() - model_started_at:.1f} s）"
            + (f"；預熱失敗：{model_box['warmup_error']}" if "warmup_error" in model_box else "。"),
            flush=True,
        )
        frame_seq = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                print("收到停止要求：結束自動瞄準，停止發射器並回 home。", flush=True)
                break
            if launcher is not None:
                launcher.check_health()
            loop_started_at = time.perf_counter()
            capture_started_at = loop_started_at
            if grabber is not None:
                ok, frame, depth_data, frame_seq, frame_ready_at = grabber.wait_next(
                    frame_seq, timeout=1.0, need_depth=use_zed_sdk)
                if not ok:
                    if grabber.failed:
                        break
                    continue  # no frame yet; re-check stop_event and health
            else:
                if use_zed_sdk:
                    ok, frame, depth_data = camera.read()
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
                last_report = now

            reference_plan = board_plan or plan
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
            if args.match_fire:
                if match_started_at is None and score_state.started_at is not None:
                    match_started_at = score_state.started_at
                    print(f"開賽：{score_state.active_hole_id or '手動開始'}；"
                          f"時限 {args.match_duration:g} 秒。", flush=True)
                stop_reason = None
                if match_started_at is not None and now - match_started_at >= args.match_duration:
                    stop_reason = "比賽時間到，停止發射"
                elif balls_remaining <= 0:
                    stop_reason = "已裝彈藥全部射出，停止發射"
                if (stop_reason is not None and match_stop_reason is None
                        and not (launcher is not None and launcher.pending)):
                    match_stop_reason = stop_reason
                    plan = EngagementPlan([])
                    driver.set_plan(plan)
                    print(f"{stop_reason}（剩餘 {balls_remaining} 顆）。", flush=True)
            if args.score_mode and match_stop_reason is None:
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
                            f"指定得分孔鎖定：{last_bonus_hole_id}；"
                            f"{strategy.total_balls} 球最佳計畫，"
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
            if launcher is not None:
                status = _update_panel_engagement(
                    driver, launcher, dashboard, plan, pose_ready, pose_blocked_reason,
                    auto_fire=auto_fire, previous_status=previous_status)
            else:
                status = driver.update()
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
                    f"仰角 目標 {record['muzzle_pitch_target_deg']}° 實際 {record['muzzle_pitch_current_deg']}°",
                    flush=True,
                )
            previous_status = status
            if args.match_fire and status.fired:
                balls_remaining -= 1
            partial_shot_gate.record(pose, status.fired)
            control_done_at = time.perf_counter()
            yaw_motor_state = motors.get_state(config.yaw_motor_id)
            pitch_motor_state = motors.get_state(config.pitch_motor_id)
            if (
                args.score_mode
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
            if preview is not None:
                preview.set_overlay(display, vision_frame)
                preview_stats = preview.stats()
                jpeg_encode_ms = preview_stats.get("preview_jpeg_encode_ms")
            elif dashboard is not None:
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
                        telemetry["match"] = {
                            "phase": args.match_phase,
                            "round_phase": score_state.phase if score_state is not None else None,
                            "bonus_hole_id": last_bonus_hole_id,
                            "elapsed_s": (round(now - match_started_at, 1)
                                          if match_started_at is not None else None),
                            "remaining_s": (round(max(0.0, args.match_duration - (now - match_started_at)), 1)
                                            if match_started_at is not None else None),
                            "balls_remaining": balls_remaining,
                            "stopped": match_stop_reason,
                        }
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
            elif key == ord("g") and args.score_mode:
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
            cleanup.callback(driver.close)
            cleanup.callback(shutdown_to_home)
            if shot_log is not None:
                cleanup.callback(shot_log.close)
            if dashboard is not None:
                cleanup.callback(dashboard.stop)


if __name__ == "__main__":
    main()

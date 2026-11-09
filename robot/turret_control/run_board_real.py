"""Real-camera, real-motor target-board aiming run.

This controls the two DAMIAO motors and can build the upper/lower-half
maximum-score plans from the 2026-09-03 rules.  The CLI defaults to an
operator-confirmed ``f`` key.  An integrating process can instead pass a real
``fire_callback`` and receive structured state through ``telemetry_callback``.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
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
    SharedMotorBackend,
)
from .run_yolo_demo import _inference_frame
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


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    fire_callback: Optional[FireCallback] = None,
    telemetry_callback: Optional[TelemetryCallback] = None,
) -> None:
    """Run the real board pipeline.

    ``fire_callback`` is an opt-in automatic launcher hook.  It is called only
    after the vision, pose, reachability and motor-settling gates all pass.  It
    must return True only after the requested ball was actually sent; False
    keeps the same target active.

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
        help="雙軸自動瞄準的平滑速度上限 rad/s（預設 0.09）",
    )
    parser.add_argument(
        "--aim-acceleration", type=float, default=DEFAULT_AIM_ACCELERATION_RAD_S2,
        help="雙軸自動瞄準的平滑加速度上限 rad/s²（預設 0.075）",
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
        help="等待目前孔位成功發射的秒數；0 表示不自動略過",
    )
    parser.add_argument("--headless", action="store_true", help="不開 OpenCV 視窗")
    parser.add_argument(
        "--aim-only",
        action="store_true",
        help="實機驗證模式：允許安全範圍內追瞄，但完全停用發射與 F 操作",
    )
    parser.add_argument("--restore-completed-aim", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        help="即時影像、孔位與砲塔資料儀表板的 port（預設 8080；0 停用）",
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
    if not math.isfinite(args.aim_speed) or args.aim_speed <= 0:
        parser.error("--aim-speed 必須是大於 0 的有限數值")
    if not math.isfinite(args.aim_acceleration) or args.aim_acceleration <= 0:
        parser.error("--aim-acceleration 必須是大於 0 的有限數值")
    if args.cooldown < 0:
        parser.error("--cooldown 不可小於 0")
    if args.target_timeout < 0:
        parser.error("--target-timeout 不可小於 0")
    if args.zed_calibration is not None and not args.zed_uvc:
        parser.error("--zed-calibration 必須配合 --zed-uvc 使用")
    if args.restore_completed_aim and (not args.aim_only or args.score_mode):
        parser.error("--restore-completed-aim 需要 --aim-only --no-score-mode")

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

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("請先安裝 ultralytics：python3 -m pip install ultralytics") from exc

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
    if args.aim_only:
        config = with_recorded_mechanism_geometry(config)
        print(
            "實機瞄準驗證模式：已套用量測機構幾何，只允許安全範圍內調整砲口，"
            "不會發射。",
            flush=True,
        )
    if args.score_mode:
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
    model = YOLO(str(args.model), task="segment")
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
    except Exception:
        camera.release()
        raise
    turret = TurretController(config, motors)
    driver = BoardEngagement(
        config,
        turret,
        shot_cooldown_s=args.cooldown,
        target_timeout_s=(
            None if args.target_timeout == 0 else args.target_timeout
        ),
    )
    pose_tracker = BoardPoseTracker(config)
    vision = BoardVision(model, config, confidence=args.conf, imgsz=args.imgsz,
                         device=args.device, virtual_board=args.virtual_board)
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
    fire_label = "AIM ONLY / no fire" if args.aim_only else (
        "external fire callback" if external_fire else "F confirm"
    )
    window = f"REAL BOARD AIM - {fire_label} / S skip / G score start / Q quit"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    dashboard = DashboardState() if args.web_port else None
    manual_confirmed_holes = []
    home_active = False
    home_error = None
    restore_completed = args.restore_completed_aim

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

    if dashboard is not None and args.aim_only and not args.score_mode:
        dashboard.enable_manual_aim()
    web_server = None
    if dashboard is not None:
        web_server = start_dashboard_server(dashboard, args.web_port)
        print(
            f"即時儀表板：http://<robot-ip>:{args.web_port}",
            flush=True,
        )
    last_report = 0.0
    last_target_id = None
    partial_shot_gate = PartialPoseShotGate()
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
        if args.restore_completed_aim:
            begin_home()
        while True:
            loop_started_at = time.perf_counter()
            capture_started_at = loop_started_at
            if use_zed_sdk:
                ok, frame, depth_data = camera.read()
            else:
                ok, frame = camera.read()
                depth_data = None
            if not ok:
                break
            frame_ready_at = time.perf_counter()
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
                    if depth_data is not None:
                        if board_surface is not None:
                            depth_plane = fit_board_depth_plane(
                                depth_data, board_surface.mask, pose, live_targets,
                                detected, config,
                            )
                            if depth_plane is not None:
                                live_targets = apply_board_depth_plane(live_targets, depth_plane, config)
                        if depth_plane is None:
                            live_targets = prefer_observed_depth(live_targets, detected)
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
            elif external_fire or fire_requested:
                driver.set_firing_enabled(True)
            else:
                # In CLI/manual mode, waiting for F is not a failed launcher
                # attempt and must not consume the target timeout window.
                driver.set_firing_enabled(
                    False,
                    "waiting for operator F confirmation",
                )

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
            score_light_scores = (
                score_probe_state.scores
                if reference_plan is not None or trusted_quad is not None
                else None
            )
            score_state = score_probe_state if args.score_mode else None
            if args.score_mode:
                if (
                    score_state.phase == "active"
                    and score_state.active_hole_id is not None
                    and strategy_round_id != score_round.round_id
                    and reference_plan is not None
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
            status = driver.update()
            partial_shot_gate.record(pose, status.fired)
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
            jpeg_encode_ms = None
            if dashboard is not None:
                encode_started_at = time.perf_counter()
                ok_jpg, encoded = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                jpeg_encode_ms = (time.perf_counter() - encode_started_at) * 1000.0
                if ok_jpg:
                    dashboard.update_frame(encoded.tobytes())
            latency = {
                "capture_ms": (frame_ready_at - capture_started_at) * 1000.0,
                "inference_ms": (inference_finished_at - inference_started_at) * 1000.0,
                "frame_age_ms": (time.perf_counter() - frame_ready_at) * 1000.0,
                "loop_ms": (time.perf_counter() - loop_started_at) * 1000.0,
                "jpeg_encode_ms": jpeg_encode_ms,
            }
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
                    operator_fire_pending=fire_requested,
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
                if telemetry_callback is not None:
                    telemetry_callback(telemetry)
                if dashboard is not None:
                    dashboard.update_telemetry(telemetry)
            key = -1
            if not args.headless:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                if args.aim_only:
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
        driver.close()
        turret.close()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if dashboard is not None:
            dashboard.stop()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()


if __name__ == "__main__":
    main()

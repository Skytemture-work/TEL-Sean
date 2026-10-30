"""Real-camera, real-motor target-board aiming run.

This controls the two DAMIAO motors and can build the upper/lower-half
maximum-score plans from the 2026-09-03 rules.  The CLI defaults to an
operator-confirmed ``f`` key.  An integrating process can instead pass a real
``fire_callback`` and receive structured state through ``telemetry_callback``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import cv2

from .backends import DamiaoSerialBackend
from .board import (
    BoardTarget,
    build_plan,
    build_plan_from_board_quad,
    build_virtual_plan,
    build_virtual_plan_from_partial,
    estimate_board_distance_from_targets,
    EngagementPlan,
)
from .board_surface import detect_dark_board_surface, filter_targets_to_surface
from .board_pose import (
    BoardPoseTracker,
    project_targets_from_pose,
    update_plan_geometry,
)
from .config import TurretConfig
from .controller import TurretController
from .dashboard import (
    DashboardState,
    build_dashboard_snapshot,
    start_dashboard_server,
)
from .detection import boxes_to_board_targets
from .engagement import BoardEngagement
from .fixed_board import calibrate_layout, load_layout
from .limits import load_motor_limits
from .run_yolo_demo import DEFAULT_MODEL, _draw_board_plan, _inference_frame
from .scoring_light import ScoreRoundController
from .scoring_strategy import (
    MATCH_PHASES,
    append_reallocated_grid_shots,
    build_max_score_strategy,
)

FireCallback = Callable[[BoardTarget], bool]
TelemetryCallback = Callable[[Dict[str, Any]], None]


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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--zed-uvc", action="store_true")
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=6.0,
        help="等待目前孔位成功發射的秒數；0 表示不自動略過",
    )
    parser.add_argument("--headless", action="store_true", help="不開 OpenCV 視窗")
    parser.add_argument(
        "--web-port",
        type=int,
        default=0,
        help="提供即時影像、孔位與砲塔資料儀表板的 port（例如 8080）",
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
    parser.add_argument(
        "--score-mode",
        action="store_true",
        help="2026 計分策略：按 G 開始，鎖定亮燈孔並建立最高分射擊計畫",
    )
    parser.add_argument(
        "--match-phase",
        choices=MATCH_PHASES,
        default="upper",
        help="比賽半場；決定 27/18 球上限與最大得分射擊順序",
    )
    args = parser.parse_args(argv)
    if args.cooldown < 0:
        parser.error("--cooldown 不可小於 0")
    if args.target_timeout < 0:
        parser.error("--target-timeout 不可小於 0")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("請先安裝 ultralytics：python3 -m pip install ultralytics") from exc

    config = TurretConfig()
    if args.virtual_board:
        config = replace(
            config,
            use_axis_geometry=False,
            compensate_parallax=False,
            compensate_gravity=False,
            require_distance_to_fire=False,
        )
        print("虛擬靶模式：只依畫面位置瞄準，已關閉距離／視差／重力補正。", flush=True)
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
    model = YOLO(str(args.model))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"無法開啟 /dev/video{args.camera}")
    requested_width = config.image_width_px * (2 if args.zed_uvc else 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_height_px)

    motors = DamiaoSerialBackend(
        motor_ids=[config.yaw_motor_id, config.pitch_motor_id],
        port=args.port,
        mode="mit",
        kp=5.0,
        kd=1.0,
        enable_on_start=False,
        position_limits=load_motor_limits(),
    )
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
    fire_label = "external fire callback" if external_fire else "F confirm"
    window = f"REAL BOARD AIM - {fire_label} / S skip / G score start / Q quit"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    dashboard = DashboardState() if args.web_port else None
    web_server = None
    if dashboard is not None:
        web_server = start_dashboard_server(dashboard, args.web_port)
        print(
            f"即時儀表板：http://<robot-ip>:{args.web_port}",
            flush=True,
        )
    last_report = 0.0
    last_target_id = None
    if args.layout:
        board_plan = load_layout(args.layout)
        if not args.score_mode:
            plan = board_plan
            driver.set_plan(plan)
        print(f"已載入固定靶板版面：{args.layout}", flush=True)
    if args.score_mode:
        # Hold position until the whistle and red-light selection establish a
        # legal competition firing plan.
        plan = EngagementPlan([])
        driver.set_plan(plan)

    try:
        motors.enable()
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            vision_frame = _inference_frame(frame, args.zed_uvc)
            frame_size_ok = (
                vision_frame.shape[1] == config.image_width_px
                and vision_frame.shape[0] == config.image_height_px
            )
            result = model.predict(
                vision_frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            raw_detected = boxes_to_board_targets(result, args.conf)
            board_surface = detect_dark_board_surface(
                vision_frame,
                value_threshold=config.board_dark_value_threshold,
                min_area_fraction=config.board_min_area_fraction,
                targets=raw_detected,
            )
            detected = filter_targets_to_surface(raw_detected, board_surface)
            quad_plan = None
            if board_surface is not None and board_surface.quad is not None:
                quad_distance = estimate_board_distance_from_targets(
                    detected,
                    board_surface.quad,
                    config.focal_length_x_px,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.board_position_tolerance_cm,
                )
                quad_plan = build_plan_from_board_quad(
                    board_surface.quad,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    board_distance_m=quad_distance,
                    large_first=config.engage_large_first,
                    left_to_right=config.engage_left_to_right,
                    bottom_to_top=config.engage_bottom_to_top,
                )
            now = time.monotonic()
            candidate_plan = None
            inferred_plan_used = False
            # Build a fresh geometric reference every frame.  This reference
            # follows the board when the robot moves; the active firing plan is
            # deliberately kept as a separate object so its shot index and
            # duplicate scoring shots are never reset.
            if quad_plan is not None and len(quad_plan) == 12:
                candidate_plan = quad_plan
                inferred_plan_used = True
            elif args.virtual_board:
                if len(detected) == 12:
                    candidate_plan = build_virtual_plan(
                        detected,
                        config.small_hole_diameter_m,
                        config.large_hole_diameter_m,
                        config.engage_large_first,
                        config.engage_left_to_right,
                        config.engage_bottom_to_top,
                        config.board_position_tolerance_cm,
                    )
                elif len(detected) >= 4:
                    candidate_plan = build_virtual_plan_from_partial(
                        detected,
                        config.small_hole_diameter_m,
                        config.large_hole_diameter_m,
                        config.engage_large_first,
                        config.engage_left_to_right,
                        config.engage_bottom_to_top,
                        focal_px=config.focal_length_x_px,
                        position_tolerance_cm=config.board_position_tolerance_cm,
                    )
                    inferred_plan_used = True
            else:
                if len(detected) == 12:
                    candidate_plan = build_plan(
                        detected,
                        config.focal_length_x_px,
                        config.small_hole_diameter_m,
                        config.large_hole_diameter_m,
                        config.engage_large_first,
                        config.engage_left_to_right,
                        config.engage_bottom_to_top,
                        config.board_position_tolerance_cm,
                    )
                    if len(candidate_plan) != 12:
                        candidate_plan = build_virtual_plan(
                            detected,
                            config.small_hole_diameter_m,
                            config.large_hole_diameter_m,
                            config.engage_large_first,
                            config.engage_left_to_right,
                            config.engage_bottom_to_top,
                            config.board_position_tolerance_cm,
                        )
                # Lighting can hide one or more holes.  Four non-collinear
                # centres are enough to recover the official metric layout.
                if (
                    (candidate_plan is None or len(candidate_plan) != 12)
                    and len(detected) >= 4
                ):
                    candidate_plan = build_virtual_plan_from_partial(
                        detected,
                        config.small_hole_diameter_m,
                        config.large_hole_diameter_m,
                        config.engage_large_first,
                        config.engage_left_to_right,
                        config.engage_bottom_to_top,
                        focal_px=config.focal_length_x_px,
                        position_tolerance_cm=config.board_position_tolerance_cm,
                    )
                    inferred_plan_used = True

            if (
                candidate_plan is not None
                and len(candidate_plan) == 12
                and all(t.hole_id is not None for t in candidate_plan.targets)
            ):
                board_plan = candidate_plan
                if not args.score_mode and plan is None:
                    plan = candidate_plan
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
                    board_surface.quad if board_surface is not None else None,
                    now=now,
                )
                if pose is not None and pose_reference is not None:
                    live_targets = project_targets_from_pose(
                        pose_reference,
                        pose,
                        config,
                    )
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

            # An operator confirmation belongs to the exact stable pose in
            # which it was given.  Never keep it latched across robot motion or
            # a lost visual pose and fire later without a fresh confirmation.
            if not pose_ready:
                fire_requested = False

            if not pose_ready:
                driver.set_firing_enabled(False, pose_blocked_reason)
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

            score_state = None
            if args.score_mode:
                reference_plan = board_plan or plan
                score_state = score_round.update(
                    vision_frame,
                    reference_plan,
                    board_surface.quad if board_surface is not None else None,
                )
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

            status = driver.update()
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
                    print(
                        f"目前瞄準 {target_id}："
                        f"yaw={status.command.target_yaw_rad:+.3f} rad，"
                        f"pitch={status.command.target_pitch_rad:+.3f} rad，"
                        + (
                            f"水平軸距離={status.command.yaw_axis_distance_m:.2f} m，"
                            if status.command.yaw_axis_distance_m is not None
                            else ""
                        )
                        + f"remaining={status.remaining}",
                        flush=True,
                    )
                last_target_id = target_id
            display = result.plot()
            if board_surface is not None:
                cv2.drawContours(
                    display,
                    [board_surface.contour],
                    -1,
                    (0, 255, 255),
                    2,
                )
            display_plan = board_plan if args.score_mode and board_plan is not None else plan
            if display_plan is not None:
                _draw_board_plan(display, display_plan)
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
            if args.score_mode and score_state is not None:
                score_text = " ".join(
                    f"{hole_id}:{score_state.scores.get(hole_id, 0.0):.2f}"
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
                    f"POSE {'READY' if pose_ready else 'SETTLING'} "
                    f"n={pose.correspondence_count} "
                    f"err={pose.reprojection_error_px:.1f}px "
                    f"v={pose.translation_speed_m_s:.2f}m/s "
                    f"w={pose.rotation_speed_rad_s:.2f}rad/s"
                )
                pose_color = (0, 220, 0) if pose_ready else (0, 165, 255)
            cv2.putText(
                display,
                pose_text,
                (12, 82 if args.score_mode else 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                pose_color,
                2,
                cv2.LINE_AA,
            )
            if dashboard is not None or telemetry_callback is not None:
                telemetry = build_dashboard_snapshot(
                    plan=plan,
                    board_plan=board_plan,
                    engagement_status=status,
                    config=config,
                    pose=pose,
                    pose_ready=pose_ready,
                    virtual_board=args.virtual_board,
                    frame_size_ok=frame_size_ok,
                    raw_detection_count=len(raw_detected),
                    board_detection_count=len(detected),
                    board_surface_found=board_surface is not None,
                    score_mode=args.score_mode,
                    match_phase=args.match_phase,
                    score_state=score_state,
                    bonus_hole_id=last_bonus_hole_id,
                    operator_fire_pending=fire_requested,
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
                if telemetry_callback is not None:
                    telemetry_callback(telemetry)
                if dashboard is not None:
                    dashboard.update_telemetry(telemetry)
            if dashboard is not None:
                ok_jpg, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_jpg:
                    dashboard.update_frame(encoded.tobytes())
            key = -1
            if not args.headless:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                if external_fire:
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

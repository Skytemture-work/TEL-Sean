"""Real-camera, real-motor 3x4 board aiming run.

This controls the two DAMIAO motors and advances through the 12-hole plan.
Firing is deliberately operator-confirmed with the ``f`` key; the launcher
callback is the integration point for the real firing actuator.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .backends import DamiaoSerialBackend
from .board import (
    build_plan,
    build_plan_from_board_quad,
    build_virtual_plan,
    build_virtual_plan_from_partial,
    estimate_board_distance_from_targets,
)
from .board_surface import detect_dark_board_surface, filter_targets_to_surface
from .config import TurretConfig
from .controller import TurretController
from .detection import boxes_to_board_targets
from .engagement import BoardEngagement
from .fixed_board import calibrate_layout, load_layout
from .run_yolo_demo import DEFAULT_MODEL, _draw_board_plan, _inference_frame


class _LiveFrame:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = True


class _LiveHandler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            body = b'<meta charset="utf-8"><h2>REAL BOARD AIM</h2><img src="/stream" style="max-width:100%">'
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while self.state.running:
                with self.state.lock:
                    frame = self.state.frame
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Real camera + real motors, 12-hole board aiming")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--zed-uvc", action="store_true")
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", help="不開 OpenCV 視窗")
    parser.add_argument("--web-port", type=int, default=0, help="提供 Mac 瀏覽器影像串流的 port")
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
    args = parser.parse_args()

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
    elif config.use_axis_geometry:
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

    motors = DamiaoSerialBackend(
        motor_ids=[config.yaw_motor_id, config.pitch_motor_id],
        port=args.port,
        mode="mit",
        kp=5.0,
        kd=1.0,
        enable_on_start=False,
        position_limits=config.motor_limits,
    )
    turret = TurretController(config, motors)
    driver = BoardEngagement(config, turret, shot_cooldown_s=args.cooldown)
    plan = None
    fire_requested = False

    def operator_fire(_target) -> bool:
        nonlocal fire_requested
        if not fire_requested:
            return False
        fire_requested = False
        print("F：確認此孔，請在這裡接入實際發射器命令。")
        return True

    driver.fire = operator_fire
    window = "REAL BOARD AIM - F confirm / S skip / Q quit"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    live = _LiveFrame() if args.web_port else None
    web_server = None
    if live is not None:
        _LiveHandler.state = live
        web_server = ThreadingHTTPServer(("0.0.0.0", args.web_port), _LiveHandler)
        threading.Thread(target=web_server.serve_forever, daemon=True).start()
        print(f"Mac 瀏覽器開啟：http://<robot-ip>:{args.web_port}", flush=True)
    last_state = None
    last_report = 0.0
    last_target_id = None
    if args.layout:
        plan = load_layout(args.layout)
        driver.set_plan(plan)
        print(f"已載入固定靶板版面：{args.layout}", flush=True)

    try:
        motors.enable()
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = model.predict(
                _inference_frame(frame, args.zed_uvc),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            raw_detected = boxes_to_board_targets(result, args.conf)
            board_surface = detect_dark_board_surface(
                frame,
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
            if now - last_report >= 2.0:
                print(
                    f"YOLO 候選孔：靶面內 {len(detected)} / 原始 {len(raw_detected)}",
                    flush=True,
                )
                last_report = now
            candidate_plan = None
            inferred_plan_used = False
            if plan is None:
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
                    # Normal field mode also accepts a geometrically valid
                    # partial board.  It is treated as a complete target plan
                    # so lighting-related missed holes do not stop the match.
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
                
            if candidate_plan is not None and len(candidate_plan) == 12 and all(t.hole_id is not None for t in candidate_plan.targets):
                plan = candidate_plan
                if last_state != tuple(t.hole_id for t in plan.targets):
                    driver.set_plan(plan)
                    last_state = tuple(t.hole_id for t in plan.targets)
                    plan_source = "推算" if inferred_plan_used else "實測"
                    print(
                        f"{plan_source}靶板計畫建立："
                        + " ".join(t.hole_id or "??" for t in plan.targets),
                        flush=True,
                    )
                    if args.calibrate_layout and not inferred_plan_used:
                        calibrate_layout(detected, config, args.calibrate_layout)
                        print(f"已保存固定靶板版面：{args.calibrate_layout}", flush=True)

            status = driver.update()
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
            if plan is not None:
                _draw_board_plan(display, plan)
            text = f"12-hole {'READY' if plan else 'WAIT'} | remaining={status.remaining}"
            cv2.putText(display, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
            if live is not None:
                ok_jpg, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_jpg:
                    with live.lock:
                        live.frame = encoded.tobytes()
            key = -1
            if not args.headless:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                fire_requested = True
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
        if live is not None:
            live.running = False
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()


if __name__ == "__main__":
    main()

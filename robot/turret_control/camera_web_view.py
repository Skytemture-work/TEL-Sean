"""Stream camera + YOLO board detections to a web browser.

Run on the robot, then open http://ROBOT_IP:8080 from a Mac on the same LAN.
This program only opens the camera; it does not connect to or move motors.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO

if "bool" not in np.__dict__:
    np.bool = bool  # type: ignore[attr-defined]

from .board import (
    build_plan,
    build_plan_from_board_quad,
    estimate_board_distance_from_targets,
)
from .board_surface import detect_dark_board_surface, filter_targets_to_surface
from .config import TurretConfig, with_zed2_hd720_uvc_calibration
from .detection import boxes_to_board_targets
from .dashboard import build_dashboard_snapshot
from .geometry import aim_angles_from_target
from .run_yolo_demo import _draw_board_plan
from .run_zed_tensorrt_test import (
    PreviewBoardTracker,
    _draw_inferred_board,
    _surface_from_estimated_quad,
)

DEFAULT_MODEL = Path.home() / "Desktop" / "best.engine"


class _TemporalGeometryFilter:
    """Stabilize preview geometry without changing the detector output."""

    def __init__(
        self,
        quad_alpha: float = 0.22,
        distance_alpha: float = 0.25,
        distance_window: int = 5,
        stale_after_s: float = 0.6,
    ):
        self.quad_alpha = quad_alpha
        self.distance_alpha = distance_alpha
        self.distance_window = distance_window
        self.stale_after_s = stale_after_s
        self.quad = None
        self.distance_m = None
        self.distance_samples = []
        self.last_update_at = None

    def reset(self) -> None:
        self.quad = None
        self.distance_m = None
        self.distance_samples.clear()
        self.last_update_at = None

    def _expire_if_stale(self, now: float) -> None:
        if (
            self.last_update_at is not None
            and now - self.last_update_at > self.stale_after_s
        ):
            self.reset()

    def update_quad(self, quad, now: float):
        self._expire_if_stale(now)
        if quad is None:
            return None
        candidate = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        if candidate.shape != (4, 2) or not np.all(np.isfinite(candidate)):
            return None
        if self.quad is None:
            self.quad = candidate.copy()
        else:
            self.quad = (
                (1.0 - self.quad_alpha) * self.quad
                + self.quad_alpha * candidate
            )
        self.last_update_at = now
        return self.quad.copy()

    def update_distance(self, distance_m, now: float):
        self._expire_if_stale(now)
        if distance_m is None:
            return self.distance_m
        value = float(distance_m)
        if not math.isfinite(value) or value <= 0.0:
            return self.distance_m
        self.distance_samples.append(value)
        if len(self.distance_samples) > self.distance_window:
            self.distance_samples.pop(0)
        robust_value = float(np.median(self.distance_samples))
        if self.distance_m is None:
            self.distance_m = robust_value
        else:
            self.distance_m = (
                (1.0 - self.distance_alpha) * self.distance_m
                + self.distance_alpha * robust_value
            )
        self.last_update_at = now
        return self.distance_m


class CameraState:
    def __init__(
        self,
        camera: int,
        model_path: Path,
        device: str,
        conf: float,
        zed_uvc: bool,
        zed_sdk: bool,
    ):
        self.frame = None
        self.status = {}
        self.lock = threading.Lock()
        self.running = True
        self.zed_uvc = zed_uvc
        self.zed_sdk = zed_sdk
        self.cap = None
        self.zed = None
        self.image = None
        if self.zed_sdk:
            import pyzed.sl as sl

            self.sl = sl
            self.zed = sl.Camera()
            init = sl.InitParameters()
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = 30
            init.coordinate_units = sl.UNIT.METER
            init.depth_mode = sl.DEPTH_MODE.ULTRA
            status = self.zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"cannot open ZED: {status}")
            self.image = sl.Mat()
        else:
            self.cap = cv2.VideoCapture(camera)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera {camera}")
            if self.zed_uvc:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        model_kwargs = {"task": "segment"} if self.zed_sdk else {}
        self.model = YOLO(str(model_path), **model_kwargs)
        self.device = device
        self.conf = conf
        self.config = TurretConfig()
        if self.zed_uvc:
            self.config = with_zed2_hd720_uvc_calibration(self.config)
        self.readonly_geometry = replace(
            self.config,
            camera_to_yaw_yaw_rad=0.0,
            camera_to_yaw_pitch_rad=math.radians(20.0),
            yaw_axis_offset_x_m=0.0,
            yaw_axis_offset_y_m=0.0,
            yaw_axis_offset_z_m=-0.101662,
            pitch_axis_offset_x_m=0.0,
            pitch_axis_offset_y_m=0.0721,
            pitch_axis_offset_z_m=0.0,
        )
        self.board_tracker = PreviewBoardTracker()
        self.geometry_filter = _TemporalGeometryFilter()
        self.confirmed_holes = []
        self.current_hole_id = None

    def _apply_manual_progress(self, plan):
        """Advance the display-only plan through operator confirmations."""

        with self.lock:
            confirmed = tuple(self.confirmed_holes)
        confirmed_set = set(confirmed)
        index = 0
        while index < len(plan.targets):
            if plan.targets[index].hole_id not in confirmed_set:
                break
            index += 1
        # This is only the local preview plan index.  It never reaches a
        # motor backend and does not mark a target as physically fired.
        plan._index = index
        return confirmed

    def confirm_current(self):
        """Confirm the currently displayed hole and move to the next one."""

        with self.lock:
            hole_id = self.current_hole_id
            if hole_id is None:
                return False, "目前沒有可確認的孔位"
            if hole_id in self.confirmed_holes:
                return False, f"{hole_id} 已經人工確認"
            self.confirmed_holes.append(hole_id)
            return True, f"已確認 {hole_id}，下一幀切換下一孔"

    def reset_manual_progress(self):
        """Reset only the display-only manual confirmation sequence."""

        with self.lock:
            self.confirmed_holes.clear()
            return True, "已重設唯讀確認順序"

    def _update_readonly_status(
        self,
        plan,
        target,
        raw_count: int,
        board_count: int,
        board_surface_found: bool,
    ) -> None:
        target_yaw = None
        target_pitch = None
        yaw_error = None
        pitch_error = None
        yaw_axis_distance = None
        pitch_axis_distance = None
        if target is not None and target.distance_m is not None:
            try:
                target_yaw, target_pitch, geometry = aim_angles_from_target(
                    target.u_px,
                    target.v_px,
                    target.distance_m,
                    self.readonly_geometry,
                )
                yaw_error = target_yaw - self.readonly_geometry.yaw_zero_rad
                pitch_error = target_pitch - self.readonly_geometry.pitch_zero_rad
                yaw_axis_distance = geometry.yaw_axis_distance_m
                pitch_axis_distance = geometry.pitch_axis_distance_m
            except (ValueError, ZeroDivisionError):
                pass

        command = SimpleNamespace(
            target_yaw_rad=target_yaw,
            target_pitch_rad=target_pitch,
            yaw_servo_error_rad=yaw_error,
            pitch_servo_error_rad=pitch_error,
            yaw_error_rad=yaw_error or 0.0,
            pitch_error_rad=pitch_error or 0.0,
            yaw_axis_distance_m=yaw_axis_distance,
            pitch_axis_distance_m=pitch_axis_distance,
            target_reachable=target is not None,
            fire_allowed=False,
            stable=False,
            blocked_reason="純偵測模式：未連接馬達",
        )
        engagement_status = SimpleNamespace(command=command, target=target)
        with self.lock:
            self.current_hole_id = getattr(target, "hole_id", None)
            confirmed_holes = tuple(self.confirmed_holes)
        snapshot = build_dashboard_snapshot(
            plan=plan,
            board_plan=plan,
            engagement_status=engagement_status,
            config=self.readonly_geometry,
            pose=None,
            pose_ready=False,
            virtual_board=False,
            frame_size_ok=True,
            raw_detection_count=raw_count,
            board_detection_count=board_count,
            board_surface_found=board_surface_found,
            score_mode=False,
            match_phase="upper",
            score_state=None,
            bonus_hole_id=None,
            operator_fire_pending=False,
            detection_only=True,
            confirmed_holes=confirmed_holes,
        )
        snapshot["system"]["ammo_capacity"] = 18
        snapshot["manual_control"] = {
            "enabled": True,
            "current_hole_id": getattr(target, "hole_id", None),
            "confirmed_holes": list(confirmed_holes),
        }
        with self.lock:
            self.status = snapshot

    def run(self) -> None:
        while self.running:
            if self.zed_sdk:
                if self.zed.grab() != self.sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.01)
                    continue
                self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
                frame = cv2.cvtColor(
                    self.image.get_data(),
                    cv2.COLOR_BGRA2BGR,
                )
                ok = True
            else:
                ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            inference_frame = (
                frame[:, : frame.shape[1] // 2].copy()
                if self.zed_uvc and not self.zed_sdk
                else frame
            )
            result = self.model.predict(
                inference_frame,
                imgsz=640,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )[0]
            raw_detected = boxes_to_board_targets(result, self.conf)
            board_surface = detect_dark_board_surface(
                inference_frame,
                value_threshold=self.config.board_dark_value_threshold,
                min_area_fraction=self.config.board_min_area_fraction,
                targets=raw_detected,
            )
            estimated_quad, tracked_estimate = self.board_tracker.update(
                raw_detected,
                inference_frame.shape,
                time.monotonic(),
            )
            estimated_surface = False
            if estimated_quad is not None and (
                board_surface is None or board_surface.quad is None
            ):
                board_surface = _surface_from_estimated_quad(
                    estimated_quad,
                    inference_frame.shape,
                )
                estimated_surface = True
            elif board_surface is not None and board_surface.quad is None:
                board_surface = None
            detected = filter_targets_to_surface(raw_detected, board_surface)
            observed_plan = build_plan(
                detected,
                self.config.focal_length_x_px,
                self.config.small_hole_diameter_m,
                self.config.large_hole_diameter_m,
                self.config.engage_large_first,
                self.config.engage_left_to_right,
                self.config.engage_bottom_to_top,
                self.config.board_position_tolerance_cm,
            )
            plan = observed_plan
            inferred = False
            using_estimated_quad = False
            if board_surface is not None and board_surface.quad is not None:
                raw_plan_quad = board_surface.quad
            elif estimated_quad is not None:
                raw_plan_quad = estimated_quad
                using_estimated_quad = True
            else:
                raw_plan_quad = None

            now = time.monotonic()
            plan_quad = self.geometry_filter.update_quad(raw_plan_quad, now)
            raw_board_distance = None
            if plan_quad is not None:
                raw_board_distance = estimate_board_distance_from_targets(
                    detected,
                    plan_quad,
                    self.config.focal_length_x_px,
                    self.config.small_hole_diameter_m,
                    self.config.large_hole_diameter_m,
                    self.config.board_position_tolerance_cm,
                )
            board_distance = self.geometry_filter.update_distance(
                raw_board_distance,
                now,
            )
            if plan_quad is not None and board_distance is not None:
                quad_plan = build_plan_from_board_quad(
                    plan_quad,
                    self.config.small_hole_diameter_m,
                    self.config.large_hole_diameter_m,
                    board_distance_m=board_distance,
                    large_first=self.config.engage_large_first,
                    left_to_right=self.config.engage_left_to_right,
                    bottom_to_top=self.config.engage_bottom_to_top,
                )
                if len(quad_plan.targets) == 12:
                    plan = quad_plan
                    inferred = (
                        using_estimated_quad
                        or len(observed_plan.targets) != 12
                    )
            self._apply_manual_progress(plan)
            display = result.plot()
            if board_surface is not None and not estimated_surface:
                cv2.drawContours(
                    display,
                    [board_surface.contour],
                    -1,
                    (0, 255, 255),
                    2,
                )
            if estimated_surface:
                cv2.polylines(
                    display,
                    [board_surface.contour],
                    True,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    "ESTIMATED BOARD" if not tracked_estimate else "TRACKED BOARD - INFERRED",
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            _draw_board_plan(display, plan)
            if inferred and len(plan.targets) == 12:
                _draw_inferred_board(display, plan, inferred=True)
            known = sum(t.diameter_m is not None for t in plan.targets)
            self._update_readonly_status(
                plan,
                plan.current() if len(plan.targets) else None,
                len(raw_detected),
                len(detected),
                board_surface is not None,
            )
            if inferred:
                status = (
                    f"PARTIAL -> INFERRED 12/12 | observed "
                    f"{len(detected)}/{len(raw_detected)}"
                )
                color = (255, 0, 255)
            else:
                status = (
                    f"Board {len(detected)}/{len(raw_detected)} raw | "
                    f"Classified {known}/{len(detected)}"
                )
                color = (0, 220, 0) if len(detected) == 12 and known == 12 else (0, 0, 255)
            cv2.putText(display, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            ok, encoded = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.frame = encoded.tobytes()
        if self.zed_sdk:
            self.zed.close()
        else:
            self.cap.release()


class Handler(BaseHTTPRequestHandler):
    state: CameraState

    def do_GET(self):  # noqa: N802
        if self.path == '/':
            body = Path(__file__).with_name("dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path == '/api/status':
            with self.state.lock:
                body = json.dumps(self.state.status).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path != '/stream':
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while self.state.running:
                with self.state.lock:
                    frame = self.state.frame
                if frame:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame); self.wfile.write(b'\r\n'); self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):  # noqa: N802
        if self.path == '/api/confirm':
            ok, message = self.state.confirm_current()
            body = json.dumps(
                {"ok": ok, "message": message}, ensure_ascii=False
            ).encode('utf-8')
            self.send_response(200 if ok else 409)
        elif self.path == '/api/reset':
            ok, message = self.state.reset_manual_progress()
            body = json.dumps(
                {"ok": ok, "message": message}, ensure_ascii=False
            ).encode('utf-8')
            self.send_response(200)
        else:
            self.send_error(404)
            return
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument(
        '--model',
        type=Path,
        default=DEFAULT_MODEL,
        help=f'TensorRT/YOLO model (default: {DEFAULT_MODEL})',
    )
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument(
        '--zed-uvc',
        action='store_true',
        help='crop the left 1280x720 image from a side-by-side ZED UVC stream',
    )
    parser.add_argument(
        '--zed-sdk',
        dest='zed_sdk',
        action='store_true',
        default=True,
        help='use the ZED SDK left image (default)',
    )
    parser.add_argument(
        '--no-zed-sdk',
        dest='zed_sdk',
        action='store_false',
        help='use an OpenCV camera instead of the ZED SDK',
    )
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    state = CameraState(
        args.camera,
        args.model,
        args.device,
        args.conf,
        args.zed_uvc,
        args.zed_sdk,
    )
    Handler.state = state
    thread = threading.Thread(target=state.run, daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Open http://<robot-ip>:{args.port} on the Mac. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        server.server_close()


if __name__ == '__main__':
    main()

"""Local real-time dashboard for board vision and turret telemetry."""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")
BOARD_HOLE_IDS = tuple(f"H{number:02d}" for number in range(1, 13))


class DashboardState:
    """Thread-safe frame and JSON snapshot shared with the HTTP server."""

    def __init__(self):
        self._frame: Optional[bytes] = None
        self._telemetry: Dict[str, Any] = {
            "generated_at_ms": 0,
            "system": {"online": False},
        }
        self._running = True
        self._lock = threading.Lock()

    def update_frame(self, frame: bytes) -> None:
        with self._lock:
            self._frame = frame

    def update_telemetry(self, telemetry: Dict[str, Any]) -> None:
        with self._lock:
            self._telemetry = telemetry

    def frame_snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def telemetry_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # JSON round-tripping gives the HTTP thread a detached structure
            # without exposing mutable plan lists across threads.
            return json.loads(json.dumps(self._telemetry, ensure_ascii=False))

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def stop(self) -> None:
        with self._lock:
            self._running = False


def _json_number(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _degrees(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return _json_number(math.degrees(value), digits)


def _axis_telemetry(
    target_motor_rad: Optional[float],
    servo_error_rad: Optional[float],
    current_motor_rad: Optional[float],
    zero_rad: float,
    sign: int,
    positive_direction: str,
    negative_direction: str,
) -> Dict[str, Any]:
    if (
        current_motor_rad is None
        and target_motor_rad is not None
        and servo_error_rad is not None
    ):
        current_motor_rad = target_motor_rad - servo_error_rad

    target_physical_rad = None
    current_physical_rad = None
    physical_delta_rad = None
    if target_motor_rad is not None:
        target_physical_rad = (target_motor_rad - zero_rad) * sign
    if current_motor_rad is not None:
        current_physical_rad = (current_motor_rad - zero_rad) * sign
    if servo_error_rad is not None:
        physical_delta_rad = servo_error_rad * sign

    delta_deg = _degrees(physical_delta_rad)
    if delta_deg is None:
        direction = "無目標" if current_motor_rad is not None else "無回饋"
    elif abs(delta_deg) < 0.05:
        direction = "已到位"
    elif delta_deg > 0:
        direction = positive_direction
    else:
        direction = negative_direction

    return {
        "current_deg": _degrees(current_physical_rad),
        "target_deg": _degrees(target_physical_rad),
        "delta_deg": delta_deg,
        "direction": direction,
        "current_motor_deg": _degrees(current_motor_rad),
        "target_motor_deg": _degrees(target_motor_rad),
        "motor_delta_deg": _degrees(servo_error_rad),
    }


def _sequence_state(position: int, plan_index: int, target: Any) -> str:
    if position < plan_index:
        return "fired" if bool(getattr(target, "engaged", False)) else "skipped"
    if position == plan_index:
        return "current"
    return "pending"


def _plan_telemetry(plan: Any, board_plan: Any) -> Dict[str, Any]:
    targets = list(getattr(plan, "targets", ()) or ())
    plan_index = int(getattr(plan, "index", 0)) if plan is not None else 0
    if plan_index >= len(targets):
        current_position = None
    else:
        current_position = plan_index

    hole_stats: Dict[str, Dict[str, Any]] = {
        hole_id: {
            "hole_id": hole_id,
            "state": "unscheduled",
            "fired_shots": 0,
            "pending_shots": 0,
            "skipped_shots": 0,
            "scheduled_shots": 0,
            "distance_m": None,
        }
        for hole_id in BOARD_HOLE_IDS
    }

    latest_geometry: Dict[str, Any] = {}
    for source_plan in (board_plan, plan):
        for target in list(getattr(source_plan, "targets", ()) or ()):
            hole_id = getattr(target, "hole_id", None)
            if hole_id in hole_stats:
                latest_geometry[hole_id] = target

    sequence = []
    for position, target in enumerate(targets):
        hole_id = getattr(target, "hole_id", None) or "H??"
        state = _sequence_state(position, plan_index, target)
        sequence.append(
            {
                "shot_number": position + 1,
                "hole_id": hole_id,
                "state": state,
                "distance_m": _json_number(getattr(target, "distance_m", None), 3),
            }
        )
        if hole_id not in hole_stats:
            continue
        stats = hole_stats[hole_id]
        stats["scheduled_shots"] += 1
        if state == "fired":
            stats["fired_shots"] += 1
        elif state == "skipped":
            stats["skipped_shots"] += 1
        else:
            stats["pending_shots"] += 1

    for hole_id, stats in hole_stats.items():
        geometry = latest_geometry.get(hole_id)
        if geometry is not None:
            stats["distance_m"] = _json_number(
                getattr(geometry, "distance_m", None), 3
            )
        is_current = any(
            item["hole_id"] == hole_id and item["state"] == "current"
            for item in sequence
        )
        if is_current:
            stats["state"] = "current"
        elif stats["pending_shots"]:
            stats["state"] = "pending"
        elif stats["fired_shots"]:
            stats["state"] = "fired"
        elif stats["skipped_shots"]:
            stats["state"] = "skipped"

    holes = [hole_stats[hole_id] for hole_id in BOARD_HOLE_IDS]
    fired_holes = [
        item["hole_id"] for item in holes if item["fired_shots"] > 0
    ]
    unfired_holes = [
        item["hole_id"] for item in holes if item["fired_shots"] == 0
    ]
    pending_holes = [
        item["hole_id"] for item in holes if item["pending_shots"] > 0
    ]
    fired_shots = sum(item["fired_shots"] for item in holes)
    skipped_shots = sum(item["skipped_shots"] for item in holes)
    pending_shots = sum(item["pending_shots"] for item in holes)

    return {
        "index": plan_index,
        "total_shots": len(targets),
        "fired_shots": fired_shots,
        "skipped_shots": skipped_shots,
        "pending_shots": pending_shots,
        "remaining_shots": max(0, len(targets) - plan_index),
        "finished": bool(targets) and current_position is None,
        "fired_holes": fired_holes,
        "unfired_holes": unfired_holes,
        "pending_holes": pending_holes,
        "holes": holes,
        "sequence": sequence,
    }


def build_dashboard_snapshot(
    *,
    plan: Any,
    board_plan: Any,
    engagement_status: Any,
    config: Any,
    pose: Any,
    pose_ready: bool,
    virtual_board: bool,
    frame_size_ok: bool,
    raw_detection_count: int,
    board_detection_count: int,
    board_surface_found: bool,
    score_mode: bool,
    match_phase: str,
    score_state: Any,
    bonus_hole_id: Optional[str],
    operator_fire_pending: bool,
    current_yaw_motor_rad: Optional[float] = None,
    current_pitch_motor_rad: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert live controller objects into browser-safe telemetry."""

    command = getattr(engagement_status, "command", None)
    target = getattr(engagement_status, "target", None)
    plan_data = _plan_telemetry(plan, board_plan)

    yaw = _axis_telemetry(
        getattr(command, "target_yaw_rad", None),
        getattr(command, "yaw_servo_error_rad", None),
        current_yaw_motor_rad,
        config.yaw_zero_rad,
        config.yaw_sign,
        "向右",
        "向左",
    )
    pitch = _axis_telemetry(
        getattr(command, "target_pitch_rad", None),
        getattr(command, "pitch_servo_error_rad", None),
        current_pitch_motor_rad,
        config.pitch_zero_rad,
        config.pitch_sign,
        "抬高",
        "降低",
    )

    blocked_reason = getattr(command, "blocked_reason", None)
    if target is None:
        blocked_reason = blocked_reason or "尚未建立射擊目標"
    elif not frame_size_ok:
        blocked_reason = "相機解析度與標定內參不一致"
    elif not virtual_board and not pose_ready:
        blocked_reason = blocked_reason or "靶面姿態尚未穩定"
    elif not bool(getattr(command, "stable", False)):
        blocked_reason = blocked_reason or "等待砲塔馬達到位"

    if virtual_board:
        pose_state = "virtual"
    elif pose is None:
        pose_state = "invalid"
    elif pose_ready:
        pose_state = "ready"
    else:
        pose_state = "settling"

    score_scores = {}
    if score_state is not None:
        score_scores = {
            str(key): _json_number(value, 3)
            for key, value in getattr(score_state, "scores", {}).items()
        }

    return {
        "generated_at_ms": int(time.time() * 1000),
        "system": {
            "online": True,
            "score_mode": bool(score_mode),
            "match_phase": match_phase,
            "frame_size_ok": bool(frame_size_ok),
        },
        "vision": {
            "raw_detections": int(raw_detection_count),
            "board_detections": int(board_detection_count),
            "board_surface_found": bool(board_surface_found),
        },
        "pose": {
            "state": pose_state,
            "ready": bool(pose_ready),
            "correspondences": (
                int(getattr(pose, "correspondence_count", 0))
                if pose is not None
                else 0
            ),
            "reprojection_error_px": _json_number(
                getattr(pose, "reprojection_error_px", None), 3
            ),
            "translation_speed_m_s": _json_number(
                getattr(pose, "translation_speed_m_s", None), 3
            ),
            "rotation_speed_rad_s": _json_number(
                getattr(pose, "rotation_speed_rad_s", None), 3
            ),
        },
        "firing": {
            "allowed": bool(getattr(command, "fire_allowed", False)),
            "stable": bool(getattr(command, "stable", False)),
            "reachable": bool(getattr(command, "target_reachable", False)),
            "blocked_reason": blocked_reason,
            "operator_confirmation_pending": bool(operator_fire_pending),
            "fired_this_frame": bool(
                getattr(engagement_status, "fired", False)
            ),
            "skipped_this_frame": bool(
                getattr(engagement_status, "skipped", False)
            ),
        },
        "target": {
            "hole_id": getattr(target, "hole_id", None),
            "distance_m": _json_number(getattr(target, "distance_m", None), 3),
            "yaw_axis_distance_m": _json_number(
                getattr(command, "yaw_axis_distance_m", None), 3
            ),
            "pitch_axis_distance_m": _json_number(
                getattr(command, "pitch_axis_distance_m", None), 3
            ),
            "confidence": _json_number(getattr(target, "confidence", None), 3),
            "pixel_u": _json_number(getattr(target, "u_px", None), 1),
            "pixel_v": _json_number(getattr(target, "v_px", None), 1),
            "diameter_cm": _json_number(
                (
                    getattr(target, "diameter_m", None) * 100.0
                    if getattr(target, "diameter_m", None) is not None
                    else None
                ),
                1,
            ),
            "command_yaw_error_deg": _degrees(
                getattr(command, "yaw_error_rad", None)
            ),
            "command_pitch_error_deg": _degrees(
                getattr(command, "pitch_error_rad", None)
            ),
            "yaw": yaw,
            "pitch": pitch,
        },
        "plan": plan_data,
        "score": {
            "enabled": bool(score_mode),
            "phase": getattr(score_state, "phase", "disabled"),
            "active_bonus_hole": (
                getattr(score_state, "active_hole_id", None) or bonus_hole_id
            ),
            "remaining_s": _json_number(
                getattr(score_state, "remaining_s", None), 1
            ),
            "scores": score_scores,
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard, MJPEG stream and telemetry JSON."""

    state: DashboardState

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            try:
                body = DASHBOARD_HTML_PATH.read_bytes()
            except OSError as exc:
                self._send_bytes(
                    f"Dashboard asset unavailable: {exc}".encode("utf-8"),
                    "text/plain; charset=utf-8",
                    status=500,
                )
                return
            self._send_bytes(body, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            body = json.dumps(
                self.state.telemetry_snapshot(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send_bytes(b"ok", "text/plain; charset=utf-8")
            return
        if path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        try:
            while self.state.running:
                frame = self.state.frame_snapshot()
                if frame:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args):
        pass


def start_dashboard_server(
    state: DashboardState,
    port: int,
    host: str = "0.0.0.0",
) -> ThreadingHTTPServer:
    """Start a background dashboard server and return its owner object."""

    handler = type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {"state": state},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

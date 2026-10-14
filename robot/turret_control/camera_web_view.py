"""Stream camera + YOLO board detections to a web browser.

Run on the robot, then open http://ROBOT_IP:8080 from a Mac on the same LAN.
This program only opens the camera; it does not connect to or move motors.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

from .board import build_plan
from .config import TurretConfig
from .detection import boxes_to_board_targets
from .run_yolo_demo import _draw_board_plan


class CameraState:
    def __init__(self, camera: int, model_path: Path, device: str, conf: float):
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera}")
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.config = TurretConfig()

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            result = self.model.predict(
                frame, imgsz=640, conf=self.conf, device=self.device, verbose=False
            )[0]
            detected = boxes_to_board_targets(result, self.conf)
            plan = build_plan(
                detected,
                self.config.focal_length_x_px,
                self.config.small_hole_diameter_m,
                self.config.large_hole_diameter_m,
                self.config.engage_large_first,
                self.config.engage_left_to_right,
                self.config.engage_bottom_to_top,
            )
            display = result.plot()
            _draw_board_plan(display, plan)
            known = sum(t.diameter_m is not None for t in detected)
            status = f"Detected {len(detected)}/12 | Classified {known}/{len(detected)}"
            color = (0, 220, 0) if len(detected) == 12 and known == 12 else (0, 0, 255)
            cv2.putText(display, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            ok, encoded = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.frame = encoded.tobytes()
        self.cap.release()


class Handler(BaseHTTPRequestHandler):
    state: CameraState

    def do_GET(self):  # noqa: N802
        if self.path == '/':
            body = b'''<!doctype html><meta charset="utf-8"><title>Turret camera</title>
<h2>Turret camera / YOLO</h2><img src="/stream" style="max-width:100%;height:auto">
<p>Camera preview only. Motors are not connected.</p>'''
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
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

    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.40)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    state = CameraState(args.camera, args.model, args.device, args.conf)
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

"""Legacy main.py adapter backed by the dashboard's board-vision pipeline.

The old UART interface is intentionally preserved:
``((vx, vy, wz), (yaw, pitch), detected)``.
Detection, board filtering, hole-plan inference, ZED depth and TensorRT model
selection are shared with ``turret_control.run_board_real``.  The legacy
caller still receives a board-centering command because its UART protocol does
not carry the dashboard's absolute motor-radian targets or firing plan.
"""

from __future__ import annotations

import math
import os
import threading
import time

import cv2
import numpy as np

import config as legacy_config
from turret_control.board_depth_plane import (
    apply_board_depth_plane,
    fit_board_depth_plane,
)
from turret_control.board_pose import (
    BoardPoseTracker,
    prefer_observed_depth,
    project_targets_from_pose,
)
from turret_control.board_vision import (
    BoardVision,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
)
from turret_control.competition_geometry import with_competition_geometry
from turret_control.config import TurretConfig
from turret_control.zed_camera import ZedDepthCamera

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class VisionTracker(threading.Thread):
    """Use BoardVision while retaining main.py's old shared-data interface."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()

        self.chassis_cmd = (0.0, 0.0, 0.0)
        self.turret_cmd = (50, 50)
        self.target_detected = False
        # main.py starts in manual mode.  Do not consume camera/GPU until the
        # ESP32 actually switches to automatic mode.
        self.is_active = False
        self.current_frame = None

        self.model_path = DEFAULT_MODEL
        self.device = os.environ.get("VISION_DEVICE", "0")
        self.model = None
        if YOLO is None:
            print("[Vision Error] 缺少 ultralytics，無法啟動新版 BoardVision。", flush=True)
        elif not self.model_path.is_file():
            print(f"[Vision Error] 找不到模型檔案: {self.model_path}", flush=True)
        else:
            print(
                f"[Vision] 使用儀表板同一模型: {self.model_path} "
                f"(device={self.device})",
                flush=True,
            )
            self.model = YOLO(str(self.model_path), task="segment")

        self.config = with_competition_geometry(TurretConfig())
        self.board_vision = None
        self.pose_tracker = None
        self._reference_plan = None

    def run(self):
        camera = None
        try:
            if self.model is None:
                return

            camera = ZedDepthCamera(self.config)
            self.config = camera.config
            self.board_vision = BoardVision(
                self.model,
                self.config,
                confidence=DEFAULT_CONFIDENCE,
                imgsz=640,
                device=self.device,
                virtual_board=False,
            )
            self.pose_tracker = BoardPoseTracker(self.config)
            print(
                "[Vision] 已切換至儀表板 BoardVision：ZED 深度、靶面篩選、"
                "孔位推算、PnP 姿態與 TensorRT/YOLO 模型。",
                flush=True,
            )

            while True:
                if not self.is_active:
                    time.sleep(0.1)
                    continue

                ok, frame, depth_data = camera.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                now = time.monotonic()
                result = self.board_vision.process(
                    frame, now=now, depth_data=depth_data
                )

                candidate_plan = result.plan
                if (
                    candidate_plan is not None
                    and len(candidate_plan) == 12
                    and all(target.hole_id is not None for target in candidate_plan.targets)
                ):
                    self._reference_plan = candidate_plan

                pose = None
                live_targets = {}
                if self._reference_plan is not None:
                    pose = self.pose_tracker.update(
                        result.targets,
                        self._reference_plan,
                        result.quad,
                        now=now,
                    )
                    if pose is not None:
                        live_targets = project_targets_from_pose(
                            self._reference_plan,
                            pose,
                            self.config,
                        )
                        if depth_data is not None:
                            depth_plane = None
                            if result.surface is not None:
                                depth_plane = fit_board_depth_plane(
                                    depth_data,
                                    result.surface.mask,
                                    pose,
                                    live_targets,
                                    result.targets,
                                    self.config,
                                )
                            if depth_plane is not None:
                                live_targets = apply_board_depth_plane(
                                    live_targets, depth_plane, self.config
                                )
                            else:
                                live_targets = prefer_observed_depth(
                                    live_targets, result.targets
                                )

                pose_ready = (
                    pose is not None
                    and pose.stable
                    and self.pose_tracker.is_fresh(now)
                    and bool(live_targets)
                )
                display = result.render(frame)
                cv2.putText(
                    display,
                    "BoardVision pose=stable" if pose_ready else "BoardVision pose=settling",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if pose_ready else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

                if pose_ready:
                    targets = list(live_targets.values())
                    center_u = float(np.median([target.u_px for target in targets]))
                    center_v = float(np.median([target.v_px for target in targets]))
                    ranges = [
                        target.distance_m
                        for target in targets
                        if target.distance_m is not None
                        and math.isfinite(target.distance_m)
                        and target.distance_m > 0.0
                    ]
                    distance_m = float(np.median(ranges)) if ranges else None
                    fx = self.config.focal_length_x_px
                    fy = self.config.focal_length_y_px
                    cx = self.config.camera_center_x_px
                    cy = self.config.camera_center_y_px
                    if distance_m is not None:
                        dx_cm = (center_u - cx) * distance_m / fx * 100.0
                        dy_cm = -(center_v - cy) * distance_m / fy * 100.0
                        vx = 1.2 * (distance_m * 100.0 - legacy_config.DISTANCE_PLATE_CM)
                    else:
                        dx_cm = (center_u - cx) / fx * 100.0
                        dy_cm = -(center_v - cy) / fy * 100.0
                        vx = 0.0
                    yaw_deg = math.degrees(math.atan2(center_u - cx, fx))
                    wz = -yaw_deg
                    vy = -1.5 * dx_cm
                    yaw = int(np.clip(50 + dx_cm * 0.5, 0, 100))
                    pitch = int(np.clip(50 - dy_cm * 0.5, 0, 100))
                    with self.lock:
                        self.chassis_cmd = (vx, vy, wz)
                        self.turret_cmd = (yaw, pitch)
                        self.target_detected = True
                        self.current_frame = display
                else:
                    with self.lock:
                        self.chassis_cmd = (0.0, 0.0, 0.0)
                        self.turret_cmd = (50, 50)
                        self.target_detected = False
                        self.current_frame = display

                time.sleep(0.001)
        except Exception as exc:
            print(f"[Vision Error] BoardVision 執行失敗：{exc}", flush=True)
            with self.lock:
                self.chassis_cmd = (0.0, 0.0, 0.0)
                self.turret_cmd = (50, 50)
                self.target_detected = False
        finally:
            if camera is not None:
                camera.release()
                print("[Vision] ZED 2 相機已安全關閉。", flush=True)

    def get_vision_data(self):
        with self.lock:
            return self.chassis_cmd, self.turret_cmd, self.target_detected

    def get_annotated_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_active(self, state: bool):
        self.is_active = bool(state)

"""Expose fresh board range, center and observed-hole count to main.py."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config as legacy_config
from turret_control.board_depth_plane import (
    apply_board_depth_plane,
    fit_board_depth_plane,
)
from turret_control.board_pose import (
    BoardPoseTracker,
    _match_observations,
    camera_matrix,
    distortion_coefficients,
    prefer_observed_depth,
    project_targets_from_pose,
)
from turret_control.board import EngagementPlan
from turret_control.official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM
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


@dataclass(frozen=True)
class BoardObservation:
    captured_at: float
    distance_cm: float | None
    center_error_px: float
    observed_holes: int
    outline_in_frame: bool

    @property
    def complete(self):
        return self.outline_in_frame and self.observed_holes == 12


class VisionTracker(threading.Thread):
    """Use BoardVision while retaining main.py's old shared-data interface."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.camera_released = False

        # main.py starts in manual mode.  Do not consume camera/GPU until the
        # ESP32 actually switches to automatic mode.
        self.is_active = False
        self.current_frame = None
        self._observation = None
        self._activation = 0
        self._tracker_activation = None

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
            print(
                "[Vision] 已切換至儀表板 BoardVision：ZED 深度、靶面篩選、"
                "孔位推算、PnP 姿態與 TensorRT/YOLO 模型。",
                flush=True,
            )

            while not self._stop_event.is_set():
                with self.lock:
                    active, activation = self.is_active, self._activation
                if not active:
                    self._stop_event.wait(0.1)
                    continue
                if activation != self._tracker_activation:
                    self.board_vision = BoardVision(
                        self.model, self.config, confidence=DEFAULT_CONFIDENCE,
                        imgsz=640, device=self.device, virtual_board=False,
                    )
                    self.pose_tracker = BoardPoseTracker(self.config)
                    self._reference_plan = None
                    self._tracker_activation = activation

                now = time.monotonic()
                ok, frame, depth_data = camera.read()
                if self._stop_event.is_set():
                    break
                if not ok:
                    self._publish(activation, None, None)
                    time.sleep(0.02)
                    continue

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

                observation = None
                if pose_ready:
                    observation = self._board_observation(
                        pose, live_targets, result.targets, frame.shape, now
                    )
                self._publish(activation, observation, display)

                time.sleep(0.001)
        except Exception as exc:
            print(f"[Vision Error] BoardVision 執行失敗：{exc}", flush=True)
            with self.lock:
                self._observation = None
        finally:
            if camera is not None:
                camera.release()
                print("[Vision] ZED 2 相機已安全關閉。", flush=True)
            self.camera_released = True

    def _board_observation(self, pose, live_targets, observed, frame_shape, now):
        width, height = BOARD_WIDTH_CM / 100.0, BOARD_HEIGHT_CM / 100.0
        board_points = np.array([
            [0.0, 0.0, 0.0], [width, 0.0, 0.0],
            [width, height, 0.0], [0.0, height, 0.0],
            [width / 2.0, height / 2.0, 0.0],
        ])
        image_points, _ = cv2.projectPoints(
            board_points, pose.rvec, pose.tvec, camera_matrix(self.config),
            distortion_coefficients(self.config),
        )
        image_points = image_points.reshape(-1, 2)
        if not np.all(np.isfinite(image_points)):
            return None
        outline = image_points[:4]
        outline_in_frame = bool(
            np.all(outline >= 0) and np.all(outline < [frame_shape[1], frame_shape[0]])
        )
        matched, _ = _match_observations(observed, EngagementPlan(list(live_targets.values())))
        measured_depths = [
            target.distance_m for target in live_targets.values()
            if target.distance_source in ("zed_plane", "zed_depth")
            and target.distance_m is not None and math.isfinite(target.distance_m)
            and target.distance_m > 0
        ]
        distance_cm = None
        if measured_depths:
            distance_cm = (float(np.median(measured_depths)) - legacy_config.ZED_TO_FRONT_M) * 100.0
        return BoardObservation(
            now, distance_cm, float(image_points[-1, 0] - self.config.camera_center_x_px),
            len(matched), outline_in_frame,
        )

    def _publish(self, activation, observation, display):
        with self.lock:
            if self.is_active and activation == self._activation:
                self._observation = observation
                self.current_frame = display

    def get_board_observation(self):
        with self.lock:
            observation = self._observation
        if (observation is None
                or time.monotonic() - observation.captured_at > legacy_config.AUTO_VISION_MAX_AGE_S):
            return None
        return observation

    def get_vision_data(self):
        """Compatibility snapshot for the standalone vision viewer."""
        observation = self.get_board_observation()
        if observation is None:
            return (0, 0, 0), (50, 50), False
        vy = legacy_config.AUTO_STRAFE_SIGN * int(np.clip(observation.center_error_px / 4, -15, 15))
        return (0, vy, 0), (50, 50), True

    def get_annotated_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_active(self, state: bool):
        with self.lock:
            self.is_active = bool(state)
            self._activation += 1
            self._observation = None
            self.current_frame = None

    def stop(self):
        """Wait for capture to exit and confirm the ZED handle was released."""
        self.set_active(False)
        self._stop_event.set()
        if self.ident is not None:
            self.join()
        return self.camera_released

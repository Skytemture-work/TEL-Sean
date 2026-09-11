# vision_target.py
import cv2
import numpy as np
import threading
import time
import config

class VisionTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        
        self.chassis_cmd = (0, 0, 0)   # 底盤建議速度 (vx, vy, wz)
        self.turret_cmd = (50, 50)     # 砲台絕對角度 (yaw, pitch)
        self.target_detected = False
        self.is_active = True          # 支援模式切換的待命控制

        w, h = config.TARGET_WIDTH_CM, config.TARGET_HEIGHT_CM
        self.object_points = np.array([
            [-w/2,  h/2, 0],
            [ w/2,  h/2, 0],
            [ w/2, -h/2, 0],
            [-w/2, -h/2, 0]
        ], dtype=np.float32)

        self.camera_matrix = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))

    def run(self):
        cap = cv2.VideoCapture(0) # 實際接上 ZED 2 可改為 pyzed 擷取
        
        while True:
            if not self.is_active:
                time.sleep(0.1) # 手動模式下進入休眠
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            corners = self._detect_target_corners(frame)

            if corners is not None:
                image_points = np.array(corners, dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points, image_points, self.camera_matrix, self.dist_coeffs
                )

                if success:
                    dx = tvec[0][0]
                    dy = tvec[1][0]
                    dz = tvec[2][0]

                    R, _ = cv2.Rodrigues(rvec)
                    yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                    # 1. 計算底盤目標移動量 (P Controller)
                    vx = 1.2 * (dz - config.DISTANCE_PLATE_CM)
                    vy = -1.5 * dx
                    wz = -1.0 * yaw_deg

                    # 2. 計算砲台獨立對準角度 (映射 0~100)
                    yaw_val = int(np.clip(50 + (dx * 0.5), 0, 100))
                    pitch_val = int(np.clip(50 - (dy * 0.5), 0, 100))

                    with self.lock:
                        self.chassis_cmd = (vx, vy, wz)
                        self.turret_cmd = (yaw_val, pitch_val)
                        self.target_detected = True
            else:
                with self.lock:
                    self.chassis_cmd = (0, 0, 0)
                    self.turret_cmd = (50, 50)
                    self.target_detected = False

            time.sleep(0.02)

    def _detect_target_corners(self, frame):
        # TODO: 填入 YOLOv8 或 OpenCV 標靶四角點提取邏輯
        return None 

    def get_vision_data(self):
        with self.lock:
            return self.chassis_cmd, self.turret_cmd, self.target_detected
            
    def set_active(self, state):
        self.is_active = state
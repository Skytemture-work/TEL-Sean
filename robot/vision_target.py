import sys
import os
import cv2
import numpy as np
import threading
import time
import pyzed.sl as sl
from ultralytics import YOLO
import config

class VisionTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        
        # 系統控制指令與狀態 (與 main.py 相容)
        self.chassis_cmd = (0, 0, 0)   # 底盤建議速度 (vx, vy, wz)
        self.turret_cmd = (50, 50)     # 砲台絕對角度 (yaw, pitch) -> 範圍 0~100
        self.target_detected = False
        self.is_active = True          # 支援模式切換待命
        
        # 提供給主執行緒渲染畫面的共享圖像
        self.current_frame = None

        # 控制一秒輸出一個動作的計時器
        self.last_output_time = 0.0    # 上次指令更新時間戳記
        self.OUTPUT_INTERVAL_SEC = 1.0 # 1 秒輸出一次動作 (1 Hz)

        # ---------------------------------------------------------
        # 1. 載入當前目錄下的 best.pt 模型
        # ---------------------------------------------------------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best.pt")

        if not os.path.exists(model_path):
            print(f"[Vision Error] 找不到模型檔案: {model_path}")
            self.model = None
        else:
            print(f"[Vision] 成功載入模型: {model_path}")
            self.model = YOLO(model_path)

        # ---------------------------------------------------------
        # 2. 標靶 3D 物理空間座標設定 (單位: cm)
        # ---------------------------------------------------------
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
        # ---------------------------------------------------------
        # 3. 初始化 ZED 2 相機與深度引擎
        # ---------------------------------------------------------
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[Vision Error] 無法開啟 ZED 相機，狀態碼: {status}")
            return

        image_zed = sl.Mat()
        depth_zed = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        print("[Vision] ZED 2 多組標靶取多數決 & 1Hz 控制線程已順利啟動！")

        try:
            while True:
                if not self.is_active:
                    time.sleep(0.1)
                    continue

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

                    color_image = image_zed.get_data()
                    color_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                    depth_np = depth_zed.get_data()  # 2D 深度矩陣 (Meters)

                    if self.model is None:
                        time.sleep(0.02)
                        continue

                    # ---------------------------------------------------------
                    # 4. YOLOv8 推論 (提取多組 Segmentation Masks)
                    # ---------------------------------------------------------
                    results = self.model(color_bgr, conf=0.5, verbose=False)
                    annotated_frame = results[0].plot()

                    # 收集多組物體/孔洞的測距與姿態數據
                    detected_depths_cm = []
                    detected_dx_cm = []
                    detected_dy_cm = []
                    detected_yaw_deg = []
                    valid_boxes = []

                    for result in results:
                        boxes = result.boxes
                        masks = result.masks  # 分割遮罩

                        if boxes is None or len(boxes) == 0:
                            continue

                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            mask = masks[idx].data[0].cpu().numpy() if masks is not None and len(masks) > idx else None

                            # A. 計算單一個物體/孔洞遮罩內最濃密的點 (Distance Transform)
                            conc_u, conc_v = self._get_most_concentrated_point(mask, (x1, y1, x2, y2), color_bgr.shape)
                            
                            # B. 採樣視窗中位數深度
                            dz_m = self._get_median_depth(depth_np, conc_u, conc_v)

                            # C. SolvePnP 解算該物體姿態
                            image_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                            success, rvec, tvec = cv2.solvePnP(
                                self.object_points, image_points, self.camera_matrix, self.dist_coeffs
                            )

                            if success and dz_m is not None and dz_m > 0:
                                dx = tvec[0][0]
                                dy = tvec[1][0]
                                R, _ = cv2.Rodrigues(rvec)
                                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                                detected_depths_cm.append(dz_m * 100.0)
                                detected_dx_cm.append(dx)
                                detected_dy_cm.append(dy)
                                detected_yaw_deg.append(yaw)
                                valid_boxes.append((x1, y1, x2, y2, conc_u, conc_v, dz_m))

                                # 繪製綠色十字標記每個找到的密集採樣點
                                cv2.drawMarker(annotated_frame, (conc_u, conc_v), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

                    # ---------------------------------------------------------
                    # 5. 多組數據共識決 (多數決 / 中位數評估) 與 1 Hz 控制更新
                    # ---------------------------------------------------------
                    now = time.time()
                    if len(detected_depths_cm) > 0:
                        # 【多數決共識機制】使用中位數（Median）濾除單一異常雜訊點
                        final_dz_cm = float(np.median(detected_depths_cm))
                        final_dx_cm = float(np.median(detected_dx_cm))
                        final_dy_cm = float(np.median(detected_dy_cm))
                        final_yaw_deg = float(np.median(detected_yaw_deg))

                        # 在畫面上印出檢測數量與中位數結果
                        summary_text = f"Objects: {len(detected_depths_cm)} | Consensus Dist: {final_dz_cm/100.0:.2f}m"
                        cv2.putText(annotated_frame, summary_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # 【1 秒輸出一個動作限制】
                        if now - self.last_output_time >= self.OUTPUT_INTERVAL_SEC:
                            self.last_output_time = now

                            # 距離與姿態 P 控制量
                            vx = 1.2 * (final_dz_cm - config.DISTANCE_PLATE_CM)
                            vy = -1.5 * final_dx_cm
                            wz = -1.0 * final_yaw_deg

                            yaw_val = int(np.clip(50 + (final_dx_cm * 0.5), 0, 100))
                            pitch_val = int(np.clip(50 - (final_dy_cm * 0.5), 0, 100))

                            with self.lock:
                                self.chassis_cmd = (vx, vy, wz)
                                self.turret_cmd = (yaw_val, pitch_val)
                                self.target_detected = True
                    else:
                        # 畫面中無有效目標
                        if now - self.last_output_time >= self.OUTPUT_INTERVAL_SEC:
                            self.last_output_time = now
                            with self.lock:
                                self.chassis_cmd = (0, 0, 0)
                                self.turret_cmd = (50, 50)
                                self.target_detected = False

                    with self.lock:
                        self.current_frame = annotated_frame

                time.sleep(0.01)

        finally:
            zed.close()
            print("[Vision] ZED 2 相機已安全關閉。")

    def _get_most_concentrated_point(self, mask, box, frame_shape):
        """尋找遮罩內距離邊界最遠、面積最濃密厚實的點 (Pole of Inaccessibility)"""
        x1, y1, x2, y2 = box
        img_h, img_w = frame_shape[:2]

        if mask is not None:
            mask_resized = cv2.resize(mask, (img_w, img_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

            if max_val > 0:
                return max_loc[0], max_loc[1]

        center_u = int((x1 + x2) / 2)
        center_v = int((y1 + y2) / 2)
        return center_u, center_v

    def _get_median_depth(self, depth_np, u, v, window_size=3):
        """採樣 3x3 視窗中位數深度，排除無效區域"""
        h, w = depth_np.shape[:2]
        half_w = window_size // 2

        u_min = max(0, u - half_w)
        u_max = min(w, u + half_w + 1)
        v_min = max(0, v - half_w)
        v_max = min(h, v + half_w + 1)

        patch = depth_np[v_min:v_max, u_min:u_max]
        valid_depths = patch[~np.isnan(patch) & ~np.isinf(patch) & (patch > 0)]

        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return None

    def get_vision_data(self):
        """提供給 main.py 讀取控制量的接口"""
        with self.lock:
            return self.chassis_cmd, self.turret_cmd, self.target_detected

    def get_annotated_frame(self):
        """提供給 test_vision.py 主執行緒繪圖視窗讀取畫面"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_active(self, state: bool):
        """手動/自動模式切換待命控制"""
        self.is_active = state

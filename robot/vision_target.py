import sys
import os
import cv2
import numpy as np
import threading
import time
import pyzed.sl as sl
from ultralytics import YOLO
import config
from turret_control.vision_geometry import (
    binary_mask_geometry,
    polygon_geometry,
    sample_board_depth,
)

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
        # 2. 相機內參：先用 HD720 的保守預設值，開啟 ZED 後換成 SDK 校正值
        # ---------------------------------------------------------
        self.camera_matrix = np.array(
            [[700, 0, 640], [0, 700, 360], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

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

        self._load_zed_calibration(zed)

        image_zed = sl.Mat()
        depth_zed = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        # 過濾立體匹配在孔洞邊緣與低紋理背景產生的不可靠深度。
        # 舊版 SDK 若沒有這些欄位則保留預設值。
        if hasattr(runtime_parameters, "confidence_threshold"):
            runtime_parameters.confidence_threshold = 50
        if hasattr(runtime_parameters, "texture_confidence_threshold"):
            runtime_parameters.texture_confidence_threshold = 50

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

                            # YOLO segmentation 的 polygon 是原始影像座標；
                            # 優先使用它，不再把圓孔當成矩形來做 PnP。
                            mask_polygon = None
                            mask_array = None
                            if masks is not None and len(masks) > idx:
                                try:
                                    mask_polygon = np.asarray(
                                        masks.xy[idx], dtype=np.float32
                                    )
                                except (AttributeError, IndexError, TypeError, ValueError):
                                    # 相容沒有 masks.xy 的舊版 Ultralytics。
                                    try:
                                        mask_array = masks[idx].data[0].cpu().numpy()
                                    except (AttributeError, IndexError, TypeError):
                                        mask_array = None

                            geometry = (
                                polygon_geometry(mask_polygon)
                                if mask_polygon is not None
                                else None
                            )
                            if geometry is None and mask_array is not None:
                                geometry = binary_mask_geometry(
                                    mask_array, color_bgr.shape[:2]
                                )
                            # 沒有 segmentation mask 時，不能安全地從孔洞中心取深度；
                            # 直接捨棄這筆，避免又把背景距離送進控制器。
                            if geometry is None:
                                continue

                            center_u = geometry.center_u
                            center_v = geometry.center_v
                            dz_m = sample_board_depth(
                                depth_np,
                                polygon=mask_polygon,
                                mask=mask_array,
                                ring_px=6,
                            )
                            if dz_m is None or dz_m <= 0:
                                continue

                            # 目標是圓孔，不使用 bbox 四角 solvePnP。
                            # 以「孔中心視線 + 孔外靶板深度」求相對位置。
                            fx = float(self.camera_matrix[0, 0])
                            fy = float(self.camera_matrix[1, 1])
                            cx = float(self.camera_matrix[0, 2])
                            cy = float(self.camera_matrix[1, 2])
                            dx = (center_u - cx) * dz_m / fx * 100.0
                            # 轉成 +Y 向上，配合原本 pitch 控制的座標約定。
                            dy = -(center_v - cy) * dz_m / fy * 100.0
                            yaw = np.degrees(np.arctan2(dx, dz_m * 100.0))

                            detected_depths_cm.append(dz_m * 100.0)
                            detected_dx_cm.append(dx)
                            detected_dy_cm.append(dy)
                            detected_yaw_deg.append(yaw)
                            valid_boxes.append(
                                (x1, y1, x2, y2, center_u, center_v, dz_m)
                            )

                            # 顯示 mask 重心；藍色輪廓方便確認深度取樣位置。
                            cv2.drawContours(
                                annotated_frame,
                                [geometry.contour],
                                -1,
                                (255, 0, 0),
                                2,
                            )
                            cv2.drawMarker(
                                annotated_frame,
                                (round(center_u), round(center_v)),
                                (0, 255, 0),
                                cv2.MARKER_CROSS,
                                10,
                                2,
                            )
                            cv2.putText(
                                annotated_frame,
                                f"board depth {dz_m:.2f}m",
                                (round(center_u) + 8, round(center_v) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

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

    def _load_zed_calibration(self, zed):
        """Use the calibrated intrinsics for the rectified left image."""
        try:
            camera_info = zed.get_camera_information()
            left = (
                camera_info.camera_configuration
                .calibration_parameters.left_cam
            )
            self.camera_matrix = np.array(
                [[left.fx, 0.0, left.cx],
                 [0.0, left.fy, left.cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            # VIEW.LEFT is rectified, so distortion is intentionally zero here.
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            print(
                "[Vision] 使用 ZED 左相機校正內參："
                f"fx={left.fx:.1f}, fy={left.fy:.1f}, "
                f"cx={left.cx:.1f}, cy={left.cy:.1f}"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"[Vision Warning] 無法讀取 ZED 校正內參，使用預設值：{exc}")

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

"""ZED SDK rectified left image and aligned stereo depth capture."""

from dataclasses import replace
import math

import cv2


class ZedDepthCamera:
    """Capture matching SDK left-image/depth pairs in metres."""

    def __init__(self, config):
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError("缺少 ZED SDK Python 套件 pyzed；可使用 --zed-uvc 切換無深度模式") from exc
        if (config.image_width_px, config.image_height_px) != (1280, 720):
            raise ValueError("ZED SDK 目前使用 HD720，單眼解析度必須為 1280×720")
        self.sl = sl
        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        serial = getattr(config, "chassis_zed_serial", None)
        if serial is not None and hasattr(init, "set_from_serial_number"):
            # Two ZED 2 units are attached; never open the muzzle camera here.
            init.set_from_serial_number(int(serial))
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"無法開啟 ZED SDK 相機：{status}")
        try:
            info = self.camera.get_camera_information()
            opened_serial = getattr(info, "serial_number", None)
            if serial is not None and opened_serial is not None and int(opened_serial) != int(serial):
                raise ValueError(f"開到的 ZED 序號 {opened_serial} 不是車體相機 {serial}")
            left = info.camera_configuration.calibration_parameters.left_cam
            fx, fy, cx, cy = (float(getattr(left, key)) for key in ("fx", "fy", "cx", "cy"))
            if (not all(math.isfinite(v) for v in (fx, fy, cx, cy))
                    or fx <= 0 or fy <= 0 or not 0 <= cx < config.image_width_px
                    or not 0 <= cy < config.image_height_px):
                raise ValueError("ZED SDK 回傳的左眼校正參數無效")
            self.config = replace(config, focal_length_x_px=fx, focal_length_y_px=fy,
                                  camera_principal_x_px=cx, camera_principal_y_px=cy,
                                  camera_distortion_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))
            self.image = sl.Mat()
            self.depth = sl.Mat()
            self.runtime = sl.RuntimeParameters()
            sensors_type = getattr(sl, "SensorsData", None)
            self.sensors = sensors_type() if sensors_type is not None else None
            # Chassis orientation (x, y, z, w) from the ZED IMU at the latest
            # grabbed frame; None when unavailable.  Used by the pose lock.
            self.imu_orientation = None
        except Exception:
            self.camera.close()
            raise

    def read(self, want_depth: bool = True):
        """Grab one frame; ``want_depth=False`` skips the stereo depth compute."""
        self.runtime.enable_depth = bool(want_depth)
        if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            self.imu_orientation = None
            return False, None, None
        self.imu_orientation = self._read_imu_orientation()
        if self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT) != self.sl.ERROR_CODE.SUCCESS:
            return False, None, None
        frame = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR)
        if not want_depth:
            return True, frame, None
        status = self.camera.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)
        depth = self.depth.get_data() if status == self.sl.ERROR_CODE.SUCCESS else None
        if depth is not None and depth.shape[:2] != frame.shape[:2]:
            depth = None
        return True, frame, depth

    def _read_imu_orientation(self):
        if self.sensors is None:
            return None
        try:
            status = self.camera.get_sensors_data(self.sensors, self.sl.TIME_REFERENCE.IMAGE)
            if status != self.sl.ERROR_CODE.SUCCESS:
                return None
            imu = self.sensors.get_imu_data()
            available = imu.is_available
            if callable(available):
                available = available()
            if not available:
                return None
            quaternion = tuple(float(v) for v in imu.get_pose().get_orientation().get())
        except Exception:
            return None
        if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
            return None
        if not 0.5 <= math.sqrt(sum(v * v for v in quaternion)) <= 1.5:
            return None
        return quaternion

    def release(self):
        self.camera.close()

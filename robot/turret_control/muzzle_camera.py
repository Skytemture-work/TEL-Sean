"""Muzzle-mounted ZED 2: upright images, barrel boresight point and preview.

The muzzle ZED is bolted to the turret rotated 90 degrees (lenses stacked
vertically, 120 mm baseline).  Its IMU reads "up" along raw image -x, so the
raw image is rotated 90 degrees clockwise to stand upright.  The SDK "left"
eye is then the upper lens.

Boresight model (team decision 2026-11-14: assume the camera optical axis is
parallel to the barrel until a calibration says otherwise).  Seen from the
camera, the barrel axis runs 66.5 mm to the left of the lens pair and level
with their midpoint, i.e. 60 mm below the upper (reference) lens.  A point on
the barrel axis at distance ``d`` therefore projects to

    u = cx + fx * offset_x / d,   v = cy + fy * offset_y / d

with ``offset_x = -0.0665 m`` and ``offset_y = +0.060 m`` (x right, y down).
Two parallel axes never converge, so the pixel shift shrinks with distance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MuzzleIntrinsics:
    """Pinhole intrinsics of the *upright* (rotated) muzzle image."""

    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_raw(cls, width_px: int, height_px: int, fx: float, fy: float,
                 cx: float, cy: float) -> "MuzzleIntrinsics":
        """Intrinsics after rotating a raw ``width x height`` image 90 deg clockwise.

        A raw pixel (u, v) lands at (height - 1 - v, u).
        """
        return cls(width_px=height_px, height_px=width_px, fx=fy, fy=fx,
                   cx=height_px - 1 - cy, cy=cx)


def upright(image: np.ndarray) -> np.ndarray:
    """Rotate a raw muzzle image (or depth map) so the board stands upright."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def barrel_aim_pixel(intrinsics: MuzzleIntrinsics, distance_m: float,
                     offset_x_m: float, offset_y_m: float) -> Tuple[float, float]:
    """Pixel where the barrel axis meets a surface ``distance_m`` ahead."""
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive")
    return (intrinsics.cx + intrinsics.fx * offset_x_m / distance_m,
            intrinsics.cy + intrinsics.fy * offset_y_m / distance_m)


def barrel_offset_angles_deg(distance_m: float, offset_x_m: float,
                             offset_y_m: float) -> Tuple[float, float]:
    """Angle between camera axis and the barrel point at ``distance_m``."""
    return (math.degrees(math.atan2(offset_x_m, distance_m)),
            math.degrees(math.atan2(offset_y_m, distance_m)))


def draw_boresight(image: np.ndarray, intrinsics: MuzzleIntrinsics,
                   distance_m: Optional[float], fallback_distance_m: float,
                   offset_x_m: float, offset_y_m: float) -> np.ndarray:
    """Overlay the optical centre and the parallax-corrected barrel point."""
    out = image.copy()
    known = distance_m is not None and math.isfinite(distance_m) and distance_m > 0
    distance = distance_m if known else fallback_distance_m
    centre = (int(round(intrinsics.cx)), int(round(intrinsics.cy)))
    u, v = barrel_aim_pixel(intrinsics, distance, offset_x_m, offset_y_m)
    aim = (int(round(u)), int(round(v)))
    cv2.drawMarker(out, centre, (160, 160, 160), cv2.MARKER_TILTED_CROSS, 14, 1, cv2.LINE_AA)
    colour = (0, 220, 255) if known else (0, 140, 255)
    cv2.circle(out, aim, 18, colour, 2, cv2.LINE_AA)
    cv2.line(out, (aim[0] - 30, aim[1]), (aim[0] - 8, aim[1]), colour, 2, cv2.LINE_AA)
    cv2.line(out, (aim[0] + 8, aim[1]), (aim[0] + 30, aim[1]), colour, 2, cv2.LINE_AA)
    cv2.line(out, (aim[0], aim[1] - 30), (aim[0], aim[1] - 8), colour, 2, cv2.LINE_AA)
    cv2.line(out, (aim[0], aim[1] + 8), (aim[0], aim[1] + 30), colour, 2, cv2.LINE_AA)
    yaw_deg, pitch_deg = barrel_offset_angles_deg(distance, offset_x_m, offset_y_m)
    label = (f"BARREL @ {distance:.2f} m{'' if known else ' (assumed)'}  "
             f"shift {u - intrinsics.cx:+.1f},{v - intrinsics.cy:+.1f} px "
             f"({yaw_deg:+.2f},{pitch_deg:+.2f} deg)")
    for thickness, text_colour in ((3, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(out, label, (10, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, text_colour, thickness, cv2.LINE_AA)
    return out


def draw_measurement(image: np.ndarray, expected_px, measured_px, text: str) -> None:
    """In place: expected hole position (cyan), detected hole (magenta), result."""
    if expected_px is not None:
        point = (int(round(expected_px[0])), int(round(expected_px[1])))
        cv2.circle(image, point, 10, (255, 255, 0), 2, cv2.LINE_AA)
    if measured_px is not None:
        point = (int(round(measured_px[0])), int(round(measured_px[1])))
        cv2.drawMarker(image, point, (255, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    if expected_px is not None and measured_px is not None:
        cv2.line(image, tuple(int(round(v)) for v in expected_px),
                 tuple(int(round(v)) for v in measured_px), (255, 0, 255), 1, cv2.LINE_AA)
    for thickness, colour in ((3, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour,
                    thickness, cv2.LINE_AA)


class MuzzleZedCamera:
    """Open the muzzle ZED 2 by serial; image only (no depth, keeps GPU free)."""

    def __init__(self, serial_number: int):
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError("缺少 ZED SDK Python 套件 pyzed") from exc
        self.sl = sl
        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 15
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.set_from_serial_number(int(serial_number))
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"無法開啟砲口 ZED（S/N {serial_number}）：{status}")
        try:
            info = self.camera.get_camera_information()
            if int(info.serial_number) != int(serial_number):
                raise ValueError(f"開到的 ZED 序號 {info.serial_number} 不是砲口相機 {serial_number}")
            config = info.camera_configuration
            left = config.calibration_parameters.left_cam
            self.intrinsics = MuzzleIntrinsics.from_raw(
                int(config.resolution.width), int(config.resolution.height),
                float(left.fx), float(left.fy), float(left.cx), float(left.cy))
            self.image = sl.Mat()
            self.runtime = sl.RuntimeParameters()
        except Exception:
            self.camera.close()
            raise

    def read(self):
        """Return ``(ok, upright_bgr_frame)``."""
        if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return False, None
        if self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT) != self.sl.ERROR_CODE.SUCCESS:
            return False, None
        return True, upright(cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR))

    def release(self):
        self.camera.close()


class MuzzlePreview:
    """Background muzzle capture -> boresight overlay -> JPEG for the dashboard.

    Runs entirely beside the control loop; a failure here only stops the
    second preview and never touches motors, plan or firing gates.
    """

    def __init__(self, camera, publish: Callable[[bytes], None], *, offset_x_m: float,
                 offset_y_m: float, fallback_distance_m: float, fps: float = 10.0,
                 scale: float = 0.5, jpeg_quality: int = 70):
        self._camera = camera
        self._publish = publish
        self._offset = (offset_x_m, offset_y_m)
        self._fallback = fallback_distance_m
        self._period = 1.0 / fps
        self._scale = scale
        self._quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._distance_m: Optional[float] = None
        self._latest = (None, 0.0)  # (upright frame, time.monotonic() at capture)
        self._measurement = None
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="muzzle-preview", daemon=True)
        self._thread.start()

    @property
    def intrinsics(self) -> MuzzleIntrinsics:
        return self._camera.intrinsics

    def latest_frame(self):
        """Newest upright frame and its ``time.monotonic()`` capture time."""
        with self._lock:
            return self._latest

    def set_measurement(self, measurement) -> None:
        """Overlay the latest pre-shot check: ``(expected_px, measured_px, text)``."""
        with self._lock:
            self._measurement = measurement

    def set_distance(self, distance_m: Optional[float]) -> None:
        """Range along the barrel (current target distance), None if unknown."""
        with self._lock:
            self._distance_m = distance_m

    def _run(self) -> None:
        next_due = time.perf_counter()
        while not self._stop.is_set():
            try:
                ok, frame = self._camera.read()
                if not ok:
                    raise RuntimeError("砲口相機取像失敗")
                with self._lock:
                    self._latest = (frame, time.monotonic())
                now = time.perf_counter()
                if now < next_due:
                    continue  # keep draining the camera; publish at the set rate
                next_due = max(next_due + self._period, now)
                with self._lock:
                    distance = self._distance_m
                    measurement = self._measurement
                display = draw_boresight(frame, self._camera.intrinsics, distance,
                                         self._fallback, *self._offset)
                if measurement is not None:
                    draw_measurement(display, *measurement)
                if self._scale != 1.0:
                    display = cv2.resize(display, None, fx=self._scale, fy=self._scale,
                                         interpolation=cv2.INTER_AREA)
                ok_jpg, encoded = cv2.imencode(".jpg", display,
                                               [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                if ok_jpg:
                    self._publish(encoded.tobytes())
            except Exception as exc:  # second preview must never stop control
                self.error = str(exc)
                print(f"砲口相機預覽停止（不影響控制）：{exc}", flush=True)
                return

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

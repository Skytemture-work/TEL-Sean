"""Decouple camera capture / dashboard preview from the YOLO control loop.

The control loop (YOLO -> PnP -> engagement gates -> motors) keeps its exact
one-decision-per-inference structure.  Only two things move to background
threads:

* ``LatestFrameGrabber`` reads the camera continuously and keeps the newest
  frame, so the control loop never waits on a stale buffered frame and the
  preview always has a fresh image.  With ``depth_on_demand`` (ZED SDK) the
  stereo depth is only computed for frames the control loop asks for, so the
  30 FPS preview does not steal GPU time from TensorRT.
* ``LivePreviewPublisher`` encodes the newest camera frame at a fixed rate for
  the dashboard and overlays the annotation layer of the most recent
  inference.  It never touches the motors, the plan or the firing gates.
"""

import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class LatestFrameGrabber:
    """Background camera reader that always exposes the newest frame."""

    def __init__(self, camera, *, has_depth: bool, depth_on_demand: bool = False):
        self._camera = camera
        self._has_depth = has_depth
        self._on_demand = bool(has_depth and depth_on_demand)
        self._cond = threading.Condition()
        self._frame = None
        self._depth = None
        self._seq = 0
        self._captured_at = 0.0
        # Newest frame that carries a depth map (kept separately so later
        # preview-only frames cannot overwrite it before the consumer reads it).
        self._depth_frame = None
        self._depth_map = None
        self._depth_seq = 0
        self._depth_captured_at = 0.0
        # Depth wanted for frames whose seq is greater than this value.
        self._depth_requested_after: Optional[int] = None
        self._failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-grabber", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._on_demand:
                    with self._cond:
                        want_depth = self._depth_requested_after is not None
                    ok, frame, depth = self._camera.read(want_depth=want_depth)
                    if ok and depth is not None:
                        depth = np.array(depth, copy=True)
                elif self._has_depth:
                    ok, frame, depth = self._camera.read()
                    # ZED sl.Mat.get_data() returns a view into a buffer that
                    # the next grab() overwrites; the consumer thread must own
                    # its copy.
                    if ok and depth is not None:
                        depth = np.array(depth, copy=True)
                else:
                    ok, frame = self._camera.read()
                    depth = None
                if not ok:
                    break
                captured_at = time.perf_counter()
                with self._cond:
                    self._frame = frame
                    self._depth = depth
                    self._captured_at = captured_at
                    self._seq += 1
                    if depth is not None:
                        self._depth_frame = frame
                        self._depth_map = depth
                        self._depth_seq = self._seq
                        self._depth_captured_at = captured_at
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._failed = True
                self._cond.notify_all()

    def wait_next(
        self, last_seq: int, timeout: Optional[float] = None, *, need_depth: bool = False
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], int, float]:
        """Block until a frame newer than ``last_seq`` exists.

        Returns ``(ok, frame, depth, seq, captured_at_perf_counter)``.  ``ok``
        is False once the camera has failed/stopped (same meaning as the
        original ``camera.read()`` returning False) or on timeout.

        ``need_depth`` (depth-on-demand mode) waits for a frame grabbed after
        this call with depth enabled, so the depth map is never stale.
        """
        if need_depth and self._on_demand:
            return self._wait_depth_frame(last_seq, timeout)
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._seq > last_seq or self._failed, timeout=timeout
            )
            if not ready or self._seq <= last_seq:
                return False, None, None, last_seq, 0.0
            return True, self._frame, self._depth, self._seq, self._captured_at

    def _wait_depth_frame(self, last_seq, timeout):
        with self._cond:
            after = max(last_seq, self._seq)
            self._depth_requested_after = after
            try:
                ready = self._cond.wait_for(
                    lambda: self._depth_seq > after or self._failed, timeout=timeout
                )
                if not ready or self._depth_seq <= after:
                    return False, None, None, last_seq, 0.0
                return (True, self._depth_frame, self._depth_map, self._depth_seq,
                        self._depth_captured_at)
            finally:
                self._depth_requested_after = None

    def latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._cond:
            return self._frame, self._seq

    @property
    def failed(self) -> bool:
        with self._cond:
            return self._failed

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


class LivePreviewPublisher:
    """Publish fresh camera frames + latest inference overlay to a sink."""

    def __init__(
        self,
        grabber: LatestFrameGrabber,
        publish: Callable[[bytes], None],
        *,
        fps: float,
        frame_transform: Callable[[np.ndarray], np.ndarray] = lambda f: f,
        jpeg_quality: int = 80,
        scale: float = 1.0,
    ):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")
        # Downscale only the published JPEG; overlays are composed at full size.
        self._scale = float(scale)
        self._grabber = grabber
        self._publish = publish
        self._period = 1.0 / fps
        self._transform = frame_transform
        self._quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._overlay = None  # (annotated, base, created_at)
        self._layer = None  # (annotated, mask, created_at)
        self._encode_ms: Optional[float] = None
        self._fps: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-preview", daemon=True)
        self._thread.start()

    def set_overlay(self, annotated: np.ndarray, base: np.ndarray) -> None:
        """Hand over the latest inference display and the frame it was drawn on.

        Cheap for the control loop: only references are stored.  The diff mask
        is computed lazily on the preview thread.
        """
        with self._lock:
            self._overlay = (annotated, base, time.perf_counter())

    def stats(self) -> dict:
        with self._lock:
            return {"preview_fps": self._fps, "preview_jpeg_encode_ms": self._encode_ms}

    def _current_layer(self):
        with self._lock:
            pending, self._overlay = self._overlay, None
        if pending is not None:
            annotated, base, created_at = pending
            if annotated.shape == base.shape:
                mask = np.any(annotated != base, axis=2)
                self._layer = (annotated, mask[..., None], created_at)
            else:
                self._layer = None
        return self._layer

    def compose(self, live: np.ndarray) -> np.ndarray:
        out = live.copy()
        layer = self._current_layer()
        if layer is not None and layer[0].shape == out.shape:
            annotated, mask, created_at = layer
            np.copyto(out, annotated, where=mask)
            age_ms = (time.perf_counter() - created_at) * 1000.0
            label = f"LIVE | vision overlay {age_ms:.0f} ms old"
        else:
            label = "LIVE | waiting for vision"
        h = out.shape[0]
        cv2.putText(out, label, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _run(self) -> None:
        seq = 0
        last_publish = None
        next_due = None
        while not self._stop.is_set():
            ok, frame, _depth, new_seq, _ = self._grabber.wait_next(seq, timeout=0.5)
            if not ok:
                if self._grabber.failed:
                    return
                continue
            seq = new_seq
            now = time.perf_counter()
            if next_due is not None and now < next_due:
                if self._stop.wait(next_due - now):
                    return
                # Publish whatever is newest right now instead of waiting for
                # yet another camera frame.
                frame, seq = self._grabber.latest()
            started_at = time.perf_counter()
            # Fixed schedule measured from publish *start*, so the compose and
            # encode time is not added on top of the period (that capped a
            # 15 FPS target at ~11.5 FPS).  Re-anchor if we fell behind.
            if next_due is None or started_at - next_due > self._period:
                next_due = started_at + self._period
            else:
                next_due += self._period
            try:
                display = self.compose(self._transform(frame))
                if self._scale != 1.0:
                    display = cv2.resize(display, None, fx=self._scale, fy=self._scale,
                                         interpolation=cv2.INTER_AREA)
                started = time.perf_counter()
                ok_jpg, encoded = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                encode_ms = (time.perf_counter() - started) * 1000.0
                if ok_jpg:
                    self._publish(encoded.tobytes())
            except Exception as exc:  # preview must never take down control
                print(f"即時預覽執行緒錯誤（不影響控制）：{exc}", flush=True)
                self._stop.wait(0.5)
                continue
            with self._lock:
                self._encode_ms = encode_ms
                if last_publish is not None:
                    inst = 1.0 / max(started_at - last_publish, 1e-6)
                    self._fps = inst if self._fps is None else 0.8 * self._fps + 0.2 * inst
            last_publish = started_at

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

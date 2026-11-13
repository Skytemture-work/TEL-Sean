import threading
import time
import unittest

import cv2
import numpy as np

from turret_control.live_preview import LatestFrameGrabber, LivePreviewPublisher


class _FakeZed:
    """Mimics ZedDepthCamera: depth is a view into a buffer reused per grab."""

    def __init__(self, frames=None, period=0.005):
        self.count = 0
        self.frames = frames
        self.period = period
        self._depth_buffer = np.zeros((4, 6), dtype=np.float32)

    def read(self):
        time.sleep(self.period)
        if self.frames is not None and self.count >= self.frames:
            return False, None, None
        self.count += 1
        frame = np.full((4, 6, 3), self.count % 256, dtype=np.uint8)
        self._depth_buffer[:] = self.count
        return True, frame, self._depth_buffer


class _FakeUvc:
    def __init__(self):
        self.count = 0

    def read(self):
        time.sleep(0.005)
        self.count += 1
        return True, np.full((4, 6, 3), self.count % 256, dtype=np.uint8)


class LatestFrameGrabberTest(unittest.TestCase):
    def test_returns_strictly_newer_frames_and_owns_depth_copy(self):
        camera = _FakeZed()
        grabber = LatestFrameGrabber(camera, has_depth=True)
        try:
            ok, frame, depth, seq, _ = grabber.wait_next(0, timeout=1.0)
            self.assertTrue(ok)
            value = float(depth[0, 0])
            self.assertEqual(value, float(frame[0, 0, 0]))
            ok2, _, _, seq2, _ = grabber.wait_next(seq, timeout=1.0)
            self.assertTrue(ok2)
            self.assertGreater(seq2, seq)
            # The camera buffer has been overwritten since; our copy must not.
            self.assertEqual(float(depth[0, 0]), value)
        finally:
            grabber.stop()

    def test_camera_failure_ends_wait_like_camera_read(self):
        grabber = LatestFrameGrabber(_FakeZed(frames=2), has_depth=True)
        try:
            seq = 0
            results = []
            for _ in range(5):
                ok, _, _, seq, _ = grabber.wait_next(seq, timeout=1.0)
                results.append(ok)
                if not ok:
                    break
            self.assertFalse(results[-1])
            self.assertTrue(grabber.failed)
        finally:
            grabber.stop()

    def test_uvc_camera_has_no_depth(self):
        grabber = LatestFrameGrabber(_FakeUvc(), has_depth=False)
        try:
            ok, frame, depth, _, _ = grabber.wait_next(0, timeout=1.0)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertIsNone(depth)
        finally:
            grabber.stop()


class _FakeZedOnDemand(_FakeZed):
    def __init__(self):
        super().__init__(period=0.01)
        self.depth_grabs = 0
        self.plain_grabs = 0

    def read(self, want_depth=True):
        ok, frame, depth = super().read()
        if want_depth:
            self.depth_grabs += 1
            return ok, frame, depth
        self.plain_grabs += 1
        return ok, frame, None


class DepthOnDemandTest(unittest.TestCase):
    def test_depth_only_computed_when_control_loop_asks(self):
        camera = _FakeZedOnDemand()
        grabber = LatestFrameGrabber(camera, has_depth=True, depth_on_demand=True)
        try:
            time.sleep(0.15)
            self.assertEqual(camera.depth_grabs, 0)  # preview-only frames
            ok, frame, depth, seq, _ = grabber.wait_next(0, timeout=1.0, need_depth=True)
            self.assertTrue(ok)
            self.assertIsNotNone(depth)
            self.assertEqual(float(depth[0, 0]), float(frame[0, 0, 0]))  # same grab
            # Frame was grabbed after the request, not an old depth frame.
            ok2, _, depth2, seq2, _ = grabber.wait_next(seq, timeout=1.0, need_depth=True)
            self.assertTrue(ok2)
            self.assertGreater(seq2, seq)
            self.assertNotEqual(float(depth2[0, 0]), float(depth[0, 0]))
            time.sleep(0.15)
            grabs_after = camera.depth_grabs
            time.sleep(0.15)
            # At most one in-flight depth grab after the request was served.
            self.assertLessEqual(camera.depth_grabs - grabs_after, 1)
            self.assertGreater(camera.plain_grabs, 10)
        finally:
            grabber.stop()


class LivePreviewPublisherTest(unittest.TestCase):
    def test_throttled_rate_reaches_target_fps(self):
        camera = _FakeUvc()
        camera_period = 0.033

        def slow_read():
            time.sleep(camera_period - 0.005)
            return _FakeUvc.read(camera)

        camera.read = slow_read
        grabber = LatestFrameGrabber(camera, has_depth=False)
        count = []
        preview = LivePreviewPublisher(grabber, lambda b: count.append(1), fps=15)
        try:
            time.sleep(2.0)
        finally:
            preview.stop()
            grabber.stop()
        self.assertGreaterEqual(len(count), 26)  # ~30 at 15 FPS, was ~24 before fix

    def test_processing_time_is_not_added_to_period(self):
        camera = _FakeUvc()

        def slow_read():
            time.sleep(0.028)
            return _FakeUvc.read(camera)

        camera.read = slow_read
        grabber = LatestFrameGrabber(camera, has_depth=False)
        count = []

        def slow_sink(_data):  # simulate ~20 ms compose + JPEG encode on AGX
            time.sleep(0.02)
            count.append(1)

        preview = LivePreviewPublisher(grabber, slow_sink, fps=15)
        try:
            time.sleep(3.0)
        finally:
            preview.stop()
            grabber.stop()
        # 15 FPS for 3 s = 45; the old end-to-start scheduling gave ~34.
        self.assertGreaterEqual(len(count), 41)

    def test_publishes_fresh_frames_faster_than_slow_inference(self):
        grabber = LatestFrameGrabber(_FakeUvc(), has_depth=False)
        published = []
        lock = threading.Lock()

        def sink(data):
            with lock:
                published.append(data)

        preview = LivePreviewPublisher(grabber, sink, fps=50)
        try:
            time.sleep(0.4)
        finally:
            preview.stop()
            grabber.stop()
        # No overlay was ever set (inference "never finished"), yet the
        # dashboard still received a steady stream of live JPEG frames.
        self.assertGreaterEqual(len(published), 8)
        self.assertTrue(all(p[:2] == b"\xff\xd8" for p in published))
        self.assertIsNotNone(preview.stats()["preview_fps"])

    def test_published_jpeg_is_downscaled(self):
        grabber = LatestFrameGrabber(_FakeUvc(), has_depth=False)
        frames = []
        preview = LivePreviewPublisher(grabber, frames.append, fps=20, scale=0.5)
        try:
            time.sleep(0.4)
        finally:
            preview.stop()
            grabber.stop()
        self.assertTrue(frames)
        image = cv2.imdecode(np.frombuffer(frames[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
        source = _FakeUvc().read()[1]
        self.assertEqual(image.shape[:2], (source.shape[0] // 2, source.shape[1] // 2))
        with self.assertRaises(ValueError):
            LivePreviewPublisher(grabber, frames.append, fps=10, scale=0.0)

    def test_overlay_annotations_are_copied_onto_live_frame(self):
        grabber = LatestFrameGrabber(_FakeUvc(), has_depth=False)
        preview = LivePreviewPublisher(grabber, lambda _b: None, fps=1)
        try:
            base = np.zeros((200, 300, 3), dtype=np.uint8)
            annotated = base.copy()
            cv2.rectangle(annotated, (10, 10), (20, 20), (0, 255, 0), 1)
            preview.set_overlay(annotated, base)
            live = np.full((200, 300, 3), 100, dtype=np.uint8)
            out = preview.compose(live)
            self.assertEqual(tuple(out[10, 15]), (0, 255, 0))  # annotation kept
            self.assertEqual(tuple(out[15, 15]), (100, 100, 100))  # live pixel
            self.assertEqual(tuple(live[10, 15]), (100, 100, 100))  # input untouched
        finally:
            preview.stop()
            grabber.stop()


if __name__ == "__main__":
    unittest.main()

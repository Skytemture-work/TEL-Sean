import math
import time
import unittest

import numpy as np

from turret_control.config import TurretConfig
from turret_control.dashboard import DashboardState
from turret_control.muzzle_camera import (
    MuzzleIntrinsics,
    MuzzlePreview,
    barrel_aim_pixel,
    barrel_offset_angles_deg,
    draw_boresight,
    upright,
)

# Measured S/N 20681321 HD720 left-eye calibration (raw, sideways image).
RAW = dict(width_px=1280, height_px=720, fx=532.7, fy=532.7, cx=637.4, cy=358.7)


class UprightImageTests(unittest.TestCase):
    def test_raw_left_edge_becomes_the_top(self):
        raw = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw[:, 0] = 255  # left column = physical "up" per the muzzle IMU
        rotated = upright(raw)
        self.assertEqual(rotated.shape, (1280, 720, 3))
        self.assertTrue(np.all(rotated[0] == 255))
        self.assertFalse(np.any(rotated[1:] == 255))

    def test_intrinsics_follow_the_rotation(self):
        intrinsics = MuzzleIntrinsics.from_raw(**RAW)
        self.assertEqual((intrinsics.width_px, intrinsics.height_px), (720, 1280))
        # The principal point moves exactly like a pixel: (u, v) -> (H-1-v, u).
        raw = np.zeros((720, 1280), dtype=np.uint8)
        raw[359, 637] = 1
        v, u = np.argwhere(upright(raw))[0]
        self.assertEqual((u, v), (720 - 1 - 359, 637))
        self.assertAlmostEqual(intrinsics.cx, 720 - 1 - RAW["cy"])
        self.assertAlmostEqual(intrinsics.cy, RAW["cx"])


class BoresightTests(unittest.TestCase):
    def setUp(self):
        config = TurretConfig()
        self.offset = (config.muzzle_barrel_offset_x_m, config.muzzle_barrel_offset_y_m)
        self.intrinsics = MuzzleIntrinsics.from_raw(**RAW)

    def test_barrel_point_is_left_of_and_below_the_optical_centre(self):
        u, v = barrel_aim_pixel(self.intrinsics, 5.0, *self.offset)
        self.assertAlmostEqual(u - self.intrinsics.cx, -532.7 * 0.0665 / 5.0, places=6)
        self.assertAlmostEqual(v - self.intrinsics.cy, 532.7 * 0.060 / 5.0, places=6)
        self.assertLess(u, self.intrinsics.cx)
        self.assertGreater(v, self.intrinsics.cy)

    def test_parallax_shrinks_with_distance_and_matches_angles(self):
        shifts = [self.intrinsics.cx - barrel_aim_pixel(self.intrinsics, d, *self.offset)[0]
                  for d in (2.0, 5.0, 8.0)]
        self.assertGreater(shifts[0], shifts[1])
        self.assertGreater(shifts[1], shifts[2])
        yaw_deg, pitch_deg = barrel_offset_angles_deg(5.0, *self.offset)
        self.assertAlmostEqual(yaw_deg, -math.degrees(math.atan(0.0665 / 5.0)))
        self.assertAlmostEqual(pitch_deg, math.degrees(math.atan(0.060 / 5.0)))

    def test_rejects_invalid_distance(self):
        for distance in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                barrel_aim_pixel(self.intrinsics, distance, *self.offset)

    def test_overlay_uses_fallback_when_range_unknown(self):
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        for distance in (None, float("nan"), 3.0):
            out = draw_boresight(image, self.intrinsics, distance, 5.0, *self.offset)
            self.assertEqual(out.shape, image.shape)
            self.assertTrue(np.any(out))
        self.assertFalse(np.any(image))  # input frame untouched


class MuzzlePreviewTests(unittest.TestCase):
    def test_publishes_upright_jpeg_to_the_muzzle_channel(self):
        class FakeCamera:
            intrinsics = MuzzleIntrinsics.from_raw(**RAW)

            def read(self):
                time.sleep(0.01)
                return True, np.zeros((1280, 720, 3), dtype=np.uint8)

        state = DashboardState()
        preview = MuzzlePreview(FakeCamera(), state.update_muzzle_frame, offset_x_m=-0.0665,
                                offset_y_m=0.06, fallback_distance_m=5.0, fps=20.0)
        try:
            frame, version = state.wait_frame(0, timeout=2.0, channel="muzzle")
            self.assertTrue(frame.startswith(b"\xff\xd8"))
            self.assertGreater(version, 0)
            self.assertIsNone(state.frame_snapshot())  # main channel untouched
        finally:
            preview.stop()

    def test_camera_failure_stops_only_the_preview(self):
        class BrokenCamera:
            intrinsics = MuzzleIntrinsics.from_raw(**RAW)

            def read(self):
                return False, None

        preview = MuzzlePreview(BrokenCamera(), lambda _frame: None, offset_x_m=-0.0665,
                                offset_y_m=0.06, fallback_distance_m=5.0)
        preview.stop()
        self.assertIn("取像失敗", preview.error)



class StartMuzzlePreviewTests(unittest.TestCase):
    def test_open_failure_only_disables_the_muzzle_view(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from turret_control import run_board_real as real

        args = SimpleNamespace(preview_fps=15.0, preview_scale=0.5, preview_quality=70)
        with patch("turret_control.muzzle_camera.MuzzleZedCamera",
                   side_effect=RuntimeError("no camera")), patch("builtins.print"):
            self.assertEqual(real._start_muzzle_preview(TurretConfig(), DashboardState(), args),
                             (None, None))

if __name__ == "__main__":
    unittest.main()

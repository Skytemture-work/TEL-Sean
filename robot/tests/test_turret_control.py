import math
import unittest

from turret_control.aiming import AimingController, TargetPixel
from turret_control.config import TurretConfig


class AimingControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig(
            stable_frames_required=3,
            aim_tolerance_rad=math.radians(2.0),
        )
        self.controller = AimingController(self.config)

    def test_image_center_has_zero_error(self):
        command = self.controller.update(
            TargetPixel(
                self.config.camera_center_x_px,
                self.config.camera_center_y_px,
            )
        )
        self.assertAlmostEqual(command.yaw_error_rad, 0.0)
        self.assertAlmostEqual(command.pitch_error_rad, 0.0)
        self.assertFalse(command.fire_allowed)

    def test_target_right_and_above_produces_positive_errors(self):
        command = self.controller.update(TargetPixel(800.0, 250.0))
        self.assertGreater(command.yaw_error_rad, 0.0)
        self.assertGreater(command.pitch_error_rad, 0.0)

    def test_firing_requires_stable_frames(self):
        target = TargetPixel(
            self.config.camera_center_x_px,
            self.config.camera_center_y_px,
        )
        self.assertFalse(self.controller.update(target).fire_allowed)
        self.assertFalse(self.controller.update(target).fire_allowed)
        self.assertTrue(self.controller.update(target).fire_allowed)

    def test_no_target_never_allows_firing(self):
        command = self.controller.update(None)
        self.assertFalse(command.has_target)
        self.assertFalse(command.fire_allowed)
        self.assertIsNone(command.target_yaw_rad)
        self.assertIsNone(command.target_pitch_rad)

    def test_angle_is_limited(self):
        command = self.controller.update(TargetPixel(100000.0, -100000.0))
        self.assertEqual(command.target_yaw_rad, self.config.yaw_max_rad)
        self.assertEqual(command.target_pitch_rad, self.config.pitch_max_rad)


if __name__ == "__main__":
    unittest.main()

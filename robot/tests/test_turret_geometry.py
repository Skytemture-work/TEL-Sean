import math
import unittest

from turret_control.config import TurretConfig
from turret_control.geometry import (
    aim_angles_from_target,
    normalized_image_coordinates,
    target_to_axis_aim,
)


class TurretGeometryTests(unittest.TestCase):
    def test_center_target_has_zero_angles_when_axes_are_coincident(self):
        config = TurretConfig(
            image_width_px=1280,
            image_height_px=720,
            yaw_zero_rad=0.0,
            yaw_sign=1,
            pitch_zero_rad=0.0,
            pitch_sign=1,
            compensate_gravity=False,
        )
        yaw, pitch, result = aim_angles_from_target(640.0, 360.0, 5.0, config)
        self.assertAlmostEqual(yaw, 0.0, places=7)
        self.assertAlmostEqual(pitch, 0.0, places=7)
        self.assertAlmostEqual(result.pitch_axis_distance_m, 5.0, places=7)

    def test_pixel_direction_maps_to_absolute_axis_angles(self):
        config = TurretConfig(
            image_width_px=1280,
            image_height_px=720,
            yaw_zero_rad=0.0,
            yaw_sign=1,
            pitch_zero_rad=0.0,
            pitch_sign=1,
            compensate_gravity=False,
        )
        yaw, pitch, _ = aim_angles_from_target(990.0, 10.0, 5.0, config)
        self.assertGreater(yaw, 0.0)
        self.assertGreater(pitch, 0.0)
        self.assertAlmostEqual(yaw, math.atan2(350.0, 700.0), places=7)
        self.assertAlmostEqual(
            pitch,
            math.atan2(350.0, math.hypot(350.0, 700.0)),
            places=7,
        )

    def test_axis_offsets_change_the_required_angles(self):
        config = TurretConfig(
            image_width_px=1280,
            image_height_px=720,
            yaw_axis_offset_x_m=0.10,
            pitch_axis_offset_y_m=0.20,
            compensate_gravity=False,
        )
        result = target_to_axis_aim(640.0, 360.0, 5.0, config)
        self.assertLess(result.yaw_rad, 0.0)
        self.assertLess(result.pitch_rad, 0.0)

    def test_calibrated_distortion_is_removed_before_angle_conversion(self):
        config = TurretConfig(
            focal_length_x_px=710.0,
            focal_length_y_px=705.0,
            camera_principal_x_px=631.0,
            camera_principal_y_px=357.0,
            camera_distortion_coeffs=(0.12, -0.04, 0.001, -0.002, 0.01),
        )
        expected_x = 0.24
        expected_y_down = -0.16
        radius2 = expected_x * expected_x + expected_y_down * expected_y_down
        radial = 1.0 + 0.12 * radius2 - 0.04 * radius2**2 + 0.01 * radius2**3
        distorted_x = (
            expected_x * radial
            + 2.0 * 0.001 * expected_x * expected_y_down
            - 0.002 * (radius2 + 2.0 * expected_x * expected_x)
        )
        distorted_y = (
            expected_y_down * radial
            + 0.001 * (radius2 + 2.0 * expected_y_down * expected_y_down)
            - 2.0 * 0.002 * expected_x * expected_y_down
        )
        u_px = config.camera_center_x_px + distorted_x * config.focal_length_x_px
        v_px = config.camera_center_y_px + distorted_y * config.focal_length_y_px

        normalized_x, normalized_y_up = normalized_image_coordinates(
            u_px,
            v_px,
            config,
        )

        self.assertAlmostEqual(normalized_x, expected_x, places=7)
        self.assertAlmostEqual(normalized_y_up, -expected_y_down, places=7)


if __name__ == "__main__":
    unittest.main()

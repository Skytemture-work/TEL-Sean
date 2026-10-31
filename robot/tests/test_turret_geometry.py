import math
import unittest
from dataclasses import replace

from turret_control.aiming import AimingController, TargetPixel
from turret_control.competition_geometry import with_competition_geometry
from turret_control.config import TurretConfig
from turret_control.geometry import (
    aim_angles_from_target,
    normalized_image_coordinates,
    target_to_axis_aim,
)


class TurretGeometryTests(unittest.TestCase):
    def test_competition_geometry_recovers_known_forward_projected_rays(self):
        config = with_competition_geometry(TurretConfig(
            yaw_zero_rad=0.0, yaw_sign=1, pitch_zero_rad=0.0, pitch_sign=1,
            yaw_min_rad=-1.0, yaw_max_rad=1.0, pitch_min_rad=-1.0, pitch_max_rad=1.0,
        ))
        controller = AimingController(config)
        self.assertFalse(config.compensate_gravity)
        # Independent forward model: CAD home dimensions first rotate to the
        # desired pitch; the whole assembly then rotates about the yaw axis.
        for yaw_deg, pitch_deg, distance in ((0, 0, 2), (15, 20, 5), (-22, 30, 8)):
            with self.subTest(yaw=yaw_deg, pitch=pitch_deg, distance=distance):
                yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
                offset_y = (0.0965 - 0.150) / math.sqrt(2.0)
                offset_z = (0.0965 + 0.150) / math.sqrt(2.0)
                level_y = 0.0721 + math.cos(pitch) * offset_y + math.sin(pitch) * (offset_z + distance)
                level_z = -math.sin(pitch) * offset_y + math.cos(pitch) * (offset_z + distance)
                x = 0.060 + math.sin(yaw) * level_z
                y = -0.001998 + level_y
                z = -0.101662 + math.cos(yaw) * level_z
                c, s = math.cos(math.radians(20)), math.sin(math.radians(20))
                camera_y, camera_z = c * y - s * z, s * y + c * z
                u = config.camera_center_x_px + config.focal_length_x_px * x / camera_z
                v = config.camera_center_y_px - config.focal_length_y_px * camera_y / camera_z
                command = controller.update(TargetPixel(u, v, distance_m=camera_z))
                self.assertTrue(command.target_reachable, command.blocked_reason)
                self.assertAlmostEqual(command.target_yaw_rad, yaw, places=7)
                self.assertAlmostEqual(command.target_pitch_rad, pitch, places=7)

    def test_rotating_exit_rejects_a_target_behind_the_exit(self):
        config = replace(TurretConfig(), pitch_exit_offset_z_m=1.0, compensate_gravity=False)
        with self.assertRaisesRegex(ValueError, "behind the launch exit"):
            target_to_axis_aim(config.camera_center_x_px, config.camera_center_y_px, 0.5, config)

    def test_gravity_and_rotating_exit_cannot_be_combined_silently(self):
        config = replace(with_competition_geometry(TurretConfig()), compensate_gravity=True)
        with self.assertRaisesRegex(ValueError, "gravity with a rotating exit offset"):
            config.validate()

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

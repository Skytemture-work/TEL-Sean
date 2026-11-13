import math
import unittest
from dataclasses import replace

from turret_control.aiming import AimCommand, AimingController, TargetPixel
from turret_control.board import BoardTarget, EngagementPlan
from turret_control.competition_geometry import with_competition_geometry
from turret_control.config import TurretConfig, with_recorded_mechanism_geometry
from turret_control.engagement import BoardEngagement
from turret_control.run_board_real import _parse_pitch_trims


def _field_config():
    return with_recorded_mechanism_geometry(with_competition_geometry(TurretConfig()))


class AimingPitchTrimTests(unittest.TestCase):
    def _command(self, config, trim_rad):
        aiming = AimingController(config)
        target = TargetPixel(640.0, 300.0, 1.0, 6.0, pitch_trim_rad=trim_rad)
        return aiming.update(target, config.yaw_zero_rad, config.pitch_zero_rad)

    def test_positive_trim_raises_the_muzzle_by_that_angle(self):
        config = _field_config()
        base = self._command(config, 0.0)
        trimmed = self._command(config, math.radians(1.0))
        self.assertTrue(base.target_reachable, base.blocked_reason)
        self.assertTrue(trimmed.target_reachable, trimmed.blocked_reason)
        # motor = zero + pitch_sign * physical elevation
        physical_change = (trimmed.target_pitch_rad - base.target_pitch_rad) * config.pitch_sign
        self.assertAlmostEqual(physical_change, math.radians(1.0), places=9)
        self.assertAlmostEqual(trimmed.target_yaw_rad, base.target_yaw_rad, places=12)

    def test_trim_cannot_push_past_mechanical_limits(self):
        config = _field_config()
        # Physical elevation is bounded by the motor-space pitch limits; a huge
        # trim must be reported unreachable, never clamped and fired.
        command = self._command(config, math.radians(80.0))
        self.assertFalse(command.target_reachable)
        self.assertFalse(command.fire_allowed)
        self.assertGreaterEqual(command.target_pitch_rad, config.pitch_min_rad)
        self.assertLessEqual(command.target_pitch_rad, config.pitch_max_rad)


class _RecordingTurret:
    def __init__(self):
        self.pixels = []

    def update(self, pixel):
        self.pixels.append(pixel)
        return AimCommand(has_target=pixel is not None, target_yaw_rad=0.0,
                          target_pitch_rad=0.0, stable=False, fire_allowed=False)


class EngagementPitchTrimTests(unittest.TestCase):
    def test_trim_applies_only_to_the_configured_hole(self):
        turret = _RecordingTurret()
        plan = EngagementPlan([
            BoardTarget(100, 100, 20, 20, hole_id="H01", distance_m=5.5),
            BoardTarget(100, 50, 10, 10, hole_id="H10", distance_m=6.2),
        ])
        driver = BoardEngagement(TurretConfig(), turret, plan,
                                 pitch_trim_rad={"H10": math.radians(0.8)})
        driver.update()
        self.assertEqual(turret.pixels[-1].pitch_trim_rad, 0.0)
        plan.skip()
        driver.update()
        self.assertAlmostEqual(turret.pixels[-1].pitch_trim_rad, math.radians(0.8))
        self.assertEqual(turret.pixels[-1].distance_m, 6.2)


class ParsePitchTrimTests(unittest.TestCase):
    def test_parses_and_normalises_hole_ids(self):
        self.assertEqual(_parse_pitch_trims(["H10=0.8", "h11=+1", "H3=-0.5"]),
                         {"H10": 0.8, "H11": 1.0, "H03": -0.5})
        self.assertEqual(_parse_pitch_trims([]), {})

    def test_rejects_bad_input(self):
        for bad in (["H10"], ["H13=1"], ["X1=1"], ["H10=abc"], ["H10=nan"],
                    ["H10=3.5"], ["H10=1", "H10=2"]):
            with self.assertRaises(ValueError, msg=bad):
                _parse_pitch_trims(bad)


class BoardLeanDiagnosticTests(unittest.TestCase):
    """A vertical board reads 0 lean only when the camera pitch is configured right."""

    @staticmethod
    def _pose_for_vertical_board(true_camera_pitch_deg):
        import cv2
        import numpy as np
        from types import SimpleNamespace
        t = math.radians(true_camera_pitch_deg)
        level_to_camera = np.array([[1, 0, 0], [0, -math.cos(t), math.sin(t)],
                                    [0, math.sin(t), math.cos(t)]])
        board_in_level = np.column_stack(([1, 0, 0], [0, -1, 0], [0, 0, 1]))
        rvec, _ = cv2.Rodrigues(level_to_camera @ board_in_level)
        return SimpleNamespace(rvec=rvec)

    def test_reads_zero_for_correct_camera_pitch(self):
        from turret_control.board_pose import board_lean_deg
        pose = self._pose_for_vertical_board(20.0)
        self.assertAlmostEqual(board_lean_deg(pose, math.radians(20.0)), 0.0, places=6)

    def test_mean_lean_reveals_camera_pitch_error(self):
        from turret_control.board_pose import board_lean_deg
        pose = self._pose_for_vertical_board(21.0)  # camera really tilted 21 deg
        lean = board_lean_deg(pose, math.radians(20.0))  # config says 20
        self.assertAlmostEqual(lean, -1.0, places=6)
        self.assertAlmostEqual(20.0 - lean, 21.0, places=6)  # documented recovery


if __name__ == "__main__":
    unittest.main()

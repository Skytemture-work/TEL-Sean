import math
import unittest
from unittest.mock import patch

import numpy as np

from turret_control import scoring_light
from turret_control.aiming import AimingController, TargetPixel
from turret_control.competition_geometry import with_competition_geometry
from turret_control.config import TurretConfig, with_recorded_mechanism_geometry
from turret_control.scoring_light import ScoreRoundController
from turret_control.scoring_strategy import build_max_score_strategy
from test_scoring_strategy import _board_plan


class LightAutoStartTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.round = ScoreRoundController(duration_s=60.0, confirm_frames=3,
                                          clock=lambda: self.now)
        self.frame = np.zeros((10, 10, 3), np.uint8)

    def step(self, scores, dt=0.4):
        self.now += dt
        with patch.object(scoring_light, "detect_score_light_scores", return_value=scores):
            return self.round.update(self.frame)

    def test_stays_still_until_light_then_times_window_from_first_sighting(self):
        self.round.arm_auto_start()
        dark = {"H10": 0.0, "H11": 0.01, "H12": 0.0}
        for _ in range(10):
            state = self.step(dark)
            self.assertEqual(state.phase, "armed")
            self.assertIsNone(state.started_at)
        lit = {"H10": 0.02, "H11": 0.6, "H12": 0.01}
        first_seen = self.now + 0.4
        self.assertEqual(self.step(lit).phase, "armed")
        self.assertEqual(self.step(lit).phase, "armed")
        state = self.step(lit)
        self.assertEqual(state.phase, "active")
        self.assertEqual(state.active_hole_id, "H11")
        self.assertAlmostEqual(state.started_at, first_seen)
        self.assertAlmostEqual(state.expires_at, first_seen + 60.0)
        self.assertEqual(self.round.round_id, 1)

    def test_flicker_or_ambiguous_light_does_not_start(self):
        self.round.arm_auto_start()
        for scores in ({"H10": 0.6, "H11": 0.0, "H12": 0.0},
                       {"H10": 0.0, "H11": 0.0, "H12": 0.0},
                       {"H10": 0.6, "H11": 0.58, "H12": 0.0},
                       {"H10": 0.6, "H11": 0.0, "H12": 0.0}):
            self.assertEqual(self.step(scores).phase, "armed")


class LowerSerpentineTests(unittest.TestCase):
    def test_each_cell_gets_two_consecutive_balls_and_moves_to_a_neighbour(self):
        strategy = build_max_score_strategy(_board_plan(), "H10", "lower")
        ids = strategy.hole_ids
        self.assertEqual(len(ids), 18)
        self.assertTrue(all(ids[i] == ids[i + 1] for i in range(0, 18, 2)))
        self.assertEqual(set(ids), {f"H{n:02d}" for n in range(1, 10)})
        # Grid numbering: column = (n-1)//3, row = (n-1)%3; every move is one step.
        cells = [int(h[1:]) - 1 for h in ids[::2]]
        for a, b in zip(cells, cells[1:]):
            self.assertEqual(abs(a // 3 - b // 3) + abs(a % 3 - b % 3), 1, (a, b))


class PerHoleToleranceTests(unittest.TestCase):
    def test_tighter_tolerance_blocks_fire_with_one_degree_servo_error(self):
        config = with_recorded_mechanism_geometry(with_competition_geometry(TurretConfig()))
        results = {}
        for name, tolerance in (("default", None), ("small", math.radians(0.25))):
            aiming = AimingController(config)
            target = TargetPixel(640.0, 300.0, 1.0, 6.0, aim_tolerance_rad=tolerance)
            first = aiming.update(target, config.yaw_zero_rad, config.pitch_zero_rad)
            yaw = first.target_yaw_rad + math.radians(1.0)
            for _ in range(config.stable_frames_required + 1):
                command = aiming.update(target, yaw, first.target_pitch_rad)
            results[name] = command.fire_allowed
        self.assertTrue(results["default"])   # 1 deg < 2 deg default
        self.assertFalse(results["small"])    # 1 deg > 0.25 deg

    def test_tolerance_can_only_tighten(self):
        config = with_recorded_mechanism_geometry(with_competition_geometry(TurretConfig()))
        aiming = AimingController(config)
        target = TargetPixel(640.0, 300.0, 1.0, 6.0, aim_tolerance_rad=math.radians(10.0))
        first = aiming.update(target, config.yaw_zero_rad, config.pitch_zero_rad)
        for _ in range(config.stable_frames_required + 1):
            command = aiming.update(target, first.target_yaw_rad + math.radians(3.0),
                                    first.target_pitch_rad)
        self.assertFalse(command.fire_allowed)  # still bounded by the 2 deg default


if __name__ == "__main__":
    unittest.main()

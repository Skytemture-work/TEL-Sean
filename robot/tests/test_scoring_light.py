import unittest
from unittest.mock import patch

from turret_control.scoring_light import ScoreRoundController


class ScoringLightRoundTests(unittest.TestCase):
    def test_requires_consecutive_clear_winner_then_locks(self):
        controller = ScoreRoundController(
            duration_s=60.0,
            confirm_frames=3,
            min_score=0.12,
            min_margin=0.05,
            clock=lambda: 0.0,
        )
        controller.start_round(now=0.0)
        scores = {"H10": 0.03, "H11": 0.55, "H12": 0.04}

        with patch(
            "turret_control.scoring_light.detect_score_light_scores",
            return_value=scores,
        ):
            self.assertIsNone(controller.update(object(), now=0.1).active_hole_id)
            self.assertIsNone(controller.update(object(), now=0.2).active_hole_id)
            state = controller.update(object(), now=0.3)

        self.assertEqual(state.phase, "active")
        self.assertEqual(state.active_hole_id, "H11")

    def test_similar_red_scores_do_not_lock(self):
        controller = ScoreRoundController(
            confirm_frames=2,
            min_score=0.12,
            min_margin=0.05,
            clock=lambda: 0.0,
        )
        controller.start_round(now=0.0)
        scores = {"H10": 0.30, "H11": 0.28, "H12": 0.02}

        with patch(
            "turret_control.scoring_light.detect_score_light_scores",
            return_value=scores,
        ):
            controller.update(object(), now=0.1)
            state = controller.update(object(), now=0.2)

        self.assertEqual(state.phase, "waiting")
        self.assertIsNone(state.active_hole_id)

    def test_locked_hole_does_not_switch_and_expires_at_60_seconds(self):
        controller = ScoreRoundController(
            duration_s=60.0,
            confirm_frames=1,
            min_score=0.12,
            min_margin=0.05,
            clock=lambda: 0.0,
        )
        controller.start_round(now=0.0)

        with patch(
            "turret_control.scoring_light.detect_score_light_scores",
            return_value={"H10": 0.60, "H11": 0.02, "H12": 0.01},
        ):
            locked = controller.update(object(), now=0.1)
        with patch(
            "turret_control.scoring_light.detect_score_light_scores",
            return_value={"H10": 0.01, "H11": 0.02, "H12": 0.70},
        ):
            unchanged = controller.update(object(), now=10.0)
            expired = controller.update(object(), now=60.0)

        self.assertEqual(locked.active_hole_id, "H10")
        self.assertEqual(unchanged.active_hole_id, "H10")
        self.assertEqual(expired.phase, "expired")
        self.assertIsNone(expired.active_hole_id)


if __name__ == "__main__":
    unittest.main()

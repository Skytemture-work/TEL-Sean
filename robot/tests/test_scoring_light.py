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


class ScoringLightDetectionTests(unittest.TestCase):
    def _board(self):
        import cv2
        import numpy as np

        from turret_control.official_board import (
            BOARD_HEIGHT_CM, BOARD_WIDTH_CM, hole_position_for_cell)

        # 1 px per cm, board fills the frame: centres are easy to draw.
        frame = np.full((int(BOARD_HEIGHT_CM), int(BOARD_WIDTH_CM), 3), 60, np.uint8)
        quad = np.float32([[0, 0], [BOARD_WIDTH_CM, 0],
                           [BOARD_WIDTH_CM, BOARD_HEIGHT_CM], [0, BOARD_HEIGHT_CM]])
        centres = []
        for column in range(3):
            x, y = hole_position_for_cell(column, 0)
            centre = (round(x * BOARD_WIDTH_CM), round(y * BOARD_HEIGHT_CM))
            centres.append(centre)
            cv2.circle(frame, centre, 10, (20, 20, 20), cv2.FILLED)  # black hole
            cv2.circle(frame, centre, 16, (235, 240, 245), 2)          # white LED strip
        return frame, quad, centres

    def test_overexposed_strip_with_red_hole_is_detected(self):
        import cv2

        from turret_control.scoring_light import detect_score_light_scores

        frame, quad, centres = self._board()
        # Official field: the lit strip saturates to white, its light turns
        # the hole itself red.
        cv2.circle(frame, centres[1], 16, (250, 250, 255), 2)
        cv2.circle(frame, centres[1], 9, (60, 60, 200), cv2.FILLED)

        scores = detect_score_light_scores(frame, None, quad)

        self.assertGreater(scores["H11"], 0.5)
        self.assertLess(scores["H10"], 0.05)
        self.assertLess(scores["H12"], 0.05)

    def test_red_strip_still_detected(self):
        import cv2

        from turret_control.scoring_light import detect_score_light_scores

        frame, quad, centres = self._board()
        cv2.circle(frame, centres[0], 16, (30, 30, 230), 3)

        scores = detect_score_light_scores(frame, None, quad)

        self.assertGreater(scores["H10"], 0.12)
        self.assertLess(scores["H11"], 0.05)


if __name__ == "__main__":
    unittest.main()

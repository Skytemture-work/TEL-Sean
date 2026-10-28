import unittest

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.official_board import hole_id_for_cell
from turret_control.scoring_strategy import (
    LOWER_LINE_FIRST_IDS,
    append_reallocated_grid_shots,
    build_max_score_strategy,
)


def _board_plan() -> EngagementPlan:
    targets = []
    for row_from_top in range(4):
        for column in range(3):
            hole_id = hole_id_for_cell(column, row_from_top)
            size = 20.0 if row_from_top == 0 else 40.0
            targets.append(
                BoardTarget(
                    u_px=100.0 + column * 100.0,
                    v_px=100.0 + row_from_top * 100.0,
                    width_px=size,
                    height_px=size,
                    confidence=0.9,
                    diameter_px=size,
                    diameter_m=0.20 if row_from_top == 0 else 0.40,
                    distance_m=5.0,
                    column=column,
                    row=3 - row_from_top,
                    hole_id=hole_id,
                )
            )
    return EngagementPlan(targets)


class ScoringStrategyTests(unittest.TestCase):
    def test_upper_uses_27_balls_without_exceeding_cell_caps(self):
        strategy = build_max_score_strategy(_board_plan(), "H11", "upper")

        self.assertEqual(len(strategy.plan), 27)
        self.assertEqual(strategy.hole_ids[:3], ("H11", "H11", "H11"))
        self.assertEqual(strategy.theoretical_max_points, 60)
        counts = {
            hole_id: strategy.hole_ids.count(hole_id)
            for hole_id in (f"H{number:02d}" for number in range(1, 10))
        }
        self.assertEqual(sum(counts.values()), 24)
        self.assertTrue(all(2 <= count <= 3 for count in counts.values()))

    def test_lower_completes_high_value_lines_before_repeat_shots(self):
        strategy = build_max_score_strategy(_board_plan(), "H12", "lower")

        self.assertEqual(len(strategy.plan), 18)
        self.assertEqual(strategy.hole_ids[:2], ("H12", "H12"))
        self.assertEqual(strategy.hole_ids[2:11], LOWER_LINE_FIRST_IDS)
        self.assertEqual(strategy.theoretical_max_points, 74)
        counts = {
            hole_id: strategy.hole_ids.count(hole_id)
            for hole_id in (f"H{number:02d}" for number in range(1, 10))
        }
        self.assertEqual(sum(counts.values()), 16)
        self.assertTrue(all(1 <= count <= 2 for count in counts.values()))

    def test_rejects_non_scoring_top_hole(self):
        with self.assertRaises(ValueError):
            build_max_score_strategy(_board_plan(), "H05", "upper")

    def test_unused_upper_bonus_balls_are_reallocated_to_grid_capacity(self):
        board = _board_plan()
        strategy = build_max_score_strategy(board, "H10", "upper")

        replacements = append_reallocated_grid_shots(
            strategy.plan, board, "upper", 3
        )

        self.assertEqual(len(replacements), 3)
        counts = {
            hole_id: sum(
                target.hole_id == hole_id for target in strategy.plan.targets
            )
            for hole_id in (f"H{number:02d}" for number in range(1, 10))
        }
        self.assertTrue(all(count == 3 for count in counts.values()))

    def test_unused_lower_bonus_balls_fill_the_two_single_grid_cells(self):
        board = _board_plan()
        strategy = build_max_score_strategy(board, "H12", "lower")

        replacements = append_reallocated_grid_shots(
            strategy.plan, board, "lower", 2
        )

        self.assertEqual(len(replacements), 2)
        counts = {
            hole_id: sum(
                target.hole_id == hole_id for target in strategy.plan.targets
            )
            for hole_id in (f"H{number:02d}" for number in range(1, 10))
        }
        self.assertTrue(all(count == 2 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()

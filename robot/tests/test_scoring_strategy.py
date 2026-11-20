import unittest

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.official_board import hole_id_for_cell
from turret_control.scoring_strategy import (
    LOWER_LINE_FIRST_IDS,
    UPPER_GRID_IDS,
    UPPER_GRID_NO_BONUS_IDS,
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
    def test_upper_is_three_bonus_then_three_per_cell_and_two_in_the_last_column(self):
        strategy = build_max_score_strategy(_board_plan(), "H11", "upper")

        self.assertEqual(len(strategy.plan), 27)
        self.assertEqual(strategy.bonus_balls, 3)
        self.assertEqual(strategy.hole_ids[:3], ("H11",) * 3)
        # From the top bonus hole, column 1 down, centre up, right column down;
        # at most 3 balls per big hole, the last column gets 2 each.
        self.assertEqual(strategy.hole_ids[3:], (
            "H03", "H03", "H03", "H02", "H02", "H02", "H01", "H01", "H01",
            "H04", "H04", "H04", "H05", "H05", "H05", "H06", "H06", "H06",
            "H09", "H09", "H08", "H08", "H07", "H07"))
        self.assertEqual(strategy.hole_ids[3:], UPPER_GRID_IDS)
        # The first magazine (18) ends exactly after the third H05.
        self.assertEqual(strategy.hole_ids[17:19], ("H05", "H06"))

    def test_lower_skips_bonus_and_shoots_each_cell_twice_in_serpentine(self):
        strategy = build_max_score_strategy(_board_plan(), "H12", "lower")

        self.assertEqual(len(strategy.plan), 18)
        self.assertNotIn("H12", strategy.hole_ids)
        self.assertEqual(strategy.bonus_balls, 0)
        self.assertEqual(strategy.hole_ids, (
            "H01", "H01", "H02", "H02", "H03", "H03", "H06", "H06", "H05",
            "H05", "H04", "H04", "H07", "H07", "H08", "H08", "H09", "H09"))
        self.assertEqual(strategy.theoretical_max_points, 70)

    def test_rejects_non_scoring_top_hole(self):
        with self.assertRaises(ValueError):
            build_max_score_strategy(_board_plan(), "H05", "upper")

    def test_unused_upper_bonus_balls_are_reallocated_to_grid_capacity(self):
        board = _board_plan()
        strategy = build_max_score_strategy(board, "H10", "upper")

        replacements = append_reallocated_grid_shots(
            strategy.plan, board, "upper", 3
        )

        self.assertEqual(sorted(replacements), ["H07", "H08", "H09"])
        counts = {
            hole_id: sum(
                target.hole_id == hole_id for target in strategy.plan.targets
            )
            for hole_id in (f"H{number:02d}" for number in range(1, 10))
        }
        self.assertTrue(all(count == 3 for count in counts.values()))

    def test_lower_grid_is_already_at_the_two_ball_cap(self):
        board = _board_plan()
        strategy = build_max_score_strategy(board, "H12", "lower")

        replacements = append_reallocated_grid_shots(
            strategy.plan, board, "lower", 2
        )

        self.assertEqual(replacements, ())
        self.assertEqual(len(strategy.plan), 18)



class NoLightTests(unittest.TestCase):
    def test_upper_without_light_puts_all_27_balls_in_big_holes(self):
        strategy = build_max_score_strategy(_board_plan(), None, "upper")
        self.assertEqual(strategy.hole_ids, UPPER_GRID_NO_BONUS_IDS)
        self.assertEqual(len(strategy.plan), 27)
        self.assertEqual(strategy.hole_ids[-9:], ("H09",) * 3 + ("H08",) * 3 + ("H07",) * 3)
        self.assertEqual(strategy.bonus_balls, 0)

    def test_lower_ignores_the_light(self):
        self.assertEqual(build_max_score_strategy(_board_plan(), None, "lower").hole_ids,
                         build_max_score_strategy(_board_plan(), "H11", "lower").hole_ids)


class LowerSmallHoleModeTests(unittest.TestCase):
    def test_two_bonus_then_sixteen_with_h01_and_h07_once(self):
        strategy = build_max_score_strategy(_board_plan(), "H12", "lower", lower_bonus=True)
        self.assertEqual(strategy.hole_ids, (
            "H12", "H12",
            "H01", "H02", "H02", "H03", "H03", "H06", "H06", "H05", "H05",
            "H04", "H04", "H07", "H08", "H08", "H09", "H09"))
        self.assertEqual(len(strategy.plan), 18)
        self.assertEqual(strategy.bonus_balls, 2)
        self.assertEqual(strategy.grid_balls, 16)

    def test_window_over_falls_back_to_the_plain_18(self):
        self.assertEqual(
            build_max_score_strategy(_board_plan(), None, "lower", lower_bonus=True).hole_ids,
            build_max_score_strategy(_board_plan(), None, "lower").hole_ids)

if __name__ == "__main__":
    unittest.main()

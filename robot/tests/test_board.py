import unittest

from turret_control.board import (
    BoardTarget,
    build_plan,
    build_virtual_plan_from_partial,
)


def make_board(swapped_size=False):
    targets = []
    for column, u in enumerate((100.0, 200.0, 300.0)):
        for row, v in enumerate((400.0, 300.0, 200.0, 100.0)):
            is_small = row == 3
            if swapped_size and column == 0 and row in (0, 3):
                is_small = not is_small
            size = 20.0 if is_small else 40.0
            targets.append(BoardTarget(u, v, size, size, confidence=0.9))
    return targets


class BoardGeometryTests(unittest.TestCase):
    def test_known_twelve_hole_layout_builds_plan(self):
        plan = build_plan(make_board(), 700.0, 0.20, 0.40)

        self.assertEqual(len(plan.targets), 12)
        self.assertEqual({target.hole_id for target in plan.targets}, {
            f"H{i:02d}" for i in range(1, 13)
        })
        self.assertEqual(
            sum(target.diameter_m == 0.20 for target in plan.targets), 3
        )
        self.assertEqual(
            sum(target.diameter_m == 0.40 for target in plan.targets), 9
        )

    def test_wrong_small_hole_row_is_rejected(self):
        plan = build_plan(make_board(swapped_size=True), 700.0, 0.20, 0.40)

        self.assertEqual(len(plan.targets), 0)

    def test_partial_board_can_be_inferred_for_visualization(self):
        partial = make_board()[:-1]
        plan = build_virtual_plan_from_partial(partial, 0.20, 0.40)

        self.assertEqual(len(plan.targets), 12)
        self.assertEqual(
            {target.hole_id for target in plan.targets},
            {f"H{i:02d}" for i in range(1, 13)},
        )


if __name__ == "__main__":
    unittest.main()

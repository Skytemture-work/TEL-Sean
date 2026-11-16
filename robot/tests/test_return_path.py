import unittest

import main


class BuildReturnPathTests(unittest.TestCase):
    def test_reverses_order_and_negates_every_component(self):
        samples = [(0.00, -40, 0, 0), (0.02, -40, 0, 0), (2.00, -40, 15, 0),
                   (2.50, 0, 0, 0)]
        path = main.build_return_path(samples)
        self.assertEqual([seg[1:] for seg in path], [(40, -15, 0), (40, 0, 0)])
        self.assertAlmostEqual(path[0][0], 0.5)
        self.assertAlmostEqual(path[1][0], 0.02 + 0.5)  # each gap capped at 0.5 s

    def test_stops_are_dropped_and_equal_neighbours_merge(self):
        samples = [(t / 50.0, -40, 0, 0) for t in range(100)]       # 2 s forward
        samples += [(2.0 + t / 50.0, 0, 0, 0) for t in range(50)]   # 1 s waiting
        samples += [(3.0 + t / 50.0, 0, 22, 0) for t in range(25)]  # 0.5 s strafe
        samples.append((3.5, 0, 0, 0))
        path = main.build_return_path(samples)
        self.assertEqual([seg[1:] for seg in path], [(0, -22, 0), (40, 0, 0)])
        self.assertAlmostEqual(path[0][0], 0.5, places=6)
        self.assertAlmostEqual(path[1][0], 2.0, places=6)

    def test_gap_longer_than_the_esp32_timeout_counts_only_the_timeout(self):
        path = main.build_return_path([(0.0, -40, 0, 0), (3.0, 0, 0, 0)])
        self.assertEqual(path, [(0.5, 40, 0, 0)])

    def test_nothing_recorded_gives_no_path(self):
        self.assertEqual(main.build_return_path([]), [])
        self.assertEqual(main.build_return_path([(0.0, 0, 0, 0), (1.0, 0, 0, 0)]), [])


if __name__ == "__main__":
    unittest.main()

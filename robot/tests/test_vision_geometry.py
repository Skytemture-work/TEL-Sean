import unittest

import cv2
import numpy as np

from turret_control.vision_geometry import (
    binary_mask_geometry,
    polygon_geometry,
    sample_board_depth,
)


class VisionGeometryTests(unittest.TestCase):
    def test_polygon_center_and_diameter_use_mask(self):
        polygon = np.array(
            [[40, 30], [60, 30], [70, 40], [70, 60],
             [60, 70], [40, 70], [30, 60], [30, 40]],
            dtype=np.float32,
        )

        geometry = polygon_geometry(polygon)

        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry.center_u, 50.0, places=1)
        self.assertAlmostEqual(geometry.center_v, 50.0, places=1)
        self.assertGreater(geometry.equivalent_diameter_px, 35.0)
        self.assertLess(geometry.equivalent_diameter_px, 45.0)

    def test_ring_depth_ignores_background_inside_hole(self):
        polygon = np.array(
            [[40, 30], [60, 30], [70, 40], [70, 60],
             [60, 70], [40, 70], [30, 60], [30, 40]],
            dtype=np.float32,
        )
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        depth[mask > 0] = 5.0  # simulated background behind the hole

        measured = sample_board_depth(depth, polygon=polygon, ring_px=4)

        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured, 2.0, places=5)

    def test_binary_mask_is_supported_as_fallback(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        cv2.circle(mask, (10, 10), 5, 1, -1)

        geometry = binary_mask_geometry(mask, (100, 100))

        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry.center_u, 50.0, delta=3.0)
        self.assertAlmostEqual(geometry.center_v, 50.0, delta=3.0)


if __name__ == "__main__":
    unittest.main()

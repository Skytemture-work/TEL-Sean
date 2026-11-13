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


def _full_frame_reference_depth(depth, polygon, ring_px=6, minimum_samples=10):
    """Original full-frame implementation, kept to prove the crop is identical."""
    from turret_control.vision_geometry import contour_geometry
    geometry = contour_geometry(polygon)
    if geometry is None:
        return None
    inside = np.zeros(depth.shape[:2], dtype=np.uint8)
    cv2.fillPoly(inside, [geometry.contour], 1)
    inside = inside.astype(bool)
    if not np.any(inside):
        return None
    k = ring_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = (cv2.dilate(inside.astype(np.uint8), kernel) > 0) & ~inside
    values = depth[ring]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < minimum_samples:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.03, 3.0 * 1.4826 * mad)
    inliers = values[np.abs(values - median) <= threshold]
    if inliers.size < minimum_samples:
        return None
    return float(np.median(inliers))


class CroppedDepthSamplingTest(unittest.TestCase):
    def test_cropped_sampling_matches_full_frame_including_edges(self):
        from turret_control.vision_geometry import sample_board_depth
        rng = np.random.default_rng(7)
        depth = (rng.random((720, 1280)) * 0.5 + 2.0).astype(np.float32)
        depth[rng.random((720, 1280)) < 0.05] = np.nan
        centres = [(640, 360), (3, 3), (1277, 716), (0, 400), (1280, 100),
                   (-10, 300), (700, 725)]
        centres += [tuple(rng.integers(0, [1280, 720])) for _ in range(40)]
        checked = 0
        for cx, cy in centres:
            r = float(rng.uniform(4, 70))
            angles = np.linspace(0, 2 * np.pi, int(rng.integers(5, 80)), endpoint=False)
            radii = r * rng.uniform(0.7, 1.0, angles.size)
            polygon = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1)
            for ring_px in (1, 6, 9):
                expected = _full_frame_reference_depth(depth, polygon, ring_px)
                actual = sample_board_depth(depth, polygon=polygon, ring_px=ring_px)
                self.assertEqual(actual, expected, (cx, cy, r, ring_px))
                checked += expected is not None
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()

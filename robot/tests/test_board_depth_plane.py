import unittest
from dataclasses import replace

import cv2
import numpy as np

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.board_depth_plane import apply_board_depth_plane, fit_board_depth_plane
from turret_control.board_pose import BoardPose, project_targets_from_pose
from turret_control.config import TurretConfig
from turret_control.official_board import OFFICIAL_HOLE_LAYOUT


class BoardDepthPlaneTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig(
            image_width_px=640, image_height_px=480,
            focal_length_x_px=400, focal_length_y_px=400,
            camera_principal_x_px=320, camera_principal_y_px=240,
        )
        self.pose = BoardPose(
            rvec=np.asarray([.05, .25, .02]).reshape(3, 1),
            tvec=np.asarray([-1.5, -1.7, 5.]).reshape(3, 1),
            timestamp=0., reprojection_error_px=0., correspondence_count=12,
            translation_speed_m_s=0., rotation_speed_rad_s=0., stable=True,
        )
        plan = EngagementPlan([
            BoardTarget(0, 0, 1, 1, hole_id=hole_id)
            for hole_id in OFFICIAL_HOLE_LAYOUT
        ])
        self.projected = project_targets_from_pose(plan, self.pose, self.config)
        rotation, _ = cv2.Rodrigues(self.pose.rvec)
        self.normal = rotation[:, 2]
        self.offset = -float(self.normal @ self.pose.tvec.reshape(3))
        y, x = np.mgrid[0:480, 0:640]
        denominator = (self.normal[0] * (x - 320) / 400
                       + self.normal[1] * (y - 240) / 400 + self.normal[2])
        self.depth = (-self.offset / denominator).astype(np.float32)
        self.surface = np.ones((480, 640), np.uint8)

    def fit(self, depth=None, surface=None, projected=None):
        return fit_board_depth_plane(
            self.depth if depth is None else depth,
            self.surface if surface is None else surface,
            self.pose, self.projected if projected is None else projected, [], self.config,
        )

    def test_slanted_plane_reconstructs_all_hole_depths_with_outliers(self):
        noisy = self.depth.copy()
        noisy[::13, ::7] = 9.0
        noisy[::17, ::11] = np.nan
        # Background through one projected hole must not bias the fit.
        center = self.projected["H05"]
        cv2.circle(noisy, (round(center.u_px), round(center.v_px)), 20, 10., -1)
        plane = self.fit(depth=noisy)
        self.assertIsNotNone(plane)
        self.assertGreaterEqual(plane.inlier_count, 120)
        result = apply_board_depth_plane(self.projected, plane, self.config)
        for hole_id, target in self.projected.items():
            self.assertEqual(result[hole_id].distance_source, "zed_plane")
            self.assertAlmostEqual(result[hole_id].distance_m, target.distance_m, delta=.02)
        self.assertGreater(abs(result["H01"].distance_m - result["H07"].distance_m), .1)

    def test_insufficient_depth_and_wrong_shape_are_rejected(self):
        self.assertIsNone(self.fit(depth=np.full_like(self.depth, np.nan)))
        self.assertIsNone(self.fit(surface=np.zeros_like(self.surface)))
        self.assertIsNone(self.fit(depth=self.depth[:100]))

    def test_conflicting_or_narrow_plane_is_rejected(self):
        self.assertIsNone(self.fit(depth=self.depth + 3.0))
        narrow = np.zeros_like(self.surface)
        narrow[:, 300:310] = 1
        self.assertIsNone(self.fit(surface=narrow))

    def test_out_of_frame_hole_keeps_previous_depth(self):
        plane = self.fit()
        self.assertIsNotNone(plane)
        target = replace(self.projected["H01"], u_px=-10, distance_source="board_pose")
        result = apply_board_depth_plane({"H01": target}, plane, self.config)
        self.assertEqual(result["H01"], target)


if __name__ == "__main__":
    unittest.main()

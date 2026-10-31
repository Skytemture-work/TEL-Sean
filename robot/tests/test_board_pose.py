import unittest
from dataclasses import replace

import cv2
import numpy as np

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.board_pose import (
    BoardPose,
    BoardPoseTracker,
    camera_matrix,
    distortion_coefficients,
    project_targets_from_pose,
    update_plan_geometry,
)
from turret_control.config import TurretConfig
from turret_control.official_board import OFFICIAL_HOLE_LAYOUT


def _template_plan():
    targets = []
    for hole_id in sorted(OFFICIAL_HOLE_LAYOUT):
        small = hole_id in {"H10", "H11", "H12"}
        targets.append(
            BoardTarget(
                0.0,
                0.0,
                1.0,
                1.0,
                diameter_m=0.20 if small else 0.40,
                hole_id=hole_id,
            )
        )
    return EngagementPlan(targets)


class BoardPoseTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig(
            image_width_px=1280,
            image_height_px=720,
            focal_length_x_px=810.0,
            focal_length_y_px=805.0,
            camera_principal_x_px=638.0,
            camera_principal_y_px=362.0,
            board_pose_stable_frames=3,
        )
        self.template = _template_plan()
        self.expected = BoardPose(
            rvec=np.asarray((0.025, -0.12, 0.018), dtype=np.float64).reshape(3, 1),
            tvec=np.asarray((-1.45, -1.15, 7.0), dtype=np.float64).reshape(3, 1),
            timestamp=0.0,
            reprojection_error_px=0.0,
            correspondence_count=12,
            translation_speed_m_s=0.0,
            rotation_speed_rad_s=0.0,
            stable=True,
        )

    def _observed_plan(self, pose=None):
        live = project_targets_from_pose(
            self.template,
            pose or self.expected,
            self.config,
        )
        return EngagementPlan(list(live.values()))

    def test_pnp_recovers_pose_and_becomes_stable(self):
        observed_plan = self._observed_plan()
        tracker = BoardPoseTracker(self.config)
        result = None
        for frame_index in range(5):
            result = tracker.update(
                observed_plan.targets,
                observed_plan,
                now=frame_index / 30.0,
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.stable)
        self.assertLess(result.reprojection_error_px, 0.1)
        np.testing.assert_allclose(result.tvec, self.expected.tvec, atol=0.02)
        expected_rotation, _ = cv2.Rodrigues(self.expected.rvec)
        actual_rotation, _ = cv2.Rodrigues(result.rvec)
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=0.01)

    def test_motion_resets_stability_but_keeps_a_live_pose(self):
        observed_plan = self._observed_plan()
        tracker = BoardPoseTracker(self.config)
        for frame_index in range(5):
            result = tracker.update(
                observed_plan.targets,
                observed_plan,
                now=frame_index / 30.0,
            )
        self.assertTrue(result.stable)

        moved = BoardPose(
            rvec=self.expected.rvec,
            tvec=self.expected.tvec + np.asarray((0.05, 0.0, 0.0)).reshape(3, 1),
            timestamp=0.0,
            reprojection_error_px=0.0,
            correspondence_count=12,
            translation_speed_m_s=0.0,
            rotation_speed_rad_s=0.0,
            stable=False,
        )
        moved_plan = self._observed_plan(moved)
        result = tracker.update(
            moved_plan.targets,
            moved_plan,
            now=5.0 / 30.0,
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.stable)
        self.assertGreater(
            result.translation_speed_m_s,
            self.config.board_pose_fire_translation_m_s,
        )

    def test_corroborated_full_board_quad_is_a_valid_fallback(self):
        corners_m = np.asarray(
            ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 3.4, 0.0), (0.0, 3.4, 0.0)),
            dtype=np.float64,
        )
        quad, _ = cv2.projectPoints(
            corners_m,
            self.expected.rvec,
            self.expected.tvec,
            camera_matrix(self.config),
            distortion_coefficients(self.config),
        )
        tracker = BoardPoseTracker(self.config)
        observed = self._observed_plan().targets[:3]

        result = tracker.update(
            observed,
            self._observed_plan(),
            quad.reshape(4, 2),
            now=0.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.correspondence_count, 4)
        self.assertLess(result.reprojection_error_px, 0.1)

    def test_dark_quad_without_hole_support_is_rejected(self):
        quad = np.asarray(
            ((100.0, 100.0), (900.0, 100.0), (900.0, 650.0), (100.0, 650.0)),
            dtype=np.float64,
        )
        tracker = BoardPoseTracker(self.config)

        result = tracker.update([], self.template, quad, now=0.0)

        self.assertIsNone(result)

    def test_single_visible_hole_bridges_occlusion_then_expires(self):
        plan = self._observed_plan()
        tracker = BoardPoseTracker(self.config)
        for index in range(5):
            stable = tracker.update(plan.targets, plan, now=index * .05)
        self.assertTrue(stable.stable)
        one = [replace(plan.targets[0], u_px=plan.targets[0].u_px + 0.2)]
        self.assertIsNone(tracker.update(one, plan, now=.25))
        partial = tracker.update_partial(one, plan, now=.25)
        self.assertIsNotNone(partial)
        self.assertEqual(partial.source, "partial")
        self.assertEqual(partial.correspondence_count, 1)
        live = project_targets_from_pose(plan, partial, self.config)
        self.assertAlmostEqual(live[one[0].hole_id].u_px, one[0].u_px, places=2)
        # Frequent partial frames must never extend the original PnP deadline.
        for timestamp in (.4, .55, .7, .85, 1.0, 1.15):
            self.assertIsNotNone(tracker.update_partial(one, plan, now=timestamp))
        self.assertIsNone(tracker.update_partial(one, plan, now=1.21))

    def test_partial_requires_stable_recent_reference_and_matching_observation(self):
        plan = self._observed_plan()
        tracker = BoardPoseTracker(self.config)
        self.assertIsNone(tracker.update_partial(plan.targets[:1], plan, now=0.0))
        for index in range(5):
            tracker.update(plan.targets, plan, now=index * .05)
        self.assertIsNone(tracker.update_partial([], plan, now=.25))
        wrong_size = replace(plan.targets[0], width_px=1, height_px=1, diameter_px=1)
        self.assertIsNone(tracker.update_partial([wrong_size], plan, now=.25))
        wrong_position = replace(plan.targets[0], u_px=plan.targets[0].u_px + 1000)
        self.assertIsNone(tracker.update_partial([wrong_position], plan, now=.25))
        self.assertIsNone(tracker.update_partial(plan.targets[:1], plan, now=.5))

    def test_fast_partial_motion_invalidates_reference(self):
        plan = self._observed_plan()
        tracker = BoardPoseTracker(self.config)
        for index in range(5):
            tracker.update(plan.targets, plan, now=index * .05)
        moved = replace(plan.targets[0], u_px=plan.targets[0].u_px + 10)
        self.assertIsNone(tracker.update_partial([moved], plan, now=.25))
        self.assertIsNone(tracker.update_partial(plan.targets[:1], plan, now=.3))

    def test_geometry_update_preserves_shot_index_and_duplicate_targets(self):
        original = self._observed_plan()
        duplicated = EngagementPlan(
            [original.targets[0], original.targets[0], original.targets[1]]
        )
        duplicated.mark_engaged()
        old_index = duplicated.index
        live = project_targets_from_pose(original, self.expected, self.config)

        update_plan_geometry(duplicated, live)

        self.assertEqual(duplicated.index, old_index)
        self.assertEqual(len(duplicated.targets), 3)
        self.assertTrue(duplicated.targets[0].engaged)
        self.assertEqual(duplicated.targets[0].hole_id, duplicated.targets[1].hole_id)


if __name__ == "__main__":
    unittest.main()

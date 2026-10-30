import unittest
from dataclasses import replace

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from turret_control.board import BoardTarget, EngagementPlan, build_plan_from_board_quad
from turret_control.config import TurretConfig
from turret_control.official_board import hole_position_for_cell
from turret_control.detection import boxes_to_board_targets
from turret_control.run_zed_tensorrt_test import (
    PreviewBoardTracker, _estimate_board_quad, _filter_preview_result, _grid_geometry_valid, _select_grid_plan,
    _surface_from_estimated_quad,
)


class BoardPreviewTests(unittest.TestCase):
    def test_estimates_outline_from_observed_grid_with_missing_hole(self):
        corners = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        matrix = cv2.getPerspectiveTransform(np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32), corners)
        targets = []
        for row in range(4):
            for column in range(3):
                if (column, row) == (2, 0):
                    continue
                point = np.asarray(hole_position_for_cell(column, row), np.float32).reshape(1, 1, 2)
                x, y = cv2.perspectiveTransform(point, matrix).reshape(2)
                size = 19 if row == 0 else 37
                targets.append(BoardTarget(float(x), float(y), size, size, 0.9))
        targets.append(BoardTarget(460, 395, 40, 40, 0.35))
        estimated = _estimate_board_quad(targets, (720, 1280, 3))
        self.assertIsNotNone(estimated)
        np.testing.assert_allclose(estimated, corners, atol=1.0)
        surface = _surface_from_estimated_quad(estimated, (720, 1280, 3))
        self.assertIsNone(surface.quad)
        self.assertTrue(surface.mask[414, 674])
        self.assertFalse(surface.mask[395, 460])

    def test_does_not_guess_with_insufficient_targets(self):
        targets = [BoardTarget(100 + i * 10, 200, 20, 20, 0.9) for i in range(7)]
        self.assertIsNone(_estimate_board_quad(targets, (720, 1280, 3)))

    def test_estimates_board_when_all_small_holes_and_one_large_hole_are_occluded(self):
        corners = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        matrix = cv2.getPerspectiveTransform(np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32), corners)
        targets = []
        for row in (1, 2, 3):
            for column in range(3):
                if (row, column) == (1, 2):
                    continue
                point = np.asarray(hole_position_for_cell(column, row), np.float32).reshape(1, 1, 2)
                x, y = cv2.perspectiveTransform(point, matrix).reshape(2)
                targets.append(BoardTarget(float(x), float(y), 37, 37, .9))
        estimated = _estimate_board_quad(targets, (720, 1280, 3))
        self.assertIsNotNone(estimated)
        np.testing.assert_allclose(estimated, corners, atol=1.0)
        grid, inferred = _select_grid_plan(False, EngagementPlan([]), None, None, None, estimated, TurretConfig())
        self.assertTrue(inferred)
        self.assertEqual(len(grid.targets), 12)

    def test_does_not_treat_wrong_sized_objects_as_large_hole_grid(self):
        targets = [BoardTarget(200 + col * 90, 200 + row * 60, 10, 10, .9)
                   for row in range(3) for col in range(3)]
        self.assertIsNone(_estimate_board_quad(targets, (720, 1280, 3)))

    def test_three_non_collinear_holes_track_recent_board_within_time_limit(self):
        shape = (720, 1280, 3)
        corners = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        full = build_plan_from_board_quad(corners, .2, .4)
        tracker = PreviewBoardTracker(max_reference_age_s=10)
        observed = [t for t in full.targets if t.row != 3 and t.hole_id != "H09"]
        fresh, tracked = tracker.update(observed, shape, 0.0)
        self.assertFalse(tracked)
        self.assertIsNotNone(fresh)
        three = [replace(t, u_px=t.u_px + 5, v_px=t.v_px - 3)
                 for t in full.targets if t.hole_id in ("H01", "H03", "H06")]
        quad, tracked = tracker.update(three, shape, 1.0)
        self.assertTrue(tracked)
        np.testing.assert_allclose(quad, corners + np.asarray((5, -3)), atol=1.5)
        expired, _ = tracker.update(three, shape, 11.0)
        self.assertIsNone(expired)
        self.assertIsNone(PreviewBoardTracker().update(three, shape, 0.0)[0])

    def test_collinear_holes_can_only_translate_a_recent_reference(self):
        corners = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        full = build_plan_from_board_quad(corners, .2, .4)
        tracker = PreviewBoardTracker()
        tracker.update(full.targets, (720, 1280, 3), 0.0)
        collinear = [replace(t, u_px=t.u_px + 3, v_px=t.v_px - 2)
                     for t in full.targets if t.hole_id in ("H01", "H04", "H07")]
        tracked, inferred = tracker.update(collinear, (720, 1280, 3), 1.0)
        self.assertTrue(inferred)
        np.testing.assert_allclose(tracked, corners + np.asarray((3, -2)), atol=1.5)

    def test_one_known_hole_tracks_translation_but_expires_without_reference(self):
        shape = (720, 1280, 3)
        corners = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        full = build_plan_from_board_quad(corners, .2, .4)
        target = next(t for t in full.targets if t.hole_id == "H03")
        observed = [replace(target, u_px=target.u_px + 4, v_px=target.v_px - 3)]
        tracker = PreviewBoardTracker(max_reference_age_s=10)
        self.assertIsNone(tracker.update(observed, shape, 0.0)[0])
        self.assertIsNotNone(tracker.update(full.targets, shape, 1.0)[0])
        quad, tracked = tracker.update(observed, shape, 2.0)
        self.assertTrue(tracked)
        np.testing.assert_allclose(quad, corners + np.asarray((4, -3)), atol=1.5)
        self.assertIsNone(tracker.update(observed, shape, 12.0)[0])
        distant = [replace(target, u_px=target.u_px + 200)]
        tracker = PreviewBoardTracker()
        tracker.update(full.targets, shape, 0.0)
        self.assertIsNone(tracker.update(distant, shape, 1.0)[0])

    def test_preview_keeps_only_in_board_boxes_and_corresponding_masks(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = torch.tensor([[10, 10, 20, 20, .9, 0],
                              [30, 30, 40, 40, .2, 0],
                              [70, 70, 80, 80, .8, 0]])
        masks = torch.zeros((3, 100, 100))
        masks[0, 10:21, 10:21] = 1
        masks[1, 30:41, 30:41] = 1
        masks[2, 70:81, 70:81] = 1
        result = Results(frame, path="test", names={0: "hole"}, boxes=boxes, masks=masks)
        raw = boxes_to_board_targets(result, .25)
        quad = np.asarray(((0, 0), (50, 0), (50, 50), (0, 50)), dtype=np.float32)
        surface = _surface_from_estimated_quad(quad, frame.shape)
        filtered = _filter_preview_result(result, raw, surface, .25)
        self.assertEqual(len(filtered.boxes), 1)
        self.assertEqual(len(filtered.masks), 1)
        self.assertAlmostEqual(float(filtered.boxes.conf[0]), .9, places=5)
        self.assertTrue(torch.equal(filtered.masks.data[0], masks[0]))
        self.assertIsNone(_filter_preview_result(result, raw, None, .25))
        unverified = _filter_preview_result(result, raw, None, .25, show_unverified=True)
        self.assertEqual(len(unverified.boxes), 2)
        self.assertTrue(torch.equal(unverified.masks.data[1], masks[2]))

    def test_grid_is_available_for_confirmed_and_estimated_boards(self):
        config = TurretConfig()
        quad = np.asarray(((540, 280), (802, 281), (835, 600), (513, 601)), np.float32)
        confirmed = build_plan_from_board_quad(quad, .2, .4)
        selected, inferred = _select_grid_plan(True, confirmed, None, None, None, quad, config)
        self.assertIs(selected, confirmed)
        self.assertFalse(inferred)
        selected, inferred = _select_grid_plan(False, EngagementPlan([]), None, None, None, quad, config)
        self.assertEqual(len(selected.targets), 12)
        self.assertTrue(inferred)
        selected, inferred = _select_grid_plan(False, EngagementPlan([]), None, None, None, None, config)
        self.assertIsNone(selected)

    def test_rejects_collapsed_or_crossed_grid_but_accepts_perspective(self):
        config = TurretConfig()
        # The picture can be tilted by perspective; straight rows/columns need
        # not align exactly to the image axes.
        quad = np.asarray(((480, 260), (820, 300), (770, 640), (510, 580)), np.float32)
        grid = build_plan_from_board_quad(quad, .2, .4)
        self.assertTrue(_grid_geometry_valid(grid, (720, 1280, 3)))
        collapsed = EngagementPlan([replace(t, u_px=600 + (t.u_px - 600) * .005)
                                    for t in grid.targets])
        self.assertFalse(_grid_geometry_valid(collapsed, (720, 1280, 3)))
        targets = list(grid.targets)
        first, second = targets[0], targets[1]
        targets[0] = replace(first, u_px=second.u_px, v_px=second.v_px)
        targets[1] = replace(second, u_px=first.u_px, v_px=first.v_px)
        crossed = EngagementPlan(targets)
        self.assertFalse(_grid_geometry_valid(crossed, (720, 1280, 3)))
        selected, _ = _select_grid_plan(False, EngagementPlan([]), None, collapsed, None, None, config, (720, 1280, 3))
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()

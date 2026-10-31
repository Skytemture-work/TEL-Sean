import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.board_pose import prefer_observed_depth, update_plan_geometry
from turret_control.config import TurretConfig
from turret_control.detection import boxes_to_board_targets
from turret_control.zed_camera import ZedDepthCamera


class ZedDepthTests(unittest.TestCase):
    def test_visible_mask_depth_is_assigned_to_matching_hole(self):
        class Box:
            conf = np.asarray([.9])
            xyxy = np.asarray([[40., 40., 60., 60.]])

        polygon = np.asarray([[40, 40], [60, 40], [60, 60], [40, 60]], np.float32)
        result = SimpleNamespace(boxes=[Box()], masks=SimpleNamespace(xy=[polygon]))
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[41:60, 41:60] = 12.0  # background through the hole
        observed = boxes_to_board_targets(result, .4, depth)
        self.assertAlmostEqual(observed[0].distance_m, 4.0)
        self.assertEqual(observed[0].distance_source, "zed_depth")
        reference = BoardTarget(50, 50, 20, 20, hole_id="H05", distance_m=5.0, distance_source="board_pose")
        other = replace(reference, u_px=85, hole_id="H06", distance_m=5.2)
        merged = prefer_observed_depth({"H05": reference, "H06": other}, observed)
        self.assertEqual(merged["H05"].distance_m, 4.0)
        self.assertEqual(merged["H05"].distance_source, "zed_depth")
        self.assertEqual(merged["H06"].distance_m, 5.2)
        self.assertEqual(merged["H06"].distance_source, "board_pose")
        plan = EngagementPlan([reference, reference])
        plan.mark_engaged()
        update_plan_geometry(plan, merged)
        self.assertEqual([t.distance_m for t in plan.targets], [4.0, 4.0])
        self.assertEqual(plan.index, 1)

    def test_invalid_or_unmatched_depth_keeps_pose_estimate(self):
        ref = BoardTarget(50, 50, 20, 20, hole_id="H05", distance_m=5., distance_source="board_pose")
        for value, x in ((float("nan"), 50), (-1., 50), (4., 200)):
            observation = replace(ref, u_px=x, distance_m=value, distance_source="zed_depth")
            merged = prefer_observed_depth({"H05": ref}, [observation])
            self.assertEqual(merged["H05"].distance_m, 5.)
            self.assertEqual(merged["H05"].distance_source, "board_pose")
        self.assertEqual(prefer_observed_depth({"H05": ref}, []), {"H05": ref})

    def test_sdk_uses_rectified_intrinsics_and_same_frame_depth(self):
        class Mat:
            def get_data(self):
                return self.data

        class Camera:
            def open(self, init):
                self.init = init
                return 0
            def get_camera_information(self):
                left = SimpleNamespace(fx=528., fy=529., cx=639., cy=361.)
                return SimpleNamespace(camera_configuration=SimpleNamespace(
                    calibration_parameters=SimpleNamespace(left_cam=left)))
            def grab(self, runtime):
                return 0
            def retrieve_image(self, image, view):
                image.data = np.zeros((720, 1280, 4), dtype=np.uint8)
                return 0
            def retrieve_measure(self, depth, measure):
                if getattr(self, "depth_failure", False):
                    return 1
                depth.data = np.full((720, 1280), 5., dtype=np.float32)
                return 0
            def close(self):
                self.closed = True

        sl = SimpleNamespace(Camera=Camera, Mat=Mat, InitParameters=SimpleNamespace,
                             RuntimeParameters=SimpleNamespace,
                             RESOLUTION=SimpleNamespace(HD720=1), UNIT=SimpleNamespace(METER=1),
                             DEPTH_MODE=SimpleNamespace(ULTRA=1), ERROR_CODE=SimpleNamespace(SUCCESS=0),
                             VIEW=SimpleNamespace(LEFT=1), MEASURE=SimpleNamespace(DEPTH=1))
        with patch.dict("sys.modules", {"pyzed": SimpleNamespace(sl=sl), "pyzed.sl": sl}):
            camera = ZedDepthCamera(TurretConfig())
            ok, image, depth = camera.read()
            self.assertTrue(ok)
            self.assertEqual(image.shape, (720, 1280, 3))
            self.assertEqual(depth[0, 0], 5.)
            self.assertEqual(camera.config.focal_length_x_px, 528.)
            self.assertEqual(camera.config.camera_distortion_coeffs, (0.,) * 5)
            camera.camera.depth_failure = True
            ok, _, missing_depth = camera.read()
            self.assertTrue(ok)
            self.assertIsNone(missing_depth)  # Never reuse the preceding frame's depth.
            camera.release()
            self.assertTrue(camera.camera.closed)


if __name__ == "__main__":
    unittest.main()

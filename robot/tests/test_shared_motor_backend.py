import math
import time
import unittest
from unittest.mock import patch
from dataclasses import replace

from turret_control.aiming import AimCommand, TargetPixel
from turret_control.config import TurretConfig
from turret_control.controller import TurretController
from turret_control.run_vertical_panel import WorkerSnapshot
from turret_control.shared_motor_backend import SharedMotorBackend


class FakeWorker:
    def __init__(self):
        self.state = WorkerSnapshot(ready=True, enabled=True, motor_err=1, yaw_motor_err=1)
        self.requests = []
        self.auto_ack = True

    def snapshot(self):
        return self.state

    def request_tagged_aim(self, yaw, pitch, request_id):
        self.requests.append((yaw, pitch))
        if self.auto_ack:
            self.state = replace(self.state, aim_request_id=request_id)

    def request_tagged_home(self, request_id):
        self.requests.append(("home", request_id))


class SharedMotorBackendTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.backend = SharedMotorBackend("unused", worker=self.worker)
        self.yaw = self.backend.yaw_calibration.motor_from_muzzle(math.radians(20))
        self.pitch = self.backend.pitch_calibration.motor_from_muzzle(math.radians(30))
        self.worker.state = replace(
            self.worker.state, yaw_position_rad=self.yaw, position_rad=self.pitch,
            yaw_target_rad=self.yaw, target_rad=self.pitch,
            velocity_rad_s=0., yaw_velocity_rad_s=0.,
            command_velocity_rad_s=0., yaw_command_velocity_rad_s=0.,
            feedback_at=time.monotonic(), yaw_feedback_at=time.monotonic(),
        )
        self.targets = {3: (self.yaw, .06, 0, 0), 1: (self.pitch, .06, 0, 0)}

    def test_worker_receives_motor_target_as_same_physical_pose_and_coalesces_jitter(self):
        self.backend.send_targets(self.targets)
        self.assertAlmostEqual(self.worker.requests[0][0], 20.)
        self.assertAlmostEqual(self.worker.requests[0][1], 30.)
        self.backend.send_targets(self.targets)
        self.assertEqual(len(self.worker.requests), 1)
        self.assertTrue(self.backend.motion_ready(self.yaw, self.pitch))

    def test_pose_jitter_below_deadband_neither_retargets_nor_blocks_firing(self):
        self.backend.send_targets(self.targets)
        jitter = math.radians(0.05)
        moved = {3: (self.yaw + jitter, .06, 0, 0), 1: (self.pitch - jitter, .06, 0, 0)}
        self.backend.send_targets(moved)
        self.assertEqual(len(self.worker.requests), 1)
        self.assertTrue(self.backend.motion_ready(self.yaw + jitter, self.pitch - jitter))

    def test_real_target_change_retargets_and_waits_for_the_worker(self):
        self.backend.send_targets(self.targets)
        step = math.radians(0.3)
        self.backend.send_targets({3: (self.yaw, .06, 0, 0), 1: (self.pitch + step, .06, 0, 0)})
        self.assertEqual(len(self.worker.requests), 2)
        # The worker is still on the old target -> not ready.
        self.assertFalse(self.backend.motion_ready(self.yaw, self.pitch + step))

    @patch("turret_control.shared_motor_backend.MotorWorker")
    def test_speed_and_acceleration_are_passed_to_motor_worker(self, worker_class):
        SharedMotorBackend("unused", speed_rad_s=0.12, acceleration_rad_s2=0.08)
        kwargs = worker_class.call_args.kwargs
        self.assertEqual(kwargs["speed_rad_s"], 0.12)
        self.assertEqual(kwargs["acceleration_rad_s2"], 0.08)

    @patch("turret_control.shared_motor_backend.MotorWorker")
    def test_default_aim_trajectory_limits(self, worker_class):
        SharedMotorBackend("unused")
        kwargs = worker_class.call_args.kwargs
        self.assertEqual(kwargs["speed_rad_s"], 0.60)
        self.assertEqual(kwargs["acceleration_rad_s2"], 3.0)
        self.assertEqual(kwargs["kp"], 50.0)
        self.assertEqual(kwargs["kd"], 1.5)
        self.assertEqual(kwargs["control_mode"], "posvel")

    @patch("turret_control.shared_motor_backend.MotorWorker")
    def test_invalid_trajectory_limits_are_rejected(self, worker_class):
        for speed, acceleration in ((0, 0.05), (-1, 0.05), (float("nan"), 0.05),
                                    (0.09, 0), (0.09, float("inf"))):
            with self.subTest(speed=speed, acceleration=acceleration):
                with self.assertRaises(ValueError):
                    SharedMotorBackend("unused", speed_rad_s=speed,
                                       acceleration_rad_s2=acceleration)
        worker_class.assert_not_called()

    def test_home_requires_acknowledged_goal_and_fresh_stopped_feedback(self):
        self.backend.request_home()
        self.assertEqual(self.worker.requests[0], ("home", 1))
        self.assertEqual(self.backend.home_status()[0], "returning")
        yaw_home, pitch_home = self.backend._home_target
        self.worker.state = replace(
            self.worker.state, aim_request_id=1,
            yaw_target_rad=yaw_home, target_rad=pitch_home,
            yaw_position_rad=yaw_home, position_rad=pitch_home,
            yaw_sequence_phase="restoring",
        )
        self.assertEqual(self.backend.home_status()[0], "returning")
        self.worker.state = replace(self.worker.state, yaw_sequence_phase=None)
        self.assertEqual(self.backend.home_status()[0], "arrived")
        self.worker.state = replace(self.worker.state, feedback_at=time.monotonic() - 1)
        self.assertEqual(self.backend.home_status()[0], "returning")
        self.worker.state = replace(self.worker.state, error="安全路徑被拒絕")
        self.assertEqual(self.backend.home_status(), ("error", "安全路徑被拒絕"))

    def test_home_is_rejected_without_fresh_feedback_or_recorded_corridor(self):
        self.worker.state = replace(self.worker.state, feedback_at=time.monotonic() - 1)
        with self.assertRaises(RuntimeError):
            self.backend.request_home()
        self.assertEqual(self.worker.requests, [])
        self.backend._workspace = None
        with self.assertRaises(ValueError):
            self.backend.request_home()

    def test_sequence_feedback_and_faults_gate_firing(self):
        controller = TurretController(TurretConfig(), self.backend)
        controller.aiming.update = lambda *args: AimCommand(
            has_target=True, target_yaw_rad=self.yaw, target_pitch_rad=self.pitch,
            stable=True, fire_allowed=True,
        )
        self.worker.state = replace(self.worker.state, yaw_sequence_phase="rotating")
        self.assertFalse(controller.update(None).fire_allowed)
        self.worker.state = replace(self.worker.state, yaw_sequence_phase=None)
        self.assertTrue(controller.update(None).fire_allowed)
        self.worker.state = replace(self.worker.state, yaw_motor_err=8)
        self.assertFalse(controller.update(None).fire_allowed)

    def test_new_unacknowledged_target_cannot_fire(self):
        self.backend.send_targets(self.targets)
        changed_yaw = self.yaw + math.radians(.5)
        self.backend.send_targets({3: (changed_yaw, .06, 0, 0), 1: self.targets[1]})
        self.assertFalse(self.backend.motion_ready(changed_yaw, self.pitch))
        self.assertEqual(len(self.worker.requests), 2)

    def test_repeated_old_target_is_not_ready_before_latest_request_is_processed(self):
        self.backend.send_targets(self.targets)
        self.worker.auto_ack = False
        other = self.yaw + math.radians(.5)
        self.backend.send_targets({3: (other, .06, 0, 0), 1: self.targets[1]})
        self.backend.send_targets(self.targets)
        self.assertFalse(self.backend.motion_ready(self.yaw, self.pitch))

    def test_stale_feedback_cannot_authorize_firing(self):
        self.backend.send_targets(self.targets)
        self.worker.state = replace(self.worker.state, yaw_feedback_at=time.monotonic() - 1.)
        self.assertIsNone(self.backend.get_state(3))
        self.assertFalse(self.backend.motion_ready(self.yaw, self.pitch))

    def test_aim_stability_must_be_reestablished_after_motion_finishes(self):
        config = TurretConfig(use_axis_geometry=False, compensate_gravity=False,
                              compensate_parallax=False, stable_frames_required=3)
        self.worker.state = replace(
            self.worker.state,
            yaw_position_rad=config.yaw_zero_rad, position_rad=config.pitch_zero_rad,
            yaw_target_rad=config.yaw_zero_rad, target_rad=config.pitch_zero_rad,
            yaw_sequence_phase="rotating",
        )
        controller = TurretController(config, self.backend)
        target = TargetPixel(config.camera_center_x_px, config.camera_center_y_px)
        for _ in range(5):
            self.assertFalse(controller.update(target).fire_allowed)
        self.worker.state = replace(self.worker.state, yaw_sequence_phase=None)
        self.assertFalse(controller.update(target).fire_allowed)
        self.assertFalse(controller.update(target).fire_allowed)
        self.assertTrue(controller.update(target).fire_allowed)


if __name__ == "__main__":
    unittest.main()

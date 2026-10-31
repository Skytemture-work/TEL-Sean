import math
import time
import unittest
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
        target = TargetPixel(640., 360.)
        for _ in range(5):
            self.assertFalse(controller.update(target).fire_allowed)
        self.worker.state = replace(self.worker.state, yaw_sequence_phase=None)
        self.assertFalse(controller.update(target).fire_allowed)
        self.assertFalse(controller.update(target).fire_allowed)
        self.assertTrue(controller.update(target).fire_allowed)


if __name__ == "__main__":
    unittest.main()

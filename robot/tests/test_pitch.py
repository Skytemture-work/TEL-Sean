import math
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

from turret_control.pitch import PitchCalibration, save_manual_origin_calibration
from turret_control.config import TurretConfig
from turret_control.limits import (
    constrain_pitch_activity_limit,
    load_axis_limits,
    load_workspace_limit,
    max_activity_pitch_for_yaw_deg,
    workspace_contains,
)
from turret_control.run_vertical_panel import (
    DEFAULT_STARTUP_POSE,
    MAX_MUZZLE_ELEVATION_DEG,
    MIN_MUZZLE_ELEVATION_DEG,
    MotorWorker,
    _prompt_and_request_pose,
    _wait_and_print_enabled_pose,
    load_pitch_calibration,
)
from turret_control.pitch import load_yaw_calibration


class PitchCalibrationTests(unittest.TestCase):
    @staticmethod
    def make_worker(calibration=None, yaw_calibration=None):
        return MotorWorker(
            port="unused",
            pitch_calibration=calibration or PitchCalibration(motor_zero_rad=0.2),
            speed_rad_s=0.08,
            acceleration_rad_s2=0.08,
            torque_ff_nm=0.0,
            torque_ramp_s=1.0,
            kp=50.0,
            kd=1.5,
            max_tracking_error_deg=10.0,
            stall_timeout_s=2.0,
            min_muzzle_deg=MIN_MUZZLE_ELEVATION_DEG,
            max_muzzle_deg=MAX_MUZZLE_ELEVATION_DEG,
            yaw_calibration=yaw_calibration,
        )

    def test_level_muzzle_uses_calibrated_motor_zero(self):
        calibration = PitchCalibration(motor_zero_rad=0.191310)
        self.assertAlmostEqual(calibration.motor_from_muzzle(0.0), 0.191310)

    def test_direction_and_ratio_are_applied_and_invertible(self):
        calibration = PitchCalibration(
            motor_zero_rad=0.2,
            motor_rad_per_muzzle_rad=2.0,
            motor_sign=-1,
        )
        muzzle = math.radians(15.0)
        motor = calibration.motor_from_muzzle(muzzle)
        self.assertAlmostEqual(motor, 0.2 - 2.0 * muzzle)
        self.assertAlmostEqual(calibration.muzzle_from_motor(motor), muzzle)

    def test_project_calibration_maps_requested_muzzle_angle(self):
        calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        stored = json.loads(DEFAULT_STARTUP_POSE.read_text())["vertical"]
        motor_at_zero = calibration.motor_from_muzzle(0.0)
        motor_at_fifteen = calibration.motor_from_muzzle(math.radians(15.0))
        motor_at_45 = float(stored["angle_rad"])
        level_reference = float(stored["level_reference_angle_rad"])
        self.assertAlmostEqual(motor_at_zero, level_reference)
        self.assertAlmostEqual(calibration.motor_from_muzzle(math.radians(45.0)), motor_at_45)
        self.assertAlmostEqual(motor_at_fifteen, level_reference + (motor_at_45 - level_reference) / 3)
        self.assertGreater(motor_at_zero, motor_at_fifteen)

    def test_centered_activity_range_is_in_workspace(self):
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        center = yaw.motor_from_muzzle(0.0)
        for elevation in (0.0, 15.0, 30.0):
            self.assertTrue(workspace_contains(
                workspace, center, pitch.motor_from_muzzle(math.radians(elevation))
            ), msg=f"centered elevation {elevation}° is outside workspace")

    def test_activity_pitch_limit_narrows_linearly_with_yaw(self):
        self.assertAlmostEqual(max_activity_pitch_for_yaw_deg(0.0), 35.0)
        self.assertAlmostEqual(max_activity_pitch_for_yaw_deg(12.5), 32.5)
        self.assertAlmostEqual(max_activity_pitch_for_yaw_deg(25.0), 30.0)
        self.assertAlmostEqual(max_activity_pitch_for_yaw_deg(34.5), 30.0)

    def test_captured_initial_motor_pose_is_forty_five_degrees_up(self):
        calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        initial_motor_rad = calibration.motor_from_muzzle(math.radians(45.0))
        muzzle_deg = math.degrees(calibration.muzzle_from_motor(initial_motor_rad))
        self.assertAlmostEqual(muzzle_deg, 45.0)

    def test_global_pitch_limits_match_measured_mechanism_limits(self):
        config = TurretConfig()
        calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        limits = load_axis_limits()["pitch"]
        self.assertEqual(config.pitch_sign, -1)
        activity = constrain_pitch_activity_limit((limits.min_rad, limits.max_rad), calibration)
        self.assertAlmostEqual(config.pitch_min_rad, activity[0])
        self.assertAlmostEqual(config.pitch_max_rad, activity[1])
        physical_limits = sorted(
            math.degrees(calibration.muzzle_from_motor(value))
            for value in (config.pitch_min_rad, config.pitch_max_rad)
        )
        self.assertGreaterEqual(physical_limits[0], 0.0)
        self.assertAlmostEqual(physical_limits[1], 35.0)
        self.assertAlmostEqual(max(math.degrees(calibration.muzzle_from_motor(v)) for v in activity), 35.0)

    def test_home_exit_corridor_allows_ten_degrees_yaw_above_activity_range(self):
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        pitch_pos = pitch.motor_from_muzzle(math.radians(32.5))
        for angle in (-10.0, 0.0, 10.0):
            self.assertTrue(
                workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
            )
        for angle in (-20.0, 20.0):
            self.assertFalse(
                workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
            )

        for elevation in (33.0, 34.0, 35.0, 45.0):
            pitch_pos = pitch.motor_from_muzzle(math.radians(elevation))
            for angle in (-10.0, 0.0, 10.0):
                self.assertTrue(
                    workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
                )
            for angle in (-10.1, 10.1):
                self.assertFalse(
                    workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
                )

    def test_explicit_home_centers_before_rising_above_activity_range(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker.max_muzzle_rad = math.radians(30)
        worker.max_pitch_for_yaw_rad = math.radians(30)
        worker.yaw_release_pitch_rad = math.radians(30)
        worker.yaw_clearance_pitch_rad = math.radians(29.5)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(20))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(math.radians(10))
        worker._enabled = True
        requested = []
        worker._motors = SimpleNamespace(set_home_return=requested.append)
        worker._set_aim_targets(0.0, 45.0, home_return=True)
        self.assertEqual(requested, [True])
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(worker.snapshot().target_muzzle_rad, math.radians(29.5))
        self.assertAlmostEqual(worker.snapshot().yaw_restore_muzzle_rad, math.radians(45))

    def test_horizontal_reference_and_limits_match_measured_workspace(self):
        calibration = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        config = TurretConfig()
        limits = load_axis_limits()["yaw"]
        self.assertAlmostEqual(
            calibration.motor_from_muzzle(0.0),
            config.yaw_zero_rad,
        )
        self.assertEqual(calibration.motor_sign, -1)
        self.assertEqual(config.yaw_sign, -1)
        self.assertAlmostEqual(config.yaw_min_rad, limits.min_rad)
        self.assertAlmostEqual(config.yaw_max_rad, limits.max_rad)
        physical_limits = sorted(
            math.degrees(calibration.muzzle_from_motor(value))
            for value in (config.yaw_min_rad, config.yaw_max_rad)
        )
        self.assertGreaterEqual(physical_limits[0], -35.0)
        self.assertLessEqual(physical_limits[1], 35.0)

    def test_horizontal_motion_uses_clearance_sequence_above_thirty_five_degrees(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=1)
        worker = self.make_worker(pitch, yaw)
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(9.9))
        worker._set_yaw_target(30.0)
        self.assertAlmostEqual(
            worker.snapshot().yaw_target_angle_rad,
            math.radians(30.0),
        )
        self.assertIsNone(worker.snapshot().yaw_sequence_phase)

        worker._set_target(30.5)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(45.0))
        worker._set_yaw_target(20.0)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(30.5))
        self.assertAlmostEqual(snapshot.yaw_restore_muzzle_rad, math.radians(30.5))
        self.assertAlmostEqual(snapshot.yaw_target_angle_rad, math.radians(20.0))

        with self.assertRaises(ValueError):
            worker._set_yaw_target(35.01)

    def test_combined_aim_sets_both_targets_atomically(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(12.0))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)

        worker._set_aim_targets(20.0, 15.0)

        snapshot = worker.snapshot()
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(15.0))
        self.assertAlmostEqual(snapshot.yaw_target_angle_rad, math.radians(20.0))
        self.assertAlmostEqual(snapshot.target_rad, 0.2 - math.radians(15.0))
        self.assertAlmostEqual(snapshot.yaw_target_rad, 0.5 - math.radians(20.0))

        worker._set_aim_targets(-10.0, 36.0)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(32.5))
        self.assertAlmostEqual(snapshot.yaw_restore_muzzle_rad, math.radians(36.0))
        self.assertAlmostEqual(snapshot.yaw_target_angle_rad, math.radians(-10.0))

    def test_yaw_clearance_sequence_rotates_then_restores_original_pitch(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(45.0))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)
        worker._set_target(30.5)

        worker._set_yaw_target(20.0)
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "clearance")

        safe_command = pitch.motor_from_muzzle(math.radians(30.5))
        boundary_feedback = pitch.motor_from_muzzle(math.radians(30.4))
        worker._last_command_rad = safe_command
        worker._feedback_position_rad = boundary_feedback
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")
        self.assertAlmostEqual(
            worker.snapshot().target_muzzle_rad,
            math.radians(30.5),
        )

        yaw_target = yaw.motor_from_muzzle(math.radians(20.0))
        worker._yaw_last_command_rad = yaw_target
        worker._yaw_feedback_position_rad = yaw_target
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "restoring")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(30.5))

        restored_motor = pitch.motor_from_muzzle(math.radians(30.5))
        worker._last_command_rad = restored_motor
        worker._feedback_position_rad = restored_motor
        worker._update_yaw_clearance_sequence()
        self.assertIsNone(worker.snapshot().yaw_sequence_phase)

    def test_aim_from_high_pitch_starts_yaw_below_35_while_pitch_continues(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(45.0))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)

        worker._set_aim_targets(20.0, 20.0)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(20.0))

        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(30.4))
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "rotating")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(20.0))

    def test_exactly_35_degree_final_uses_safe_intermediate_pose(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(45.0))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)

        worker._set_aim_targets(10.0, 35.0)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(32.5))
        self.assertAlmostEqual(snapshot.yaw_restore_muzzle_rad, math.radians(35.0))

        safe_command = pitch.motor_from_muzzle(math.radians(32.5))
        boundary_feedback = pitch.motor_from_muzzle(math.radians(32.4))
        worker._last_command_rad = safe_command
        worker._feedback_position_rad = boundary_feedback
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")

        yaw_target = yaw.motor_from_muzzle(math.radians(10.0))
        worker._yaw_last_command_rad = yaw_target
        worker._yaw_feedback_position_rad = yaw_target
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "restoring")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(35.0))

    def test_home_waits_for_yaw_zero_and_safe_pitch_then_returns_to_45(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(20.0))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(math.radians(15.0))

        worker._set_aim_targets(0.0, 45.0)
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "rotating")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(34.5))

        yaw_zero = yaw.motor_from_muzzle(0.0)
        worker._yaw_last_command_rad = yaw_zero
        worker._yaw_feedback_position_rad = yaw_zero
        pitch_at_30 = pitch.motor_from_muzzle(math.radians(30.0))
        worker._last_command_rad = pitch_at_30
        worker._feedback_position_rad = pitch_at_30
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")

        safe_pitch = pitch.motor_from_muzzle(math.radians(34.5))
        worker._last_command_rad = safe_pitch
        worker._feedback_position_rad = safe_pitch
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "restoring")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(45.0))

    def test_yaw_interlock_latches_position_instead_of_following_drift(self):
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(yaw_calibration=yaw)
        worker._yaw_feedback_position_rad = 0.50
        self.assertEqual(worker._yaw_interlock_hold_position(), 0.50)

        worker._yaw_feedback_position_rad = 0.60
        self.assertEqual(worker._yaw_interlock_hold_position(), 0.50)

        worker._yaw_interlock_hold_rad = None
        self.assertEqual(worker._yaw_interlock_hold_position(), 0.60)

    def test_enable_confirmation_prints_physical_two_axis_pose(self):
        pitch = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._publish(
            enabled=True,
            position_rad=pitch.motor_from_muzzle(math.radians(45.0)),
            yaw_position_rad=yaw.motor_from_muzzle(math.radians(-12.5)),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _wait_and_print_enabled_pose(worker, timeout_s=0.0)

        self.assertIn("垂直：+45.00°", output.getvalue())
        self.assertIn("水平：-12.50°", output.getvalue())

    def test_interactive_pose_prompt_queues_vertical_then_horizontal_input(self):
        yaw = PitchCalibration(motor_zero_rad=0.5, motor_sign=-1)
        worker = self.make_worker(yaw_calibration=yaw)
        output = io.StringIO()
        with patch("builtins.input", side_effect=("20", "-15")):
            with redirect_stdout(output):
                accepted = _prompt_and_request_pose(worker, 0.0, 45.0)

        self.assertTrue(accepted)
        self.assertEqual(worker._commands.get_nowait(), ("aim", (-15.0, 20.0)))
        self.assertIn("垂直 +20.00°、水平 -15.00°", output.getvalue())

    def test_manual_origin_capture_is_atomic_and_keeps_a_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "startup_pose.json"
            shutil.copy2(DEFAULT_STARTUP_POSE, calibration_path)
            original_pitch = load_pitch_calibration(calibration_path)
            captured_yaw = 0.52
            captured_pitch = -0.61

            backup_path = save_manual_origin_calibration(
                calibration_path,
                yaw_motor_rad=captured_yaw,
                pitch_motor_rad=captured_pitch,
                pitch_reference_deg=45.0,
            )

            self.assertTrue(backup_path.is_file())
            yaw = load_yaw_calibration(calibration_path)
            pitch = load_pitch_calibration(calibration_path)
            self.assertAlmostEqual(yaw.motor_from_muzzle(0.0), captured_yaw)
            self.assertAlmostEqual(
                pitch.motor_from_muzzle(math.radians(45.0)),
                captured_pitch,
            )
            old_pitch = load_pitch_calibration(backup_path)
            self.assertEqual(old_pitch, original_pitch)

    def test_invalid_ratio_and_sign_are_rejected(self):
        with self.assertRaises(ValueError):
            PitchCalibration(0.0, motor_rad_per_muzzle_rad=0.0)
        with self.assertRaises(ValueError):
            PitchCalibration(0.0, motor_sign=0)

    def test_worker_converts_muzzle_target_before_motion_control(self):
        calibration = PitchCalibration(motor_zero_rad=0.2, motor_sign=-1)
        worker = self.make_worker(calibration)
        worker._set_target(15.0)
        snapshot = worker.snapshot()
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(15.0))
        self.assertAlmostEqual(snapshot.target_rad, 0.2 - math.radians(15.0))

    def test_worker_rejects_muzzle_targets_outside_zero_to_forty_five(self):
        worker = self.make_worker()
        for angle in (-0.01, 45.01):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                worker._set_target(angle)

    def test_stall_warning_preserves_requested_target(self):
        worker = self.make_worker()
        worker._feedback_position_rad = 0.2
        worker._set_target(15.0)
        requested_target = worker._target_rad
        first_window_end = worker._stall_reference_at + worker.stall_timeout_s + 0.01

        worker._update_stall_monitor(0.2, first_window_end)

        snapshot = worker.snapshot()
        self.assertEqual(worker._target_rad, requested_target)
        self.assertEqual(snapshot.target_rad, requested_target)
        self.assertEqual(snapshot.error, "stall_warning")

        worker._update_stall_monitor(
            0.2 + math.radians(0.5),
            first_window_end + worker.stall_timeout_s + 0.01,
        )
        self.assertIsNone(worker.snapshot().error)


if __name__ == "__main__":
    unittest.main()

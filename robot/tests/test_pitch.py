import math
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

from turret_control.pitch import PitchCalibration, save_level_reference_calibration, save_manual_origin_calibration
from turret_control.config import TurretConfig
from turret_control.limits import (
    constrain_pitch_activity_limit,
    WorkspaceLimit,
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
        home_motor = float(stored["angle_rad"])
        level_reference = float(stored["level_reference_angle_rad"])
        self.assertAlmostEqual(motor_at_zero, level_reference)
        self.assertAlmostEqual(calibration.motor_rad_per_muzzle_rad, 1.0)
        self.assertAlmostEqual(motor_at_fifteen, level_reference - math.radians(15.0))
        self.assertAlmostEqual(math.degrees(calibration.muzzle_from_motor(home_motor)), 44.63186942390103)
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

    def test_home_motor_pose_is_independent_of_design_forty_five_degrees(self):
        calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        home_motor = workspace.home_corridor.home_pitch_rad
        self.assertAlmostEqual(home_motor, -0.5579079985618591)
        self.assertAlmostEqual(math.degrees(calibration.muzzle_from_motor(home_motor)), 44.63186942390103)
        self.assertNotAlmostEqual(calibration.motor_from_muzzle(math.radians(45.0)), home_motor)

    def test_legacy_pitch_calibration_without_level_reference_remains_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"vertical": {
                "angle_rad": -.5, "muzzle_elevation_deg_at_angle": 45.,
                "motor_rad_per_muzzle_rad": .9, "motor_sign": -1,
            }}))
            calibration = load_pitch_calibration(path)
            self.assertAlmostEqual(calibration.motor_from_muzzle(math.radians(45.)), -.5)

    def test_home_command_uses_recorded_encoder_target_not_45_degrees(self):
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        worker = self.make_worker(pitch, yaw)
        worker._enabled = True
        worker._motors = SimpleNamespace(_workspace=workspace, set_home_return=lambda enabled: None)
        worker._yaw_feedback_position_rad = worker._yaw_last_command_rad = yaw.motor_from_muzzle(0.)
        worker._feedback_position_rad = worker._last_command_rad = pitch.motor_from_muzzle(math.radians(20.))
        worker._handle_command(("home", None))
        self.assertIsNone(worker.snapshot().error)
        self.assertAlmostEqual(worker.snapshot().target_rad, workspace.home_corridor.home_pitch_rad)
        self.assertAlmostEqual(math.degrees(worker.snapshot().target_muzzle_rad), 44.63186942390103)

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

        home_deg = math.degrees(pitch.muzzle_from_motor(workspace.home_corridor.home_pitch_rad))
        for elevation in (33.0, 34.0, 35.0, home_deg):
            pitch_pos = pitch.motor_from_muzzle(math.radians(elevation))
            for angle in (-10.0, 0.0, 10.0):
                self.assertTrue(
                    workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
                )
            for angle in (-10.1, 10.1):
                self.assertFalse(
                    workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(angle)), pitch_pos)
                )

    def test_confirmed_level_based_workspace_boundaries(self):
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()

        def inside(horizontal, elevation):
            return workspace_contains(
                workspace, yaw.motor_from_muzzle(math.radians(horizontal)),
                pitch.motor_from_muzzle(math.radians(elevation)),
            )

        for elevation, half_width in ((0., 34.5), (29.99, 34.5), (30., 34.5),
                                      (31., 20.), (32., 15.), (32.5, 12.5),
                                      (33., 10.), (34., 10.), (35., 10.)):
            for sign in (-1, 1):
                with self.subTest(elevation=elevation, sign=sign):
                    self.assertTrue(inside(sign * half_width, elevation))
                    self.assertFalse(inside(sign * (half_width + .01), elevation))
        home = math.degrees(pitch.muzzle_from_motor(workspace.home_corridor.home_pitch_rad))
        self.assertTrue(inside(10., home))
        self.assertFalse(inside(10.01, home))
        self.assertFalse(inside(0., home + .01))

    def test_restored_30_degree_wide_target_has_valid_planned_path(self):
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        worker = self.make_worker(pitch, yaw)
        worker._motors = SimpleNamespace(_workspace=workspace, set_home_return=lambda value: None)
        worker._yaw_feedback_position_rad = worker._yaw_last_command_rad = yaw.motor_from_muzzle(0.)
        worker._feedback_position_rad = worker._last_command_rad = pitch.motor_from_muzzle(math.radians(10.))
        worker._set_aim_targets(30., 30.)
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(math.degrees(worker.yaw_clearance_pitch_rad), 29.5)

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
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "clearance")

        worker = self.make_worker(pitch, yaw)
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.0)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(9.9))
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
        self.assertEqual(worker._deferred_aim, (-10.0, 36.0, False))
        self.assertAlmostEqual(snapshot.yaw_target_angle_rad, math.radians(20.0))
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(15.0))
        worker._last_command_rad = worker._feedback_position_rad
        worker._update_yaw_clearance_sequence()
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(math.radians(20.0))
        worker._yaw_last_command_rad = worker._yaw_feedback_position_rad
        worker._update_yaw_clearance_sequence()
        worker._update_yaw_clearance_sequence()
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.yaw_sequence_phase, "clearance")
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(30.5))
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

    def test_clearance_accounts_for_start_yaw_and_path_stays_inside_workspace(self):
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        worker = self.make_worker(pitch, yaw)
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(math.radians(30))
        worker._yaw_last_command_rad = worker._yaw_feedback_position_rad
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(29.7))
        worker._last_command_rad = worker._feedback_position_rad
        worker._set_aim_targets(0, 34.5)
        self.assertAlmostEqual(math.degrees(worker.yaw_clearance_pitch_rad), 29.5)
        points = [(30., 29.7), (30., 29.5), (0., 29.5), (0., 34.5)]
        for start, end in zip(points, points[1:]):
            for index in range(101):
                fraction = index / 100
                y = start[0] + fraction * (end[0] - start[0])
                p = start[1] + fraction * (end[1] - start[1])
                self.assertTrue(workspace_contains(
                    workspace, yaw.motor_from_muzzle(math.radians(y)),
                    pitch.motor_from_muzzle(math.radians(p)),
                ), (y, p))

    def test_yaw_waits_for_clearance_and_stops_if_pitch_drifts_high(self):
        pitch = PitchCalibration(motor_zero_rad=.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(30.4))
        worker._set_aim_targets(20., 30.)
        worker._feedback_velocity_rad_s = -.02  # physical pitch is rising
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "clearance")
        worker._feedback_velocity_rad_s = 0.
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")
        self.assertTrue(worker._yaw_feedback_clear(math.radians(30.5)))
        self.assertFalse(worker._yaw_feedback_clear(math.radians(30.6)))

    def test_planned_segments_cover_activity_and_home_corridor_grid(self):
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        workspace = load_workspace_limit()
        yaws = (-34., -25., -10., 0., 10., 25., 34.)
        pitches = (0., 15., 29., 30., 31., 32., 33., 34., 35., 40., 45.)

        def inside(y, p):
            return workspace_contains(workspace, yaw.motor_from_muzzle(math.radians(y)),
                                      pitch.motor_from_muzzle(math.radians(p)))

        starts = [(y, p) for y in yaws for p in pitches if inside(y, p)]
        goals = [(y, p) for y in yaws for p in pitches if p <= 35. and inside(y, p)]
        for start in starts:
            for goal in goals:
                worker = self.make_worker(pitch, yaw)
                worker._yaw_feedback_position_rad = worker._yaw_last_command_rad = yaw.motor_from_muzzle(math.radians(start[0]))
                worker._feedback_position_rad = worker._last_command_rad = pitch.motor_from_muzzle(math.radians(start[1]))
                if not worker._pose_allowed_by_activity(math.radians(goal[0]), math.radians(goal[1])):
                    continue
                worker._set_aim_targets(*goal)
                stage = math.degrees(worker._target_muzzle_rad)
                path = ([start, (start[0], stage), (goal[0], stage), goal]
                        if worker._yaw_sequence_phase else [start, goal])
                for a, b in zip(path, path[1:]):
                    for index in range(11):
                        fraction = index / 10.
                        point = (a[0] + fraction * (b[0] - a[0]),
                                 a[1] + fraction * (b[1] - a[1]))
                        self.assertTrue(inside(*point), (start, goal, point))

    def test_recorded_nonconvex_workspace_rejects_unsafe_intermediate_path(self):
        pitch = PitchCalibration(0., motor_sign=1)
        yaw = PitchCalibration(0., motor_sign=1)
        worker = self.make_worker(pitch, yaw)
        workspace = WorkspaceLimit(
            yaw_motor_id=3, pitch_motor_id=1,
            vertices=((-0.6, 0.), (.6, 0.), (.6, .6), (.1, .6),
                      (.1, .15), (-.1, .15), (-.1, .6), (-.6, .6)),
            anchor=(-.4, .1),
        )
        worker._motors = SimpleNamespace(_workspace=workspace, set_home_return=lambda value: None)
        worker._yaw_feedback_position_rad = worker._yaw_last_command_rad = -.4
        worker._feedback_position_rad = worker._last_command_rad = .3
        with self.assertRaisesRegex(ValueError, "規劃路徑"):
            worker._set_aim_targets(math.degrees(.4), math.degrees(.3))
        self.assertIsNone(worker.snapshot().yaw_sequence_phase)

    def test_running_worker_holds_when_feedback_stales_and_replans_after_recovery(self):
        class FakeBackend:
            def __init__(self, **kwargs):
                now = time.monotonic()
                self.states = {
                    1: SimpleNamespace(position_rad=.2, velocity_rad_s=0., torque_nm=0., err=0, feedback_at=now),
                    3: SimpleNamespace(position_rad=.5, velocity_rad_s=0., torque_nm=0., err=0, feedback_at=now),
                }
                self.drop = False
                self.hold_count = 0
                self._home_return_requested = False

            def refresh(self):
                if not self.drop:
                    now = time.monotonic()
                    for state in self.states.values():
                        state.feedback_at = now

            def get_state(self, motor_id):
                return self.states[motor_id]

            def enable(self):
                for state in self.states.values():
                    state.err = 1
                self.refresh()

            def disable(self):
                for state in self.states.values():
                    state.err = 0

            def set_home_return(self, enabled):
                self._home_return_requested = enabled

            def send_targets(self, targets):
                for motor_id, values in targets.items():
                    self.states[motor_id].position_rad = values[0]

            def hold(self):
                self.hold_count += 1

            def close(self):
                pass

        def wait_for(predicate, timeout=2.):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(.01)
            return False

        worker = self.make_worker(PitchCalibration(.2, motor_sign=-1),
                                  PitchCalibration(.5, motor_sign=-1))
        with patch("turret_control.run_vertical_panel.DamiaoSerialBackend", FakeBackend):
            worker.start()
            try:
                self.assertTrue(wait_for(lambda: worker.snapshot().ready))
                worker.request_enable()
                self.assertTrue(wait_for(lambda: worker.snapshot().enabled))
                worker.request_aim(20., 15.)
                self.assertTrue(wait_for(lambda: worker.snapshot().yaw_sequence_phase is not None))
                worker._motors.drop = True
                self.assertTrue(wait_for(lambda: worker.snapshot().error == "feedback_stale"))
                self.assertGreater(worker._motors.hold_count, 0)
                worker._motors.drop = False
                self.assertTrue(wait_for(lambda: worker.snapshot().error is None))
                self.assertTrue(worker.snapshot().enabled)
            finally:
                worker.request_disable()
                wait_for(lambda: not worker.snapshot().enabled)
                worker.request_shutdown()
                worker.wait_closed()

    def test_does_not_restore_at_home_corridor_edge_until_yaw_is_inside_and_stopped(self):
        pitch = PitchCalibration(motor_zero_rad=.2, motor_sign=-1)
        yaw = PitchCalibration(motor_zero_rad=.5, motor_sign=-1)
        worker = self.make_worker(pitch, yaw)
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(0.)
        worker._feedback_position_rad = pitch.motor_from_muzzle(math.radians(32.5))
        worker._last_command_rad = worker._feedback_position_rad
        worker._set_aim_targets(10., 35.)
        worker._update_yaw_clearance_sequence()
        worker._yaw_last_command_rad = yaw.motor_from_muzzle(math.radians(10))
        worker._yaw_feedback_position_rad = yaw.motor_from_muzzle(math.radians(10.03))
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")
        worker._yaw_feedback_position_rad = worker._yaw_last_command_rad
        worker._yaw_feedback_velocity_rad_s = .02
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")
        worker._yaw_feedback_velocity_rad_s = 0.
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "restoring")

    def test_deferred_invalid_path_reports_error_without_stopping_worker(self):
        worker = self.make_worker(PitchCalibration(.2, motor_sign=-1),
                                  PitchCalibration(.5, motor_sign=-1))
        worker._yaw_sequence_phase = "restoring"
        worker._yaw_restore_muzzle_rad = math.radians(15.)
        worker._feedback_position_rad = worker._last_command_rad = worker.pitch_calibration.motor_from_muzzle(math.radians(15.))
        worker._deferred_aim = (20., 15., False)
        with patch.object(worker, "_set_aim_targets", side_effect=ValueError("路徑不安全")):
            worker._update_yaw_clearance_sequence()
        self.assertIsNone(worker.snapshot().yaw_sequence_phase)
        self.assertEqual(worker.snapshot().error, "路徑不安全")

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
        self.assertAlmostEqual(snapshot.target_muzzle_rad, math.radians(31.5))

        yaw_zero = yaw.motor_from_muzzle(0.0)
        worker._yaw_last_command_rad = yaw_zero
        worker._yaw_feedback_position_rad = yaw_zero
        pitch_at_30 = pitch.motor_from_muzzle(math.radians(30.0))
        worker._last_command_rad = pitch_at_30
        worker._feedback_position_rad = pitch_at_30
        worker._update_yaw_clearance_sequence()
        self.assertEqual(worker.snapshot().yaw_sequence_phase, "rotating")

        safe_pitch = pitch.motor_from_muzzle(math.radians(31.5))
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
            self.assertAlmostEqual(pitch.motor_zero_rad, original_pitch.motor_zero_rad)
            saved = json.loads(calibration_path.read_text())["vertical"]
            self.assertAlmostEqual(saved["angle_rad"], captured_pitch)
            self.assertAlmostEqual(
                math.radians(saved["muzzle_elevation_deg_at_angle"]),
                pitch.muzzle_from_motor(captured_pitch),
            )
            old_pitch = load_pitch_calibration(backup_path)
            self.assertEqual(old_pitch, original_pitch)

    def test_level_capture_updates_zero_but_preserves_home_and_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup_pose.json"
            shutil.copy2(DEFAULT_STARTUP_POSE, path)
            before = json.loads(path.read_text())["vertical"]
            new_level = before["level_reference_angle_rad"] - .002
            backup = save_level_reference_calibration(path, new_level)
            after = json.loads(path.read_text())["vertical"]
            self.assertTrue(backup.is_file())
            self.assertAlmostEqual(load_pitch_calibration(path).motor_zero_rad, new_level)
            self.assertEqual(after["angle_rad"], before["angle_rad"])
            self.assertEqual(after["motor_rad_per_muzzle_rad"], before["motor_rad_per_muzzle_rad"])
            self.assertAlmostEqual(math.radians(after["muzzle_elevation_deg_at_angle"]),
                                   load_pitch_calibration(path).muzzle_from_motor(before["angle_rad"]))

    def test_recorded_home_is_preserved_and_corridor_begins_at_33_degrees(self):
        workspace = load_workspace_limit()
        calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        self.assertAlmostEqual(workspace.home_corridor.home_pitch_rad, -.5579079985618591)
        self.assertAlmostEqual(workspace.home_corridor.transition_pitch_rad, -.3548935789527302)
        self.assertAlmostEqual(math.degrees(calibration.muzzle_from_motor(
            workspace.home_corridor.transition_pitch_rad)), 33.0)

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

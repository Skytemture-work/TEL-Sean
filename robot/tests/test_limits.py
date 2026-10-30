import math
import io
import json
import numpy as np
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from turret_control.backends import DamiaoSerialBackend, MotorState, PositionCommand
from turret_control.config import TurretConfig
from turret_control.limits import (
    HomeCorridor,
    WorkspaceLimit,
    clamp_workspace_target,
    clamp_motor_position,
    constrain_yaw_design_limit,
    constrain_pitch_activity_limit,
    load_axis_limits,
    load_motor_limits,
    load_workspace_limit,
    save_manual_limits,
    save_workspace_limits,
    workspace_contains,
)
from turret_control.run_limit_calibration import _save_recovery_trace
from turret_control.pitch import PitchCalibration, load_pitch_calibration
from turret_control.run_vertical_panel import DEFAULT_STARTUP_POSE


class MotorLimitTests(unittest.TestCase):
    def test_yaw_design_limit_uses_calibration_and_preserves_stricter_limits(self):
        for sign in (-1, 1):
            calibration = PitchCalibration(0.5, 2.0, sign)
            lower, upper = constrain_yaw_design_limit((-10.0, 10.0), calibration)
            physical = sorted(math.degrees(calibration.muzzle_from_motor(v)) for v in (lower, upper))
            self.assertAlmostEqual(physical[0], -34.5)
            self.assertAlmostEqual(physical[1], 34.5)
            self.assertEqual(constrain_yaw_design_limit((0.4, 0.6), calibration), (0.4, 0.6))

    def test_enable_rejects_yaw_outside_fixed_limit(self):
        backend = DamiaoSerialBackend.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: object()}
        backend._position_limits = {0x03: (-0.6, 0.6)}
        backend.refresh = lambda: None
        backend.get_state = lambda motor_id: MotorState(motor_id, 0.61, 0.0, 0.0, 0)
        with self.assertRaisesRegex(RuntimeError, "outside.*safety limit"):
            backend.enable()

    def test_pitch_activity_limit_and_enable_reject_outside_position(self):
        for sign in (-1, 1):
            calibration = PitchCalibration(0.5, 2.0, sign)
            lower, upper = constrain_pitch_activity_limit((-10.0, 10.0), calibration)
            physical = sorted(math.degrees(calibration.muzzle_from_motor(v)) for v in (lower, upper))
            self.assertAlmostEqual(physical[0], 0.0)
            self.assertAlmostEqual(physical[1], 30.0)
        backend = DamiaoSerialBackend.__new__(DamiaoSerialBackend)
        backend._motors = {0x01: object()}
        backend._position_limits = {0x01: (0.0, 0.6)}
        backend.refresh = lambda: None
        backend.get_state = lambda motor_id: MotorState(motor_id, 0.61, 0.0, 0.0, 0)
        with self.assertRaisesRegex(RuntimeError, "pitch.*0–45° travel limit"):
            backend.enable()

    def test_home_hold_exit_and_activity_pitch_cap_use_feedback(self):
        from turret_control.pitch import load_yaw_calibration

        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        yaw = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        center = yaw.motor_from_muzzle(0.0)
        home = pitch.motor_from_muzzle(math.radians(45.0))
        feedback = {0x03: center, 0x01: home}
        backend = DamiaoSerialBackend.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: object(), 0x01: object()}
        backend._position_limits = load_motor_limits()
        backend._pitch_calibration = pitch
        backend._workspace = load_workspace_limit()
        backend._last_workspace_notice = None
        backend._last = {}
        backend._enabled = True
        backend.get_state = lambda motor_id: MotorState(motor_id, feedback[motor_id], 0.0, 0.0, 1)
        sent = {}
        backend._send_target = lambda motor_id, position, speed, velocity, torque: sent.update({motor_id: position})
        backend.send_targets({0x03: (center, 0.2, 0.0, 0.0), 0x01: (home, 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x01], home)
        feedback[0x03] = yaw.motor_from_muzzle(math.radians(9))
        backend.send_targets({0x03: (feedback[0x03], 0.2, 0.0, 0.0), 0x01: (home, 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x03], feedback[0x03])
        self.assertAlmostEqual(sent[0x01], home)
        backend.send_targets({0x03: (yaw.motor_from_muzzle(math.radians(10)), 0.2, 0.0, 0.0),
                              0x01: (pitch.motor_from_muzzle(math.radians(25)), 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x03], center)
        self.assertAlmostEqual(sent[0x01], pitch.motor_from_muzzle(math.radians(25)))
        feedback[0x01] = pitch.motor_from_muzzle(math.radians(20))
        feedback[0x03] = center
        backend.send_targets({0x03: (center, 0.2, 0.0, 0.0), 0x01: (home, 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x01], pitch.motor_from_muzzle(math.radians(30)))
        feedback[0x03] = yaw.motor_from_muzzle(math.radians(9))
        backend.set_home_return(True)
        backend.send_targets({0x03: (center, 0.2, 0.0, 0.0), 0x01: (home, 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x03], center)
        self.assertAlmostEqual(sent[0x01], feedback[0x01])
        feedback[0x03] = center
        backend.send_targets({0x03: (center, 0.2, 0.0, 0.0), 0x01: (home, 0.2, 0.0, 0.0)})
        self.assertAlmostEqual(sent[0x01], home)

    def test_recovery_trace_serializes_numpy_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism_limits_trace_latest.json"
            _save_recovery_trace(
                path,
                samples=[
                    {
                        "elapsed_s": np.float32(0.05),
                        "yaw_rad": np.float32(0.52),
                        "pitch_rad": np.float32(-0.31),
                    }
                ],
                duration_s=np.float32(30.0),
                sample_hz=np.float32(20.0),
            )

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(stored["samples"][0]["elapsed_s"], 0.05, places=6)
            self.assertAlmostEqual(stored["samples"][0]["yaw_rad"], 0.52, places=6)
            self.assertAlmostEqual(stored["samples"][0]["pitch_rad"], -0.31, places=6)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_default_limits_are_loaded_in_motor_space(self):
        limits = load_motor_limits()
        config = TurretConfig()
        self.assertEqual(limits[config.yaw_motor_id], (config.yaw_min_rad, config.yaw_max_rad))
        pitch = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        self.assertEqual(constrain_pitch_activity_limit(limits[config.pitch_motor_id], pitch),
                         (config.pitch_min_rad, config.pitch_max_rad))

    def test_clamp_reports_out_of_range_target(self):
        self.assertEqual(clamp_motor_position(0.5, (0.0, 1.0)), (0.5, False))
        clipped, changed = clamp_motor_position(-0.1, (0.0, 1.0))
        self.assertEqual(clipped, 0.0)
        self.assertTrue(changed)

    def test_manual_capture_normalizes_endpoints_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism_limits.json"
            save_manual_limits(
                path,
                yaw_positions={"left_rad": 1.0, "right_rad": -1.0},
                pitch_positions={"low_rad": 0.8, "high_rad": -0.4},
                margin_motor_deg=math.degrees(0.1),
            )
            limits = load_axis_limits(path)
            self.assertAlmostEqual(limits["yaw"].min_rad, -0.9)
            self.assertAlmostEqual(limits["yaw"].max_rad, 0.9)
            self.assertAlmostEqual(limits["pitch"].min_rad, -0.3)
            self.assertAlmostEqual(limits["pitch"].max_rad, 0.7)

            save_manual_limits(
                path,
                yaw_positions={"left_rad": 1.1, "right_rad": -1.1},
                pitch_positions={"low_rad": 0.9, "high_rad": -0.5},
            )
            self.assertTrue(path.with_suffix(".json.bak").is_file())

    def test_backend_clamps_position_before_sending(self):
        class FakeMotorControl:
            def __init__(self):
                self.position = None

            def control_Pos_Vel(self, _motor, position, _speed, receive=False):
                self.position = position
                self.receive = receive

        backend = object.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: object()}
        backend._position_limits = {0x03: (0.0, 1.0)}
        backend._last_limit_notice = {}
        backend.mode = "posvel"
        backend.vmax = 2.0
        backend.tmax = 3.0
        backend.kp = 1.0
        backend.kd = 1.0
        backend._mc = FakeMotorControl()
        backend._last = {}

        output = io.StringIO()
        with redirect_stdout(output):
            backend._send_target(0x03, -0.5, 0.5)
        self.assertEqual(backend._mc.position, 0.0)
        self.assertIn("SAFETY LIMIT", output.getvalue())
        self.assertEqual(backend._last[0x03].position_rad, 0.0)

    def test_concave_workspace_rejects_safe_axis_pair_in_collision_area(self):
        # This L shape is non-rectangular: (2, 2) is inside the independent
        # min/max box, but outside the safe mechanical workspace.
        workspace = WorkspaceLimit(
            yaw_motor_id=0x03,
            pitch_motor_id=0x01,
            vertices=((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)),
            anchor=(0.5, 0.5),
            margin_rad=0.05,
        )
        self.assertTrue(workspace_contains(workspace, 0.5, 3.0))
        self.assertFalse(workspace_contains(workspace, 2.0, 2.0))
        (yaw, pitch), clipped = clamp_workspace_target(
            workspace, 0.5, 0.5, 2.0, 2.0
        )
        self.assertTrue(clipped)
        self.assertTrue(workspace_contains(workspace, yaw, pitch))
        self.assertLessEqual(yaw, 1.0)
        self.assertLessEqual(pitch, 1.0)

    def test_symmetric_workspace_uses_conservative_mirrored_intersection(self):
        workspace = WorkspaceLimit(
            yaw_motor_id=0x03,
            pitch_motor_id=0x01,
            vertices=((-1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-1.0, 1.0)),
            anchor=(0.0, 0.0),
            symmetry_yaw_center_rad=0.0,
        )
        self.assertTrue(workspace_contains(workspace, 0.8, 0.0))
        # The recorded right side reaches +2, but its mirror would be -2,
        # outside the recorded left side. Symmetry therefore shrinks to +/-1.
        self.assertFalse(workspace_contains(workspace, 1.2, 0.0))
        self.assertFalse(workspace_contains(workspace, -1.2, 0.0))

    def test_workspace_capture_persists_polygon_and_anchor(self):
        # A 1x1 square with 20+ ordered boundary samples and a closed final point.
        samples = []
        for yaw, pitch in (
            (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
            (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0), (0.75, 1.0),
            (0.5, 1.0), (0.25, 1.0), (0.0, 1.0), (0.0, 0.75), (0.0, 0.5),
            (0.0, 0.25), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        ):
            samples.append({"yaw_rad": yaw, "pitch_rad": pitch, "elapsed_s": len(samples) / 20.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism_limits.json"
            save_workspace_limits(
                path,
                samples=samples,
                anchor_position={"yaw_rad": 0.5, "pitch_rad": 0.5},
                margin_motor_deg=math.degrees(0.05),
                simplify_motor_deg=math.degrees(0.01),
                closure_motor_deg=1.0,
                hard_motor_limits={0x03: (0.1, 0.9), 0x01: (0.2, 0.8)},
            )
            workspace = load_workspace_limit(path)
            self.assertIsNotNone(workspace)
            self.assertTrue(workspace_contains(workspace, 0.5, 0.5))
            self.assertFalse(workspace_contains(workspace, 1.2, 0.5))
            limits = load_motor_limits(path)
            self.assertGreaterEqual(limits[0x03][0], 0.1)
            self.assertLessEqual(limits[0x03][1], 0.9)
            self.assertGreaterEqual(limits[0x01][0], 0.2)
            self.assertLessEqual(limits[0x01][1], 0.8)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreater(stored["workspace"]["hard_limit_violation_count"], 0)

    def test_workspace_capture_repairs_only_closure_seam_overshoot(self):
        def sample(yaw_deg, pitch_deg):
            return {
                "yaw_rad": math.radians(yaw_deg),
                "pitch_rad": math.radians(pitch_deg),
                "elapsed_s": len(samples) / 20.0,
            }

        # Start just below/left of the lower edge, then finish by descending
        # through that edge. Values mirror the shape seen in the real trace.
        samples = []
        samples.append(sample(-4.142, -20.338))
        samples.append(sample(-4.142, -19.092))
        for yaw in (-4.0, -3.77, -3.53, 0.0, 15.0, 30.0, 45.0, 60.0, 63.134):
            samples.append(sample(yaw, -19.092))
        for pitch in (-10.0, 0.0, 10.0, 13.912):
            samples.append(sample(63.134, pitch))
        for yaw in (50.0, 30.0, 10.0, 0.0, -3.705):
            samples.append(sample(yaw, 13.912))
        for pitch in (8.0, 0.0, -10.0, -18.0, -19.092, -19.420):
            samples.append(sample(-3.705, pitch))
        samples.append(sample(-3.705, -19.420))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism_limits.json"
            save_workspace_limits(
                path,
                samples=samples,
                anchor_position={"yaw_rad": math.radians(30.0), "pitch_rad": 0.0},
                margin_motor_deg=0.0,
                simplify_motor_deg=0.25,
                closure_motor_deg=2.0,
            )
            workspace = load_workspace_limit(path)
            self.assertTrue(workspace_contains(workspace, math.radians(30.0), 0.0))
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreater(stored["workspace"]["closure_seam_trimmed_samples"], 0)

    def test_workspace_capture_still_rejects_middle_self_intersection(self):
        points = [(0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)]
        samples = []
        for index in range(25):
            segment = min(index // 6, 3)
            fraction = (index % 6) / 6.0
            if index == 24:
                segment, fraction = 3, 1.0
            start = points[segment]
            end = points[segment + 1]
            samples.append(
                {
                    "yaw_rad": start[0] + (end[0] - start[0]) * fraction,
                    "pitch_rad": start[1] + (end[1] - start[1]) * fraction,
                    "elapsed_s": index / 20.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "intersects itself"):
                save_workspace_limits(
                    Path(directory) / "mechanism_limits.json",
                    samples=samples,
                    anchor_position={"yaw_rad": 1.0, "pitch_rad": 1.0},
                    margin_motor_deg=0.0,
                    simplify_motor_deg=0.01,
                    closure_motor_deg=0.1,
                )

    def test_backend_clamps_coordinated_targets_to_workspace(self):
        class FakeMotorControl:
            def __init__(self):
                self.positions = {}

            def control_Pos_Vel(self, motor, position, _speed, receive=False):
                self.positions[id(motor)] = (position, receive)

        yaw_motor = object()
        pitch_motor = object()
        workspace = WorkspaceLimit(
            yaw_motor_id=0x03,
            pitch_motor_id=0x01,
            vertices=((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)),
            anchor=(0.5, 0.5),
            margin_rad=0.05,
        )
        backend = object.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: yaw_motor, 0x01: pitch_motor}
        backend._position_limits = {0x03: (0.0, 4.0), 0x01: (0.0, 4.0)}
        backend._last_limit_notice = {}
        backend._last_workspace_notice = None
        backend._workspace = workspace
        backend._last = {
            0x03: PositionCommand(0x03, 0.5, 0.5),
            0x01: PositionCommand(0x01, 0.5, 0.5),
        }
        backend.mode = "posvel"
        backend.vmax = 2.0
        backend.tmax = 3.0
        backend.kp = 1.0
        backend.kd = 1.0
        backend._mc = FakeMotorControl()

        backend.send_targets({0x03: (2.0, 0.5, 0.0, 0.0), 0x01: (2.0, 0.5, 0.0, 0.0)})
        self.assertTrue(
            workspace_contains(
                workspace,
                backend._last[0x03].position_rad,
                backend._last[0x01].position_rad,
            )
        )

    def test_backend_stages_vertical_home_corridor_before_yaw(self):
        class FakeMotorControl:
            def __init__(self):
                self.positions = {}

            def control_Pos_Vel(self, motor, position, _speed, receive=False):
                self.positions[id(motor)] = (position, receive)

        yaw_motor = object()
        pitch_motor = object()
        workspace = WorkspaceLimit(
            yaw_motor_id=0x03,
            pitch_motor_id=0x01,
            vertices=((-2.0, 0.0), (2.0, 0.0), (2.0, 1.0), (-2.0, 1.0)),
            anchor=(0.0, 0.5),
            home_corridor=HomeCorridor(
                yaw_center_rad=0.0,
                yaw_tolerance_rad=0.1,
                home_pitch_rad=2.0,
                transition_pitch_rad=1.0,
            ),
        )
        backend = object.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: yaw_motor, 0x01: pitch_motor}
        backend._position_limits = {0x03: (-2.0, 2.0), 0x01: (0.0, 2.0)}
        backend._last_limit_notice = {}
        backend._last_workspace_notice = None
        backend._workspace = workspace
        backend._last = {
            0x03: PositionCommand(0x03, 0.0, 0.5),
            0x01: PositionCommand(0x01, 2.0, 0.5),
        }
        backend.mode = "posvel"
        backend.vmax = 2.0
        backend.tmax = 3.0
        backend.kp = 1.0
        backend.kd = 1.0
        backend._mc = FakeMotorControl()

        # Leaving home lowers at centered yaw before horizontal travel.
        backend.send_targets({0x03: (1.0, 0.5, 0.0, 0.0), 0x01: (0.5, 0.5, 0.0, 0.0)})
        self.assertAlmostEqual(backend._last[0x03].position_rad, 0.0)
        self.assertAlmostEqual(backend._last[0x01].position_rad, 0.5)

        # Returning home first centers at low pitch, then rises next cycle.
        backend._last[0x03] = PositionCommand(0x03, 1.0, 0.5)
        backend._last[0x01] = PositionCommand(0x01, 0.5, 0.5)
        backend.send_targets({0x03: (0.0, 0.5, 0.0, 0.0), 0x01: (2.0, 0.5, 0.0, 0.0)})
        self.assertAlmostEqual(backend._last[0x03].position_rad, 0.0)
        self.assertAlmostEqual(backend._last[0x01].position_rad, 0.5)
        backend.send_targets({0x03: (0.0, 0.5, 0.0, 0.0), 0x01: (2.0, 0.5, 0.0, 0.0)})
        self.assertAlmostEqual(backend._last[0x01].position_rad, 2.0)

    def test_enabled_backend_uses_feedback_not_command_to_unlock_yaw(self):
        class FeedbackMotor:
            def __init__(self, position):
                self.position = position

            def getPosition(self):
                return self.position

            def getVelocity(self):
                return 0.0

            def getTorque(self):
                return 0.0

            def getError(self):
                return 1  # enabled, not a fault

        class FakeMotorControl:
            def control_Pos_Vel(self, _motor, _position, _speed, receive=False):
                self.receive = receive

        workspace = WorkspaceLimit(
            yaw_motor_id=0x03,
            pitch_motor_id=0x01,
            vertices=((-2.0, 0.0), (2.0, 0.0), (2.0, 1.0), (-2.0, 1.0)),
            anchor=(0.0, 0.5),
            home_corridor=HomeCorridor(0.0, 0.1, 2.0, 1.0),
        )
        backend = object.__new__(DamiaoSerialBackend)
        backend._motors = {0x03: FeedbackMotor(0.0), 0x01: FeedbackMotor(2.0)}
        backend._position_limits = {0x03: (-2.0, 2.0), 0x01: (0.0, 2.0)}
        backend._last_limit_notice = {}
        backend._last_workspace_notice = None
        backend._workspace = workspace
        # Commands claim pitch has already lowered, but feedback is still home.
        backend._last = {
            0x03: PositionCommand(0x03, 0.0, 0.5),
            0x01: PositionCommand(0x01, 0.5, 0.5),
        }
        backend._enabled = True
        backend.mode = "posvel"
        backend.vmax = 2.0
        backend.tmax = 3.0
        backend.kp = 1.0
        backend.kd = 1.0
        backend._mc = FakeMotorControl()

        backend.send_targets(
            {0x03: (1.0, 0.5, 0.0, 0.0), 0x01: (0.5, 0.5, 0.0, 0.0)}
        )
        self.assertAlmostEqual(backend._last[0x03].position_rad, 0.0)


if __name__ == "__main__":
    unittest.main()

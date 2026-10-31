"""Adapt the shared manual motor worker to the automatic aiming controller."""

import math
import time

from .backends import MotorState
from .limits import load_workspace_limit
from .pitch import load_pitch_calibration, load_yaw_calibration
from .run_vertical_panel import (
    DEFAULT_STARTUP_POSE, FEEDBACK_TIMEOUT_S, MotorWorker, SEQUENCE_STOP_SPEED_RAD_S,
)


class SharedMotorBackend:
    """The worker owns the serial port and runs the same 200 Hz safe sequence."""

    def __init__(self, port, *, worker=None):
        self.pitch_calibration = load_pitch_calibration(DEFAULT_STARTUP_POSE)
        self.yaw_calibration = load_yaw_calibration(DEFAULT_STARTUP_POSE)
        self._workspace = load_workspace_limit()
        self.worker = worker or MotorWorker(
            port=port, pitch_calibration=self.pitch_calibration,
            yaw_calibration=self.yaw_calibration,
            speed_rad_s=.06, acceleration_rad_s2=.05,
            torque_ff_nm=0., torque_ramp_s=1.5, kp=50., kd=1.5,
            max_tracking_error_deg=10., stall_timeout_s=2.,
            min_muzzle_deg=0., max_muzzle_deg=45., control_mode="posvel",
        )
        self._last_requested = None
        self._request_id = 0

    def _wait_for(self, predicate, timeout_s=5.):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.worker.snapshot()
            if snapshot.closed:
                raise RuntimeError(snapshot.error or "雙軸馬達控制執行緒已停止")
            if predicate(snapshot):
                return
            time.sleep(.01)
        raise RuntimeError("等待雙軸馬達連線或使能逾時")

    def enable(self):
        self.worker.start()
        try:
            self._wait_for(lambda snapshot: snapshot.ready)
            self.worker.request_enable()
            self._wait_for(lambda snapshot: snapshot.enabled)
        except Exception:
            self.close()
            raise

    def refresh(self):
        snapshot = self.worker.snapshot()
        if snapshot.closed:
            raise RuntimeError(snapshot.error or "雙軸馬達控制執行緒已停止")

    def get_state(self, motor_id):
        snapshot = self.worker.snapshot()
        if not snapshot.enabled or snapshot.closed:
            return None
        if motor_id == 0x03:
            pos, velocity, torque, err, feedback_at = (snapshot.yaw_position_rad, snapshot.yaw_velocity_rad_s,
                                                        snapshot.yaw_feedback_torque_nm, snapshot.yaw_motor_err,
                                                        snapshot.yaw_feedback_at)
        elif motor_id == 0x01:
            pos, velocity, torque, err, feedback_at = (snapshot.position_rad, snapshot.velocity_rad_s,
                                                        snapshot.feedback_torque_nm, snapshot.motor_err,
                                                        snapshot.feedback_at)
        else:
            return None
        if (pos is None or velocity is None or not math.isfinite(pos)
                or not math.isfinite(velocity) or err != 1 or feedback_at is None
                or not math.isfinite(feedback_at)
                or not 0 <= time.monotonic() - feedback_at <= FEEDBACK_TIMEOUT_S):
            return None
        return MotorState(motor_id, pos, velocity or 0., torque or 0., err, feedback_at)

    def send_targets(self, targets):
        if 0x03 not in targets or 0x01 not in targets:
            raise ValueError("共用雙軸控制需要同時提供水平及垂直目標")
        desired = (float(targets[0x03][0]), float(targets[0x01][0]))
        if not all(math.isfinite(value) for value in desired):
            raise ValueError("雙軸目標必須是有限數字")
        # Suppress tiny frame-to-frame jitter without continually restarting
        # the clearance sequence. Motor angles remain the caller's units.
        epsilon = math.radians(.02)
        if self._last_requested is not None and all(
            abs(a - b) <= epsilon for a, b in zip(desired, self._last_requested)
        ):
            return
        self._last_requested = desired
        self._request_id += 1
        self.worker.request_tagged_aim(
            math.degrees(self.yaw_calibration.muzzle_from_motor(desired[0])),
            math.degrees(self.pitch_calibration.muzzle_from_motor(desired[1])),
            self._request_id,
        )

    def motion_ready(self, yaw_target, pitch_target):
        snapshot = self.worker.snapshot()
        if (not snapshot.enabled or snapshot.closed or snapshot.error is not None
                or snapshot.yaw_sequence_phase is not None
                or snapshot.aim_request_id != self._request_id
                or self._last_requested is None
                or snapshot.yaw_target_rad is None or snapshot.target_rad is None):
            return False
        velocities = (snapshot.velocity_rad_s, snapshot.yaw_velocity_rad_s,
                      snapshot.command_velocity_rad_s, snapshot.yaw_command_velocity_rad_s)
        if any(value is None or not math.isfinite(value)
               or abs(value) > SEQUENCE_STOP_SPEED_RAD_S for value in velocities):
            return False
        epsilon = math.radians(.025)
        return (
            abs(snapshot.yaw_target_rad - yaw_target) <= epsilon
            and abs(snapshot.target_rad - pitch_target) <= epsilon
            and abs(self._last_requested[0] - yaw_target) <= epsilon
            and abs(self._last_requested[1] - pitch_target) <= epsilon
            and self.get_state(0x03) is not None
            and self.get_state(0x01) is not None
        )

    def hold(self):
        # The worker continuously holds its current active target at 200 Hz.
        pass

    def close(self):
        if self.worker.snapshot().closed:
            return
        if self.worker.snapshot().enabled:
            self.worker.request_disable()
            self._wait_for(lambda snapshot: not snapshot.enabled, timeout_s=3.)
        self.worker.request_shutdown()
        self.worker.wait_closed(timeout=3.)

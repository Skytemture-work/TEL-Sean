"""Safe operator panel for the DAMIAO vertical axis.

The panel starts with the motor disabled.  The operator can enable/disable the
vertical motor, enter a physical muzzle elevation, and make small +/- step
changes.  The calibrated motor zero, direction, and mechanism ratio convert
that elevation to an absolute motor target.  The worker thread owns the serial
port and sends an acceleration-limited, feedback-aware target at 200 Hz while
enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # The motor worker also runs on headless robots.
    tk = messagebox = ttk = None

from .backends import DamiaoSerialBackend
from .limits import (
    HOME_CORRIDOR_TRANSITION_DEG,
    PITCH_ACTIVITY_TRANSITION_DEG,
    PITCH_MAX_DEG,
    load_motor_limits,
    load_workspace_limit,
    constrain_pitch_travel_limit,
    max_activity_pitch_for_yaw_deg,
    workspace_contains,
    workspace_segment_is_safe,
)
from .pitch import (
    PitchCalibration,
    load_pitch_calibration,
    load_yaw_calibration,
    save_manual_origin_calibration,
    save_level_reference_calibration,
)
from .trajectory import acceleration_limited_step


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_STARTUP_POSE = Path(__file__).with_name("startup_pose.json")
PITCH_MOTOR_ID = 0x01
YAW_MOTOR_ID = 0x03
CONTROL_PERIOD_S = 0.005
MIN_MUZZLE_ELEVATION_DEG = 0.0
MAX_MUZZLE_ELEVATION_DEG = 45.0
MIN_TURRET_YAW_DEG = -34.5
MAX_TURRET_YAW_DEG = 34.5
MAX_PITCH_FOR_YAW_DEG = 35.0
YAW_RELEASE_MARGIN_DEG = 0.5
YAW_CLEARANCE_PITCH_DEG = MAX_PITCH_FOR_YAW_DEG - YAW_RELEASE_MARGIN_DEG
HOME_CORRIDOR_YAW_TOLERANCE_DEG = 10.0
YAW_MOVE_EPSILON_DEG = 0.000001
SEQUENCE_ARRIVAL_TOLERANCE_DEG = 0.05
# Leave room for the DAMIAO feedback velocity's finite encoder resolution.
SEQUENCE_STOP_SPEED_RAD_S = math.radians(0.5)
FEEDBACK_TIMEOUT_S = 0.25


def _fresh_enabled_feedback(state, now: float) -> bool:
    return (
        state is not None and state.err == 1
        and state.feedback_at is not None and math.isfinite(state.feedback_at)
        and 0 <= now - state.feedback_at <= FEEDBACK_TIMEOUT_S
        and math.isfinite(state.position_rad) and math.isfinite(state.velocity_rad_s)
    )


def load_startup_pitch(path: Path) -> float:
    """Backward-compatible access to the calibrated level-muzzle motor angle."""
    return load_pitch_calibration(path).motor_zero_rad


@dataclass
class WorkerSnapshot:
    status: str = "正在連接..."
    enabled: bool = False
    ready: bool = False
    closed: bool = False
    position_rad: Optional[float] = None
    velocity_rad_s: Optional[float] = None
    feedback_torque_nm: Optional[float] = None
    target_rad: Optional[float] = None
    target_muzzle_rad: Optional[float] = None
    effective_target_rad: Optional[float] = None
    command_rad: Optional[float] = None
    command_velocity_rad_s: Optional[float] = None
    tracking_error_rad: Optional[float] = None
    torque_ff_nm: Optional[float] = None
    motor_err: Optional[int] = None
    feedback_at: Optional[float] = None
    yaw_position_rad: Optional[float] = None
    yaw_velocity_rad_s: Optional[float] = None
    yaw_feedback_torque_nm: Optional[float] = None
    yaw_target_rad: Optional[float] = None
    yaw_target_angle_rad: Optional[float] = None
    yaw_effective_target_rad: Optional[float] = None
    yaw_command_rad: Optional[float] = None
    yaw_command_velocity_rad_s: Optional[float] = None
    yaw_tracking_error_rad: Optional[float] = None
    yaw_motor_err: Optional[int] = None
    yaw_feedback_at: Optional[float] = None
    yaw_interlock_ready: bool = False
    yaw_sequence_phase: Optional[str] = None
    yaw_restore_muzzle_rad: Optional[float] = None
    aim_request_id: int = 0
    error: Optional[str] = None


class MotorWorker:
    """Own the serial port and run the 200 Hz hold/trajectory loop."""

    def __init__(
        self,
        port: str,
        pitch_calibration: PitchCalibration,
        speed_rad_s: float,
        acceleration_rad_s2: float,
        torque_ff_nm: float,
        torque_ramp_s: float,
        kp: float,
        kd: float,
        max_tracking_error_deg: float,
        stall_timeout_s: float,
        min_muzzle_deg: float,
        max_muzzle_deg: float,
        control_mode: str = "posvel",
        yaw_calibration: Optional[PitchCalibration] = None,
        min_yaw_deg: float = MIN_TURRET_YAW_DEG,
        max_yaw_deg: float = MAX_TURRET_YAW_DEG,
        max_pitch_for_yaw_deg: float = MAX_PITCH_FOR_YAW_DEG,
        yaw_clearance_pitch_deg: float = YAW_CLEARANCE_PITCH_DEG,
        position_limits=None,
    ):
        self.port = port
        self.pitch_calibration = pitch_calibration
        # Kept as a public alias for scripts that displayed the old startup
        # reference.  It is now explicitly the level-muzzle motor angle.
        self.startup_pitch_rad = pitch_calibration.motor_zero_rad
        self.speed_rad_s = speed_rad_s
        self.acceleration_rad_s2 = acceleration_rad_s2
        self.torque_ff_nm = torque_ff_nm
        self.torque_ramp_s = torque_ramp_s
        self.kp = kp
        self.kd = kd
        if control_mode not in ("mit", "posvel"):
            raise ValueError("control_mode must be 'mit' or 'posvel'")
        self.control_mode = control_mode
        self.yaw_calibration = yaw_calibration
        self.max_tracking_error_rad = math.radians(max_tracking_error_deg)
        self.stall_timeout_s = stall_timeout_s
        if (
            not math.isfinite(min_muzzle_deg)
            or not math.isfinite(max_muzzle_deg)
            or min_muzzle_deg < MIN_MUZZLE_ELEVATION_DEG
            or max_muzzle_deg > MAX_MUZZLE_ELEVATION_DEG
            or min_muzzle_deg >= max_muzzle_deg
        ):
            raise ValueError(
                "砲口仰角控制範圍必須位於 "
                f"{MIN_MUZZLE_ELEVATION_DEG:.1f}° 到 "
                f"{MAX_MUZZLE_ELEVATION_DEG:.1f}° 內"
            )
        self.min_muzzle_rad = math.radians(min_muzzle_deg)
        self.max_muzzle_rad = math.radians(max_muzzle_deg)
        if (
            not math.isfinite(min_yaw_deg)
            or not math.isfinite(max_yaw_deg)
            or min_yaw_deg < MIN_TURRET_YAW_DEG
            or max_yaw_deg > MAX_TURRET_YAW_DEG
            or min_yaw_deg >= max_yaw_deg
        ):
            raise ValueError("水平角度控制範圍必須位於 -34.5° 到 +34.5° 內")
        if (
            not math.isfinite(max_pitch_for_yaw_deg)
            or max_pitch_for_yaw_deg < MIN_MUZZLE_ELEVATION_DEG
            or max_pitch_for_yaw_deg > MAX_MUZZLE_ELEVATION_DEG
        ):
            raise ValueError("水平聯鎖的垂直安全角度無效")
        self.min_yaw_rad = math.radians(min_yaw_deg)
        self.max_yaw_rad = math.radians(max_yaw_deg)
        self.max_pitch_for_yaw_rad = math.radians(max_pitch_for_yaw_deg)
        if (
            not math.isfinite(yaw_clearance_pitch_deg)
            or yaw_clearance_pitch_deg < min_muzzle_deg
            or yaw_clearance_pitch_deg > max_pitch_for_yaw_deg
        ):
            raise ValueError(
                "水平避讓角度必須位於垂直軟限制下限與水平安全角度之間"
            )
        self.yaw_clearance_pitch_rad = math.radians(yaw_clearance_pitch_deg)
        self.position_limits = dict(position_limits or load_motor_limits())
        # 35 degrees is the centered apex of the normal activity region.  The
        # actual clearance target is calculated from the requested yaw so the
        # 30–35 degree triangular workspace is respected.
        self.yaw_release_pitch_rad = self.max_pitch_for_yaw_rad
        self.sequence_arrival_tolerance_rad = math.radians(
            SEQUENCE_ARRIVAL_TOLERANCE_DEG
        )

        self._commands = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._snapshot = WorkerSnapshot()

        self._motors: Optional[DamiaoSerialBackend] = None
        self._enabled = False
        self._target_rad = pitch_calibration.motor_zero_rad
        self._target_muzzle_rad = 0.0
        self._last_command_rad = pitch_calibration.motor_zero_rad
        self._command_velocity_rad_s = 0.0
        self._feedback_position_rad: Optional[float] = None
        self._feedback_velocity_rad_s = 0.0
        self._enabled_at = time.monotonic()
        self._last_tick_at = time.monotonic()
        self._stall_reference_rad: Optional[float] = None
        self._stall_reference_at = time.monotonic()
        self._stall_warning_active = False
        self._yaw_target_rad = (
            yaw_calibration.motor_zero_rad if yaw_calibration is not None else 0.0
        )
        self._yaw_target_angle_rad = 0.0
        self._yaw_last_command_rad = self._yaw_target_rad
        self._yaw_command_velocity_rad_s = 0.0
        self._yaw_feedback_position_rad: Optional[float] = None
        self._yaw_feedback_velocity_rad_s = 0.0
        self._yaw_interlock_hold_rad: Optional[float] = None
        self._yaw_sequence_phase: Optional[str] = None
        self._yaw_restore_muzzle_rad: Optional[float] = None
        self._deferred_aim: Optional[Tuple[float, float, bool]] = None
        self._feedback_waiting = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vertical-motor", daemon=True)
        self._thread.start()

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            return WorkerSnapshot(**self._snapshot.__dict__)

    def _publish(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._snapshot, key, value)

    def request_enable(self) -> None:
        self._commands.put(("enable", None))

    def request_disable(self) -> None:
        self._commands.put(("disable", None))

    def request_muzzle_angle(self, muzzle_deg: float) -> None:
        self._commands.put(("target", float(muzzle_deg)))

    def request_yaw_angle(self, yaw_deg: float) -> None:
        self._commands.put(("yaw_target", float(yaw_deg)))

    def request_aim(self, yaw_deg: float, muzzle_deg: float) -> None:
        self._commands.put(("aim", (float(yaw_deg), float(muzzle_deg))))

    def request_tagged_aim(self, yaw_deg: float, muzzle_deg: float, request_id: int) -> None:
        self._commands.put(("aim_tagged", (float(yaw_deg), float(muzzle_deg), int(request_id))))

    def request_home(self) -> None:
        self._commands.put(("home", None))

    def request_calibration_reload(
        self,
        pitch_calibration: PitchCalibration,
        yaw_calibration: PitchCalibration,
    ) -> None:
        self._commands.put(
            ("calibration", (pitch_calibration, yaw_calibration))
        )

    def request_target_offset(self, offset_deg: float) -> None:
        """Compatibility alias; the value now means physical muzzle degrees."""
        self.request_muzzle_angle(offset_deg)

    def request_shutdown(self) -> None:
        self._commands.put(("shutdown", None))

    def wait_closed(self, timeout: float = 3.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _apply_pitch_target(self, muzzle_rad: float, status: str) -> None:
        """Set the active pitch target used by the trajectory generator."""
        target = self.pitch_calibration.motor_from_muzzle(muzzle_rad)
        reversing = (
            abs(self._command_velocity_rad_s) > 1e-9
            and self._command_velocity_rad_s * (target - self._last_command_rad) < 0.0
        )
        self._target_rad = target
        self._target_muzzle_rad = muzzle_rad
        self._stall_reference_rad = self._feedback_position_rad
        self._stall_reference_at = time.monotonic()
        self._stall_warning_active = False
        self._publish(
            target_rad=target,
            target_muzzle_rad=muzzle_rad,
            status=(
                f"{status}；正在煞車，隨後反轉"
                if reversing
                else status
            ),
            error=None,
        )

    def _set_target(self, muzzle_deg: float) -> None:
        if not math.isfinite(muzzle_deg):
            raise ValueError("砲口仰角必須是有限數字")
        muzzle_rad = math.radians(muzzle_deg)
        if not self.min_muzzle_rad <= muzzle_rad <= self.max_muzzle_rad:
            raise ValueError(
                "砲口仰角超出目前軟限制："
                f"{math.degrees(self.min_muzzle_rad):+.1f}° 到 "
                f"{math.degrees(self.max_muzzle_rad):+.1f}°"
            )
        if (
            self.yaw_calibration is not None
            and self._yaw_feedback_position_rad is not None
        ):
            current_yaw_rad = self.yaw_calibration.muzzle_from_motor(
                self._yaw_feedback_position_rad
            )
            if not self._pose_allowed_by_activity(current_yaw_rad, muzzle_rad):
                raise ValueError(
                    self._pose_limit_error(math.degrees(current_yaw_rad), muzzle_deg)
                )

        if self.yaw_calibration is not None and self._yaw_feedback_position_rad is not None:
            self._set_aim_targets(math.degrees(self._yaw_target_angle_rad), muzzle_deg)
            return

        if self._motors is not None:
            self._motors.set_home_return(False)

        self._apply_pitch_target(
            muzzle_rad,
            f"砲口目標已設定：{muzzle_deg:+.2f}°",
        )

    def _apply_yaw_target(self, yaw_rad: float) -> None:
        """Set a validated yaw target without making a clearance decision."""
        assert self.yaw_calibration is not None

        target = self.yaw_calibration.motor_from_muzzle(yaw_rad)
        self._yaw_target_rad = target
        self._yaw_target_angle_rad = yaw_rad
        self._publish(
            yaw_target_rad=target,
            yaw_target_angle_rad=yaw_rad,
            error=None,
        )

    def _clearance_pitch_for_yaw(self, yaw_rad: float) -> float:
        """Return a margin-safe pitch for the requested horizontal target."""

        boundary = math.radians(self._activity_pitch_limit_for_yaw(yaw_rad))
        return max(
            self.min_muzzle_rad,
            boundary - math.radians(YAW_RELEASE_MARGIN_DEG),
        )

    def _activity_pitch_limit_for_yaw(self, yaw_rad: float) -> float:
        """Return the triangle's normal-activity pitch ceiling."""

        return max_activity_pitch_for_yaw_deg(
            math.degrees(yaw_rad),
            activity_max_deg=math.degrees(self.max_pitch_for_yaw_rad),
            transition_deg=PITCH_ACTIVITY_TRANSITION_DEG,
        )

    def _pose_allowed_by_activity(self, yaw_rad: float, muzzle_rad: float) -> bool:
        """Check the normal triangle or the centered high-pitch corridor."""

        pitch_deg = math.degrees(muzzle_rad)
        if pitch_deg <= self._activity_pitch_limit_for_yaw(yaw_rad) + 1e-6:
            return True
        return (
            pitch_deg >= HOME_CORRIDOR_TRANSITION_DEG - 1e-6
            and abs(math.degrees(yaw_rad))
            <= HOME_CORRIDOR_YAW_TOLERANCE_DEG + 1e-6
        )

    def _recorded_home_muzzle_rad(self) -> float:
        """Convert the recorded home encoder position using the level reference."""
        workspace = getattr(self._motors, "_workspace", None)
        if workspace is None:
            workspace = load_workspace_limit()
        if workspace is None or workspace.home_corridor is None:
            raise ValueError("尚未記錄 home 馬達位置，不能回 home")
        return self.pitch_calibration.muzzle_from_motor(workspace.home_corridor.home_pitch_rad)

    @staticmethod
    def _pose_limit_error(yaw_deg: float, muzzle_deg: float) -> str:
        return (
            f"水平 {yaw_deg:+.1f}° 與垂直 {muzzle_deg:.1f}° 的組合超出活動區；"
            "30°～35° 的活動範圍會隨水平角收窄，33°以上只能在水平 ±10° home corridor"
        )

    def _begin_yaw_clearance_sequence(self, restore_muzzle_rad: float) -> None:
        """Lower into the yaw-dependent activity region, then restore pitch."""
        self._yaw_sequence_phase = "clearance"
        self._yaw_restore_muzzle_rad = restore_muzzle_rad
        current_yaw = self.yaw_calibration.muzzle_from_motor(self._yaw_feedback_position_rad)
        commanded_yaw = self.yaw_calibration.muzzle_from_motor(self._yaw_last_command_rad)
        widest_yaw = max((current_yaw, commanded_yaw, self._yaw_target_angle_rad), key=abs)
        self.yaw_clearance_pitch_rad = self._clearance_pitch_for_yaw(widest_yaw)
        self.yaw_release_pitch_rad = self.yaw_clearance_pitch_rad
        active_pitch_rad = (
            restore_muzzle_rad
            if restore_muzzle_rad < self.yaw_clearance_pitch_rad
            else self.yaw_clearance_pitch_rad
        )
        self._apply_pitch_target(
            active_pitch_rad,
            f"水平自動避讓：依目標水平角先降到 {math.degrees(self.yaw_clearance_pitch_rad):.1f}° 以下，確認安全後雙軸同時移動",
        )
        self._publish(
            yaw_sequence_phase="clearance",
            yaw_restore_muzzle_rad=restore_muzzle_rad,
        )

    def _clear_yaw_sequence(self) -> None:
        self._yaw_sequence_phase = None
        self._yaw_restore_muzzle_rad = None
        self._deferred_aim = None
        self._publish(
            yaw_sequence_phase=None,
            yaw_restore_muzzle_rad=None,
        )

    def _yaw_motion_required(self, target_rad: float) -> bool:
        if self._yaw_feedback_position_rad is None:
            raise ValueError("尚未取得水平馬達回饋，禁止水平移動")
        return (
            abs(target_rad - self._yaw_feedback_position_rad)
            > math.radians(YAW_MOVE_EPSILON_DEG)
        )

    def _yaw_interlock_hold_position(self) -> float:
        """Latch one yaw position instead of following drifting feedback."""
        if self._yaw_interlock_hold_rad is None:
            self._yaw_interlock_hold_rad = (
                self._yaw_feedback_position_rad
                if self._yaw_feedback_position_rad is not None
                else self._yaw_last_command_rad
            )
        return self._yaw_interlock_hold_rad

    def _yaw_feedback_clear(self, feedback_pitch_rad: float) -> bool:
        limit = self.max_pitch_for_yaw_rad
        if self._yaw_sequence_phase == "rotating":
            limit = min(limit, self.yaw_clearance_pitch_rad)
        return feedback_pitch_rad <= limit + 1e-10

    def _recorded_path_is_safe(self, start, end) -> bool:
        workspace = getattr(self._motors, "_workspace", None)
        if workspace is None:
            return True
        return workspace_segment_is_safe(workspace, start, end)

    def _set_yaw_target(self, yaw_deg: float) -> None:
        if self.yaw_calibration is None:
            raise ValueError("水平馬達控制未啟用")
        if not math.isfinite(yaw_deg):
            raise ValueError("水平角度必須是有限數字")
        yaw_rad = math.radians(yaw_deg)
        if not self.min_yaw_rad <= yaw_rad <= self.max_yaw_rad:
            raise ValueError(
                f"水平角度超出安全限制：{math.degrees(self.min_yaw_rad):+.1f}° 到 "
                f"{math.degrees(self.max_yaw_rad):+.1f}°"
            )
        if self._feedback_position_rad is None:
            raise ValueError("尚未取得垂直馬達回饋，禁止水平移動")

        restore_muzzle = (
            self._yaw_restore_muzzle_rad
            if self._yaw_restore_muzzle_rad is not None
            else self._target_muzzle_rad
        )
        self._set_aim_targets(yaw_deg, math.degrees(restore_muzzle))

    def _set_aim_targets(self, yaw_deg: float, muzzle_deg: float, *, home_return: bool = False) -> None:
        """Set a final two-axis pose, staging through clearance when needed."""
        if self.yaw_calibration is None:
            raise ValueError("水平馬達控制未啟用")
        if not math.isfinite(yaw_deg) or not math.isfinite(muzzle_deg):
            raise ValueError("水平與垂直角度都必須是有限數字")
        yaw_rad = math.radians(yaw_deg)
        muzzle_rad = math.radians(muzzle_deg)
        if not self.min_yaw_rad <= yaw_rad <= self.max_yaw_rad:
            raise ValueError(
                f"水平角度超出安全限制：{math.degrees(self.min_yaw_rad):+.1f}° 到 "
                f"{math.degrees(self.max_yaw_rad):+.1f}°"
            )
        pitch_upper = (
            max(math.radians(MAX_MUZZLE_ELEVATION_DEG), self._recorded_home_muzzle_rad())
            if home_return else self.max_muzzle_rad
        )
        if not self.min_muzzle_rad <= muzzle_rad <= pitch_upper:
            raise ValueError(
                f"砲口仰角超出安全限制：{math.degrees(self.min_muzzle_rad):.1f}° 到 "
                f"{math.degrees(self.max_muzzle_rad):.1f}°"
            )
        if not home_return and not self._pose_allowed_by_activity(yaw_rad, muzzle_rad):
            raise ValueError(self._pose_limit_error(yaw_deg, muzzle_deg))
        if self._feedback_position_rad is None:
            raise ValueError("尚未取得垂直馬達回饋，禁止雙軸移動")
        if self._yaw_feedback_position_rad is None:
            raise ValueError("尚未取得水平馬達回饋，禁止雙軸移動")
        if home_return and (not self._enabled or self._motors is None):
            raise ValueError("回 home 前必須先使能馬達")
        if self._yaw_sequence_phase is not None:
            same_goal = (
                abs(yaw_rad - self._yaw_target_angle_rad) <= math.radians(.02)
                and self._yaw_restore_muzzle_rad is not None
                and abs(muzzle_rad - self._yaw_restore_muzzle_rad) <= math.radians(.02)
            )
            self._deferred_aim = None if same_goal else (yaw_deg, muzzle_deg, home_return)
            return
        yaw_target = self.yaw_calibration.motor_from_muzzle(yaw_rad)
        yaw_move_required = self._yaw_motion_required(yaw_target)

        start = (self._yaw_feedback_position_rad, self._feedback_position_rad)
        final = (yaw_target, self.pitch_calibration.motor_from_muzzle(muzzle_rad))
        if yaw_move_required:
            current_yaw = self.yaw_calibration.muzzle_from_motor(start[0])
            commanded_yaw = self.yaw_calibration.muzzle_from_motor(self._yaw_last_command_rad)
            widest_yaw = max((current_yaw, commanded_yaw, yaw_rad), key=abs)
            stage_pitch = min(muzzle_rad, self._clearance_pitch_for_yaw(widest_yaw))
            stage_motor = self.pitch_calibration.motor_from_muzzle(stage_pitch)
            path = (start, (start[0], stage_motor), (yaw_target, stage_motor), final)
        else:
            path = (start, final)
        if not all(self._recorded_path_is_safe(a, b) for a, b in zip(path, path[1:])):
            raise ValueError("規劃路徑超出已記錄的雙軸工作區，目標未套用")

        if self._motors is not None:
            self._motors.set_home_return(home_return)
        self._apply_yaw_target(yaw_rad)
        if yaw_move_required:
            self._begin_yaw_clearance_sequence(muzzle_rad)
            self._publish(
                status=(
                    f"雙軸自動避讓：依水平目標先到 {math.degrees(self.yaw_clearance_pitch_rad):.1f}°，"
                    f"水平轉到 {yaw_deg:+.2f}°，再到垂直 {muzzle_deg:+.2f}°"
                ),
                error=None,
            )
            return

        self._clear_yaw_sequence()
        self._apply_pitch_target(
            muzzle_rad,
            f"砲口目標已設定：{muzzle_deg:+.2f}°",
        )
        self._publish(
            status=f"雙軸目標已設定：水平 {yaw_deg:+.2f}°、垂直 {muzzle_deg:+.2f}°",
            error=None,
        )

    def _update_yaw_clearance_sequence(self) -> None:
        """Advance clearance -> yaw -> restore using command and feedback."""
        phase = self._yaw_sequence_phase
        if phase is None:
            return

        if self._feedback_position_rad is None:
            return
        feedback_pitch = self.pitch_calibration.muzzle_from_motor(
            self._feedback_position_rad
        )

        if phase == "clearance":
            # An upward-moving pitch may still coast across the clearance;
            # release yaw only once the measured and commanded pitch are not rising.
            pitch_not_rising = (
                math.isfinite(self._command_velocity_rad_s)
                and math.isfinite(self._feedback_velocity_rad_s)
                and self._command_velocity_rad_s * self.pitch_calibration.motor_sign <= SEQUENCE_STOP_SPEED_RAD_S
                and self._feedback_velocity_rad_s * self.pitch_calibration.motor_sign <= SEQUENCE_STOP_SPEED_RAD_S
            )
            if feedback_pitch <= self.yaw_release_pitch_rad and pitch_not_rising:
                self._yaw_sequence_phase = "rotating"
                self._publish(
                    yaw_sequence_phase="rotating",
                    status=(
                        "垂直已進入水平目標的安全活動範圍；"
                        "開始水平、垂直同時移動"
                    ),
                    error=None,
                )
            return

        if phase == "rotating":
            if self._yaw_feedback_position_rad is None:
                return
            command_arrived = (
                abs(self._yaw_last_command_rad - self._yaw_target_rad)
                <= self.sequence_arrival_tolerance_rad
            )
            feedback_arrived = (
                abs(self._yaw_feedback_position_rad - self._yaw_target_rad)
                <= self.sequence_arrival_tolerance_rad
            )
            restore = self._yaw_restore_muzzle_rad
            if restore is None:
                self._clear_yaw_sequence()
                return
            if restore >= self.yaw_clearance_pitch_rad:
                stage_motor = self.pitch_calibration.motor_from_muzzle(
                    self.yaw_clearance_pitch_rad
                )
                pitch_command_arrived = (
                    abs(self._last_command_rad - stage_motor)
                    <= self.sequence_arrival_tolerance_rad
                )
                pitch_feedback_arrived = (
                    feedback_pitch <= self.yaw_clearance_pitch_rad
                )
                if not pitch_command_arrived or not pitch_feedback_arrived:
                    return
            if command_arrived and feedback_arrived:
                current_yaw = self.yaw_calibration.muzzle_from_motor(self._yaw_feedback_position_rad)
                if (not self._pose_allowed_by_activity(current_yaw, restore)
                        or not math.isfinite(self._yaw_command_velocity_rad_s)
                        or not math.isfinite(self._yaw_feedback_velocity_rad_s)
                        or abs(self._yaw_command_velocity_rad_s) > SEQUENCE_STOP_SPEED_RAD_S
                        or abs(self._yaw_feedback_velocity_rad_s) > SEQUENCE_STOP_SPEED_RAD_S):
                    return
                self._yaw_sequence_phase = "restoring"
                if restore >= self.yaw_clearance_pitch_rad:
                    self._apply_pitch_target(
                        restore,
                        "水平已到位；回復最終垂直角度",
                    )
                else:
                    self._publish(status="水平已到位；等待垂直到最終角度")
                self._publish(yaw_sequence_phase="restoring")
            return

        if phase == "restoring":
            restore = self._yaw_restore_muzzle_rad
            if restore is None:
                self._clear_yaw_sequence()
                return
            restore_motor = self.pitch_calibration.motor_from_muzzle(restore)
            command_arrived = (
                abs(self._last_command_rad - restore_motor)
                <= self.sequence_arrival_tolerance_rad
            )
            feedback_arrived = (
                abs(feedback_pitch - restore)
                <= self.sequence_arrival_tolerance_rad
            )
            if (command_arrived and feedback_arrived
                    and math.isfinite(self._command_velocity_rad_s)
                    and math.isfinite(self._feedback_velocity_rad_s)
                    and abs(self._command_velocity_rad_s) <= SEQUENCE_STOP_SPEED_RAD_S
                    and abs(self._feedback_velocity_rad_s) <= SEQUENCE_STOP_SPEED_RAD_S):
                final_pitch_deg = math.degrees(restore)
                deferred = self._deferred_aim
                self._clear_yaw_sequence()
                if deferred is not None:
                    try:
                        self._set_aim_targets(deferred[0], deferred[1], home_return=deferred[2])
                    except (ValueError, RuntimeError) as exc:
                        self._publish(status="新目標未套用；保持目前位置", error=str(exc))
                    return
                self._publish(
                    status=(
                        f"水平旋轉及垂直回復完成：砲口 {final_pitch_deg:+.2f}°"
                    ),
                    error=None,
                )

    def _apply_calibrations(
        self,
        pitch_calibration: PitchCalibration,
        yaw_calibration: PitchCalibration,
    ) -> None:
        if self._enabled:
            raise ValueError("套用手動校正前必須先失能")
        if self._feedback_position_rad is None:
            raise ValueError("尚未取得垂直馬達回饋")
        if self._yaw_feedback_position_rad is None:
            raise ValueError("尚未取得水平馬達回饋")

        if self._motors is not None:
            self._motors.set_yaw_calibration(yaw_calibration)
            self._motors.set_pitch_calibration(pitch_calibration)
        self.pitch_calibration = pitch_calibration
        self.yaw_calibration = yaw_calibration
        self.startup_pitch_rad = pitch_calibration.motor_zero_rad
        self._target_rad = self._feedback_position_rad
        self._target_muzzle_rad = pitch_calibration.muzzle_from_motor(
            self._feedback_position_rad
        )
        self._last_command_rad = self._feedback_position_rad
        self._command_velocity_rad_s = 0.0
        self._yaw_target_rad = self._yaw_feedback_position_rad
        self._yaw_target_angle_rad = yaw_calibration.muzzle_from_motor(
            self._yaw_feedback_position_rad
        )
        self._yaw_last_command_rad = self._yaw_feedback_position_rad
        self._yaw_command_velocity_rad_s = 0.0
        self._yaw_interlock_hold_rad = None
        self._yaw_sequence_phase = None
        self._yaw_restore_muzzle_rad = None
        self._deferred_aim = None
        self._publish(
            target_rad=self._target_rad,
            target_muzzle_rad=self._target_muzzle_rad,
            command_rad=self._last_command_rad,
            command_velocity_rad_s=0.0,
            yaw_target_rad=self._yaw_target_rad,
            yaw_target_angle_rad=self._yaw_target_angle_rad,
            yaw_command_rad=self._yaw_last_command_rad,
            yaw_command_velocity_rad_s=0.0,
            yaw_sequence_phase=None,
            yaw_restore_muzzle_rad=None,
            status="手動原點校正已套用；仍為失能，確認人員離開後才能使能",
            error=None,
        )

    def _update_stall_monitor(self, position_rad: float, now: float) -> None:
        """Warn about low motion without discarding the requested target.

        The previous implementation replaced ``_target_rad`` with feedback
        after one slow two-second window.  That made an enabled motor appear
        intermittent: every new command moved briefly and then software
        silently converted it into a hold command.  A stall is useful
        diagnostic information, but changing the operator's target here is
        both surprising and impossible to distinguish from a motor failure.
        """
        target_error = abs(self._target_rad - position_rad)
        if target_error <= math.radians(1.0):
            self._stall_reference_rad = position_rad
            self._stall_reference_at = now
            if self._stall_warning_active:
                self._stall_warning_active = False
                self._publish(
                    status="已到達砲口目標",
                    error=None,
                )
            return

        if self._stall_reference_rad is None:
            self._stall_reference_rad = position_rad
            self._stall_reference_at = now
            return

        if now - self._stall_reference_at < self.stall_timeout_s:
            return

        travelled = abs(position_rad - self._stall_reference_rad)
        if travelled < math.radians(0.25):
            self._stall_warning_active = True
            self._publish(
                status="停滯警告：移動量過小，但原砲口目標仍保留",
                error="stall_warning",
            )
        elif self._stall_warning_active:
            self._stall_warning_active = False
            self._publish(
                status="移動已恢復，繼續前往原砲口目標",
                error=None,
            )

        self._stall_reference_rad = position_rad
        self._stall_reference_at = now

    def _enable(self) -> None:
        if self._motors is None or self._enabled:
            return
        self._publish(status="使能中...", error=None)
        self._motors.enable()
        state = self._motors.get_state(PITCH_MOTOR_ID)
        if state is not None:
            # Enabling always holds the measured pose.  A move requires a new
            # explicit target, so stale targets cannot produce a surprise jump.
            self._last_command_rad = state.position_rad
            self._command_velocity_rad_s = 0.0
            self._feedback_position_rad = state.position_rad
            self._target_rad = state.position_rad
            self._target_muzzle_rad = self.pitch_calibration.muzzle_from_motor(
                state.position_rad
            )
            self._stall_reference_rad = state.position_rad
            self._stall_reference_at = time.monotonic()
            self._stall_warning_active = False
        yaw_state = (
            self._motors.get_state(YAW_MOTOR_ID)
            if self.yaw_calibration is not None
            else None
        )
        if self.yaw_calibration is not None:
            if yaw_state is None:
                raise RuntimeError("使能前讀不到水平馬達回饋")
            self._yaw_last_command_rad = yaw_state.position_rad
            self._yaw_command_velocity_rad_s = 0.0
            self._yaw_feedback_position_rad = yaw_state.position_rad
            self._yaw_target_rad = yaw_state.position_rad
            self._yaw_target_angle_rad = self.yaw_calibration.muzzle_from_motor(
                yaw_state.position_rad
            )
            self._yaw_interlock_hold_rad = None
        self._enabled_at = time.monotonic()
        self._last_tick_at = self._enabled_at
        self._enabled = True
        self._feedback_waiting = False
        self._yaw_sequence_phase = None
        self._yaw_restore_muzzle_rad = None
        self._deferred_aim = None
        self._publish(
            enabled=True,
            target_rad=self._target_rad,
            target_muzzle_rad=self._target_muzzle_rad,
            command_rad=self._last_command_rad,
            command_velocity_rad_s=0.0,
            torque_ff_nm=0.0,
            yaw_target_rad=(
                self._yaw_target_rad if self.yaw_calibration is not None else None
            ),
            yaw_target_angle_rad=(
                self._yaw_target_angle_rad
                if self.yaw_calibration is not None
                else None
            ),
            yaw_command_rad=(
                self._yaw_last_command_rad
                if self.yaw_calibration is not None
                else None
            ),
            yaw_command_velocity_rad_s=0.0,
            yaw_sequence_phase=None,
            yaw_restore_muzzle_rad=None,
            status="已使能並保持目前位置；可設定砲口仰角與水平角度",
            error=None,
        )

    def _disable(self) -> None:
        if self._motors is None:
            return
        # The panel's command flag can be false while the motor is still
        # physically enabled (for example after a previous process exited).
        # Always send the disable command; the feedback ERR field confirms
        # whether it actually took effect.
        self._motors.disable()
        self._enabled = False
        self._feedback_waiting = False
        self._command_velocity_rad_s = 0.0
        self._yaw_command_velocity_rad_s = 0.0
        self._yaw_interlock_hold_rad = None
        self._yaw_sequence_phase = None
        self._yaw_restore_muzzle_rad = None
        self._deferred_aim = None
        self._publish(
            enabled=False,
            yaw_sequence_phase=None,
            yaw_restore_muzzle_rad=None,
            status="已送出失能命令，等待馬達回饋確認；垂直軸可能下墜，請支撐機構",
        )

    def _handle_command(self, command: Tuple[str, object]) -> bool:
        action, value = command
        if action == "enable":
            self._enable()
        elif action == "disable":
            self._disable()
        elif action == "target":
            try:
                self._set_target(float(value))
            except ValueError as exc:
                self._publish(status="砲口目標未套用", error=str(exc))
        elif action == "yaw_target":
            try:
                self._set_yaw_target(float(value))
            except ValueError as exc:
                self._publish(status="水平目標未套用", error=str(exc))
        elif action == "aim":
            try:
                yaw_deg, muzzle_deg = value
                self._set_aim_targets(float(yaw_deg), float(muzzle_deg))
            except (TypeError, ValueError) as exc:
                self._publish(status="雙軸目標未套用", error=str(exc))
        elif action == "aim_tagged":
            try:
                yaw_deg, muzzle_deg, request_id = value
                self._set_aim_targets(float(yaw_deg), float(muzzle_deg))
                self._publish(aim_request_id=int(request_id))
            except (TypeError, ValueError) as exc:
                self._publish(status="雙軸目標未套用", error=str(exc))
        elif action == "home":
            try:
                home_deg = math.degrees(self._recorded_home_muzzle_rad())
                self._set_aim_targets(0.0, home_deg, home_return=True)
            except (ValueError, RuntimeError) as exc:
                self._publish(status="home 目標未套用", error=str(exc))
        elif action == "calibration":
            try:
                pitch_calibration, yaw_calibration = value
                self._apply_calibrations(pitch_calibration, yaw_calibration)
            except (TypeError, ValueError) as exc:
                self._publish(status="手動原點校正未套用", error=str(exc))
        elif action == "shutdown":
            if self._enabled:
                self._publish(status="請先按失能，再關閉程式", error="motor_enabled")
                return False
            self._stop.set()
        return True

    def _run(self) -> None:
        try:
            motor_ids = [PITCH_MOTOR_ID]
            if self.yaw_calibration is not None:
                motor_ids.append(YAW_MOTOR_ID)
            self._motors = DamiaoSerialBackend(
                motor_ids=motor_ids,
                port=self.port,
                mode=self.control_mode,
                kp=self.kp,
                kd=self.kd,
                enable_on_start=False,
                startup_hold_s=0.5,
                position_limits=self.position_limits,
                yaw_calibration=self.yaw_calibration,
                pitch_calibration=self.pitch_calibration,
            )
            # The USB-CAN read is non-blocking by design.  Give the adapter a
            # few cycles to return the first feedback frame before showing it.
            for _ in range(6):
                self._motors.refresh()
                time.sleep(0.03)
            state = self._motors.get_state(PITCH_MOTOR_ID)
            yaw_state = (
                self._motors.get_state(YAW_MOTOR_ID)
                if self.yaw_calibration is not None
                else None
            )
            if state is not None:
                self._last_command_rad = state.position_rad
                self._feedback_position_rad = state.position_rad
                self._publish(
                    position_rad=state.position_rad,
                    command_rad=state.position_rad,
                    motor_err=state.err,
                    feedback_at=state.feedback_at,
                    ready=True,
                    status="已連線，等待馬達回饋狀態",
                )
            else:
                self._publish(ready=True, status="已連線，尚無回饋")
            if self.yaw_calibration is not None:
                if yaw_state is None:
                    raise RuntimeError("讀不到水平馬達回饋")
                self._yaw_last_command_rad = yaw_state.position_rad
                self._yaw_feedback_position_rad = yaw_state.position_rad
                self._yaw_target_rad = yaw_state.position_rad
                self._yaw_target_angle_rad = self.yaw_calibration.muzzle_from_motor(
                    yaw_state.position_rad
                )
                self._publish(
                    yaw_position_rad=yaw_state.position_rad,
                    yaw_command_rad=yaw_state.position_rad,
                    yaw_target_rad=yaw_state.position_rad,
                    yaw_target_angle_rad=self._yaw_target_angle_rad,
                    yaw_motor_err=yaw_state.err,
                    yaw_feedback_at=yaw_state.feedback_at,
                )

            next_tick = time.monotonic()
            next_feedback = next_tick
            while not self._stop.is_set():
                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_command(command)

                now = time.monotonic()
                if self._enabled and self._motors is not None:
                    pitch_state = self._motors.get_state(PITCH_MOTOR_ID)
                    yaw_state = (self._motors.get_state(YAW_MOTOR_ID)
                                 if self.yaw_calibration is not None else None)
                    fresh = _fresh_enabled_feedback(pitch_state, now) and (
                        self.yaw_calibration is None or _fresh_enabled_feedback(yaw_state, now)
                    )
                    workspace = getattr(self._motors, "_workspace", None)
                    inside_workspace = (not fresh or yaw_state is None or workspace is None
                                        or workspace_contains(workspace, yaw_state.position_rad,
                                                              pitch_state.position_rad))
                    if pitch_state is not None:
                        self._publish(position_rad=pitch_state.position_rad,
                                      velocity_rad_s=pitch_state.velocity_rad_s,
                                      motor_err=pitch_state.err, feedback_at=pitch_state.feedback_at)
                    if yaw_state is not None:
                        self._publish(yaw_position_rad=yaw_state.position_rad,
                                      yaw_velocity_rad_s=yaw_state.velocity_rad_s,
                                      yaw_motor_err=yaw_state.err, yaw_feedback_at=yaw_state.feedback_at)
                    stopped_after_gap = fresh and (
                        abs(pitch_state.velocity_rad_s) <= SEQUENCE_STOP_SPEED_RAD_S
                        and (yaw_state is None or abs(yaw_state.velocity_rad_s) <= SEQUENCE_STOP_SPEED_RAD_S)
                    )
                    if not fresh or not inside_workspace or (self._feedback_waiting and not stopped_after_gap):
                        self._feedback_waiting = True
                        reason = "feedback_outside_workspace" if fresh and not inside_workspace else "feedback_stale"
                        self._publish(error=reason, status="馬達回饋失效、尚未停止或超出工作區；保持上一命令")
                        self._motors.hold()
                        self._motors.refresh()
                        next_tick = time.monotonic()
                        time.sleep(CONTROL_PERIOD_S)
                        continue
                    if self._feedback_waiting:
                        self._feedback_waiting = False
                        self._feedback_position_rad = self._last_command_rad = pitch_state.position_rad
                        self._feedback_velocity_rad_s = self._command_velocity_rad_s = pitch_state.velocity_rad_s
                        self._publish(error=None)
                        if yaw_state is not None:
                            self._yaw_feedback_position_rad = self._yaw_last_command_rad = yaw_state.position_rad
                            self._yaw_feedback_velocity_rad_s = self._yaw_command_velocity_rad_s = yaw_state.velocity_rad_s
                            desired = self._deferred_aim or (
                                math.degrees(self._yaw_target_angle_rad),
                                math.degrees(self._yaw_restore_muzzle_rad if self._yaw_restore_muzzle_rad is not None else self._target_muzzle_rad),
                                bool(getattr(self._motors, "_home_return_requested", False)),
                            )
                            self._clear_yaw_sequence()
                            self._yaw_interlock_hold_rad = None
                            try:
                                self._set_aim_targets(desired[0], desired[1], home_return=desired[2])
                            except (ValueError, RuntimeError) as exc:
                                self._apply_yaw_target(self.yaw_calibration.muzzle_from_motor(yaw_state.position_rad))
                                self._apply_pitch_target(self.pitch_calibration.muzzle_from_motor(pitch_state.position_rad),
                                                         "回饋恢復後路徑無效；保持目前位置")
                                self._publish(error=str(exc))
                    dt = min(max(now - self._last_tick_at, 0.001), 0.02)
                    self._last_tick_at = now
                    self._update_yaw_clearance_sequence()

                    # Limit the trajectory's moving target instead of clipping
                    # its output afterward.  The old post-step clamp also set
                    # velocity straight to zero, which defeated acceleration
                    # continuity exactly when a loaded mechanism fell behind.
                    trajectory_target = self._target_rad
                    if self._feedback_position_rad is not None:
                        trajectory_target = max(
                            self._feedback_position_rad - self.max_tracking_error_rad,
                            min(
                                self._feedback_position_rad + self.max_tracking_error_rad,
                                trajectory_target,
                            ),
                        )
                    position, velocity = acceleration_limited_step(
                        self._last_command_rad,
                        self._command_velocity_rad_s,
                        trajectory_target,
                        self.speed_rad_s,
                        self.acceleration_rad_s2,
                        dt,
                    )

                    yaw_position: Optional[float] = None
                    yaw_velocity = 0.0
                    yaw_trajectory_target: Optional[float] = None
                    yaw_interlock_ready = False
                    if self.yaw_calibration is not None:
                        if self._feedback_position_rad is not None:
                            feedback_pitch = self.pitch_calibration.muzzle_from_motor(
                                self._feedback_position_rad
                            )
                            yaw_interlock_ready = self._yaw_feedback_clear(feedback_pitch)
                        if (
                            yaw_interlock_ready
                            and self._yaw_sequence_phase != "clearance"
                        ):
                            self._yaw_interlock_hold_rad = None
                            yaw_trajectory_target = self._yaw_target_rad
                            if self._yaw_feedback_position_rad is not None:
                                yaw_trajectory_target = max(
                                    self._yaw_feedback_position_rad
                                    - self.max_tracking_error_rad,
                                    min(
                                        self._yaw_feedback_position_rad
                                        + self.max_tracking_error_rad,
                                        yaw_trajectory_target,
                                    ),
                                )
                            yaw_position, yaw_velocity = acceleration_limited_step(
                                self._yaw_last_command_rad,
                                self._yaw_command_velocity_rad_s,
                                yaw_trajectory_target,
                                self.speed_rad_s,
                                self.acceleration_rad_s2,
                                dt,
                            )
                        else:
                            # Hold yaw while pitch is outside its collision-safe
                            # range or while the automatic sequence is still
                            # moving all the way to its 10-degree clearance pose.
                            yaw_position = self._yaw_interlock_hold_position()
                            yaw_trajectory_target = yaw_position
                            yaw_velocity = 0.0

                    torque_ramp = min(1.0, max(0.0, (now - self._enabled_at) / self.torque_ramp_s))
                    active_torque_ff = self.torque_ff_nm * torque_ramp
                    targets = {
                        PITCH_MOTOR_ID: (
                            position,
                            self.speed_rad_s,
                            velocity,
                            active_torque_ff,
                        )
                    }
                    if yaw_position is not None:
                        targets[YAW_MOTOR_ID] = (
                            yaw_position,
                            self.speed_rad_s,
                            yaw_velocity,
                            0.0,
                        )
                    self._motors.send_targets(targets)
                    self._last_command_rad = position
                    self._command_velocity_rad_s = velocity
                    if yaw_position is not None:
                        self._yaw_last_command_rad = yaw_position
                        self._yaw_command_velocity_rad_s = yaw_velocity
                    self._publish(
                        effective_target_rad=trajectory_target,
                        command_rad=position,
                        command_velocity_rad_s=velocity,
                        tracking_error_rad=(
                            position - self._feedback_position_rad
                            if self._feedback_position_rad is not None
                            else None
                        ),
                        torque_ff_nm=active_torque_ff,
                        yaw_effective_target_rad=yaw_trajectory_target,
                        yaw_command_rad=yaw_position,
                        yaw_command_velocity_rad_s=yaw_velocity,
                        yaw_tracking_error_rad=(
                            yaw_position - self._yaw_feedback_position_rad
                            if yaw_position is not None
                            and self._yaw_feedback_position_rad is not None
                            else None
                        ),
                        yaw_interlock_ready=yaw_interlock_ready,
                    )
                    self._motors.refresh()
                    next_tick += CONTROL_PERIOD_S
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.monotonic()
                elif self._motors is not None:
                    if now >= next_feedback:
                        self._motors.refresh()
                        next_feedback = now + 0.1
                    time.sleep(0.01)

                if self._motors is not None:
                    state = self._motors.get_state(PITCH_MOTOR_ID)
                    if state is not None:
                        self._feedback_position_rad = state.position_rad
                        self._feedback_velocity_rad_s = state.velocity_rad_s
                        self._publish(
                            position_rad=state.position_rad,
                            velocity_rad_s=state.velocity_rad_s,
                            feedback_torque_nm=state.torque_nm,
                            tracking_error_rad=(
                                self._last_command_rad - state.position_rad
                            ),
                            motor_err=state.err,
                            feedback_at=state.feedback_at,
                            yaw_interlock_ready=self._yaw_feedback_clear(
                                self.pitch_calibration.muzzle_from_motor(state.position_rad)
                            ),
                        )
                        if self._enabled:
                            self._update_stall_monitor(state.position_rad, now)
                    if self.yaw_calibration is not None:
                        yaw_state = self._motors.get_state(YAW_MOTOR_ID)
                        if yaw_state is not None:
                            self._yaw_feedback_position_rad = yaw_state.position_rad
                            self._yaw_feedback_velocity_rad_s = yaw_state.velocity_rad_s
                            self._publish(
                                yaw_position_rad=yaw_state.position_rad,
                                yaw_velocity_rad_s=yaw_state.velocity_rad_s,
                                yaw_feedback_torque_nm=yaw_state.torque_nm,
                                yaw_tracking_error_rad=(
                                    self._yaw_last_command_rad
                                    - yaw_state.position_rad
                                ),
                                yaw_motor_err=yaw_state.err,
                                yaw_feedback_at=yaw_state.feedback_at,
                            )
        except Exception as exc:  # surfaced in the GUI
            self._publish(status="控制器錯誤", error=str(exc), ready=False, enabled=False)
            self._enabled = False
        finally:
            if self._motors is not None:
                try:
                    self._motors.close()
                except Exception:
                    pass
            self._publish(closed=True, enabled=False)


class VerticalPanel:
    def __init__(self, root: tk.Tk, worker: MotorWorker, min_angle: float, max_angle: float):
        self.root = root
        self.worker = worker
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._closing = False

        root.title("垂直馬達控制面板")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.minsize(600, 440)

        self.status_var = tk.StringVar(value="正在連接...")
        self.enabled_var = tk.StringVar(value="失能")
        self.feedback_var = tk.StringVar(value="尚未讀到")
        self.current_var = tk.StringVar(value="--")
        self.muzzle_var = tk.StringVar(value="--")
        self.target_var = tk.StringVar(value="0.00")
        self.target_abs_var = tk.StringVar(value="--")
        self.torque_var = tk.StringVar(value="--")
        self.tracking_var = tk.StringVar(value="--")
        self.error_var = tk.StringVar(value="")

        self._build()
        self.worker.start()
        self._poll()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="垂直馬達控制面板", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="輸入砲口仰角；0° 使用已記錄的水平砲口校正點。",
            foreground="#245b78",
        ).pack(anchor="w", pady=(3, 12))

        info = ttk.LabelFrame(frame, text="狀態", padding=8)
        info.pack(fill="x")
        self._row(info, 0, "連線／控制", self.status_var)
        self._row(info, 1, "使能狀態", self.enabled_var)
        self._row(info, 2, "馬達回饋 ERR", self.feedback_var)
        self._row(info, 3, "目前馬達角度", self.current_var)
        self._row(info, 4, "目前砲口仰角", self.muzzle_var)
        self._row(info, 5, "馬達絕對目標（非方向）", self.target_abs_var)
        self._row(info, 6, "馬達實際扭矩", self.torque_var)
        self._row(info, 7, "命令－回饋誤差", self.tracking_var)

        controls = ttk.LabelFrame(frame, text="目標砲口仰角（度）", padding=8)
        controls.pack(fill="x", pady=12)
        entry = ttk.Entry(controls, textvariable=self.target_var, width=12)
        entry.grid(row=0, column=0, padx=(0, 6), pady=4)
        ttk.Button(controls, text="套用目標", command=self._apply_target).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="回水平 0°", command=self._return_level).grid(row=0, column=2, padx=4)
        ttk.Label(
            controls,
            text=f"砲口軟限制：{self.min_angle:+.1f}° 到 {self.max_angle:+.1f}°",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        steps = ttk.Frame(controls)
        steps.grid(row=2, column=0, columnspan=3, pady=(9, 0), sticky="w")
        for label, step in (("-5°", -5.0), ("-1°", -1.0), ("+1°", 1.0), ("+5°", 5.0)):
            ttk.Button(steps, text=label, command=lambda value=step: self._nudge(value)).pack(
                side="left", padx=3
            )

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="使能（開始保持）", command=self._enable).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="失能（允許手動調整）", command=self._disable).pack(side="left")
        ttk.Label(frame, textvariable=self.error_var, foreground="#b00020").pack(anchor="w", pady=(12, 0))
        ttk.Label(
            frame,
            text="警告：垂直軸失能可能下墜；按失能前請先支撐機構。",
            foreground="#b00020",
        ).pack(anchor="w", pady=(4, 0))

    @staticmethod
    def _row(parent, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable).grid(row=row, column=1, sticky="w", pady=2)

    def _parse_target(self) -> Optional[float]:
        try:
            value = float(self.target_var.get())
        except ValueError:
            messagebox.showerror("角度格式錯誤", "請輸入數字，例如 0、+5 或 -3.5。")
            return None
        if not math.isfinite(value):
            messagebox.showerror("角度格式錯誤", "砲口仰角必須是有限數字。")
            return None
        if not self.min_angle <= value <= self.max_angle:
            messagebox.showerror("超出軟限制", f"請輸入 {self.min_angle:+.1f}° 到 {self.max_angle:+.1f}°。")
            return None
        return value

    def _apply_target(self) -> None:
        value = self._parse_target()
        if value is not None:
            self.worker.request_muzzle_angle(value)

    def _return_level(self) -> None:
        self.target_var.set("0.00")
        self._apply_target()

    def _nudge(self, step: float) -> None:
        try:
            value = float(self.target_var.get()) + step
        except ValueError:
            value = step
        self.target_var.set(f"{value:.2f}")
        self._apply_target()

    def _enable(self) -> None:
        if messagebox.askyesno(
            "確認使能",
            "請確認機構安全、急停可用，並且目標角度正確。\n\n要使能垂直馬達嗎？",
        ):
            self.worker.request_enable()

    def _disable(self) -> None:
        if messagebox.askyesno(
            "確認失能",
            "垂直軸失能後可能下墜。\n\n請先支撐機構，再按「是」。",
        ):
            self.worker.request_disable()

    def _on_close(self) -> None:
        snapshot = self.worker.snapshot()
        if snapshot.enabled:
            messagebox.showwarning("請先失能", "請先按「失能」，確認機構安全後再關閉面板。")
            return
        self._closing = True
        self.worker.request_shutdown()
        self.root.after(100, self._wait_closed)

    def _wait_closed(self) -> None:
        if self.worker.snapshot().closed:
            self.root.destroy()
        else:
            self.root.after(100, self._wait_closed)

    def _poll(self) -> None:
        snapshot = self.worker.snapshot()
        self.status_var.set(snapshot.status)
        self.enabled_var.set("使能" if snapshot.enabled else "失能")
        if snapshot.motor_err is None:
            self.feedback_var.set("尚未讀到")
        else:
            err_names = {0: "0（失能）", 1: "1（使能）", 8: "8（超壓）", 9: "9（欠壓）", 10: "A（過流）", 11: "B（MOS 過溫）", 12: "C（線圈過溫）", 13: "D（通訊丟失）", 14: "E（過載）"}
            self.feedback_var.set(err_names.get(snapshot.motor_err, f"{snapshot.motor_err:X}（故障／未知）"))
        if snapshot.position_rad is not None:
            self.current_var.set(
                f"{snapshot.position_rad:+.6f} rad ({math.degrees(snapshot.position_rad):+.2f}°)"
            )
            muzzle_deg = math.degrees(
                self.worker.pitch_calibration.muzzle_from_motor(snapshot.position_rad)
            )
            self.muzzle_var.set(f"{muzzle_deg:+.2f}°")
        if snapshot.target_rad is not None:
            muzzle_text = ""
            if snapshot.target_muzzle_rad is not None:
                muzzle_text = f"；砲口 {math.degrees(snapshot.target_muzzle_rad):+.2f}°"
            self.target_abs_var.set(
                f"{snapshot.target_rad:+.6f} rad ({math.degrees(snapshot.target_rad):+.2f}°)"
                f"{muzzle_text}"
            )
        if snapshot.feedback_torque_nm is not None:
            self.torque_var.set(f"{snapshot.feedback_torque_nm:+.3f} Nm")
        if snapshot.tracking_error_rad is not None:
            self.tracking_var.set(
                f"{snapshot.tracking_error_rad:+.6f} rad "
                f"({math.degrees(snapshot.tracking_error_rad):+.2f}°)"
            )
        self.error_var.set(snapshot.error or "")
        if not self._closing:
            self.root.after(100, self._poll)


def _print_terminal_status(worker: MotorWorker) -> None:
    snapshot = worker.snapshot()
    print("\n--- 砲塔雙軸狀態 ---")
    print(f"狀態：{snapshot.status}")
    print(f"使能：{'是' if snapshot.enabled else '否'}")
    if snapshot.motor_err is None:
        print("馬達回饋 ERR：尚未讀到")
    else:
        err_names = {0: "0（失能）", 1: "1（使能）", 8: "8（超壓）", 9: "9（欠壓）", 10: "A（過流）", 11: "B（MOS 過溫）", 12: "C（線圈過溫）", 13: "D（通訊丟失）", 14: "E（過載）"}
        print(f"馬達回饋 ERR：{err_names.get(snapshot.motor_err, f'{snapshot.motor_err:X}（故障／未知）')}")
    if snapshot.position_rad is not None:
        muzzle_deg = math.degrees(
            worker.pitch_calibration.muzzle_from_motor(snapshot.position_rad)
        )
        print(
            "馬達回饋：%+.6f rad (%+.2f°)；砲口仰角：%+.2f°"
            % (
                snapshot.position_rad,
                math.degrees(snapshot.position_rad),
                muzzle_deg,
            )
        )
    if snapshot.target_rad is not None:
        target_muzzle_deg = math.degrees(
            snapshot.target_muzzle_rad
            if snapshot.target_muzzle_rad is not None
            else worker.pitch_calibration.muzzle_from_motor(snapshot.target_rad)
        )
        print(
            "目標：砲口 %+.2f° -> 馬達 %+.6f rad (%+.2f°)"
            % (target_muzzle_deg, snapshot.target_rad, math.degrees(snapshot.target_rad))
        )
    if snapshot.effective_target_rad is not None:
        print(
            "追蹤限制後目標：%+.6f rad (%+.2f°)"
            % (
                snapshot.effective_target_rad,
                math.degrees(snapshot.effective_target_rad),
            )
        )
    if snapshot.command_rad is not None:
        print(
            "平滑命令：%+.6f rad (%+.2f°)，速度 %+.3f rad/s，前饋 %+.2f Nm"
            % (
                snapshot.command_rad,
                math.degrees(snapshot.command_rad),
                snapshot.command_velocity_rad_s or 0.0,
                snapshot.torque_ff_nm or 0.0,
            )
        )
    if snapshot.feedback_torque_nm is not None:
        print(f"馬達實際扭矩：{snapshot.feedback_torque_nm:+.3f} Nm")
    if snapshot.tracking_error_rad is not None:
        print(
            "命令－回饋誤差：%+.6f rad (%+.2f°)"
            % (
                snapshot.tracking_error_rad,
                math.degrees(snapshot.tracking_error_rad),
            )
        )
    if worker.yaw_calibration is not None:
        if snapshot.yaw_motor_err is None:
            print("水平馬達 ERR：尚未讀到")
        else:
            yaw_err_names = {
                0: "0（失能）",
                1: "1（使能）",
                8: "8（超壓）",
                9: "9（欠壓）",
                10: "A（過流）",
                11: "B（MOS 過溫）",
                12: "C（線圈過溫）",
                13: "D（通訊丟失）",
                14: "E（過載）",
            }
            print(
                "水平馬達 ERR："
                f"{yaw_err_names.get(snapshot.yaw_motor_err, f'{snapshot.yaw_motor_err:X}（故障／未知）')}"
            )
        if snapshot.yaw_position_rad is not None:
            yaw_deg = math.degrees(
                worker.yaw_calibration.muzzle_from_motor(
                    snapshot.yaw_position_rad
                )
            )
            print(
                "水平回饋：%+.6f rad (%+.2f° motor)；砲塔水平角：%+.2f°"
                % (
                    snapshot.yaw_position_rad,
                    math.degrees(snapshot.yaw_position_rad),
                    yaw_deg,
                )
            )
        if snapshot.yaw_target_rad is not None:
            target_yaw_deg = math.degrees(
                snapshot.yaw_target_angle_rad
                if snapshot.yaw_target_angle_rad is not None
                else worker.yaw_calibration.muzzle_from_motor(
                    snapshot.yaw_target_rad
                )
            )
            print(
                "水平目標：%+.2f° -> 馬達 %+.6f rad (%+.2f°)"
                % (
                    target_yaw_deg,
                    snapshot.yaw_target_rad,
                    math.degrees(snapshot.yaw_target_rad),
                )
            )
        if snapshot.yaw_command_rad is not None:
            print(
                "水平平滑命令：%+.6f rad，速度 %+.3f rad/s，實際扭矩 %+.3f Nm"
                % (
                    snapshot.yaw_command_rad,
                    snapshot.yaw_command_velocity_rad_s or 0.0,
                    snapshot.yaw_feedback_torque_nm or 0.0,
                )
            )
        if snapshot.yaw_tracking_error_rad is not None:
            print(
                "水平命令－回饋誤差：%+.6f rad (%+.2f°)"
                % (
                    snapshot.yaw_tracking_error_rad,
                    math.degrees(snapshot.yaw_tracking_error_rad),
                )
            )
        phase_names = {
            "clearance": "正在降到 35° 以下",
            "rotating": "水平、垂直同時移動",
            "restoring": "正在完成最終垂直角度",
        }
        if snapshot.yaw_sequence_phase is not None:
            restore_text = ""
            if snapshot.yaw_restore_muzzle_rad is not None:
                restore_text = (
                    f"，完成後回到 {math.degrees(snapshot.yaw_restore_muzzle_rad):.2f}°"
                )
            print(
                "水平自動避讓："
                f"{phase_names.get(snapshot.yaw_sequence_phase, snapshot.yaw_sequence_phase)}"
                f"{restore_text}"
            )
        print(
            "水平安全條件："
            + (
                "目前可直接旋轉（垂直 ≤ 35°）"
                if snapshot.yaw_interlock_ready
                else "目前高於 35°；水平會等實際回饋低於 35° 後自動開始"
            )
        )
    if snapshot.error:
        print(f"錯誤：{snapshot.error}")


def _wait_and_print_enabled_pose(
    worker: MotorWorker,
    timeout_s: float = 3.0,
) -> None:
    """Wait for actual enable feedback and report the physical two-axis pose."""
    deadline = time.monotonic() + timeout_s
    snapshot = worker.snapshot()
    while not snapshot.enabled and time.monotonic() < deadline:
        if snapshot.error:
            print(f"使能失敗：{snapshot.error}")
            return
        time.sleep(0.02)
        snapshot = worker.snapshot()

    if not snapshot.enabled:
        print("使能命令已送出，但尚未在期限內收到使能確認；請輸入 status 檢查。")
        return

    if snapshot.position_rad is None:
        vertical_text = "尚無回饋"
    else:
        vertical_deg = math.degrees(
            worker.pitch_calibration.muzzle_from_motor(snapshot.position_rad)
        )
        vertical_text = f"{vertical_deg:+.2f}°"

    if worker.yaw_calibration is None or snapshot.yaw_position_rad is None:
        horizontal_text = "尚無回饋"
    else:
        horizontal_deg = math.degrees(
            worker.yaw_calibration.muzzle_from_motor(snapshot.yaw_position_rad)
        )
        horizontal_text = f"{horizontal_deg:+.2f}°"

    print("使能完成，正在保持目前位置。")
    print(f"垂直：{vertical_text}")
    print(f"水平：{horizontal_text}")


def _prompt_and_request_pose(
    worker: MotorWorker,
    min_pitch_deg: float,
    max_pitch_deg: float,
) -> bool:
    """Prompt for a physical pitch/yaw pair and queue one final pose."""
    while True:
        vertical_text = input(
            f"垂直目標角度（{min_pitch_deg:.0f}～{max_pitch_deg:.0f}°，直接 Enter 取消）："
        ).strip()
        if not vertical_text:
            print("取消角度設定。")
            return False
        try:
            vertical_deg = float(vertical_text)
        except ValueError:
            print("垂直角度必須是數字。")
            continue
        if not math.isfinite(vertical_deg) or not (
            min_pitch_deg <= vertical_deg <= max_pitch_deg
        ):
            print(
                f"垂直角度必須在 {min_pitch_deg:.1f}° 到 {max_pitch_deg:.1f}° 之間。"
            )
            continue
        break

    while True:
        horizontal_text = input(
            "水平目標角度（-34.5～+34.5°，正向右、負向左，直接 Enter 取消）："
        ).strip()
        if not horizontal_text:
            print("取消角度設定。")
            return False
        try:
            horizontal_deg = float(horizontal_text)
        except ValueError:
            print("水平角度必須是數字。")
            continue
        if not math.isfinite(horizontal_deg) or not (
            MIN_TURRET_YAW_DEG <= horizontal_deg <= MAX_TURRET_YAW_DEG
        ):
            print("水平角度必須在 -34.5° 到 +34.5° 之間。")
            continue
        break

    worker.request_aim(horizontal_deg, vertical_deg)
    print(
        f"已設定砲口目標：垂直 {vertical_deg:+.2f}°、"
        f"水平 {horizontal_deg:+.2f}°"
    )
    return True


def _run_terminal_panel(
    worker: MotorWorker,
    min_angle: float,
    max_angle: float,
    startup_pose: Path = DEFAULT_STARTUP_POSE,
) -> None:
    """Fallback interface for SSH/headless sessions without an X display."""
    print("沒有圖形顯示環境，改用終端控制介面。")
    print(
        "指令：status | enable | disable | move | set <垂直角度> | "
        "yaw <水平角度> | aim <水平角度> <垂直角度> | "
        "calibrate | capture-level | capture-origin | +1 | -1 | +5 | -5 | home | quit"
    )
    print(f"set 的角度是砲口仰角，軟限制 {min_angle:+.1f}° 到 {max_angle:+.1f}°。")
    if worker.yaw_calibration is not None:
        print("yaw 是水平絕對角度：正向右、負向左，上限 -34.5° 到 +34.5°；實際範圍也受記錄的工作區限制。")
        print(
            f"若垂直高於 {math.degrees(worker.max_pitch_for_yaw_rad):.1f}°，"
            "yaw/aim 會先降低垂直；確認進入活動範圍後再水平移動。"
        )
    worker.start()
    target_angle: Optional[float] = None

    try:
        while True:
            snapshot = worker.snapshot()
            if snapshot.error and not snapshot.ready:
                print(f"控制器錯誤：{snapshot.error}")
                break
            try:
                command = input("motor> ").strip().lower()
            except EOFError:
                command = "quit"

            if command in ("status", "s"):
                _print_terminal_status(worker)
                continue
            if command in ("enable", "e"):
                worker.request_enable()
                _wait_and_print_enabled_pose(worker)
                if worker.snapshot().enabled:
                    _prompt_and_request_pose(worker, min_angle, max_angle)
                continue
            if command in ("move", "m", "angle"):
                if not worker.snapshot().enabled:
                    print("馬達尚未使能；請先輸入 enable。")
                else:
                    _prompt_and_request_pose(worker, min_angle, max_angle)
                continue
            if command in ("disable", "d"):
                worker.request_disable()
                print("已送出失能命令；垂直軸可能下墜，請支撐機構。")
                continue
            if command in ("calibrate", "manual-calibration"):
                latest = worker.snapshot()
                if latest.enabled:
                    answer = input(
                        "手動校正會失能兩軸；確認已支撐機構？輸入 yes："
                    ).strip().lower()
                    if answer != "yes":
                        print("取消手動校正。")
                        continue
                    worker.request_disable()
                    print("已送出失能命令；請先用 status 確認兩顆 ERR 都為 0。")
                else:
                    print("兩軸目前已失能。")
                print(
                    "請手動將水平對準 0°、垂直放到已記錄的 home 位置；"
                    "完成後輸入 capture-origin。"
                )
                continue
            if command == "capture-level":
                latest = worker.snapshot()
                if latest.enabled or latest.motor_err != 0 or latest.yaw_motor_err != 0:
                    print("拒絕校正：必須先失能，且兩顆馬達回饋 ERR 都必須為 0。")
                    continue
                answer = input("請支撐機構並用水平儀確認砲口仰角 0°；輸入 capture：").strip().lower()
                if answer != "capture":
                    print("取消水平基準校正。")
                    continue
                latest = worker.snapshot()
                if (latest.enabled or latest.motor_err != 0 or latest.yaw_motor_err != 0
                        or latest.position_rad is None or latest.feedback_at is None
                        or latest.velocity_rad_s is None
                        or not math.isfinite(latest.velocity_rad_s)
                        or abs(latest.velocity_rad_s) > SEQUENCE_STOP_SPEED_RAD_S
                        or not 0 <= time.monotonic() - latest.feedback_at <= .5):
                    print("拒絕校正：未取得新鮮、靜止的失能馬達位置回饋。")
                    continue
                try:
                    workspace = load_workspace_limit()
                    if workspace is None or workspace.home_corridor is None:
                        raise ValueError("未記錄 home 馬達位置")
                    lower, upper = worker.position_limits[PITCH_MOTOR_ID]
                    if not lower <= latest.position_rad <= upper:
                        raise ValueError("水平 0° 馬達位置超出既有硬限制")
                    proposed = PitchCalibration(latest.position_rad,
                                                worker.pitch_calibration.motor_rad_per_muzzle_rad,
                                                worker.pitch_calibration.motor_sign)
                    constrain_pitch_travel_limit((lower, upper), proposed,
                                                 workspace.home_corridor.home_pitch_rad)
                    backup_path = save_level_reference_calibration(startup_pose, latest.position_rad)
                    worker.request_calibration_reload(load_pitch_calibration(startup_pose),
                                                      load_yaw_calibration(startup_pose))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    print(f"水平基準校正失敗：{exc}")
                    continue
                print(f"已記錄水平 0°；home 編碼器位置未變。舊檔備份：{backup_path}")
                continue
            if command == "capture-origin":
                latest = worker.snapshot()
                if latest.enabled or latest.motor_err != 0 or latest.yaw_motor_err != 0:
                    print("拒絕校正：必須先失能，且兩顆馬達回饋 ERR 都必須為 0。")
                    continue
                if latest.position_rad is None or latest.yaw_position_rad is None:
                    print("拒絕校正：尚未取得兩顆馬達的位置回饋。")
                    continue
                answer = input(
                    "確認目前是水平 0°、垂直已記錄的 home？輸入 capture："
                ).strip().lower()
                if answer != "capture":
                    print("取消擷取原點。")
                    continue
                try:
                    workspace = load_workspace_limit()
                    if (workspace is None or workspace.home_corridor is None
                            or abs(latest.position_rad - workspace.home_corridor.home_pitch_rad) > math.radians(.05)
                            or abs(latest.yaw_position_rad - workspace.home_corridor.yaw_center_rad) > math.radians(.05)):
                        raise ValueError("目前位置與記錄的 home 不同；請使用工作區校正工具重新記錄")
                    backup_path = save_manual_origin_calibration(
                        startup_pose,
                        yaw_motor_rad=latest.yaw_position_rad,
                        pitch_motor_rad=latest.position_rad,
                    )
                    pitch_calibration = load_pitch_calibration(startup_pose)
                    yaw_calibration = load_yaw_calibration(startup_pose)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    print(f"原點校正檔寫入失敗：{exc}")
                    continue
                worker.request_calibration_reload(
                    pitch_calibration,
                    yaw_calibration,
                )
                print(
                    "已擷取手動 home；垂直水平 0° 量測基準保持不變。"
                    f"舊檔備份：{backup_path}"
                )
                print("馬達仍為失能；人員離開機構後才能輸入 enable。")
                continue
            if command.startswith("yaw "):
                try:
                    requested_yaw = float(command.split(None, 1)[1])
                except ValueError:
                    print("格式：yaw 水平角度，例如 yaw 40 或 yaw -20")
                    continue
                if not math.isfinite(requested_yaw):
                    print("水平角度必須是有限數字。")
                elif not MIN_TURRET_YAW_DEG <= requested_yaw <= MAX_TURRET_YAW_DEG:
                    print("超出水平安全限制：-34.5° 到 +34.5°")
                else:
                    worker.request_yaw_angle(requested_yaw)
                    if worker.yaw_calibration is not None:
                        target_motor = worker.yaw_calibration.motor_from_muzzle(
                            math.radians(requested_yaw)
                        )
                        print(
                            f"水平目標 {requested_yaw:+.2f}°（正向右、負向左） -> "
                            f"馬達 {target_motor:+.6f} rad"
                        )
                continue
            if command.startswith("aim "):
                parts = command.split()
                if len(parts) != 3:
                    print("格式：aim 水平角度 垂直角度，例如 aim 20 15")
                    continue
                try:
                    requested_yaw = float(parts[1])
                    requested_pitch = float(parts[2])
                except ValueError:
                    print("aim 的水平與垂直角度都必須是數字。")
                    continue
                if not math.isfinite(requested_yaw) or not math.isfinite(
                    requested_pitch
                ):
                    print("aim 的水平與垂直角度都必須是有限數字。")
                elif not MIN_TURRET_YAW_DEG <= requested_yaw <= MAX_TURRET_YAW_DEG:
                    print("超出水平安全限制：-34.5° 到 +34.5°")
                elif not min_angle <= requested_pitch <= max_angle:
                    print(
                        f"垂直角度必須在 {min_angle:.1f}° 到 {max_angle:.1f}° 之間"
                    )
                else:
                    worker.request_aim(requested_yaw, requested_pitch)
                    print(
                        f"雙軸同步目標：水平 {requested_yaw:+.2f}°、"
                        f"垂直 {requested_pitch:+.2f}°"
                    )
                continue
            if command in ("home", "initial", "i"):
                if not worker.snapshot().enabled:
                    print("馬達尚未使能；請先輸入 enable。")
                else:
                    worker.request_home()
                    target_angle = math.degrees(worker._recorded_home_muzzle_rad())
                    print(f"回 home：先水平置中，再回到記錄的馬達位置（推算仰角 {target_angle:.2f}°）。")
                continue
            if target_angle is None:
                latest = worker.snapshot()
                if latest.target_muzzle_rad is not None:
                    target_angle = math.degrees(latest.target_muzzle_rad)
                elif latest.position_rad is not None:
                    target_angle = math.degrees(
                        worker.pitch_calibration.muzzle_from_motor(latest.position_rad)
                    )
                else:
                    target_angle = 0.0

            previous_target_angle = target_angle
            requested_angle = target_angle
            if command in ("+1", "-1", "+5", "-5"):
                requested_angle += float(command)
            elif command.startswith("set "):
                try:
                    requested_angle = float(command.split(None, 1)[1])
                except ValueError:
                    print("格式：set 砲口仰角，例如 set 5 或 set -2.5")
                    continue
            elif command in ("quit", "q", "exit"):
                if worker.snapshot().enabled:
                    print("請先輸入 disable，再離開程式。")
                    continue
                worker.request_shutdown()
                worker.wait_closed()
                return
            else:
                print(
                    "未知指令；可用 status、enable、disable、move、set、yaw、aim、"
                    "calibrate、capture-origin、+1、-1、home、quit。"
                )
                continue

            if not math.isfinite(requested_angle):
                print("砲口仰角必須是有限數字。")
            elif not min_angle <= requested_angle <= max_angle:
                print(f"超出砲口軟限制：{min_angle:+.1f}° 到 {max_angle:+.1f}°")
            else:
                target_angle = requested_angle
                worker.request_muzzle_angle(target_angle)
                target_motor = worker.pitch_calibration.motor_from_muzzle(
                    math.radians(target_angle)
                )
                previous_motor = worker.pitch_calibration.motor_from_muzzle(
                    math.radians(previous_target_angle)
                )
                muzzle_delta = target_angle - previous_target_angle
                motor_delta = target_motor - previous_motor
                if motor_delta > 1e-12:
                    motor_direction = "馬達角增加"
                elif motor_delta < -1e-12:
                    motor_direction = "馬達角減少"
                else:
                    motor_direction = "馬達目標不變"
                print(
                    f"砲口目標 {previous_target_angle:+.2f}° -> {target_angle:+.2f}° "
                    f"（變化 {muzzle_delta:+.2f}°）"
                )
                print(
                    f"馬達絕對目標 {target_motor:+.6f} rad "
                    f"({math.degrees(target_motor):+.2f}°)；"
                    f"目標變化 {motor_delta:+.6f} rad，{motor_direction}"
                )
    except KeyboardInterrupt:
        print("\n已中斷終端介面；請支撐機構並確認馬達狀態。")
    finally:
        if not worker.snapshot().closed:
            worker.request_shutdown()
            worker.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="砲塔水平／垂直 DAMIAO 馬達角度控制面板")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--startup-pose", type=Path, default=DEFAULT_STARTUP_POSE)
    parser.add_argument(
        "--pitch-sign",
        type=int,
        choices=(-1, 1),
        help="覆寫校正檔：馬達角增加時砲口上升用 1，下降用 -1",
    )
    parser.add_argument(
        "--motor-rad-per-muzzle-rad",
        type=float,
        help="覆寫校正檔：砲口轉 1 rad 所需的馬達角度；直驅為 1",
    )
    parser.add_argument("--speed", type=float, default=0.10, help="平滑移動速度上限 rad/s")
    parser.add_argument("--acceleration", type=float, default=0.09, help="平滑加速度上限 rad/s^2")
    parser.add_argument(
        "--torque-ff",
        type=float,
        default=0.0,
        help="MIT 模式重力補償前饋 Nm；方向與大小必須實機校正，預設 0",
    )
    parser.add_argument("--torque-ramp", type=float, default=1.5, help="使能後前饋爬升時間，秒")
    parser.add_argument("--kp", type=float, default=50.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument(
        "--control-mode",
        choices=("mit", "posvel"),
        default="posvel",
        help="馬達控制模式；實機垂直軸預設使用已驗證可移動的 posvel",
    )
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=10.0,
        help="平滑命令最多領先實際位置的角度，度",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=2.0,
        help="無明顯移動多久後觸發停滯保護，秒",
    )
    parser.add_argument(
        "--min-angle",
        "--min-offset",
        dest="min_angle",
        type=float,
        default=None,
        help="砲口仰角軟限制下限，度；預設使用 mechanism_limits.json",
    )
    parser.add_argument(
        "--max-angle",
        "--max-offset",
        dest="max_angle",
        type=float,
        default=None,
        help="砲口仰角軟限制上限，度；預設使用 mechanism_limits.json",
    )
    parser.add_argument("--cli", action="store_true", help="強制使用終端控制介面")
    args = parser.parse_args()

    if (
        args.speed <= 0
        or args.acceleration <= 0
        or args.torque_ramp <= 0
        or args.kp < 0
        or args.kd <= 0
        or args.max_tracking_error <= 0
        or args.stall_timeout <= 0
        or (
            args.motor_rad_per_muzzle_rad is not None
            and args.motor_rad_per_muzzle_rad <= 0
        )
    ):
        raise SystemExit(
            "speed、acceleration、torque-ramp、max-tracking-error、stall-timeout、"
            "motor-rad-per-muzzle-rad 必須 > 0；"
            "kp 必須 >= 0；kd 必須 > 0"
        )
    if not all(
        math.isfinite(value)
        for value in (
            args.speed,
            args.acceleration,
            args.torque_ramp,
            args.kp,
            args.kd,
            args.max_tracking_error,
            args.stall_timeout,
            args.torque_ff,
            *(
                (args.motor_rad_per_muzzle_rad,)
                if args.motor_rad_per_muzzle_rad is not None
                else ()
            ),
        )
    ):
        raise SystemExit("所有控制參數都必須是有限數字")
    stored_calibration = load_pitch_calibration(args.startup_pose)
    yaw_calibration = load_yaw_calibration(args.startup_pose)
    pitch_calibration = PitchCalibration(
        motor_zero_rad=stored_calibration.motor_zero_rad,
        motor_rad_per_muzzle_rad=(
            args.motor_rad_per_muzzle_rad
            if args.motor_rad_per_muzzle_rad is not None
            else stored_calibration.motor_rad_per_muzzle_rad
        ),
        motor_sign=(
            args.pitch_sign
            if args.pitch_sign is not None
            else stored_calibration.motor_sign
        ),
    )
    position_limits = load_motor_limits()

    def physical_bounds(calibration, motor_id, absolute_min, absolute_max):
        motor_min, motor_max = position_limits[motor_id]
        physical = sorted(
            math.degrees(calibration.muzzle_from_motor(value))
            for value in (motor_min, motor_max)
        )
        return max(absolute_min, physical[0]), min(absolute_max, physical[1])

    pitch_safe_min, pitch_safe_max = physical_bounds(
        pitch_calibration,
        PITCH_MOTOR_ID,
        MIN_MUZZLE_ELEVATION_DEG,
        PITCH_MAX_DEG,
    )
    # Avoid rejecting a displayed 30.0° endpoint due to floating point
    # conversion (e.g. 29.999999999999996°); the backend still clamps motors.
    pitch_safe_min, pitch_safe_max = round(pitch_safe_min, 10), round(pitch_safe_max, 10)
    yaw_safe_min, yaw_safe_max = physical_bounds(
        yaw_calibration,
        YAW_MOTOR_ID,
        MIN_TURRET_YAW_DEG,
        MAX_TURRET_YAW_DEG,
    )
    if pitch_safe_min >= pitch_safe_max or yaw_safe_min >= yaw_safe_max:
        raise SystemExit("mechanism_limits.json 與 startup_pose.json 無法形成有效物理範圍")

    if args.min_angle is None:
        args.min_angle = pitch_safe_min
    if args.max_angle is None:
        args.max_angle = pitch_safe_max
    if not all(math.isfinite(value) for value in (args.min_angle, args.max_angle)):
        raise SystemExit("--min-angle 與 --max-angle 必須是有限數字")
    if args.min_angle >= args.max_angle:
        raise SystemExit("--min-angle 必須小於 --max-angle")
    if args.min_angle < pitch_safe_min or args.max_angle > pitch_safe_max:
        raise SystemExit(
            "面板角度不能超過 mechanism_limits.json 的實測範圍："
            f"{pitch_safe_min:.2f}° 到 {pitch_safe_max:.2f}°"
        )

    max_pitch_for_yaw_deg = min(MAX_PITCH_FOR_YAW_DEG, args.max_angle)
    yaw_clearance_pitch_deg = max(
        args.min_angle,
        max_pitch_for_yaw_deg - YAW_RELEASE_MARGIN_DEG,
    )
    worker = MotorWorker(
        port=args.port,
        pitch_calibration=pitch_calibration,
        speed_rad_s=args.speed,
        acceleration_rad_s2=args.acceleration,
        torque_ff_nm=args.torque_ff,
        torque_ramp_s=args.torque_ramp,
        kp=args.kp,
        kd=args.kd,
        max_tracking_error_deg=args.max_tracking_error,
        stall_timeout_s=args.stall_timeout,
        min_muzzle_deg=args.min_angle,
        max_muzzle_deg=args.max_angle,
        control_mode=args.control_mode,
        yaw_calibration=yaw_calibration,
        min_yaw_deg=yaw_safe_min,
        max_yaw_deg=yaw_safe_max,
        max_pitch_for_yaw_deg=max_pitch_for_yaw_deg,
        yaw_clearance_pitch_deg=yaw_clearance_pitch_deg,
        position_limits=position_limits,
    )

    if tk is None or args.cli or (os.name != "nt" and not os.environ.get("DISPLAY")):
        _run_terminal_panel(worker, args.min_angle, args.max_angle, args.startup_pose)
        return

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"無法建立圖形介面（{exc}），改用終端控制介面。")
        _run_terminal_panel(worker, args.min_angle, args.max_angle, args.startup_pose)
        return
    VerticalPanel(root, worker, args.min_angle, args.max_angle)
    root.mainloop()


if __name__ == "__main__":
    main()

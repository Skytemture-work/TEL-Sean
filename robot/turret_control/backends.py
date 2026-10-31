"""Motor backends for laptop simulation and AGX DAMIAO control.

IMPORTANT: the hardware is a DAMIAO USB-to-CAN debug board, which is *not* a
socketcan device.  The board exposes a USB CDC-ACM virtual serial port and
speaks a proprietary framing (0x55 0xAA header, 921600 bps) that wraps the CAN
frames.  ``python-can``/socketcan cannot talk to it, and this AGX has no
``can0`` interface at all.  All real motor traffic therefore goes through the
vendor library ``DM_CAN.py``, which is already verified working on this AGX.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .limits import (
    PITCH_MAX_DEG,
    WorkspaceLimit,
    clamp_motor_position,
    clamp_workspace_target,
    constrain_pitch_travel_limit,
    constrain_yaw_design_limit,
    load_motor_limits,
    load_workspace_limit,
    workspace_base_contains,
    workspace_contains,
    workspace_in_home_corridor,
)
from .pitch import load_pitch_calibration, load_yaw_calibration

DEFAULT_BAUD = 921600

# Where DM_CAN.py / dm_port.py live if they are not already importable.
DEFAULT_DM_PATH = Path(__file__).resolve().parent.parent / "dm_motor_test"


@dataclass(frozen=True)
class PositionCommand:
    motor_id: int
    position_rad: float
    max_speed_rad_s: float
    velocity_rad_s: float = 0.0
    torque_nm: float = 0.0


@dataclass(frozen=True)
class MotorState:
    """Feedback decoded from the motor, all on the *output shaft*."""

    motor_id: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    # DAMIAO feedback ERR nibble: 0=disabled, 1=enabled, 8..E=fault.
    err: Optional[int] = None


class MotorBackend(Protocol):
    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        ...

    def hold(self) -> None:
        ...

    def refresh(self) -> None:
        ...

    def get_state(self, motor_id: int) -> Optional[MotorState]:
        ...

    def close(self) -> None:
        ...


class MockMotorBackend:
    """A safe backend that records commands instead of moving hardware.

    It models a motor that tracks perfectly and instantly.  Pass ``motor_ids``
    so it knows which motors exist from the start -- the real backend can read
    a position before anything has been commanded, and ``hold()`` relies on
    that to keep feeding the comms-loss watchdog on the very first ticks.
    """

    def __init__(self, motor_ids: Optional[Iterable[int]] = None, verbose: bool = True):
        self.commands: List[PositionCommand] = []
        self.verbose = verbose
        self._last: Dict[int, PositionCommand] = {}
        self._position: Dict[int, float] = {int(m): 0.0 for m in (motor_ids or ())}

    def _emit(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        command = PositionCommand(motor_id, position_rad, max_speed_rad_s)
        self.commands.append(command)
        self._last[motor_id] = command
        self._position[motor_id] = position_rad

    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        self._emit(motor_id, position_rad, max_speed_rad_s)
        if self.verbose:
            print(
                f"[MOCK MOTOR] id={motor_id} "
                f"position={position_rad:.4f} rad "
                f"max_speed={max_speed_rad_s:.4f} rad/s"
            )

    def hold(self) -> None:
        """Re-send the current target so the watchdog stays fed."""
        for motor_id in sorted(set(self._position) | set(self._last)):
            command = self._last.get(motor_id)
            if command is None:
                self._emit(motor_id, self._position.get(motor_id, 0.0), 0.5)
            else:
                self.commands.append(command)

    def refresh(self) -> None:
        pass

    def get_state(self, motor_id: int) -> Optional[MotorState]:
        if motor_id not in self._position:
            return None
        return MotorState(motor_id, self._position[motor_id], 0.0, 0.0)

    def close(self) -> None:
        pass


def _load_dm_can(dm_path: Optional[Path] = None):
    """Import the vendor library, adding the dm_motor_test directory if needed."""
    try:
        import DM_CAN  # noqa: F401
        import dm_port  # noqa: F401
    except ImportError:
        candidate = Path(dm_path) if dm_path else DEFAULT_DM_PATH
        if not (candidate / "DM_CAN.py").is_file():
            raise RuntimeError(
                f"DM_CAN.py not found in {candidate}. Pass dm_path=... pointing at "
                "the directory holding DM_CAN.py and dm_port.py."
            )
        sys.path.insert(0, str(candidate))

    import DM_CAN
    import dm_port

    return DM_CAN, dm_port


class DamiaoSerialBackend:
    """Two-axis DAMIAO backend over the USB-to-CAN debug board.

    Unlike the previous socketcan version this owns the full motor lifecycle:
    it aligns the decode ranges, switches control mode, enables the motors, and
    disables them cleanly on close.

    ``mode="posvel"`` uses the trapezoidal position-speed mode.  It is simple
    and was verified on this AGX, but the motor's internal MAX_SPD register
    caps the output shaft near 1.3 rad/s and peaks lag the command by ~0.57 s.
    ``mode="mit"`` tracks far better (measured lag 0.0375 rad) but kp/kd must be
    retuned once the real mechanism inertia exists.
    """

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: Optional[str] = None,
        mode: str = "posvel",
        kp: float = 30.0,
        kd: float = 1.0,
        master_ids: Optional[Sequence[int]] = None,
        motor_type_name: str = "DM4340",
        baud: int = DEFAULT_BAUD,
        dm_path: Optional[Path] = None,
        enable_on_start: bool = True,
        startup_hold_s: float = 0.5,
        position_limits: Optional[Mapping[int, Tuple[float, float]]] = None,
        workspace: Optional[WorkspaceLimit] = None,
        yaw_calibration=None,
        pitch_calibration=None,
    ):
        import serial

        mode = mode.lower()
        if mode not in ("posvel", "mit"):
            raise ValueError("mode must be 'posvel' or 'mit'")
        if startup_hold_s < 0.0:
            raise ValueError("startup_hold_s must be non-negative")
        if mode == "mit" and kd <= 0.0:
            # Documented in the vendor manual: kd=0 in MIT position control
            # oscillates and can run away.
            raise ValueError("kd must be > 0 in MIT mode")

        motor_ids = list(motor_ids)
        if len(set(motor_ids)) != len(motor_ids):
            raise ValueError(
                f"duplicate motor ids {motor_ids}. Both motors ship as ESC_ID 0x01; "
                "change one of them before putting both on the bus."
            )

        # None means "read MST_ID off each motor".  The two motors on this robot
        # do not share a MST_ID (0x00 and 0x13), and addMotor only registers the
        # feedback route when MasterID != 0, so guessing it silently loses the
        # feedback frames of any motor whose MST_ID is non-zero.
        if master_ids is not None and len(master_ids) != len(motor_ids):
            raise ValueError("master_ids must be the same length as motor_ids")

        DM_CAN, dm_port = _load_dm_can(dm_path)
        self._dm = DM_CAN

        self.mode = mode
        self.kp = kp
        self.kd = kd
        self.startup_hold_s = startup_hold_s
        if position_limits is None:
            try:
                position_limits = load_motor_limits()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "mechanism_limits.json is missing; run "
                    "turret_control.run_limit_calibration before enabling motors"
                ) from exc
        self._position_limits = {
            int(motor_id): (float(bounds[0]), float(bounds[1]))
            for motor_id, bounds in position_limits.items()
        }
        self._recorded_position_limits = dict(self._position_limits)
        if 0x03 in self._position_limits:
            if yaw_calibration is None:
                yaw_calibration = load_yaw_calibration(Path(__file__).with_name("startup_pose.json"))
            self.set_yaw_calibration(yaw_calibration)
        if 0x01 in self._position_limits:
            if pitch_calibration is None:
                pitch_calibration = load_pitch_calibration(Path(__file__).with_name("startup_pose.json"))
            self.set_pitch_calibration(pitch_calibration)
        for motor_id, bounds in self._position_limits.items():
            if bounds[0] >= bounds[1] or not all(math.isfinite(value) for value in bounds):
                raise ValueError(f"invalid motor position limits for 0x{motor_id:02X}")
        self._last_limit_notice: Dict[int, Tuple[float, float]] = {}
        if workspace is None:
            workspace = load_workspace_limit()
        self._workspace = workspace
        if self._workspace is not None:
            required_ids = {
                self._workspace.yaw_motor_id,
                self._workspace.pitch_motor_id,
            }
            if not required_ids.issubset(set(motor_ids)):
                raise RuntimeError(
                    "non-rectangular workspace needs both yaw and pitch motors; "
                    "do not control either axis without the other feedback"
                )
        self._last_workspace_notice: Optional[Tuple[float, float, float]] = None

        motor_type = getattr(DM_CAN.DM_Motor_Type, motor_type_name)
        self._motor_type = motor_type

        self.port = dm_port.find_port(port)
        self._serial = serial.Serial(self.port, baud, timeout=0.5)
        self._mc = DM_CAN.MotorControl(self._serial)

        self._motors: Dict[int, object] = {}
        self._last: Dict[int, PositionCommand] = {}
        try:
            self._probe_and_build(motor_ids, master_ids)
            self._switch_modes()
            self._enabled = False
            if enable_on_start:
                self.enable()
        except Exception:
            self._serial.close()
            raise

    # ---------------------------------------------------------------- setup

    def _probe_and_build(
        self,
        motor_ids: Sequence[int],
        master_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Probe every configured id, then register the motors for real.

        Two things are discovered here rather than assumed:

        * ``MST_ID`` -- the id the motor sends feedback on.  ``addMotor`` only
          routes feedback through ``MasterID`` when it is non-zero, so a wrong
          guess silently throws away that motor's position.  The two motors on
          this robot genuinely differ (0x00 and 0x13).
        * ``PMAX/VMAX/TMAX`` -- this motor reports VMAX=10.0 while the library's
          built-in DM4340 entry says 8.0.  Without aligning them every decoded
          position, velocity and torque is wrong.

        Probing also proves each id actually answers, which catches both a dead
        motor and the "both motors still on ESC_ID 0x01" mistake.
        """
        DM_variable = self._dm.DM_variable
        discovered: Dict[int, int] = {}
        limits = None

        for index, motor_id in enumerate(motor_ids):
            self._mc.motors_map.clear()
            probe = self._dm.Motor(self._motor_type, motor_id, 0x00)
            self._mc.addMotor(probe)

            pmax = self._mc.read_motor_param(probe, DM_variable.PMAX)
            vmax = self._mc.read_motor_param(probe, DM_variable.VMAX)
            tmax = self._mc.read_motor_param(probe, DM_variable.TMAX)
            if None in (pmax, vmax, tmax):
                raise RuntimeError(
                    f"motor id 0x{motor_id:02X} did not answer. Check the 24V supply, "
                    "the CAN wiring, and that this id is really programmed into the "
                    "motor (run 03_set_id.py with one motor on the bus to scan)."
                )

            if limits is None:
                limits = (pmax, vmax, tmax)
            elif (pmax, vmax, tmax) != limits:
                raise RuntimeError(
                    f"motor 0x{motor_id:02X} reports limits {(pmax, vmax, tmax)} but the "
                    f"first motor reports {limits}. The library keeps one range per "
                    "motor type, so mixed limits would decode wrong."
                )

            if master_ids is not None:
                discovered[motor_id] = int(master_ids[index])
            else:
                mst = self._mc.read_motor_param(probe, DM_variable.MST_ID)
                if mst is None:
                    raise RuntimeError(
                        f"could not read MST_ID from motor 0x{motor_id:02X}; "
                        "pass master_ids=... explicitly"
                    )
                discovered[motor_id] = int(mst)

        # Registering two motors on the same MST_ID would make motors_map route
        # both feedback streams to whichever was added last.
        non_zero = [m for m in discovered.values() if m != 0]
        if len(set(non_zero)) != len(non_zero):
            raise RuntimeError(
                f"motors share a MST_ID: {discovered}. Give each motor a distinct "
                "MST_ID (or 0) so their feedback can be told apart."
            )

        self.master_ids = discovered
        self.pmax, self.vmax, self.tmax = limits
        self._mc.change_limit_param(self._motor_type, self.pmax, self.vmax, self.tmax)

        self._mc.motors_map.clear()
        self._motors = {}
        for motor_id in motor_ids:
            motor = self._dm.Motor(self._motor_type, motor_id, discovered[motor_id])
            self._mc.addMotor(motor)
            self._motors[motor_id] = motor

    def _switch_modes(self) -> None:
        control_type = (
            self._dm.Control_Type.POS_VEL if self.mode == "posvel" else self._dm.Control_Type.MIT
        )
        for motor_id, motor in self._motors.items():
            if not self._mc.switchControlMode(motor, control_type):
                raise RuntimeError(f"failed to switch motor 0x{motor_id:02X} to {self.mode}")

    # ------------------------------------------------------------ lifecycle

    def set_yaw_calibration(self, calibration) -> None:
        """Update the design boundary without loosening recorded limits."""
        if getattr(self, "_enabled", False):
            raise RuntimeError("yaw calibration cannot change while motors are enabled")
        self._position_limits[0x03] = constrain_yaw_design_limit(
            self._recorded_position_limits[0x03], calibration
        )
        self._yaw_calibration = calibration

    def set_pitch_calibration(self, calibration) -> None:
        """Update the calibrated pitch travel range while disabled."""
        if getattr(self, "_enabled", False):
            raise RuntimeError("pitch calibration cannot change while motors are enabled")
        self._position_limits[0x01] = constrain_pitch_travel_limit(
            self._recorded_position_limits[0x01], calibration
        )
        self._pitch_calibration = calibration

    def set_home_return(self, enabled: bool) -> None:
        """Allow explicit return through the centered home corridor."""
        if enabled and (self._workspace is None or self._workspace.home_corridor is None):
            raise RuntimeError("home return requires a calibrated workspace corridor")
        self._home_return_requested = bool(enabled)

    def enable(self) -> None:
        # Capture the actual pose before enabling.  Sending the first control
        # frame only after enable can otherwise create a sudden torque/position
        # error from a stale feedback sample.  Pre-loading the current pose and
        # holding it briefly makes startup much softer, especially in MIT mode.
        self.refresh()
        startup_positions = {}
        for motor_id in self._motors:
            state = self.get_state(motor_id)
            if state is None or state.err is None:
                raise RuntimeError(f"no feedback from motor 0x{motor_id:02X} before enable")
            startup_positions[motor_id] = state.position_rad

        if 0x03 in startup_positions and 0x03 in self._position_limits:
            lower, upper = self._position_limits[0x03]
            if not lower <= startup_positions[0x03] <= upper:
                raise RuntimeError(
                    "current yaw is outside the recorded or ±34.5° safety limit; "
                    "move it by hand into the allowed range before enabling"
                )

        if 0x01 in startup_positions and 0x01 in self._position_limits:
            lower, upper = self._position_limits[0x01]
            if not lower <= startup_positions[0x01] <= upper:
                raise RuntimeError(
                    "current pitch is outside the recorded or 0–45° travel limit; "
                    "move it by hand into the allowed range before enabling"
                )

        workspace = getattr(self, "_workspace", None)
        if workspace is not None and not workspace_contains(
            workspace,
            startup_positions[workspace.yaw_motor_id],
            startup_positions[workspace.pitch_motor_id],
        ):
            raise RuntimeError(
                "current yaw/pitch pose is outside the recorded safe workspace; "
                "move it by hand to the recorded interior before enabling"
            )

        for motor in self._motors.values():
            self._mc.enable(motor)
        self._enabled = True

        deadline = time.monotonic() + self.startup_hold_s
        while time.monotonic() < deadline:
            self.send_targets(
                {
                    motor_id: (position, 0.2, 0.0, 0.0)
                    for motor_id, position in startup_positions.items()
                }
            )
            self.refresh()
            time.sleep(0.005)

    def disable(self) -> None:
        for motor in self._motors.values():
            self._mc.disable(motor)
        self._enabled = False
        self._home_return_requested = False

    def set_zero_here(self) -> None:
        """Define the current pose as zero.  Only valid while disabled."""
        if self._enabled:
            raise RuntimeError("set_zero_here() requires the motors to be disabled first")
        for motor in self._motors.values():
            self._mc.set_zero_position(motor)

    # -------------------------------------------------------------- control

    def _send_target(
        self,
        motor_id: int,
        position_rad: float,
        max_speed_rad_s: float,
        velocity_rad_s: float = 0.0,
        torque_nm: float = 0.0,
    ) -> None:
        """Send one frame without waiting for a serial response.

        Several targets can therefore be put on the CAN bus back-to-back and
        one feedback read can be done after the whole control cycle.  Waiting
        after every axis command was the source of the visible stop-and-go
        motion on the USB-CAN adapter.
        """
        motor = self._motors.get(motor_id)
        if motor is None:
            raise KeyError(f"motor id {motor_id} was not configured; have {sorted(self._motors)}")

        position = float(position_rad)
        limits = self._position_limits.get(motor_id)
        if limits is not None:
            position, clipped = clamp_motor_position(position, limits)
            if clipped:
                now = time.monotonic()
                previous = self._last_limit_notice.get(motor_id)
                if (
                    previous is None
                    or abs(previous[0] - position) > 1e-9
                    or now - previous[1] > 2.0
                ):
                    print(
                        f"[SAFETY LIMIT] motor 0x{motor_id:02X}: "
                        f"target {position_rad:+.6f} rad -> {position:+.6f} rad",
                        flush=True,
                    )
                self._last_limit_notice[motor_id] = (position, now)

        speed = min(abs(max_speed_rad_s), self.vmax)
        velocity = max(-self.vmax, min(self.vmax, float(velocity_rad_s)))
        torque = max(-self.tmax, min(self.tmax, float(torque_nm)))
        if self.mode == "posvel":
            self._mc.control_Pos_Vel(motor, position, speed, receive=False)
        else:
            self._mc.controlMIT(
                motor,
                self.kp,
                self.kd,
                position,
                velocity,
                torque,
                receive=False,
            )

        self._last[motor_id] = PositionCommand(
            motor_id,
            position,
            speed,
            velocity,
            torque,
        )

    def send_targets(
        self,
        targets: Mapping[int, Tuple[float, float, float, float]],
    ) -> None:
        """Send a coordinated batch of yaw/pitch targets.

        Values are ``(position_rad, speed_rad_s, velocity_rad_s, torque_nm)``.
        With a traced workspace present, both axes are checked together and an
        unsafe pair is stopped at the first recorded boundary instead of being
        independently clamped into an unsafe rectangle.
        """

        prepared: Dict[int, Tuple[float, float, float, float]] = {}
        for motor_id, values in targets.items():
            if len(values) != 4:
                raise ValueError("each target must be (position, speed, velocity, torque)")
            position, speed, velocity, torque = values
            if motor_id not in self._motors:
                raise KeyError(
                    f"motor id {motor_id} was not configured; have {sorted(self._motors)}"
                )
            limits = self._position_limits.get(motor_id)
            if limits is not None:
                position, _ = clamp_motor_position(float(position), limits)
            if motor_id == 0x01 and hasattr(self, "_pitch_calibration"):
                calibration = self._pitch_calibration
                desired_elevation = calibration.muzzle_from_motor(position)
                if desired_elevation > math.radians(PITCH_MAX_DEG) and not getattr(self, "_home_return_requested", False):
                    state = self.get_state(motor_id)
                    measured = (
                        calibration.muzzle_from_motor(state.position_rad)
                        if state is not None and state.err is not None
                        and state.err < 0x8 and math.isfinite(state.position_rad)
                        else math.radians(PITCH_MAX_DEG)
                    )
                    # Home and the vertical-only exit corridor may be held or
                    # exited. Ordinary commands cannot raise beyond the larger
                    # of 35° and the actual feedback position.
                    allowed_elevation = min(desired_elevation, max(math.radians(PITCH_MAX_DEG), measured))
                    position = calibration.motor_from_muzzle(allowed_elevation)
            prepared[motor_id] = (position, speed, velocity, torque)

        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            yaw_id = workspace.yaw_motor_id
            pitch_id = workspace.pitch_motor_id

            def previous_position(motor_id: int, anchor_index: int) -> float:
                # While enabled, safety decisions must use actual feedback.
                # Commanded position can be far ahead of a gravity-loaded axis;
                # using it here could unlock yaw before pitch has really lowered.
                if getattr(self, "_enabled", False):
                    state = self.get_state(motor_id)
                    if (
                        state is not None
                        and state.err is not None
                        and state.err < 0x8
                        and math.isfinite(state.position_rad)
                    ):
                        return state.position_rad
                command = self._last.get(motor_id)
                if command is not None:
                    return command.position_rad
                state = self.get_state(motor_id)
                if (
                    state is not None
                    and state.err is not None
                    and math.isfinite(state.position_rad)
                ):
                    return state.position_rad
                return workspace.anchor[anchor_index]

            def fallback_target(motor_id: int, anchor_index: int) -> Tuple[float, float, float, float]:
                command = self._last.get(motor_id)
                if command is not None:
                    return (
                        command.position_rad,
                        command.max_speed_rad_s,
                        command.velocity_rad_s,
                        command.torque_nm,
                    )
                return (previous_position(motor_id, anchor_index), 0.2, 0.0, 0.0)

            if yaw_id not in prepared:
                prepared[yaw_id] = fallback_target(yaw_id, 0)
            if pitch_id not in prepared:
                prepared[pitch_id] = fallback_target(pitch_id, 1)
            previous_yaw = previous_position(yaw_id, 0)
            previous_pitch = previous_position(pitch_id, 1)
            requested_yaw = prepared[yaw_id][0]
            requested_pitch = prepared[pitch_id][0]
            original_requested_yaw = requested_yaw
            original_requested_pitch = requested_pitch
            staged_for_home = False
            corridor = workspace.home_corridor
            if corridor is not None:
                previous_in_corridor = workspace_in_home_corridor(
                    workspace, previous_yaw, previous_pitch
                )
                requested_in_corridor = workspace_in_home_corridor(
                    workspace, requested_yaw, requested_pitch
                )
                previous_in_base = workspace_base_contains(
                    workspace, previous_yaw, previous_pitch
                )
                yaw_is_centered = (
                    abs(previous_yaw - corridor.yaw_center_rad)
                    <= min(
                        corridor.yaw_tolerance_rad,
                        math.radians(0.25) * getattr(
                            getattr(self, "_yaw_calibration", None),
                            "motor_rad_per_muzzle_rad", 1.0,
                        ),
                    )
                )
                if previous_in_corridor and not previous_in_base and not requested_in_corridor:
                    # Leaving 45-degree home: lower vertically into the traced
                    # base before permitting any horizontal movement.
                    requested_yaw = corridor.yaw_center_rad
                    staged_for_home = True
                elif requested_in_corridor and not previous_in_corridor and not yaw_is_centered:
                    # Returning home: center yaw at the current safe low pitch,
                    # then permit the vertical rise on a later control cycle.
                    requested_yaw = corridor.yaw_center_rad
                    requested_pitch = previous_pitch
                    staged_for_home = True
            (safe_yaw, safe_pitch), clipped = clamp_workspace_target(
                workspace,
                previous_yaw,
                previous_pitch,
                requested_yaw,
                requested_pitch,
            )
            clipped = clipped or staged_for_home
            if clipped:
                now = time.monotonic()
                notice = getattr(self, "_last_workspace_notice", None)
                if (
                    notice is None
                    or abs(notice[0] - safe_yaw) > math.radians(0.25)
                    or abs(notice[1] - safe_pitch) > math.radians(0.25)
                    or now - notice[2] > 2.0
                ):
                    print(
                        "[SAFETY WORKSPACE] yaw/pitch target "
                        f"({original_requested_yaw:+.6f}, {original_requested_pitch:+.6f}) -> "
                        f"({safe_yaw:+.6f}, {safe_pitch:+.6f})",
                        flush=True,
                    )
                self._last_workspace_notice = (safe_yaw, safe_pitch, now)
            yaw_values = prepared[yaw_id]
            pitch_values = prepared[pitch_id]
            prepared[yaw_id] = (safe_yaw, yaw_values[1], yaw_values[2], yaw_values[3])
            prepared[pitch_id] = (safe_pitch, pitch_values[1], pitch_values[2], pitch_values[3])

        for motor_id in sorted(prepared):
            position, speed, velocity, torque = prepared[motor_id]
            self._send_target(motor_id, position, speed, velocity, torque)

    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        """Queue a position command; call ``refresh()`` once per cycle."""
        self.send_targets({motor_id: (position_rad, max_speed_rad_s, 0.0, 0.0)})

    def send_target(
        self,
        motor_id: int,
        position_rad: float,
        max_speed_rad_s: float,
        velocity_rad_s: float = 0.0,
        torque_nm: float = 0.0,
    ) -> None:
        """Queue a target with MIT velocity and torque feed-forward."""
        self.send_targets(
            {
                motor_id: (
                    position_rad,
                    max_speed_rad_s,
                    velocity_rad_s,
                    torque_nm,
                )
            }
        )

    def hold(self) -> None:
        """Re-send the last target for every motor.

        The motor auto-disables if it stops receiving frames (comms-loss
        protection), and the vertical axis has no brake, so the control loop
        must call this on every tick where no new target is available instead
        of simply sending nothing.
        """
        targets = {}
        for motor_id, motor in self._motors.items():
            command = self._last.get(motor_id)
            if command is None:
                self.refresh()
                state = self.get_state(motor_id)
                if state is None or state.err is None:
                    continue
                targets[motor_id] = (state.position_rad, 0.5, 0.0, 0.0)
            else:
                targets[motor_id] = (
                    command.position_rad,
                    command.max_speed_rad_s,
                    command.velocity_rad_s,
                    command.torque_nm,
                )
        if targets:
            self.send_targets(targets)
        self._mc.recv()

    # ------------------------------------------------------------- feedback

    def refresh(self) -> None:
        for motor in self._motors.values():
            self._mc.refresh_motor_status(motor, receive=False)
        self._mc.recv()

    def get_state(self, motor_id: int) -> Optional[MotorState]:
        motor = self._motors.get(motor_id)
        if motor is None:
            return None
        return MotorState(
            motor_id=motor_id,
            position_rad=motor.getPosition(),
            velocity_rad_s=motor.getVelocity(),
            torque_nm=motor.getTorque(),
            err=motor.getError(),
        )

    # ---------------------------------------------------------------- close

    def close(self, send_hold: bool = True) -> None:
        """Ramp the command to the present pose, then disable and close.

        Disabling drops an unbalanced vertical axis, so this only removes the
        software hold; the mechanism still needs its own counterweight or
        mechanical stop.

        ``send_hold=False`` is used by manual calibration, where the motors
        are already disabled and the program must not send any position frame
        while the operator is moving the mechanism by hand.
        """
        try:
            self.refresh()
            if send_hold:
                targets = {}
                for motor_id, motor in self._motors.items():
                    state = self.get_state(motor_id)
                    if state is None or state.err is None:
                        continue
                    targets[motor_id] = (state.position_rad, 0.5, 0.0, 0.0)
                for _ in range(20):
                    if targets:
                        self.send_targets(targets)
                    time.sleep(0.005)
            self.disable()
        finally:
            self._serial.close()

    def __enter__(self) -> "DamiaoSerialBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

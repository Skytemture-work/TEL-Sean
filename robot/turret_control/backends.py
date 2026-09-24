"""Motor backends for laptop simulation and AGX DAMIAO control.

IMPORTANT: the hardware is a DAMIAO USB-to-CAN debug board, which is *not* a
socketcan device.  The board exposes a USB CDC-ACM virtual serial port and
speaks a proprietary framing (0x55 0xAA header, 921600 bps) that wraps the CAN
frames.  ``python-can``/socketcan cannot talk to it, and this AGX has no
``can0`` interface at all.  All real motor traffic therefore goes through the
vendor library ``DM_CAN.py``, which is already verified working on this AGX.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

DEFAULT_BAUD = 921600

# Where DM_CAN.py / dm_port.py live if they are not already importable.
DEFAULT_DM_PATH = Path(__file__).resolve().parent.parent / "dm_motor_test"


@dataclass(frozen=True)
class PositionCommand:
    motor_id: int
    position_rad: float
    max_speed_rad_s: float


@dataclass(frozen=True)
class MotorState:
    """Feedback decoded from the motor, all on the *output shaft*."""

    motor_id: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float


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
    ):
        import serial

        mode = mode.lower()
        if mode not in ("posvel", "mit"):
            raise ValueError("mode must be 'posvel' or 'mit'")
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

        motor_type = getattr(DM_CAN.DM_Motor_Type, motor_type_name)
        self._motor_type = motor_type

        self.port = dm_port.find_port(port)
        self._serial = serial.Serial(self.port, baud, timeout=0.5)
        self._mc = DM_CAN.MotorControl(self._serial)

        self._motors: Dict[int, object] = {}
        try:
            self._probe_and_build(motor_ids, master_ids)
            self._switch_modes()
            self._enabled = False
            if enable_on_start:
                self.enable()
        except Exception:
            self._serial.close()
            raise

        self._last: Dict[int, PositionCommand] = {}

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

    def enable(self) -> None:
        for motor in self._motors.values():
            self._mc.enable(motor)
        self._enabled = True
        time.sleep(0.1)

    def disable(self) -> None:
        for motor in self._motors.values():
            self._mc.disable(motor)
        self._enabled = False

    def set_zero_here(self) -> None:
        """Define the current pose as zero.  Only valid while disabled."""
        if self._enabled:
            raise RuntimeError("set_zero_here() requires the motors to be disabled first")
        for motor in self._motors.values():
            self._mc.set_zero_position(motor)

    # -------------------------------------------------------------- control

    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        motor = self._motors.get(motor_id)
        if motor is None:
            raise KeyError(f"motor id {motor_id} was not configured; have {sorted(self._motors)}")

        speed = min(abs(max_speed_rad_s), self.vmax)
        if self.mode == "posvel":
            self._mc.control_Pos_Vel(motor, float(position_rad), speed)
        else:
            self._mc.controlMIT(motor, self.kp, self.kd, float(position_rad), 0.0, 0.0)

        self._last[motor_id] = PositionCommand(motor_id, float(position_rad), speed)

    def hold(self) -> None:
        """Re-send the last target for every motor.

        The motor auto-disables if it stops receiving frames (comms-loss
        protection), and the vertical axis has no brake, so the control loop
        must call this on every tick where no new target is available instead
        of simply sending nothing.
        """
        for motor_id, motor in self._motors.items():
            command = self._last.get(motor_id)
            if command is None:
                self.refresh()
                state = self.get_state(motor_id)
                if state is None:
                    continue
                position, speed = state.position_rad, 0.5
            else:
                position, speed = command.position_rad, command.max_speed_rad_s

            if self.mode == "posvel":
                self._mc.control_Pos_Vel(motor, position, speed)
            else:
                self._mc.controlMIT(motor, self.kp, self.kd, position, 0.0, 0.0)

    # ------------------------------------------------------------- feedback

    def refresh(self) -> None:
        for motor in self._motors.values():
            self._mc.refresh_motor_status(motor)

    def get_state(self, motor_id: int) -> Optional[MotorState]:
        motor = self._motors.get(motor_id)
        if motor is None:
            return None
        return MotorState(
            motor_id=motor_id,
            position_rad=motor.getPosition(),
            velocity_rad_s=motor.getVelocity(),
            torque_nm=motor.getTorque(),
        )

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """Ramp the command to the present pose, then disable and close.

        Disabling drops an unbalanced vertical axis, so this only removes the
        software hold; the mechanism still needs its own counterweight or
        mechanical stop.
        """
        try:
            self.refresh()
            for motor_id, motor in self._motors.items():
                state = self.get_state(motor_id)
                if state is None:
                    continue
                for _ in range(20):
                    if self.mode == "posvel":
                        self._mc.control_Pos_Vel(motor, state.position_rad, 0.5)
                    else:
                        self._mc.controlMIT(motor, self.kp, self.kd, state.position_rad, 0.0, 0.0)
                    time.sleep(0.005)
            self.disable()
        finally:
            self._serial.close()

    def __enter__(self) -> "DamiaoSerialBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

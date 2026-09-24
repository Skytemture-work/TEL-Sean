"""Motor backends for laptop simulation and AGX CAN control."""

from dataclasses import dataclass
import struct
from typing import Optional, Protocol


class MotorBackend(Protocol):
    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class PositionCommand:
    motor_id: int
    position_rad: float
    max_speed_rad_s: float


class MockMotorBackend:
    """A safe backend that records commands instead of moving hardware."""

    def __init__(self):
        self.commands: list[PositionCommand] = []

    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        command = PositionCommand(motor_id, position_rad, max_speed_rad_s)
        self.commands.append(command)
        print(
            f"[MOCK MOTOR] id={motor_id} "
            f"position={position_rad:.4f} rad "
            f"max_speed={max_speed_rad_s:.4f} rad/s"
        )

    def close(self) -> None:
        pass


class DamiaoCanBackend:
    """Damiao position-speed CAN backend.

    This sends the documented position-mode frame (0x100 + CAN_ID, two
    little-endian float32 values). Motor enable/disable and feedback parsing
    remain intentionally separate because they depend on the exact motor
    firmware and the team's existing working AGX code.
    """

    def __init__(self, channel: str = "can0", bitrate: int = 1_000_000):
        try:
            import can
        except ImportError as exc:
            raise RuntimeError(
                "python-can is required for the AGX CAN backend; "
                "install it with: pip install python-can"
            ) from exc

        self._bus = can.interface.Bus(
            channel=channel,
            interface="socketcan",
            bitrate=bitrate,
        )

    def set_position(self, motor_id: int, position_rad: float, max_speed_rad_s: float) -> None:
        import can

        if not 0 <= motor_id <= 0x7FF:
            raise ValueError("motor_id must fit in a standard CAN identifier")

        payload = struct.pack("<ff", float(position_rad), float(max_speed_rad_s))
        self._bus.send(
            can.Message(
                arbitration_id=0x100 + motor_id,
                data=payload,
                is_extended_id=False,
            )
        )

    def close(self) -> None:
        self._bus.shutdown()

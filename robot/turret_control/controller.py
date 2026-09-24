"""High-level turret controller joining aiming and motor output."""

from __future__ import annotations

from .aiming import AimCommand, AimingController, TargetPixel
from .backends import MotorBackend
from .config import TurretConfig


class TurretController:
    """Send safe position commands for the two-axis turret.

    The controller does not fire the launcher. ``fire_allowed`` is only a
    software condition; a separate physical safety switch should remain in
    series with the launcher actuator.
    """

    def __init__(self, config: TurretConfig, motors: MotorBackend):
        self.config = config
        self.aiming = AimingController(config)
        self.motors = motors

    def update(self, target: TargetPixel | None) -> AimCommand:
        """Run one tick.  Call this at a fixed rate, every tick, forever.

        A tick always sends motor frames, even with no target.  The motors
        auto-disable if they stop receiving frames, and the vertical axis has
        no brake, so "no detection" must mean *hold*, never *stop talking*.
        """
        self.motors.refresh()
        yaw_state = self.motors.get_state(self.config.yaw_motor_id)
        pitch_state = self.motors.get_state(self.config.pitch_motor_id)

        command = self.aiming.update(
            target,
            yaw_state.position_rad if yaw_state is not None else None,
            pitch_state.position_rad if pitch_state is not None else None,
        )

        if command.has_target:
            # The values are guaranteed non-None when has_target is true.
            self.motors.set_position(
                self.config.yaw_motor_id,
                command.target_yaw_rad,
                self.config.max_motor_speed_rad_s,
            )
            self.motors.set_position(
                self.config.pitch_motor_id,
                command.target_pitch_rad,
                self.config.max_motor_speed_rad_s,
            )
        else:
            self.motors.hold()

        return command

    def close(self) -> None:
        self.motors.close()

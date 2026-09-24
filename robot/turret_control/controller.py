"""High-level turret controller joining aiming and motor output."""

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
        command = self.aiming.update(target)

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

        return command

    def close(self) -> None:
        self.motors.close()

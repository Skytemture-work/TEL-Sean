"""High-level turret controller joining aiming and motor output."""

from __future__ import annotations

from dataclasses import replace

from .aiming import AimCommand, AimingController, TargetPixel
from .backends import MotorBackend
from .config import TurretConfig
from .limits import workspace_contains


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

        workspace = getattr(self.motors, "_workspace", None)
        outside_workspace = (command.has_target and workspace is not None
                             and not workspace_contains(workspace, command.target_yaw_rad, command.target_pitch_rad))
        if outside_workspace:
            command = replace(command, target_reachable=False, fire_allowed=False,
                              blocked_reason="target is outside the motor workspace")

        if outside_workspace or command.blocked_reason == "target is outside the mechanical limits":
            self.aiming.reset_stability()
            self.motors.hold()
            return command

        if command.has_target:
            # The values are guaranteed non-None when has_target is true.
            send_targets = getattr(self.motors, "send_targets", None)
            if callable(send_targets):
                send_targets(
                    {
                        self.config.yaw_motor_id: (
                            command.target_yaw_rad,
                            self.config.max_motor_speed_rad_s,
                            0.0,
                            0.0,
                        ),
                        self.config.pitch_motor_id: (
                            command.target_pitch_rad,
                            self.config.max_motor_speed_rad_s,
                            0.0,
                            0.0,
                        ),
                    }
                )
            else:
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
            motion_ready = getattr(self.motors, "motion_ready", None)
            if callable(motion_ready) and not motion_ready(command.target_yaw_rad, command.target_pitch_rad):
                self.aiming.reset_stability()
                command = replace(command, stable=False, fire_allowed=False,
                                  blocked_reason=command.blocked_reason or "waiting for safe two-axis movement")
        else:
            self.motors.hold()

        return command

    def close(self) -> None:
        self.motors.close()

import unittest

from turret_control.aiming import TargetPixel
from turret_control.backends import MotorState
from turret_control.config import TurretConfig
from turret_control.controller import TurretController


class _RecordingMotors:
    def __init__(self, config):
        self.config = config
        self.positions = {
            config.yaw_motor_id: 0.0,
            config.pitch_motor_id: 0.0,
        }
        self.sent = []
        self.holds = 0

    def refresh(self):
        pass

    def get_state(self, motor_id):
        return MotorState(motor_id, self.positions[motor_id], 0.0, 0.0, 1)

    def send_targets(self, targets):
        self.sent.append(targets)
        for motor_id, values in targets.items():
            self.positions[motor_id] = values[0]

    def hold(self):
        self.holds += 1

    def close(self):
        pass


class ControllerSafetyTests(unittest.TestCase):
    def test_safe_target_is_sent_to_both_axes(self):
        config = TurretConfig()
        motors = _RecordingMotors(config)
        controller = TurretController(config, motors)

        command = controller.update(
            TargetPixel(
                config.camera_center_x_px + 20.0,
                config.camera_center_y_px,
                distance_m=5.0,
            )
        )

        self.assertTrue(command.target_reachable)
        self.assertIsNone(command.blocked_reason)
        self.assertEqual(len(motors.sent), 1)
        self.assertEqual(
            set(motors.sent[0]),
            {config.yaw_motor_id, config.pitch_motor_id},
        )
        self.assertEqual(motors.holds, 0)

    def test_blocked_target_holds_instead_of_moving_to_a_limit(self):
        config = TurretConfig()
        motors = _RecordingMotors(config)
        controller = TurretController(config, motors)

        command = controller.update(
            TargetPixel(
                config.image_width_px * 8.0,
                config.camera_center_y_px,
                distance_m=5.0,
            )
        )

        self.assertFalse(command.target_reachable)
        self.assertIsNotNone(command.blocked_reason)
        self.assertEqual(motors.sent, [])
        self.assertEqual(motors.holds, 1)


if __name__ == "__main__":
    unittest.main()

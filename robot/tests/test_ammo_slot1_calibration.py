import unittest
from unittest.mock import Mock, patch

from mks_servo_can.mks_enums import MotorStatus

import calibrate_ammo_slot1 as calibration


class Slot1CalibrationTests(unittest.TestCase):
    def servo(self, readings, status=MotorStatus.MotorStop):
        motor = Mock()
        motor.query_motor_status.return_value = status
        motor.read_encoder_value_addition.side_effect = readings
        return motor

    @patch.object(calibration.time, "sleep")
    def test_stable_signed_encoder_values_are_accepted_without_motion(self, _sleep):
        motor = self.servo([-1, 0, -1])
        self.assertEqual(calibration.read_stopped_encoder(motor), [-1, 0, -1])
        motor.run_motor_relative_motion_by_pulses.assert_not_called()
        motor.set_current_axis_to_zero.assert_not_called()

    @patch.object(calibration.time, "sleep")
    def test_rejects_changing_encoder_position(self, _sleep):
        motor = self.servo([10, 20, 40])
        with self.assertRaisesRegex(RuntimeError, "不穩定"):
            calibration.read_stopped_encoder(motor)

    @patch.object(calibration.time, "sleep")
    def test_rejects_missing_feedback(self, _sleep):
        motor = self.servo([None])
        with self.assertRaisesRegex(RuntimeError, "讀取失敗"):
            calibration.read_stopped_encoder(motor)

    @patch.object(calibration.time, "sleep")
    def test_rejects_motor_that_is_not_confirmed_stopped(self, _sleep):
        motor = self.servo([123], status=MotorStatus.MotorFullSpeed)
        with self.assertRaisesRegex(RuntimeError, "未確認停止"):
            calibration.read_stopped_encoder(motor)
        motor.read_encoder_value_addition.assert_not_called()


if __name__ == "__main__":
    unittest.main()

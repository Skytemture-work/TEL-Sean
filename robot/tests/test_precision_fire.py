import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import manual
from turret_control.pitch import PitchCalibration
from turret_control import run_precision_fire as precision


class AbsoluteAimTests(unittest.TestCase):
    def setUp(self):
        self.dm = manual.DMStartupController()
        self.dm._cal = {3: PitchCalibration(0.3, 1.5, -1),
                        1: PitchCalibration(-0.8, 0.8, 1)}
        self.dm._limits = {3: (-2.0, 2.0), 1: (-2.0, 2.0)}
        self.dm._workspace = None
        self.dm.yaw_min_deg, self.dm.yaw_max_deg = -30.0, 30.0
        self.dm.pitch_min_deg, self.dm.pitch_max_deg = 10.0, 30.0
        self.dm.yaw_deg, self.dm.pitch_deg = 0.0, 20.0
        self.dm.moving = False
        self.dm._destination = {i: self.dm._cal[i].motor_from_muzzle(math.radians(angle))
                                for i, angle in ((3, 0.0), (1, 20.0))}
        self.dm._states = {
            i: SimpleNamespace(err=1, feedback_at=100.0, velocity_rad_s=0.0, position_rad=value)
            for i, value in self.dm._destination.items()
        }
        patcher = patch.object(manual.time, 'monotonic', return_value=100.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_absolute_target_uses_calibration_and_locks_until_arrival(self):
        self.assertTrue(self.dm.set_absolute_angles(12.5, 22.25))
        expected = {i: self.dm._cal[i].motor_from_muzzle(math.radians(angle))
                    for i, angle in ((3, 12.5), (1, 22.25))}
        self.assertEqual(self.dm._pending, expected)
        self.assertFalse(self.dm.aim_status()['ready'])
        self.assertFalse(self.dm.set_absolute_angles(15, 23))
        self.assertEqual(self.dm._pending, expected)
        self.dm._destination = self.dm._pending
        self.dm._pending = None
        self.dm.moving = False
        self.assertFalse(self.dm.aim_status()['ready'])  # Target accepted, feedback still at old pose.
        for i in (3, 1):
            self.dm._states[i].position_rad = expected[i]
        status = self.dm.aim_status()
        self.assertTrue(status['ready'])
        self.assertAlmostEqual(status['actual_yaw'], 12.5)
        self.assertAlmostEqual(status['actual_pitch'], 22.25)

    def test_rejects_clipped_or_nonfinite_targets_without_moving(self):
        for yaw, pitch in ((31, 20), (0, 35), (float('nan'), 20), (0, float('inf'))):
            with self.subTest(yaw=yaw, pitch=pitch), self.assertRaises(ValueError):
                self.dm.set_absolute_angles(yaw, pitch)
            self.assertIsNone(self.dm._pending)
            self.assertFalse(self.dm.moving)

    def test_motor_limits_cannot_silently_replace_numeric_target(self):
        self.dm._limits[3] = (0.2, 0.4)
        with self.assertRaises(ValueError):
            self.dm.set_absolute_angles(20, 20)
        self.assertIsNone(self.dm._pending)

    def test_loaded_activity_range_accepts_endpoints_and_enforces_triangle(self):
        for yaw, pitch, accepted in ((0, 0, True), (0, 35, True),
                                     (34.5, 30, True), (10, 33, True),
                                     (10, 35, False), (25, 31, False)):
            with self.subTest(yaw=yaw, pitch=pitch), patch('builtins.print'):
                dm = manual.DMStartupController()
                dm._load_recorded_angle_limits()
                self.assertAlmostEqual(dm.pitch_min_deg, 0.0)
                self.assertAlmostEqual(dm.pitch_max_deg, 35.0)
                dm._states = {
                    i: SimpleNamespace(err=1, feedback_at=100.0, velocity_rad_s=0.0,
                                       position_rad=dm._cal[i].motor_from_muzzle(math.radians(angle)))
                    for i, angle in ((3, 0.0), (1, 20.0))
                }
                dm.moving = False
                if accepted:
                    self.assertTrue(dm.set_absolute_angles(yaw, pitch))
                    self.assertAlmostEqual(math.degrees(dm._cal[1].muzzle_from_motor(dm._pending[1])), pitch)
                else:
                    with self.assertRaises(ValueError):
                        dm.set_absolute_angles(yaw, pitch)
                    self.assertIsNone(dm._pending)

    def test_stale_faulted_or_fast_feedback_blocks_aim_or_ready(self):
        for field, value in (('feedback_at', 98), ('feedback_at', 101),
                             ('feedback_at', None), ('err', 8),
                             ('position_rad', float('nan')), ('velocity_rad_s', float('nan'))):
            state = self.dm._states[3]
            old = getattr(state, field)
            with self.subTest(field=field, value=value):
                setattr(state, field, value)
                self.assertFalse(self.dm.aim_status()['ready'])
                self.assertFalse(self.dm.set_absolute_angles(10, 20))
            setattr(state, field, old)
        self.dm._states[3].velocity_rad_s = 0.1
        self.assertFalse(self.dm.aim_status()['ready'])


def ready_status():
    return dict(yaw=10.0, pitch=20.0, actual_yaw=10.0, actual_pitch=20.0,
                ready=True, yaw_limits=(-30.0, 30.0), pitch_limits=(10.0, 30.0))


class PrecisionSessionTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock(shot_error=None, busy=False, current_slot=1, shot_stage=0)
        self.controller.dm_controller.aim_status.return_value = ready_status()
        self.session = precision.PrecisionSession(self.controller)
        patcher = patch('builtins.print')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_commands_aim_spin_fire_and_stop_without_uart_packets(self):
        self.session.execute('aim 10 20')
        self.session.execute('speed 50')
        self.session.execute('fire')
        self.session.execute('stop')
        self.controller.dm_controller.set_absolute_angles.assert_called_once_with(10.0, 20.0)
        self.controller.vesc_controller.set_speed.assert_has_calls([call(50), call(0)])
        self.controller.request_shot.assert_called_once_with(wait=True)
        self.controller.handle_packet.assert_not_called()
        self.controller.uart.send_packet.assert_not_called()

    def test_fire_requires_explicit_aim_speed_and_current_arrival(self):
        with self.assertRaises(ValueError):
            self.session.fire()
        self.session.aim(10, 20)
        with self.assertRaises(ValueError):
            self.session.fire()
        self.session.set_speed(50)
        self.controller.dm_controller.aim_status.return_value['ready'] = False
        with self.assertRaises(ValueError):
            self.session.fire()
        self.controller.request_shot.assert_not_called()

    def test_invalid_speed_and_commands_do_not_command_hardware(self):
        for line in ('speed -1', 'speed 101', 'speed 50.5', 'speed nan', 'fire 3', 'aim 10'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.session.execute(line)
        self.controller.vesc_controller.set_speed.assert_not_called()
        self.controller.request_shot.assert_not_called()

    def test_waits_for_feedback_and_aborts_if_health_fails(self):
        self.controller.dm_controller.aim_status.return_value['ready'] = False
        self.controller.check_health.side_effect = [None, None, RuntimeError('DM disconnected')]
        with patch.object(precision.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'disconnected'):
            self.session.aim(10, 20)
        self.assertFalse(self.session.aimed)
        self.controller.request_shot.assert_not_called()

    def test_shot_failure_is_fatal_and_blocks_subsequent_fire(self):
        self.session.aimed = True
        self.session.speed = 50
        self.controller.shot_error = 'home failed'
        with self.assertRaisesRegex(RuntimeError, 'home failed'):
            self.session.fire()
        self.controller.request_shot.assert_not_called()


class SharedShotSequenceTests(unittest.TestCase):
    def make_controller(self):
        controller = manual.ManualController.__new__(manual.ManualController)
        controller.lock = threading.Lock()
        controller.busy = False
        controller.current_mode = 'M'
        controller.current_slot = 1
        controller.shot_stage = 0
        controller.shot_error = None
        controller.reload_pending = False
        controller._shoot_home_encoder = None
        controller.shoot_servo = Mock()
        controller.ammo_servo = Mock()
        controller.check_health = Mock()
        controller._go_home = Mock(return_value=123)
        controller._move_relative = Mock()
        controller._rotate_ammo_one_slot = Mock()
        return controller

    @patch('builtins.print')
    @patch.object(manual.time, 'sleep')
    def test_three_separate_fire_commands_keep_existing_feed_home_reload_order(self, *_):
        controller = self.make_controller()
        with patch.object(manual.threading, 'Thread') as thread:
            self.assertTrue(controller.request_shot(wait=True))
            self.assertEqual(controller.shot_stage, 1)
            controller._rotate_ammo_one_slot.assert_not_called()
            self.assertTrue(controller.request_shot(wait=True))
            self.assertEqual(controller.shot_stage, 2)
            controller._rotate_ammo_one_slot.assert_not_called()
            self.assertTrue(controller.request_shot(wait=True))
            thread.assert_not_called()
        controller._move_relative.assert_has_calls([
            call(controller.shoot_servo, manual.TARGET_POSITIONS[0],
                 manual.MOVE_DIRECTION, manual.SHOOTING_SPEED_RPM),
            call(controller.shoot_servo, manual.TARGET_POSITIONS[1] - manual.TARGET_POSITIONS[0],
                 manual.MOVE_DIRECTION, manual.SHOOTING_SPEED_RPM),
        ])
        self.assertEqual(controller._go_home.call_count, 3)
        controller._rotate_ammo_one_slot.assert_called_once()
        self.assertEqual(controller.current_slot, 2)
        self.assertEqual(controller.shot_stage, 0)
        self.assertFalse(controller.busy)

    @patch('builtins.print')
    def test_home_failure_stops_both_mks_motors_and_blocks_reload(self, _):
        controller = self.make_controller()
        controller.shot_stage = 2
        controller._go_home.side_effect = RuntimeError('home failed')
        controller.request_shot(wait=True)
        self.assertIn('home failed', controller.shot_error)
        controller._rotate_ammo_one_slot.assert_not_called()
        controller.shoot_servo.emergency_stop_motor.assert_called_once()
        controller.ammo_servo.emergency_stop_motor.assert_called_once()
        self.assertFalse(controller.request_shot(wait=True))

    @patch('builtins.print')
    def test_interrupt_unwinds_synchronous_shot_without_worker_thread(self, _):
        controller = self.make_controller()
        controller._go_home.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            controller.request_shot(wait=True)
        self.assertFalse(controller.busy)
        controller._move_relative.assert_not_called()
        controller._rotate_ammo_one_slot.assert_not_called()

    @patch('builtins.print')
    def test_vesc_failure_during_push_stops_feed_and_prevents_reload(self, _):
        controller = self.make_controller()
        controller.shot_stage = 1
        controller._move_relative = manual.ManualController._move_relative.__get__(controller)
        controller.check_health.side_effect = [None, None, RuntimeError('VESC lost')]
        controller.request_shot(wait=True)
        self.assertIn('VESC lost', controller.shot_error)
        self.assertEqual(controller.shot_stage, 1)
        controller._rotate_ammo_one_slot.assert_not_called()
        controller.shoot_servo.emergency_stop_motor.assert_called_once()
        controller.ammo_servo.emergency_stop_motor.assert_called_once()

    @patch('builtins.print')
    def test_default_remote_shot_still_runs_in_background(self, _):
        controller = self.make_controller()
        with patch.object(manual.threading, 'Thread') as thread:
            self.assertTrue(controller.request_shot())
            thread.assert_called_once_with(target=controller._shot_worker, daemon=True)
            thread.return_value.start.assert_called_once()
        self.assertEqual(controller.shot_stage, 0)


class PrecisionEntryTests(unittest.TestCase):
    @patch('builtins.print')
    def test_idle_prompt_checks_health_without_waiting_for_input(self, _):
        health = Mock(side_effect=[None, RuntimeError('DM lost')])
        with patch.object(precision.select, 'select', return_value=([], [], [])), \
             self.assertRaisesRegex(RuntimeError, 'DM lost'):
            precision.read_command(health)

    @patch('builtins.print')
    def test_quit_eof_and_health_failure_all_close_controller(self, _):
        for command in ('quit', EOFError(), RuntimeError('VESC lost')):
            with self.subTest(command=command):
                controller = Mock(shot_error=None, current_slot=1, shot_stage=0)
                controller.dm_controller.aim_status.return_value = ready_status()
                with patch.object(precision, 'ManualController', return_value=controller) as factory, \
                     patch.object(precision, 'read_command', side_effect=[command]):
                    if isinstance(command, RuntimeError):
                        with self.assertRaises(SystemExit) as raised:
                            precision.main([])
                        self.assertEqual(raised.exception.code, 1)
                    else:
                        precision.main([])
                factory.assert_called_once_with(uart=None, log_status=False)
                controller.close.assert_called_once()
                controller.request_shot.assert_not_called()


if __name__ == '__main__':
    unittest.main()

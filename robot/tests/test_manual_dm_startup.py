import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import manual
from turret_control import run_manual_move as move


class ManualDMStartupTests(unittest.TestCase):
    def test_startup_returns_while_dm_holds_and_close_stops_worker(self):
        stopped = threading.Event()

        def run(argv, *, confirm, stop_event, ready_event, on_shutdown, update_targets, log_status):
            self.assertFalse(confirm)
            self.assertTrue(log_status)
            self.assertIs(on_shutdown, manual.return_dm_before_shutdown)
            self.assertTrue(callable(update_targets))
            self.assertEqual(argv[argv.index('--pitch') + 1], '20.0')
            ready_event.set()
            stop_event.wait(2)
            stopped.set()

        with patch.object(move, 'main_keep_enabled', side_effect=run):
            dm = manual.DMStartupController()
            try:
                dm.start()
                self.assertTrue(dm.thread.is_alive())
                self.assertFalse(stopped.is_set())
            finally:
                dm.close()
            self.assertTrue(stopped.is_set())
            self.assertFalse(dm.thread.is_alive())

    def test_startup_error_reaches_caller(self):
        with patch.object(move, 'main_keep_enabled', side_effect=RuntimeError('CAN failed')):
            dm = manual.DMStartupController()
            try:
                with self.assertRaisesRegex(RuntimeError, 'CAN failed'):
                    dm.start()
            finally:
                dm.close()

    def test_background_error_reaches_health_check(self):
        fail = threading.Event()

        def run(argv, **kwargs):
            kwargs['ready_event'].set()
            fail.wait(2)
            raise RuntimeError('connection lost')

        with patch.object(move, 'main_keep_enabled', side_effect=run):
            dm = manual.DMStartupController()
            try:
                dm.start()
                fail.set()
                dm.thread.join(2)
                with self.assertRaisesRegex(RuntimeError, 'connection lost'):
                    dm.check_health()
            finally:
                fail.set()
                dm.close()

    def test_manual_initializes_other_motors_and_closes_dm(self):
        order = []
        dm, vesc = Mock(), Mock()
        dm.start.side_effect = lambda: order.append('dm')
        vesc.start.side_effect = lambda: order.append('vesc')
        with patch.object(manual, 'DMStartupController', return_value=dm), \
             patch.object(manual, 'VESCDualMotorController', return_value=vesc), \
             patch.object(manual, 'validate_mks_can_interfaces'), \
             patch.object(manual.ManualController, '_initialize_can', side_effect=lambda: order.append('mks')):
            controller = manual.ManualController(Mock())
            self.assertEqual(order, ['dm', 'mks', 'vesc'])
            controller.close()
            dm.close.assert_called_once()
            vesc.close.assert_called_once()

    def test_mks_startup_failure_also_stops_dm(self):
        dm, vesc = Mock(), Mock()
        with patch.object(manual, 'DMStartupController', return_value=dm), \
             patch.object(manual, 'VESCDualMotorController', return_value=vesc), \
             patch.object(manual, 'validate_mks_can_interfaces'), \
             patch.object(manual.ManualController, '_initialize_can', side_effect=RuntimeError('MKS failed')):
            with self.assertRaisesRegex(RuntimeError, 'MKS failed'):
                manual.ManualController(Mock())
            dm.close.assert_called_once()
            vesc.close.assert_called_once()

    def test_real_motion_entry_holds_support_then_disables_on_stop(self):
        stop, ready = threading.Event(), threading.Event()
        motors = Mock(tmax=10.0, kp=manual.DM_STARTUP_TEST_MOVE_KP, kd=manual.DM_STARTUP_TEST_KD)
        motors.gains_for.side_effect = lambda i: (
            manual.DM_STARTUP_TEST_YAW_KP if i == 3 else manual.DM_STARTUP_TEST_PITCH_KP,
            manual.DM_STARTUP_TEST_KD,
        )
        clock = [0.0]

        def now():
            clock[0] += 0.01
            return clock[0]

        motors.get_state.side_effect = lambda i: SimpleNamespace(
            err=1, position_rad=-0.52 if i == 1 else 0.5,
            torque_nm=3.0, feedback_at=clock[0],
        )

        def send(targets):
            if ready.is_set() and motors.send_targets.call_count > 100:
                stop.set()

        motors.send_targets.side_effect = send
        shutdown = Mock(side_effect=lambda m: m.close.assert_not_called())
        calibration = SimpleNamespace(motor_from_muzzle=lambda r: -0.3898)
        with patch.object(move, 'DamiaoSerialBackend', return_value=motors), \
             patch.object(move, 'load_pitch_calibration', return_value=calibration), \
             patch.object(move, 'load_yaw_calibration'), \
             patch.object(move, 'load_motor_limits', return_value={}), \
             patch.object(move, 'time', SimpleNamespace(monotonic=now, sleep=lambda s: None)), \
             patch('builtins.input', side_effect=AssertionError('unexpected prompt')), \
             patch('builtins.print'):
            move.main_keep_enabled(
                manual.dm_startup_arguments(), confirm=False,
                stop_event=stop, ready_event=ready,
                on_shutdown=shutdown,
            )
        self.assertTrue(ready.is_set())
        self.assertTrue(stop.is_set())
        forces = [c.args[0][1][3] for c in motors.send_targets.call_args_list]
        self.assertLess(forces[0], 0.01)
        self.assertEqual(forces[-1], manual.DM_STARTUP_TEST_PITCH_TAU_FF)
        self.assertEqual(motors.kp, manual.DM_STARTUP_TEST_KP)
        motors.set_motor_gains.assert_called_once_with({
            3: (manual.DM_STARTUP_TEST_YAW_KP, manual.DM_STARTUP_TEST_KD),
            1: (manual.DM_STARTUP_TEST_PITCH_KP, manual.DM_STARTUP_TEST_KD),
        })
        motors.close.assert_called_once()
        shutdown.assert_called_once_with(motors)

    def test_return_before_shutdown_reaches_calibrated_pose_without_disabling(self):
        clock = [0.0]

        def now():
            clock[0] += 0.01
            return clock[0]

        motors = Mock()
        positions = {3: 0.5, 1: -0.52}

        def send(targets):
            positions.update({i: values[0] for i, values in targets.items()})

        motors.send_targets.side_effect = send
        motors.get_state.side_effect = lambda i: SimpleNamespace(
            err=1, position_rad=positions[i], feedback_at=clock[0],
        )
        from turret_control.pitch import load_pitch_calibration, load_yaw_calibration
        expected = {
            3: load_yaw_calibration(move.DEFAULT_STARTUP_POSE).motor_from_muzzle(0.0),
            1: load_pitch_calibration(move.DEFAULT_STARTUP_POSE).motor_from_muzzle(
                math.radians(max(manual.DM_PITCH_MIN_DEG, min(manual.DM_PITCH_MAX_DEG, 35.0)))
            ),
        }
        with patch.object(move, 'read_enabled_start', return_value=dict(positions)), \
             patch.object(manual, 'time', SimpleNamespace(monotonic=now, sleep=lambda s: None)), \
             patch('builtins.print'):
            manual.return_dm_before_shutdown(motors)
        self.assertEqual(positions, expected)
        motors.close.assert_not_called()
        motors.disable.assert_not_called()

    def test_shutdown_callback_failure_still_disables(self):
        motors = Mock(tmax=10.0, kp=20.0)
        motors.get_state.return_value = SimpleNamespace(
            err=1, feedback_at=None, position_rad=0.0, torque_nm=0.0,
        )
        clock = [0.0]

        def now():
            clock[0] += 100.0
            return clock[0]

        with patch.object(move, 'DamiaoSerialBackend', return_value=motors), \
             patch.object(move, 'read_enabled_start', return_value={3: 0.5, 1: -0.52}), \
             patch.object(move, 'hold_position_forever'), \
             patch.object(move, 'time', SimpleNamespace(monotonic=now, sleep=lambda s: None)), \
             patch('builtins.print'):
            with self.assertRaisesRegex(TimeoutError, 'return failed'):
                move.main_keep_enabled(
                    manual.dm_startup_arguments(), confirm=False,
                    on_shutdown=Mock(side_effect=TimeoutError('return failed')),
                )
        motors.close.assert_called_once()


class ManualMoveVESCTests(unittest.TestCase):
    def setUp(self):
        self.motors = Mock(tmax=10.0)
        self.motors.get_state.return_value = None
        self.vesc = Mock()
        self.clock = 0.0

        def now():
            self.clock += 100.0
            return self.clock

        patches = [
            patch.object(move, 'DamiaoSerialBackend', return_value=self.motors),
            patch.object(move, 'read_enabled_start', return_value={3: 0.5, 1: -0.52}),
            patch.object(move, 'time', SimpleNamespace(monotonic=now, sleep=lambda s: None)),
            patch.object(manual, 'VESCDualMotorController', return_value=self.vesc),
            patch('builtins.print'),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.hold = self.enter_patch(patch.object(move, 'hold_position_forever'))

    def enter_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_move(self, **kwargs):
        move.main_keep_enabled(['--yaw', '0', '--pitch', '15'], confirm=False, **kwargs)

    def test_cli_starts_vesc_and_closes_both_controllers(self):
        with patch('sys.argv', ['run_manual_move', '--yaw', '0', '--pitch', '15']), \
             patch('builtins.input', return_value=''):
            move.main()
        self.vesc.start.assert_called_once()
        self.vesc.set_speed.assert_not_called()
        self.assertEqual(self.hold.call_args.kwargs['check_health'], self.vesc.check_health)
        self.vesc.close.assert_called_once()
        self.motors.close.assert_called_once()

    def test_shared_entry_does_not_start_duplicate_vesc(self):
        self.run_move()
        self.vesc.start.assert_not_called()
        self.vesc.close.assert_not_called()

    def test_vesc_startup_failure_prevents_motion_and_closes_both(self):
        self.vesc.start.side_effect = RuntimeError('VESC startup failed')
        with self.assertRaisesRegex(RuntimeError, 'VESC startup failed'):
            self.run_move(enable_vesc=True)
        self.motors.enable.assert_not_called()
        self.motors.close.assert_called_once()
        self.vesc.close.assert_called_once()

    def test_vesc_failure_during_motion_closes_both(self):
        self.vesc.check_health.side_effect = [None, RuntimeError('VESC lost')]
        with self.assertRaisesRegex(RuntimeError, 'VESC lost'):
            self.run_move(enable_vesc=True)
        self.hold.assert_not_called()
        self.motors.close.assert_called_once()
        self.vesc.close.assert_called_once()

    def test_vesc_failure_during_hold_closes_both(self):
        self.vesc.check_health.side_effect = [None, None, RuntimeError('VESC lost')]
        self.hold.side_effect = lambda *args, **kwargs: kwargs['check_health']()
        with self.assertRaisesRegex(RuntimeError, 'VESC lost'):
            self.run_move(enable_vesc=True)
        self.motors.close.assert_called_once()
        self.vesc.close.assert_called_once()

    def test_keyboard_interrupt_closes_both(self):
        self.hold.side_effect = KeyboardInterrupt
        self.run_move(enable_vesc=True)
        self.motors.close.assert_called_once()
        self.vesc.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

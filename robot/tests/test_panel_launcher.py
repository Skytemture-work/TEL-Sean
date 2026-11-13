import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import manual
from turret_control.aiming import AimCommand
from turret_control.board import BoardTarget, EngagementPlan
from turret_control.board_pose import update_plan_geometry
from turret_control.config import TurretConfig
from turret_control.engagement import BoardEngagement
from turret_control.panel_launcher import PanelLauncher, SharedAimMonitor
from turret_control import run_board_real as real


class ReadyTurret:
    def __init__(self):
        self.targets = []
        self.ready = True

    def update(self, target):
        self.targets.append(target)
        return AimCommand(has_target=target is not None, stable=self.ready,
                          target_yaw_rad=0.0 if target is not None else None,
                          target_pitch_rad=0.0 if target is not None else None,
                          target_reachable=True,
                          fire_allowed=self.ready and target is not None)


class PanelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = lambda: self.now
        self.controller = Mock(busy=False, shot_error=None, current_slot=1, shot_stage=0)

        def start_shot():
            self.controller.busy = True
            return True

        self.controller.request_shot.side_effect = start_shot
        self.factory = Mock(return_value=self.controller)
        self.motors = Mock()
        self.launcher = PanelLauncher(self.motors, controller_factory=self.factory, clock=self.clock)
        self.plan = EngagementPlan([
            BoardTarget(640, 360, 40, 40, hole_id=f'H{i:02d}') for i in range(1, 5)
        ])
        self.turret = ReadyTurret()
        self.driver = BoardEngagement(
            TurretConfig(), self.turret, plan=self.plan,
            fire=self.launcher.fire, fire_ready=self.launcher.fire_ready,
            target_timeout_s=None, clock=self.clock,
        )
        self.action = None
        self.dashboard = Mock()
        self.dashboard.take_panel_fire.side_effect = self.take_action

    def take_action(self):
        action, self.action = self.action, None
        return action

    def click(self):
        self.action = (self.driver.plan_revision, self.plan.index, self.plan.current().hole_id, self.now)

    def tick(self, pose_ready=True):
        return real._update_panel_engagement(
            self.driver, self.launcher, self.dashboard, self.driver.plan,
            pose_ready, 'pose invalid' if not pose_ready else None,
        )

    def complete_stage(self):
        self.controller.busy = False
        self.controller.shot_stage = (self.controller.shot_stage + 1) % 3
        if self.controller.shot_stage == 0:
            self.controller.current_slot += 1
        self.now += 1.1

    def test_startup_spins_at_speed_100_without_opening_second_aiming_worker(self):
        self.controller.vesc_controller.set_speed.assert_called_once_with(100)
        self.factory.assert_called_once()
        kwargs = self.factory.call_args.kwargs
        self.assertIsNone(kwargs['uart'])
        self.assertFalse(kwargs['log_status'])
        self.assertIsInstance(kwargs['dm_controller'], SharedAimMonitor)
        kwargs['dm_controller'].close()
        self.motors.close.assert_not_called()
        self.controller.request_shot.assert_not_called()

    def test_no_click_never_fires_or_consumes_cooldown_or_advances(self):
        for _ in range(20):
            self.assertFalse(self.tick().fired)
            self.now += 10
        self.controller.request_shot.assert_not_called()
        self.assertTrue(self.driver.fire_cooldown_ready)
        self.assertEqual(self.plan.index, 0)

    def test_one_click_holds_while_busy_and_advances_once_only_after_completion(self):
        self.click()
        self.assertFalse(self.tick().fired)
        self.assertTrue(self.launcher.pending)
        self.assertEqual(self.plan.index, 0)
        self.click()  # A second queued click cannot start another stage.
        self.assertFalse(self.tick().fired)
        self.assertIsNone(self.turret.targets[-1])
        self.complete_stage()
        self.assertTrue(self.tick().fired)
        self.assertEqual(self.plan.index, 1)
        self.assertFalse(self.launcher.pending)
        for _ in range(3):
            self.now += 2
            self.assertFalse(self.tick().fired)
        self.controller.request_shot.assert_called_once_with()

    def test_three_clicks_use_three_existing_stages_then_next_slot(self):
        for stage in range(3):
            self.now += 1.1
            self.click()
            self.tick()
            self.complete_stage()
            self.assertTrue(self.tick().fired)
            self.assertEqual(self.plan.index, stage + 1)
        self.assertEqual(self.controller.request_shot.call_count, 3)
        snapshot = self.launcher.snapshot()
        self.assertEqual(snapshot['slot'], 2)
        self.assertEqual(snapshot['next_stage'], 1)
        self.assertEqual(snapshot['speed'], 100)

    def test_live_pose_updates_preserve_pending_shot_identity(self):
        self.click()
        self.tick()
        target = self.plan.current()
        update_plan_geometry(self.plan, {'H01': replace(target, u_px=target.u_px + 1)})
        self.assertIsNot(self.plan.current(), target)
        self.tick()
        self.complete_stage()
        update_plan_geometry(self.plan, {'H01': replace(self.plan.current(), v_px=361)})
        self.assertTrue(self.tick().fired)
        self.assertEqual(self.plan.index, 1)
        self.controller.request_shot.assert_called_once()

    def test_unstable_pose_or_unsettled_turret_discards_click_instead_of_firing_later(self):
        for pose_ready in (False, True):
            self.turret.ready = False
            self.click()
            self.tick(pose_ready)
            self.turret.ready = True
            self.now += 2
            self.tick()
        self.controller.request_shot.assert_not_called()
        self.assertEqual(self.plan.index, 0)

    def test_expired_or_wrong_target_request_never_arms(self):
        for action in ((0, 0, 'H01', -2.0), (0, 0, 'H02', 0.0), (0, 1, 'H01', 0.0)):
            self.action = action
            self.tick()
        self.controller.request_shot.assert_not_called()

    def test_old_click_cannot_apply_to_new_plan_with_same_first_hole(self):
        self.click()
        self.driver.set_plan(EngagementPlan([BoardTarget(100, 100, 20, 20, hole_id='H01')]))
        self.tick()
        self.controller.request_shot.assert_not_called()

    def test_loss_of_pose_during_active_stage_is_fatal(self):
        self.click()
        self.tick()
        with self.assertRaisesRegex(RuntimeError, '姿態失效'):
            self.tick(False)
        self.assertEqual(self.controller.request_shot.call_count, 1)

    def test_launcher_or_aiming_failure_cannot_advance_plan(self):
        self.click()
        self.tick()
        self.controller.shot_error = 'Encoder timeout'
        self.controller.busy = False
        with self.assertRaisesRegex(RuntimeError, 'Encoder timeout'):
            self.tick()
        self.assertEqual(self.plan.index, 0)
        self.assertFalse(self.plan.targets[0].engaged)

    def test_plan_change_while_stage_is_pending_is_fatal_without_second_command(self):
        self.click()
        self.tick()
        self.complete_stage()
        self.driver.set_plan(EngagementPlan([BoardTarget(100, 100, 20, 20, hole_id='H12')]))
        with self.assertRaisesRegex(RuntimeError, '孔位計畫已改變'):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)


class SharedLauncherLifecycleTests(unittest.TestCase):
    def test_aim_monitor_rejects_missing_feedback_without_issuing_commands(self):
        motors = Mock()
        motors.get_state.side_effect = lambda i: None if i == 1 else object()
        monitor = SharedAimMonitor(motors)
        with self.assertRaisesRegex(RuntimeError, '回授無效'):
            monitor.start(timeout_s=0)
        motors.enable.assert_not_called()
        motors.send_targets.assert_not_called()

    @patch('builtins.print')
    def test_close_cancels_worker_before_stopping_can_receivers(self, _):
        dm, vesc = Mock(), Mock()
        with patch.object(manual, 'VESCDualMotorController', return_value=vesc), \
             patch.object(manual, 'validate_mks_can_interfaces'), \
             patch.object(manual.ManualController, '_initialize_can'):
            controller = manual.ManualController(None, dm_controller=dm)
        started = threading.Event()

        def long_stage():
            started.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                controller.check_health()
                time.sleep(0.005)

        controller._run_one_shot_cycle = long_stage
        notifier, bus = Mock(), Mock()
        controller.ammo_notifier = controller.shoot_notifier = notifier
        controller.ammo_bus = controller.shoot_bus = bus
        notifier.stop.side_effect = lambda: self.assertFalse(controller._shot_thread.is_alive())
        try:
            self.assertTrue(controller.request_shot())
            self.assertTrue(started.wait(1))
        finally:
            controller.close()
        self.assertFalse(controller._shot_thread.is_alive())
        self.assertFalse(controller.request_shot())
        self.assertIn('關閉', controller.shot_error)
        notifier.stop.assert_called_once()
        bus.shutdown.assert_called_once()
        vesc.close.assert_called_once()


class MagazineRotationOverlapTests(unittest.TestCase):
    """Shot 3: once ID1 is home the turret may aim at the next hole while ID2
    rotates, but no new shot can start until the rotation is finished."""

    def setUp(self):
        self.now = 0.0
        self.clock = lambda: self.now
        self.controller = Mock(busy=False, shot_error=None, current_slot=1, shot_stage=2,
                               shot_request_seq=0, shot_released_seq=0)

        def start_shot():
            self.controller.busy = True
            self.controller.shot_request_seq += 1
            return True

        self.controller.request_shot.side_effect = start_shot
        self.launcher = PanelLauncher(Mock(), controller_factory=Mock(return_value=self.controller),
                                      clock=self.clock)
        self.plan = EngagementPlan([BoardTarget(640, 360, 40, 40, hole_id=f'H{i:02d}')
                                    for i in range(1, 5)])
        self.turret = ReadyTurret()
        self.driver = BoardEngagement(TurretConfig(), self.turret, plan=self.plan,
                                      fire=self.launcher.fire, fire_ready=self.launcher.fire_ready,
                                      target_timeout_s=None, clock=self.clock)
        self.action = None
        self.dashboard = Mock()
        self.dashboard.take_panel_fire.side_effect = self._take

    def _take(self):
        action, self.action = self.action, None
        return action

    def click(self):
        self.action = (self.driver.plan_revision, self.plan.index, self.plan.current().hole_id, self.now)

    def tick(self):
        return real._update_panel_engagement(self.driver, self.launcher, self.dashboard,
                                             self.driver.plan, True, None)

    def test_turret_moves_on_while_magazine_rotates_but_cannot_fire(self):
        self.click()
        self.assertFalse(self.tick().fired)
        self.assertTrue(self.launcher.busy)          # muzzle holds while ID1 pushes
        self.assertFalse(self.tick().fired)
        self.assertIsNone(self.turret.targets[-1])
        # Third ball out and ID1 home; ID2 rotation still running (busy=True).
        self.controller.shot_released_seq = 1
        self.now += 1.1
        self.assertTrue(self.tick().fired)
        self.assertEqual(self.plan.index, 1)
        self.assertFalse(self.launcher.pending)
        self.assertFalse(self.launcher.busy)
        self.assertTrue(self.launcher.mechanism_busy)
        self.assertTrue(self.launcher.snapshot()['magazine_rotating'])
        # Next tick aims at H02 already (not held).
        self.now += 1.1
        self.tick()
        self.assertEqual(self.turret.targets[-1].u_px, 640)
        self.assertIsNotNone(self.turret.targets[-1])
        # A click during rotation is refused and never queued.
        self.click()
        self.assertFalse(self.tick().fired)
        self.assertIn('轉格中', self.launcher.message)
        self.assertEqual(self.controller.request_shot.call_count, 1)
        self.now += 5
        self.assertFalse(self.tick().fired)
        self.assertEqual(self.controller.request_shot.call_count, 1)
        # Rotation done: a fresh click starts the next stage.
        self.controller.busy = False
        self.click()
        self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 2)
        self.assertTrue(self.launcher.pending)

    def test_without_release_signal_still_waits_for_full_completion(self):
        self.click()
        self.tick()
        self.now += 1.1
        self.assertFalse(self.tick().fired)          # busy, not released
        self.assertEqual(self.plan.index, 0)

    @patch('builtins.print')
    def test_manual_worker_releases_before_rotating_magazine(self, _):
        with patch.object(manual, 'VESCDualMotorController', return_value=Mock()), \
             patch.object(manual, 'validate_mks_can_interfaces'), \
             patch.object(manual.ManualController, '_initialize_can'):
            controller = manual.ManualController(None, dm_controller=Mock())
        controller.shot_stage = 2
        seen = {}
        controller._run_one_shot_cycle = lambda: None

        def rotate():
            seen['released'] = controller.shot_released_seq
            seen['request'] = controller.shot_request_seq
            seen['busy'] = controller.busy

        controller._rotate_ammo_one_slot = rotate
        try:
            self.assertTrue(controller.request_shot())
            controller._shot_thread.join(2)
        finally:
            controller.close()
        self.assertEqual(seen, {'released': 1, 'request': 1, 'busy': True})
        self.assertEqual(controller.current_slot, 2)
        self.assertEqual(controller.shot_stage, 0)


class AutoFireTests(unittest.TestCase):
    """--auto-fire: the trigger only replaces the click; every gate still applies."""

    def setUp(self):
        from turret_control.panel_launcher import AutoFireTrigger
        self.now = 0.0
        self.clock = lambda: self.now
        self.controller = Mock(busy=False, shot_error=None, current_slot=1, shot_stage=0,
                               shot_request_seq=0, shot_released_seq=0)

        def start_shot():
            self.controller.busy = True
            self.controller.shot_request_seq += 1
            return True

        self.controller.request_shot.side_effect = start_shot
        self.launcher = PanelLauncher(Mock(), controller_factory=Mock(return_value=self.controller),
                                      clock=self.clock)
        self.plan = EngagementPlan([BoardTarget(640, 360, 40, 40, hole_id=f'H{i:02d}')
                                    for i in range(1, 5)])
        self.turret = ReadyTurret()
        self.driver = BoardEngagement(TurretConfig(), self.turret, plan=self.plan,
                                      fire=self.launcher.fire, fire_ready=self.launcher.fire_ready,
                                      target_timeout_s=None, clock=self.clock)
        self.auto = AutoFireTrigger(0.5, clock=self.clock)
        self.dashboard = Mock()
        self.dashboard.take_panel_fire.return_value = None
        self.previous = None

    def tick(self, pose_ready=True, dt=0.1):
        self.now += dt
        status = real._update_panel_engagement(
            self.driver, self.launcher, self.dashboard, self.driver.plan, pose_ready,
            None if pose_ready else 'pose invalid', auto_fire=self.auto,
            previous_status=self.previous)
        self.previous = status
        return status

    def finish_stage(self):
        self.controller.shot_released_seq = self.controller.shot_request_seq
        self.controller.busy = False

    def test_fires_without_click_only_after_settle_and_once_per_hole(self):
        for _ in range(5):          # 0.1 .. 0.5 s: still settling
            self.tick()
        self.controller.request_shot.assert_not_called()
        for _ in range(3):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)
        self.assertTrue(self.launcher.pending)
        for _ in range(10):         # stage running: no second request
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)
        self.finish_stage()
        fired = [self.tick().fired for _ in range(3)]
        self.assertIn(True, fired)
        self.assertEqual(self.plan.index, 1)
        # Next hole: a fresh settle period is required before the next request.
        self.now += 1.1
        for _ in range(3):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)
        for _ in range(6):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 2)

    def test_never_fires_while_aim_not_ready(self):
        self.turret.ready = False
        for _ in range(50):
            self.tick()
        self.controller.request_shot.assert_not_called()

    def test_losing_ready_restarts_settle(self):
        for _ in range(4):
            self.tick()
        self.turret.ready = False
        self.tick()
        self.turret.ready = True
        for _ in range(5):
            self.tick()
        self.controller.request_shot.assert_not_called()
        for _ in range(3):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)

    def test_never_fires_without_pose(self):
        for _ in range(50):
            self.tick(pose_ready=False)
        self.controller.request_shot.assert_not_called()

    def test_waits_for_magazine_rotation(self):
        self.controller.busy = True   # e.g. ID2 still rotating from the previous slot
        for _ in range(30):
            self.tick()
        self.controller.request_shot.assert_not_called()
        self.controller.busy = False
        for _ in range(8):
            self.tick()
        self.assertEqual(self.controller.request_shot.call_count, 1)

    @patch('builtins.print')
    def test_cli_auto_fire_validation_before_hardware(self, _):
        for args in (['--auto-fire', '--aim-only'], ['--auto-fire', '--panel-fire'],
                     ['--auto-fire', '--web-port', '0'], ['--auto-fire', '--auto-fire-settle', '0.1'],
                     ['--auto-fire', '--auto-fire-settle', 'nan'],
                     ['--match-fire', '--no-score-mode'], ['--match-fire', '--auto-fire'],
                     ['--match-fire', '--balls-loaded', '19'], ['--match-fire', '--balls-loaded', '0'],
                     ['--match-fire', '--match-duration', '10'],
                     ['--match-fire', '--small-fire-tolerance-deg', '0.01'],
                     ['--match-fire', '--fire-tolerance-deg', '5']):
            with self.subTest(args=args), patch.object(real, 'SharedMotorBackend') as factory, \
                 patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
                real.main(args)
            self.assertEqual(raised.exception.code, 2)
            factory.assert_not_called()


class PusherSpeedTests(unittest.TestCase):
    @patch('builtins.print')
    def _manual(self, _):
        with patch.object(manual, 'VESCDualMotorController', return_value=Mock()), \
             patch.object(manual, 'validate_mks_can_interfaces'), \
             patch.object(manual.ManualController, '_initialize_can'):
            return manual.ManualController(None, dm_controller=Mock())

    def test_defaults_keep_manual_constants(self):
        controller = self._manual()
        self.assertEqual(controller.shooting_speed_rpm, manual.SHOOTING_SPEED_RPM)
        self.assertEqual(controller.shooting_acceleration, manual.SHOOTING_ACCELERATION)
        self.assertEqual(controller.home_speed_rpm, manual.HOME_SPEED_RPM)
        # Faster pusher defaults shared by manual and auto modes; magazine unchanged.
        self.assertEqual((manual.SHOOTING_SPEED_RPM, manual.SHOOTING_ACCELERATION,
                          manual.HOME_SPEED_RPM, manual.ACCELERATION), (200, 200, 60, 2))
        launcher = PanelLauncher(Mock(), controller_factory=Mock(return_value=controller))
        self.assertEqual(controller.shooting_speed_rpm, manual.SHOOTING_SPEED_RPM)
        launcher.close()

    def test_launcher_overrides_only_the_pusher(self):
        controller = self._manual()
        launcher = PanelLauncher(Mock(), controller_factory=Mock(return_value=controller),
                                 push_rpm=250, push_acceleration=40, push_home_rpm=80)
        self.assertEqual((controller.shooting_speed_rpm, controller.shooting_acceleration,
                          controller.home_speed_rpm), (250, 40, 80))
        shoot, ammo = Mock(), Mock()
        controller.shoot_servo, controller.ammo_servo = shoot, ammo
        for servo in (shoot, ammo):
            servo.read_encoder_value_addition.side_effect = [0, 5, 5, 5, 5, 5, 5, 5]
            servo.is_motor_running.return_value = False
            servo.run_motor_relative_motion_by_pulses.return_value = "ok"
        with patch('builtins.print'):
            controller._move_relative(shoot, 100, manual.MOVE_DIRECTION, controller.shooting_speed_rpm)
            controller._move_relative(ammo, 100, manual.MOVE_DIRECTION, 10)
        self.assertEqual(shoot.run_motor_relative_motion_by_pulses.call_args.args[1:3], (250, 40))
        # Magazine moves keep the original acceleration constant.
        self.assertEqual(ammo.run_motor_relative_motion_by_pulses.call_args.args[2], manual.ACCELERATION)
        launcher.close()

    @patch('builtins.print')
    def test_cli_rejects_out_of_range_or_wrong_mode(self, _):
        for args in (['--auto-fire', '--push-rpm', '1000'], ['--auto-fire', '--push-accel', '0'],
                     ['--auto-fire', '--push-home-rpm', '300'], ['--aim-only', '--push-rpm', '200'],
                     ['--auto-fire', '--aim-speed', '2.0'], ['--auto-fire', '--aim-acceleration', '5']):
            with self.subTest(args=args), patch.object(real, 'SharedMotorBackend') as factory, \
                 patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
                real.main(args)
            self.assertEqual(raised.exception.code, 2)
            factory.assert_not_called()


class SharedMotionProfileTests(unittest.TestCase):
    def test_manual_and_auto_use_the_same_dm_limits(self):
        from turret_control import motion_profile, shared_motor_backend
        self.assertEqual(manual.DM_STARTUP_TEST_SPEED_RAD_S, motion_profile.DM_SPEED_RAD_S)
        self.assertEqual(shared_motor_backend.DEFAULT_AIM_SPEED_RAD_S, motion_profile.DM_SPEED_RAD_S)
        self.assertEqual(shared_motor_backend.DEFAULT_AIM_ACCELERATION_RAD_S2,
                         motion_profile.DM_ACCELERATION_RAD_S2)

    def test_smoothstep_duration_respects_speed_and_acceleration(self):
        from turret_control.motion_profile import smoothstep_duration
        for distance in (0.01, 0.15, 0.7, 1.5):
            duration = smoothstep_duration(distance, 0.6, 2.5)
            self.assertLessEqual(1.5 * distance / duration, 0.6 + 1e-9)
            self.assertLessEqual(6.0 * distance / duration ** 2, 2.5 + 1e-9)
        self.assertEqual(smoothstep_duration(0.0), 0.25)


class PanelEntryTests(unittest.TestCase):
    @patch('builtins.print')
    def test_mode_validation_happens_before_any_hardware_startup(self, _):
        for args in (['--panel-fire', '--aim-only'], ['--panel-fire', '--web-port', '0'],
                     ['--panel-fire', '--virtual-board']):
            with self.subTest(args=args), patch.object(real, 'SharedMotorBackend') as factory, \
                 patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
                real.main(args)
            self.assertEqual(raised.exception.code, 2)
            factory.assert_not_called()

    @patch('builtins.print')
    def test_camera_end_or_launcher_startup_failure_closes_all_owned_resources(self, _):
        for startup_error in (False, True):
            with self.subTest(startup_error=startup_error):
                camera, motors, launcher, server = Mock(), Mock(), Mock(), Mock()
                camera.read.return_value = (False, None)
                launcher.target_rpm = 6000
                with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=Mock())}), \
                     patch.object(real.cv2, 'VideoCapture', return_value=camera), \
                     patch.object(real, 'SharedMotorBackend', return_value=motors), \
                     patch.object(real, 'BoardVision'), \
                     patch.object(real, 'start_dashboard_server', return_value=server), \
                     patch('turret_control.panel_launcher.PanelLauncher', return_value=launcher,
                           side_effect=RuntimeError('VESC init failed') if startup_error else None) as factory:
                    args = ['--panel-fire', '--no-score-mode', '--headless', '--opencv-camera']
                    if startup_error:
                        with self.assertRaisesRegex(RuntimeError, 'VESC init failed'):
                            real.main(args)
                    else:
                        real.main(args)
                        launcher.close.assert_called_once()
                    factory.assert_called_once_with(motors, push_rpm=None, push_acceleration=None,
                                                push_home_rpm=None)
                motors.enable.assert_called_once()
                motors.close.assert_called_once()
                camera.release.assert_called_once()
                server.shutdown.assert_called_once()
                server.server_close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

import math
import time
import unittest

from turret_control import ballistics, board, control_loop, engagement
from turret_control.backends import MockMotorBackend
from turret_control.controller import TurretController
from turret_control.aiming import AimingController, TargetPixel
from turret_control.config import TurretConfig


class BaseFixedCameraTests(unittest.TestCase):
    """This robot: the camera is bolted to the base and never moves."""

    def setUp(self):
        self.config = TurretConfig(
            stable_frames_required=3,
            aim_tolerance_rad=math.radians(2.0),
            camera_on_turret=False,
            # This class covers the aiming/stability gate; the ballistic
            # corrections have their own tests below.
            compensate_gravity=False,
            compensate_parallax=False,
        )
        self.controller = AimingController(self.config)

    def test_image_center_has_zero_error(self):
        command = self.controller.update(
            TargetPixel(self.config.camera_center_x_px, self.config.camera_center_y_px)
        )
        self.assertAlmostEqual(command.yaw_error_rad, 0.0)
        self.assertAlmostEqual(command.pitch_error_rad, 0.0)
        self.assertFalse(command.fire_allowed)

    def test_target_right_and_above_produces_positive_errors(self):
        command = self.controller.update(TargetPixel(800.0, 250.0))
        self.assertGreater(command.yaw_error_rad, 0.0)
        self.assertGreater(command.pitch_error_rad, 0.0)

    def test_no_target_never_allows_firing(self):
        command = self.controller.update(None)
        self.assertFalse(command.has_target)
        self.assertFalse(command.fire_allowed)
        self.assertIsNone(command.target_yaw_rad)
        self.assertIsNone(command.target_pitch_rad)

    def test_angle_is_limited(self):
        command = self.controller.update(TargetPixel(100000.0, -100000.0))
        self.assertEqual(command.target_yaw_rad, self.config.yaw_max_rad)
        self.assertEqual(command.target_pitch_rad, self.config.pitch_max_rad)

    def test_without_feedback_never_fires(self):
        """The regression: a base-fixed camera cannot confirm arrival alone.

        The pixel error never shrinks as the turret moves, so judging it would
        either never fire or fire while still slewing.  Fail closed instead.
        """
        target = TargetPixel(800.0, 250.0)
        for _ in range(20):
            command = self.controller.update(target)
            self.assertFalse(command.fire_allowed)
            self.assertIsNone(command.yaw_servo_error_rad)

    def test_off_centre_target_fires_once_motors_arrive(self):
        """The case the old code could never reach."""
        target = TargetPixel(800.0, 250.0)
        first = self.controller.update(target, 0.0, 0.0)
        self.assertFalse(first.fire_allowed)
        self.assertTrue(first.target_reachable)
        self.assertGreater(abs(first.yaw_servo_error_rad), math.radians(2.0))

        # Motors reach the commanded angle and stay there.
        yaw, pitch = first.target_yaw_rad, first.target_pitch_rad
        self.assertFalse(self.controller.update(target, yaw, pitch).fire_allowed)
        self.assertFalse(self.controller.update(target, yaw, pitch).fire_allowed)
        self.assertTrue(self.controller.update(target, yaw, pitch).fire_allowed)

    def test_still_slewing_does_not_fire(self):
        target = TargetPixel(800.0, 250.0)
        aimed = self.controller.update(target, 0.0, 0.0)
        yaw, pitch = aimed.target_yaw_rad, aimed.target_pitch_rad
        for _ in range(10):
            # Pitch has arrived but yaw is still 10 degrees away.
            command = self.controller.update(target, yaw - math.radians(10.0), pitch)
            self.assertFalse(command.fire_allowed)

    def test_arriving_then_drifting_resets_the_counter(self):
        target = TargetPixel(800.0, 250.0)
        aimed = self.controller.update(target, 0.0, 0.0)
        yaw, pitch = aimed.target_yaw_rad, aimed.target_pitch_rad
        self.controller.update(target, yaw, pitch)
        self.controller.update(target, yaw, pitch)
        # Knocked off target before the streak completed.
        self.assertFalse(
            self.controller.update(target, yaw - math.radians(30.0), pitch).fire_allowed
        )
        self.assertFalse(self.controller.update(target, yaw, pitch).fire_allowed)

    def test_unreachable_target_never_fires(self):
        """Clamped to the end stop means the turret is not actually aimed."""
        target = TargetPixel(100000.0, -100000.0)
        for _ in range(10):
            command = self.controller.update(
                target, self.config.yaw_max_rad, self.config.pitch_max_rad
            )
            self.assertFalse(command.target_reachable)
            self.assertFalse(command.fire_allowed)

    def test_low_confidence_is_ignored(self):
        weak = TargetPixel(800.0, 250.0, confidence=0.1)
        command = self.controller.update(weak, 0.0, 0.0)
        self.assertFalse(command.has_target)
        self.assertFalse(command.fire_allowed)


class TurretMountedCameraTests(unittest.TestCase):
    """Kept correct so switching the flag does not silently misbehave."""

    def setUp(self):
        self.config = TurretConfig(
            stable_frames_required=3,
            aim_tolerance_rad=math.radians(2.0),
            camera_on_turret=True,
            compensate_gravity=False,
            compensate_parallax=False,
        )
        self.controller = AimingController(self.config)

    def test_centred_target_fires_without_feedback(self):
        target = TargetPixel(self.config.camera_center_x_px, self.config.camera_center_y_px)
        self.assertFalse(self.controller.update(target).fire_allowed)
        self.assertFalse(self.controller.update(target).fire_allowed)
        self.assertTrue(self.controller.update(target).fire_allowed)

    def test_target_is_relative_to_current_angle(self):
        target = TargetPixel(800.0, 360.0)
        command = self.controller.update(target, current_yaw_rad=0.5, current_pitch_rad=0.0)
        self.assertAlmostEqual(command.target_yaw_rad, 0.5 + command.yaw_error_rad)

    def test_off_centre_target_does_not_fire(self):
        target = TargetPixel(800.0, 250.0)
        for _ in range(10):
            self.assertFalse(self.controller.update(target, 0.0, 0.0).fire_allowed)


class BallisticsMathTests(unittest.TestCase):
    """Pure maths, no config involved."""

    def test_gravity_needs_elevation_for_a_level_target(self):
        pitch = ballistics.gravity_pitch(5.0, 0.0, 20.0)
        self.assertGreater(pitch, 0.0)
        # Verify it actually passes through the point.
        v, g, d = 20.0, 9.81, 5.0
        height = d * math.tan(pitch) - g * d * d / (2 * v * v * math.cos(pitch) ** 2)
        self.assertAlmostEqual(height, 0.0, places=6)

    def test_flatter_of_the_two_solutions_is_returned(self):
        low = ballistics.gravity_pitch(5.0, 0.0, 20.0)
        self.assertLess(low, math.radians(45.0))

    def test_faster_ball_needs_less_elevation(self):
        slow = ballistics.gravity_pitch(5.0, 0.0, 10.0)
        fast = ballistics.gravity_pitch(5.0, 0.0, 30.0)
        self.assertGreater(slow, fast)

    def test_unreachable_target_raises(self):
        with self.assertRaises(ballistics.OutOfRange):
            ballistics.gravity_pitch(100.0, 0.0, 5.0)

    def test_zero_gravity_is_the_straight_line(self):
        pitch = ballistics.gravity_pitch(5.0, 1.0, 20.0, gravity_m_s2=0.0)
        self.assertAlmostEqual(pitch, math.atan2(1.0, 5.0))

    def test_parallax_shrinks_with_distance(self):
        near, _, _ = ballistics.apply_parallax(0.0, 0.0, 2.0, (0.10, 0.0, 0.0))
        far, _, _ = ballistics.apply_parallax(0.0, 0.0, 10.0, (0.10, 0.0, 0.0))
        self.assertGreater(abs(near), abs(far))

    def test_parallax_corrects_towards_the_muzzle_side(self):
        # Muzzle sits to the right of the camera, so it must aim further left.
        yaw, _, _ = ballistics.apply_parallax(0.0, 0.0, 2.0, (0.10, 0.0, 0.0))
        self.assertLess(yaw, 0.0)

    def test_zero_offset_changes_nothing(self):
        yaw, pitch, rng = ballistics.apply_parallax(0.3, 0.1, 4.0, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(yaw, 0.3)
        self.assertAlmostEqual(pitch, 0.1)
        self.assertAlmostEqual(rng, 4.0)

    def test_solve_aim_without_distance_is_a_passthrough(self):
        yaw, pitch = ballistics.solve_aim(
            0.2, 0.1, None, compensate_parallax=True, compensate_gravity=True
        )
        self.assertEqual((yaw, pitch), (0.2, 0.1))


class BallisticIntegrationTests(unittest.TestCase):
    """The corrections as the aiming layer applies them."""

    def _config(self, **kw):
        base = dict(stable_frames_required=1, aim_tolerance_rad=math.radians(2.0))
        base.update(kw)
        return TurretConfig(**base)

    def test_missing_range_blocks_firing(self):
        controller = AimingController(self._config(compensate_gravity=True))
        command = controller.update(TargetPixel(640.0, 360.0), 0.0, 0.0)
        self.assertFalse(command.target_reachable)
        self.assertFalse(command.fire_allowed)
        self.assertIn("no range", command.blocked_reason)

    def test_range_enables_the_drop_correction(self):
        controller = AimingController(self._config(compensate_gravity=True))
        command = controller.update(
            TargetPixel(640.0, 360.0, distance_m=5.0), 0.0, 0.0
        )
        self.assertIsNone(command.blocked_reason)
        # Level target 5 m away still needs the barrel raised.
        self.assertGreater(command.target_pitch_rad, math.radians(3.0))

    def test_opting_out_restores_the_straight_line(self):
        controller = AimingController(
            self._config(compensate_gravity=False, compensate_parallax=False)
        )
        command = controller.update(TargetPixel(640.0, 360.0), 0.0, 0.0)
        self.assertTrue(command.target_reachable)
        self.assertAlmostEqual(command.target_pitch_rad, 0.0)

    def test_require_distance_false_allows_the_assumption(self):
        controller = AimingController(
            self._config(compensate_gravity=True, require_distance_to_fire=False)
        )
        command = controller.update(TargetPixel(640.0, 360.0), 0.0, 0.0)
        self.assertIsNone(command.blocked_reason)
        self.assertAlmostEqual(command.target_pitch_rad, 0.0)

    def test_target_beyond_ballistic_range_never_fires(self):
        controller = AimingController(
            self._config(compensate_gravity=True, muzzle_speed_m_s=5.0)
        )
        for _ in range(5):
            command = controller.update(
                TargetPixel(640.0, 360.0, distance_m=100.0), 0.0, 0.0
            )
            self.assertFalse(command.fire_allowed)
            self.assertIn("cannot reach", command.blocked_reason)

    def test_parallax_shifts_the_yaw_target(self):
        controller = AimingController(
            self._config(compensate_gravity=False, muzzle_offset_x_m=0.10)
        )
        command = controller.update(
            TargetPixel(640.0, 360.0, distance_m=2.0), 0.0, 0.0
        )
        self.assertLess(command.target_yaw_rad, math.radians(-1.0))



class LatestDetectionTests(unittest.TestCase):
    def test_empty_slot_reads_none(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        self.assertIsNone(slot.get())

    def test_fresh_detection_reads_back(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        target = TargetPixel(800.0, 250.0)
        slot.publish(target)
        self.assertIs(slot.get(), target)

    def test_stale_detection_reads_as_no_target(self):
        """Aiming at where the target was a second ago is worse than holding."""
        slot = control_loop.LatestDetection(max_age_s=0.05)
        slot.publish(TargetPixel(800.0, 250.0))
        time.sleep(0.12)
        self.assertIsNone(slot.get())

    def test_publishing_none_clears(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        slot.publish(TargetPixel(800.0, 250.0))
        slot.publish(None)
        self.assertIsNone(slot.get())

    def test_rejects_non_positive_age(self):
        with self.assertRaises(ValueError):
            control_loop.LatestDetection(max_age_s=0.0)


class DetectionThreadTests(unittest.TestCase):
    def test_publishes_what_the_detector_returns(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        target = TargetPixel(800.0, 250.0)
        with control_loop.DetectionThread(lambda: target, slot):
            deadline = time.monotonic() + 2.0
            while slot.publish_count == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
        self.assertGreater(slot.publish_count, 0)

    def test_detector_crash_clears_the_target_and_is_reported(self):
        slot = control_loop.LatestDetection(max_age_s=10.0)
        slot.publish(TargetPixel(800.0, 250.0))

        def boom():
            raise RuntimeError("camera unplugged")

        thread = control_loop.DetectionThread(boom, slot).start()
        thread.stop()
        self.assertIsNone(slot.get())
        self.assertIsInstance(thread.error, RuntimeError)


class ControlLoopTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig(
            stable_frames_required=2,
            aim_tolerance_rad=math.radians(2.0),
            compensate_gravity=False,
            compensate_parallax=False,
        )
        self.motors = MockMotorBackend(
            motor_ids=[self.config.yaw_motor_id, self.config.pitch_motor_id],
            verbose=False,
        )
        self.turret = TurretController(self.config, self.motors)

    def test_every_tick_sends_motor_frames_even_with_no_target(self):
        """The whole point: silence disables the motors and drops the axis."""
        slot = control_loop.LatestDetection(max_age_s=1.0)
        loop = control_loop.ControlLoop(self.turret, slot, rate_hz=200.0)
        stats = loop.run(duration_s=0.25)

        self.assertGreater(stats.ticks, 0)
        self.assertEqual(stats.stale_ticks, stats.ticks)
        self.assertGreaterEqual(len(self.motors.commands), stats.ticks)

    def test_holds_the_last_command_when_the_target_disappears(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        slot.publish(TargetPixel(800.0, 250.0))
        loop = control_loop.ControlLoop(self.turret, slot, rate_hz=200.0)
        loop.run(duration_s=0.05)

        def last_per_motor():
            seen = {}
            for command in self.motors.commands:
                seen[command.motor_id] = command.position_rad
            return seen

        aimed = last_per_motor()
        self.assertEqual(
            sorted(aimed), sorted([self.config.yaw_motor_id, self.config.pitch_motor_id])
        )

        before = len(self.motors.commands)
        slot.publish(None)
        loop.run(duration_s=0.05)

        self.assertGreater(len(self.motors.commands), before, "hold() must keep sending")
        self.assertEqual(last_per_motor(), aimed)

    def test_keeps_the_requested_rate(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        loop = control_loop.ControlLoop(self.turret, slot, rate_hz=200.0)
        stats = loop.run(duration_s=0.5)
        self.assertAlmostEqual(stats.actual_hz, 200.0, delta=15.0)

    def test_should_stop_ends_the_loop(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        loop = control_loop.ControlLoop(self.turret, slot, rate_hz=200.0)
        counter = {"n": 0}

        def stop():
            counter["n"] += 1
            return counter["n"] > 10

        stats = loop.run(duration_s=5.0, should_stop=stop)
        self.assertLessEqual(stats.ticks, 11)

    def test_rejects_non_positive_rate(self):
        slot = control_loop.LatestDetection(max_age_s=1.0)
        with self.assertRaises(ValueError):
            control_loop.ControlLoop(self.turret, slot, rate_hz=0.0)



def _grid(distance_m=6.0, focal=700.0, columns=(400.0, 640.0, 880.0)):
    """A 3x4 board: top row small holes, three rows of large ones below."""
    small = focal * 0.20 / distance_m
    large = focal * 0.40 / distance_m
    out = []
    for u in columns:
        out.append(board.BoardTarget(u, 150.0, small, small, 0.9))
        for v in (300.0, 450.0, 600.0):
            out.append(board.BoardTarget(u, v, large, large, 0.9))
    return out


class RangingTests(unittest.TestCase):
    def test_known_diameter_recovers_the_distance(self):
        d = board.distance_from_size(46.666, 0.40, 700.0)
        self.assertAlmostEqual(d, 6.0, places=2)

    def test_closer_target_looks_bigger(self):
        near = board.distance_from_size(100.0, 0.40, 700.0)
        far = board.distance_from_size(20.0, 0.40, 700.0)
        self.assertLess(near, far)

    def test_rejects_nonsense(self):
        for args in [(0.0, 0.4, 700.0), (10.0, 0.0, 700.0), (10.0, 0.4, 0.0)]:
            with self.assertRaises(ValueError):
                board.distance_from_size(*args)


class SizeClassificationTests(unittest.TestCase):
    def test_splits_the_two_hole_sizes(self):
        classified = board.classify_by_size(_grid(), 0.20, 0.40)
        small = [t for t in classified if t.diameter_m == 0.20]
        large = [t for t in classified if t.diameter_m == 0.40]
        self.assertEqual(len(small), 3)
        self.assertEqual(len(large), 9)

    def test_one_size_only_stays_unknown(self):
        """Guessing here would range off the wrong diameter."""
        only_large = [t for t in _grid() if t.v_px > 200.0]
        classified = board.classify_by_size(only_large, 0.20, 0.40)
        self.assertTrue(all(t.diameter_m is None for t in classified))

    def test_unknown_diameter_means_unknown_range(self):
        ranged = board.apply_ranging(
            [board.BoardTarget(0.0, 0.0, 10.0, 10.0)], 700.0
        )
        self.assertIsNone(ranged[0].distance_m)

    def test_empty_input(self):
        self.assertEqual(board.classify_by_size([], 0.2, 0.4), [])


class ColumnGroupingTests(unittest.TestCase):
    def test_three_columns_are_found(self):
        grouped = board.assign_columns(_grid())
        self.assertEqual(sorted({t.column for t in grouped}), [0, 1, 2])

    def test_small_jitter_stays_in_one_column(self):
        jittered = [
            board.BoardTarget(400.0, 300.0, 40.0, 40.0),
            board.BoardTarget(404.0, 450.0, 40.0, 40.0),
            board.BoardTarget(397.0, 600.0, 40.0, 40.0),
        ]
        grouped = board.assign_columns(jittered)
        self.assertEqual({t.column for t in grouped}, {0})


class EngagementOrderTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig()
        self.plan = board.build_plan(
            _grid(),
            self.config.focal_length_x_px,
            self.config.small_hole_diameter_m,
            self.config.large_hole_diameter_m,
        )

    def test_all_twelve_holes_are_planned(self):
        self.assertEqual(len(self.plan), 12)

    def test_large_holes_come_first(self):
        diameters = [t.diameter_m for t in self.plan.targets]
        self.assertEqual(diameters[:9], [0.40] * 9)
        self.assertEqual(diameters[9:], [0.20] * 3)

    def test_columns_run_left_to_right(self):
        large = self.plan.targets[:9]
        self.assertEqual([t.column for t in large], [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_within_a_column_bottom_comes_first(self):
        first_column = [t for t in self.plan.targets[:9] if t.column == 0]
        # Pixel v grows downward, so bottom first means descending v.
        self.assertEqual([t.v_px for t in first_column], [600.0, 450.0, 300.0])

    def test_every_hole_is_ranged(self):
        for target in self.plan.targets:
            self.assertAlmostEqual(target.distance_m, 6.0, places=2)

    def test_order_flags_flip_the_sequence(self):
        flipped = board.build_plan(
            _grid(), 700.0, 0.20, 0.40,
            large_first=False, left_to_right=False, bottom_to_top=False,
        )
        self.assertEqual(flipped.targets[0].diameter_m, 0.20)
        self.assertEqual(flipped.targets[0].column, 2)


class EngagementPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = board.build_plan(_grid(), 700.0, 0.20, 0.40)

    def test_walks_the_whole_list(self):
        for _ in range(12):
            self.assertFalse(self.plan.finished)
            self.plan.mark_engaged()
        self.assertTrue(self.plan.finished)
        self.assertIsNone(self.plan.current())
        self.assertTrue(all(t.engaged for t in self.plan.targets))

    def test_skip_does_not_mark_engaged(self):
        self.plan.skip()
        self.assertFalse(self.plan.targets[0].engaged)
        self.assertEqual(self.plan.index, 1)

    def test_advancing_past_the_end_is_safe(self):
        for _ in range(20):
            self.plan.mark_engaged()
        self.assertTrue(self.plan.finished)
        self.assertEqual(self.plan.remaining, 0)


class FakeClock:
    """Exact test clock.

    Accumulating 0.005 four hundred times lands on 1.9999999999999793, which
    is a hair under a 1.0 s cooldown boundary and makes timing assertions
    flake.  Counting whole microseconds keeps the arithmetic exact so the test
    measures the cooldown, not float drift.
    """

    def __init__(self):
        self._microseconds = 0

    def __call__(self):
        return self._microseconds / 1_000_000.0

    @property
    def now(self):
        return self()

    def advance(self, seconds):
        self._microseconds += int(round(seconds * 1_000_000))


class BoardEngagementTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig(
            stable_frames_required=1,
            aim_tolerance_rad=math.radians(90.0),  # always "on target"
            compensate_gravity=False,
            compensate_parallax=False,
        )
        self.motors = MockMotorBackend(
            motor_ids=[self.config.yaw_motor_id, self.config.pitch_motor_id],
            verbose=False,
        )
        self.turret = TurretController(self.config, self.motors)
        self.clock = FakeClock()
        self.shots = []
        self.plan = board.build_plan(_grid(), 700.0, 0.20, 0.40)

    def _engagement(self, **kw):
        kw.setdefault("shot_cooldown_s", 1.0)
        return engagement.BoardEngagement(
            self.config,
            self.turret,
            self.plan,
            fire=lambda t: (self.shots.append(t) or True),
            clock=self.clock,
            **kw,
        )

    def test_cooldown_stops_a_200hz_burst(self):
        """Without this the loop fires every tick the gate stays open."""
        driver = self._engagement()
        for _ in range(401):          # ticks at t = 0.000 .. 2.000
            driver.update()
            self.clock.advance(0.005)
        self.assertEqual(len(self.shots), 3)   # t=0.0, 1.0, 2.0

    def test_each_shot_advances_to_the_next_hole(self):
        driver = self._engagement(shot_cooldown_s=0.0)
        first = driver.update()
        self.assertTrue(first.fired)
        self.assertEqual(driver.plan.index, 1)
        second = driver.update()
        self.assertNotEqual(second.target.v_px, first.target.v_px)

    def test_works_through_every_hole(self):
        driver = self._engagement(shot_cooldown_s=0.0)
        for _ in range(12):
            driver.update()
        self.assertEqual(len(self.shots), 12)
        self.assertTrue(driver.plan.finished)

    def test_a_refused_shot_keeps_the_same_hole(self):
        driver = engagement.BoardEngagement(
            self.config, self.turret, self.plan,
            fire=lambda t: False, shot_cooldown_s=0.0, clock=self.clock,
        )
        target = driver.update()
        self.assertFalse(target.fired)
        self.assertEqual(driver.plan.index, 0)

    def test_unreachable_hole_times_out_instead_of_stalling(self):
        # A tiny tolerance will not do it: the mock tracks perfectly, so the
        # servo error is exactly 0.0 and passes any tolerance.  Block the gate
        # the way the real thing would -- a hole with no usable range.
        blocked = TurretConfig(
            stable_frames_required=1,
            aim_tolerance_rad=math.radians(90.0),
            compensate_gravity=True,
            require_distance_to_fire=True,
        )
        turret = TurretController(
            blocked,
            MockMotorBackend(
                motor_ids=[blocked.yaw_motor_id, blocked.pitch_motor_id], verbose=False
            ),
        )
        # Unknown diameter -> unknown range -> the gate stays shut.
        unranged = board.EngagementPlan(
            [board.BoardTarget(400.0, 600.0, 46.0, 46.0, 0.9) for _ in range(3)]
        )
        driver = engagement.BoardEngagement(
            blocked, turret, unranged,
            fire=lambda t: self.shots.append(t) or True,
            target_timeout_s=1.0, clock=self.clock,
        )
        driver.update()
        self.clock.advance(1.5)
        status = driver.update()
        self.assertTrue(status.skipped)
        self.assertEqual(driver.plan.index, 1)
        self.assertEqual(len(self.shots), 0)

    def test_still_holds_the_motors_when_the_board_is_done(self):
        driver = self._engagement(shot_cooldown_s=0.0)
        for _ in range(12):
            driver.update()
        before = len(self.motors.commands)
        status = driver.update()
        self.assertTrue(status.finished)
        self.assertGreater(len(self.motors.commands), before)

    def test_rejects_negative_cooldown(self):
        with self.assertRaises(ValueError):
            engagement.BoardEngagement(
                self.config, self.turret, self.plan, shot_cooldown_s=-1.0
            )


if __name__ == "__main__":
    unittest.main()

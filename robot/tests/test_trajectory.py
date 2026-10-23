import unittest

from turret_control.trajectory import acceleration_limited_step, smoothstep_sample


class SmoothTrajectoryTests(unittest.TestCase):
    def test_profile_starts_and_ends_at_rest(self):
        self.assertEqual(smoothstep_sample(1.0, 3.0, 0.0, 2.0), (1.0, 0.0))
        self.assertEqual(smoothstep_sample(1.0, 3.0, 2.0, 2.0), (3.0, 0.0))

    def test_profile_is_continuous_and_monotonic(self):
        samples = [
            smoothstep_sample(0.0, 1.0, t, 1.0)[0]
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(samples[-1], 1.0)
        self.assertEqual(samples, sorted(samples))
        self.assertAlmostEqual(samples[2], 0.5)

    def test_velocity_is_zero_at_endpoints_and_has_correct_sign(self):
        self.assertEqual(smoothstep_sample(2.0, -2.0, 0.0, 4.0)[1], 0.0)
        self.assertEqual(smoothstep_sample(2.0, -2.0, 4.0, 4.0)[1], 0.0)
        self.assertLess(smoothstep_sample(2.0, -2.0, 2.0, 4.0)[1], 0.0)


class AccelerationLimitedTrajectoryTests(unittest.TestCase):
    def test_starts_without_position_or_velocity_jump(self):
        position, velocity = acceleration_limited_step(0.0, 0.0, 1.0, 1.0, 2.0, 0.1)
        self.assertAlmostEqual(velocity, 0.2)
        self.assertAlmostEqual(position, 0.01)

    def test_speed_and_acceleration_are_bounded(self):
        position = 0.0
        velocity = 0.0
        previous_velocity = velocity
        for _ in range(100):
            position, velocity = acceleration_limited_step(
                position, velocity, 10.0, 0.5, 1.0, 0.01
            )
            self.assertLessEqual(abs(velocity), 0.5 + 1e-9)
            self.assertLessEqual(abs(velocity - previous_velocity), 0.01 + 1e-9)
            previous_velocity = velocity

    def test_reaches_positive_and_negative_targets_without_overshoot(self):
        for target in (1.0, -1.0):
            position = 0.0
            velocity = 0.0
            positions = [position]
            for _ in range(2000):
                position, velocity = acceleration_limited_step(
                    position, velocity, target, 0.8, 1.2, 0.005
                )
                positions.append(position)
                if position == target and velocity == 0.0:
                    break
            self.assertEqual(position, target)
            self.assertEqual(velocity, 0.0)
            if target > 0:
                self.assertTrue(all(value <= target for value in positions))
            else:
                self.assertTrue(all(value >= target for value in positions))

    def test_acceleration_remains_bounded_while_stopping(self):
        position = 0.0
        velocity = 0.0
        max_acceleration = 0.08
        dt = 0.005
        previous_velocity = velocity
        for _ in range(10000):
            position, velocity = acceleration_limited_step(
                position,
                velocity,
                1.0,
                0.08,
                max_acceleration,
                dt,
            )
            self.assertLessEqual(
                abs(velocity - previous_velocity),
                max_acceleration * dt + 1e-12,
            )
            previous_velocity = velocity
            if position == 1.0 and velocity == 0.0:
                break
        self.assertEqual((position, velocity), (1.0, 0.0))

    def test_retargeting_brakes_continuously_before_returning(self):
        position = 0.0
        velocity = 0.0
        max_acceleration = 0.08
        dt = 0.005

        for _ in range(300):
            position, velocity = acceleration_limited_step(
                position, velocity, 1.0, 0.08, max_acceleration, dt
            )

        # Put the new target just in front of a command moving at full speed.
        # It is physically impossible to stop there without either overshoot
        # or a velocity jump, so the profile must brake and return smoothly.
        target = position + 0.0001
        previous_velocity = velocity
        passed_target = False
        for _ in range(10000):
            position, velocity = acceleration_limited_step(
                position, velocity, target, 0.08, max_acceleration, dt
            )
            self.assertLessEqual(
                abs(velocity - previous_velocity),
                max_acceleration * dt + 1e-12,
            )
            passed_target = passed_target or position > target
            previous_velocity = velocity
            if position == target and velocity == 0.0:
                break

        self.assertTrue(passed_target)
        self.assertEqual((position, velocity), (target, 0.0))


if __name__ == "__main__":
    unittest.main()

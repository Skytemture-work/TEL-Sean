import math
import unittest

import numpy as np

from turret_control.board_pose import BoardPose, BoardPoseLock, quaternion_angle_deg
from turret_control.config import TurretConfig


def _pose(t, *, x=0.0, yaw_deg=0.0, source="pnp", stable=True):
    return BoardPose(
        rvec=np.array([[0.0], [math.radians(yaw_deg)], [0.0]]),
        tvec=np.array([[x], [0.0], [5.0]]),
        timestamp=t,
        reprojection_error_px=1.0,
        correspondence_count=12,
        translation_speed_m_s=0.0,
        rotation_speed_rad_s=0.0,
        stable=stable,
        source=source,
    )


def _quat_about_z(deg):
    half = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


LEVEL = _quat_about_z(0.0)


class QuaternionAngleTests(unittest.TestCase):
    def test_angle_and_sign_invariance(self):
        self.assertAlmostEqual(quaternion_angle_deg(LEVEL, _quat_about_z(2.0)), 2.0, places=6)
        flipped = tuple(-v for v in _quat_about_z(2.0))
        self.assertAlmostEqual(quaternion_angle_deg(LEVEL, flipped), 2.0, places=6)
        self.assertEqual(quaternion_angle_deg(LEVEL, (0, 0, 0, 0)), math.inf)


DRIFT_DEG_S = 0.07  # measured ZED 2 fused-yaw drift on the robot


def _drifting(t, extra_deg=0.0):
    return _quat_about_z(DRIFT_DEG_S * t + extra_deg)


class BoardPoseLockTests(unittest.TestCase):
    def setUp(self):
        self.config = TurretConfig()
        self.lock = BoardPoseLock(self.config)

    def learn(self, until, imu=_drifting, start=0.0, **pose_kwargs):
        """Stable PnP frames at 10 Hz: the chassis is provably still."""
        t = start
        while t <= until + 1e-9:
            self.lock.update(_pose(t, **pose_kwargs), True, t, imu(t) if imu else None)
            t += 0.1
        return until

    def test_nothing_to_lock_before_first_stable_pnp(self):
        self.assertFalse(self.lock.update(None, False, 0.0, LEVEL).usable)
        self.assertFalse(self.lock.update(_pose(0.0, stable=False), False, 0.1, LEVEL).usable)

    def test_steady_imu_drift_does_not_release_a_long_occlusion(self):
        t0 = self.learn(5.0)
        for t in (t0 + 1.0, t0 + 10.0, t0 + 60.0):
            decision = self.lock.update(None, False, t, _drifting(t))
            self.assertTrue(decision.usable, t)
        self.assertAlmostEqual(self.lock.pose.timestamp, t0)

    def test_turn_beyond_drift_releases_until_next_stable_pnp(self):
        t0 = self.learn(5.0)
        decision = self.lock.update(None, False, t0 + 3.0, _drifting(t0 + 3.0, extra_deg=1.0))
        self.assertFalse(decision.usable)
        self.assertIn("IMU", decision.reason)
        # Turning back does not restore trust in the old pose.
        self.assertFalse(self.lock.update(None, False, t0 + 3.1, _drifting(t0 + 3.1)).usable)
        # Drift history before the motion is discarded: a fresh 2 s is needed.
        moved = lambda t: _drifting(t, extra_deg=1.0)
        t1 = self.learn(t0 + 6.0, imu=moved, start=t0 + 5.0, x=0.2)
        self.assertTrue(self.lock.update(None, False, t1 + 0.5, moved(t1 + 0.5)).usable)
        limit = self.config.pose_lock_max_blind_s_without_imu
        self.assertFalse(self.lock.update(None, False, t1 + limit + 0.1,
                                          moved(t1 + limit + 0.1)).usable)

    def test_lost_imu_reading_releases(self):
        t0 = self.learn(5.0)
        decision = self.lock.update(None, False, t0 + 0.5, None)
        self.assertFalse(decision.usable)
        self.assertIn("IMU", decision.reason)

    def test_without_imu_lock_expires(self):
        t0 = self.learn(5.0, imu=None)
        limit = self.config.pose_lock_max_blind_s_without_imu
        self.assertTrue(self.lock.update(None, False, t0 + limit - 0.1, None).usable)
        decision = self.lock.update(None, False, t0 + limit + 0.1, None)
        self.assertFalse(decision.usable)
        self.assertIn("無 IMU", decision.reason)

    def test_too_short_drift_history_falls_back_to_time_limit(self):
        t0 = self.learn(1.0)
        limit = self.config.pose_lock_max_blind_s_without_imu
        # Even a large IMU turn is not trusted without a drift estimate ...
        self.assertTrue(self.lock.update(None, False, t0 + 1.0, _drifting(t0, 5.0)).usable)
        # ... so the blind time limit applies instead.
        self.assertFalse(self.lock.update(None, False, t0 + limit + 0.1,
                                          _drifting(t0 + limit)).usable)

    def test_settling_pnp_close_to_lock_keeps_it(self):
        t0 = self.learn(5.0)
        settling = _pose(t0 + 1.0, x=0.01, yaw_deg=0.3, stable=False)
        self.assertTrue(self.lock.update(settling, False, t0 + 1.0, _drifting(t0 + 1.0)).usable)

    def test_settling_pnp_far_from_lock_releases(self):
        for moved in (dict(x=0.05), dict(yaw_deg=2.0)):
            with self.subTest(moved=moved):
                self.lock = BoardPoseLock(self.config)
                t0 = self.learn(5.0)
                decision = self.lock.update(_pose(t0 + 1.0, stable=False, **moved), False,
                                            t0 + 1.0, _drifting(t0 + 1.0))
                self.assertFalse(decision.usable)
                self.assertIn("解除姿態鎖定", decision.reason)

    def test_partial_pose_never_refreshes_lock(self):
        t0 = self.learn(5.0, imu=None)
        partial = _pose(t0 + 5.0, x=0.5, source="partial")
        self.lock.update(partial, True, t0 + 5.0, None)
        self.assertAlmostEqual(self.lock.pose.timestamp, t0)
        limit = self.config.pose_lock_max_blind_s_without_imu
        self.assertFalse(self.lock.update(None, False, t0 + limit + 0.1, None).usable)


if __name__ == "__main__":
    unittest.main()

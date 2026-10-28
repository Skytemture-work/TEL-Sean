import unittest

from turret_control.aiming import AimCommand
from turret_control.board import BoardTarget, EngagementPlan
from turret_control.config import TurretConfig
from turret_control.engagement import BoardEngagement


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _AlwaysReadyTurret:
    def update(self, _target):
        return AimCommand(
            has_target=True,
            target_yaw_rad=0.0,
            target_pitch_rad=0.0,
            stable=True,
            fire_allowed=True,
        )


class EngagementSafetyGateTests(unittest.TestCase):
    def test_unstable_pose_blocks_fire_without_timing_out_plan(self):
        clock = _Clock()
        fired = []
        target = BoardTarget(
            640.0,
            360.0,
            40.0,
            40.0,
            hole_id="H05",
            distance_m=5.0,
        )
        plan = EngagementPlan([target])
        engagement = BoardEngagement(
            TurretConfig(),
            _AlwaysReadyTurret(),
            plan=plan,
            fire=lambda item: fired.append(item.hole_id) or True,
            target_timeout_s=1.0,
            clock=clock,
        )

        engagement.set_firing_enabled(False, "board pose unstable")
        first = engagement.update()
        clock.now = 10.0
        second = engagement.update()

        self.assertFalse(first.command.fire_allowed)
        self.assertEqual(first.command.blocked_reason, "board pose unstable")
        self.assertFalse(second.skipped)
        self.assertEqual(plan.index, 0)
        self.assertEqual(fired, [])

        engagement.set_firing_enabled(True)
        ready = engagement.update()
        self.assertTrue(ready.fired)
        self.assertEqual(fired, ["H05"])
        self.assertTrue(plan.finished)


if __name__ == "__main__":
    unittest.main()

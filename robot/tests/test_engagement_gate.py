import unittest
from types import SimpleNamespace

from turret_control.aiming import AimCommand
from turret_control.board import BoardTarget, EngagementPlan
from turret_control.config import TurretConfig
from turret_control.engagement import BoardEngagement, PartialPoseShotGate
from turret_control.controller import TurretController
from turret_control.limits import WorkspaceLimit


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
    def test_motor_workspace_blocks_a_stable_but_unreachable_pose(self):
        workspace = WorkspaceLimit(
            yaw_motor_id=3, pitch_motor_id=1,
            vertices=((0, 0), (2, 0), (0, 2)), anchor=(.2, .2),
        )

        class Motors:
            _workspace = workspace
            def refresh(self):
                pass
            def get_state(self, motor_id):
                return SimpleNamespace(position_rad=1.5)
            def send_targets(self, targets):
                self.targets = targets
            def hold(self):
                self.held = True

        controller = TurretController(TurretConfig(), Motors())
        controller.aiming.update = lambda *args: AimCommand(
            has_target=True, target_yaw_rad=1.5, target_pitch_rad=1.5,
            stable=True, fire_allowed=True,
        )
        command = controller.update(None)
        self.assertFalse(command.fire_allowed)
        self.assertFalse(command.target_reachable)
        self.assertEqual(command.blocked_reason, "target is outside the motor workspace")
        self.assertTrue(controller.motors.held)
        self.assertFalse(hasattr(controller.motors, "targets"))

    def test_partial_fallback_allows_one_successful_shot_until_stable_pnp(self):
        gate = PartialPoseShotGate()
        partial = SimpleNamespace(source="partial", stable=True)
        pnp = SimpleNamespace(source="pnp", stable=True)
        self.assertTrue(gate.allows(partial))
        gate.record(partial, False)
        self.assertTrue(gate.allows(partial))
        gate.record(partial, True)
        self.assertFalse(gate.allows(partial))
        self.assertFalse(gate.allows(None))
        self.assertTrue(gate.allows(pnp))
        self.assertTrue(gate.allows(partial))

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

    def test_failed_launcher_attempt_is_retried_only_after_cooldown(self):
        clock = _Clock()
        attempts = []
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
            fire=lambda item: attempts.append(item.hole_id) or False,
            shot_cooldown_s=1.0,
            target_timeout_s=None,
            clock=clock,
        )

        engagement.update()
        clock.now = 0.2
        engagement.update()
        clock.now = 0.99
        engagement.update()
        self.assertEqual(attempts, ["H05"])
        self.assertEqual(plan.index, 0)

        clock.now = 1.0
        engagement.update()
        self.assertEqual(attempts, ["H05", "H05"])
        self.assertEqual(plan.index, 0)


if __name__ == "__main__":
    unittest.main()

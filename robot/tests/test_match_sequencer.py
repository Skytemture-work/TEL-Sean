import unittest

from test_scoring_strategy import _board_plan
from turret_control.match_sequencer import MatchSequencer
from turret_control.scoring_strategy import UPPER_GRID_IDS, UPPER_GRID_NO_BONUS_IDS


class Harness:
    """Drive a MatchSequencer like the control loop does."""

    def __init__(self, phase, duration=240.0, balls=18, light_wait=5.0, rotation_ok=True,
                 clock_origin=None):
        self.plans = []
        self.logs = []
        self.rotations = 0
        self.rotation_ok = rotation_ok
        self.flywheels = []

        def rotate():
            self.rotations += 1
            return self.rotation_ok

        self.match = MatchSequencer(phase, duration, balls, self.plans.append, self.logs.append,
                                    light_wait_s=light_wait, start_reload_rotation=rotate,
                                    stop_flywheels=lambda: self.flywheels.append("off"),
                                    start_flywheels=lambda: self.flywheels.append("on"),
                                    clock_origin=clock_origin)
        self.board = _board_plan()
        self.now = 0.0

    def tick(self, lit=None, idle=True, busy=False, reload=False, board=True, dt=0.1):
        self.match.step(self.now, self.board if board else None, lit, idle, busy, reload)
        self.now += dt

    def fire(self, n=1):
        """Take n shots from the current plan."""
        shots = []
        for _ in range(n):
            plan = self.match.plan
            if plan is None or plan.current() is None or self.match.hold_reason:
                break
            shots.append(plan.current().hole_id)
            plan.mark_engaged()
            self.match.record_fired()
            self.tick()
        return shots


class LowerHalfTests(unittest.TestCase):
    def test_starts_as_soon_as_the_board_is_seen_and_ends_with_the_balls(self):
        h = Harness("lower")
        h.tick(board=False)
        self.assertIsNone(h.match.started_at)
        h.tick()
        self.assertIsNotNone(h.match.started_at)
        shots = h.fire(30)
        self.assertEqual(shots, ["H01", "H01", "H02", "H02", "H03", "H03", "H06", "H06", "H05",
                                 "H05", "H04", "H04", "H07", "H07", "H08", "H08", "H09", "H09"])
        h.tick()
        self.assertEqual(h.match.stopped, "已裝彈藥全部射出，停止發射")
        self.assertFalse(h.match.reload_wait)


class UpperHalfTests(unittest.TestCase):
    def test_bonus_then_grid_with_a_reload_after_18(self):
        h = Harness("upper")
        h.tick()
        self.assertTrue(h.match.waiting_light)
        self.assertEqual(len(h.match.plan), 0)  # hold still until the light is found
        h.tick(lit="H11")
        first = h.fire(30)
        self.assertEqual(first, ["H11"] * 3 + ["H03"] * 3 + ["H02"] * 3 + ["H01"] * 3
                         + ["H04"] * 3 + ["H05"] * 3)
        h.tick()
        self.assertTrue(h.match.reload_wait)
        self.assertEqual(h.match.reload_balls, 9)
        self.assertEqual(h.match.hold_reason, "等待裝填：完成後在儀表板按「裝填完成」")
        self.assertEqual(h.fire(1), [])  # held: no shot while reloading
        h.tick(reload=True, busy=True)    # magazine still rotating: refused
        self.assertTrue(h.match.reload_wait)
        self.assertEqual(h.rotations, 0)
        self.assertEqual(h.flywheels, ["off"])  # flywheels stopped while reloading
        h.tick(reload=True)
        self.assertFalse(h.match.reload_wait)
        self.assertEqual(h.rotations, 1)  # magazine turned 2 slots before shooting on
        self.assertEqual(h.flywheels, ["off", "on"])
        self.assertEqual(h.match.balls_remaining, 9)
        second = h.fire(30)
        self.assertEqual(second, ["H06"] * 3 + ["H09"] * 2 + ["H08"] * 2 + ["H07"] * 2)
        h.tick()
        self.assertEqual(h.match.stopped, "已裝彈藥全部射出，停止發射")

    def test_bonus_window_closes_60_s_after_start(self):
        h = Harness("upper")
        h.tick()
        h.tick(lit="H10")
        self.assertEqual(h.fire(2), ["H10"] * 2)
        h.now = 60.0
        h.tick(idle=False)  # a stage is running: wait for it
        self.assertEqual(h.match.plan.current().hole_id, "H10")
        h.tick()
        rest = h.fire(30)
        self.assertNotIn("H10", rest)
        self.assertEqual(rest, list(UPPER_GRID_IDS[:16]))
        self.assertEqual(len(["H10"] * 2 + rest), 18)
        # The skipped ball is still in the magazine: a big hole below the cap.
        self.assertTrue(any("略過 1 顆小孔" in line for line in h.logs))
        h.tick(reload=True)
        second = h.fire(30)
        self.assertEqual(len(second), 9)
        self.assertEqual(second[:8], list(UPPER_GRID_IDS[16:]))
        self.assertIn(second[8], ("H07", "H08", "H09"))

    def test_no_light_in_time_starts_on_h11_then_follows_the_real_light(self):
        h = Harness("upper", light_wait=5.0)
        h.tick()
        h.now = 4.9
        h.tick()
        self.assertTrue(h.match.waiting_light)
        h.now = 5.0
        h.tick()
        self.assertEqual(h.match.bonus_hole_id, "H11")
        self.assertEqual(h.fire(1), ["H11"])
        h.tick(lit="H12", idle=False)   # wait for the running stage
        self.assertEqual(h.match.plan.current().hole_id, "H11")
        h.tick(lit="H12")
        rest = h.fire(2)
        self.assertEqual(rest, ["H12"] * 2)
        self.assertEqual(h.match.bonus_hole_id, "H12")
        self.assertEqual(h.fire(1), ["H03"])

    def test_no_wait_starts_on_h11_at_once_and_switches_to_the_light(self):
        # Default since 2026-11-16: light_wait 0, the light is on from the start.
        h = Harness("upper", light_wait=0.0)
        h.tick()
        self.assertFalse(h.match.waiting_light)
        self.assertEqual(h.match.plan.current().hole_id, "H11")
        h.tick(lit="H10")               # light confirmed before the first push
        self.assertEqual(h.match.bonus_hole_id, "H10")
        self.assertEqual(h.fire(3), ["H10"] * 3)
        self.assertEqual(h.fire(1), ["H03"])

    def test_light_confirmed_late_after_60_s_never_shoots_a_small_hole(self):
        h = Harness("upper", light_wait=55.0)
        h.tick()
        h.now = 60.0
        h.tick(lit="H12")
        self.assertFalse(h.match.waiting_light)
        self.assertIsNone(h.match.bonus_hole_id)
        self.assertEqual(h.fire(18), list(UPPER_GRID_NO_BONUS_IDS[:18]))
        h.tick(reload=True)
        # All 27 balls go to big holes, three each.
        self.assertEqual(h.fire(30), list(UPPER_GRID_NO_BONUS_IDS[18:]))

    def test_reload_rotation_refused_keeps_waiting(self):
        h = Harness("lower", balls=4, rotation_ok=False)
        h.tick()
        h.fire(4)
        h.tick()
        self.assertTrue(h.match.reload_wait)
        h.tick(reload=True)
        self.assertTrue(h.match.reload_wait)
        self.assertEqual(h.rotations, 1)
        self.assertTrue(any("無法開始裝填轉格" in line for line in h.logs))

    def test_plan_done_with_balls_left_ends_the_match(self):
        h = Harness("lower", balls=18)
        h.tick()
        plan = h.match.plan
        for _ in range(3):
            plan.skip()  # e.g. targets that timed out
        h.fire(30)
        h.tick()
        self.assertEqual(h.match.stopped, "射擊計畫完成")
        self.assertEqual(h.match.balls_remaining, 3)

    def test_match_time_ends_the_match(self):
        h = Harness("upper", duration=100.0)
        h.tick()
        h.tick(lit="H11")
        h.fire(3)
        h.now = 100.0
        h.tick(idle=False)
        self.assertIsNone(h.match.stopped)  # never cut a running stage
        h.tick()
        self.assertEqual(h.match.stopped, "比賽時間到，停止發射")
        self.assertEqual(len(h.match.plan), 0)

    def test_telemetry(self):
        h = Harness("upper")
        h.tick()
        data = h.match.telemetry(h.now)
        self.assertTrue(data["waiting_light"])
        self.assertEqual(data["balls_remaining"], 18)
        self.assertEqual(data["remaining_s"], 239.9)


class DepartureClockTests(unittest.TestCase):
    """main.py passes the VRA departure; positioning took 30 s before now=0."""

    def test_time_limit_counts_from_the_departure(self):
        h = Harness("lower", duration=180.0, clock_origin=-30.0)
        h.tick()
        self.assertEqual(h.match.started_at, 0.0)
        self.assertEqual(h.match.telemetry(h.now)["remaining_s"], 149.9)
        h.fire(2)
        h.now = 149.9
        h.tick()
        self.assertIsNone(h.match.stopped)
        h.now = 150.0
        h.tick()
        self.assertEqual(h.match.stopped, "比賽時間到，停止發射")

    def test_bonus_window_counts_from_the_departure(self):
        h = Harness("upper", duration=180.0, clock_origin=-30.0)
        h.tick()
        h.tick(lit="H10")
        self.assertEqual(h.fire(2), ["H10"] * 2)
        h.now = 30.0  # 60 s after the departure
        h.tick()
        rest = h.fire(30)
        self.assertNotIn("H10", rest)
        self.assertTrue(any("略過 1 顆小孔" in line for line in h.logs))

    def test_light_wait_still_counts_from_the_first_shot_chance(self):
        h = Harness("upper", duration=180.0, light_wait=5.0, clock_origin=-30.0)
        h.tick()
        h.now = 4.9
        h.tick()
        self.assertTrue(h.match.waiting_light)
        h.now = 5.0
        h.tick()
        self.assertEqual(h.match.bonus_hole_id, "H11")

    def test_departure_more_than_60_s_ago_skips_the_bonus(self):
        h = Harness("upper", duration=180.0, clock_origin=-61.0)
        h.tick(lit="H10")
        self.assertNotIn("H10", h.fire(18))


if __name__ == "__main__":
    unittest.main()

import math
import unittest
from types import SimpleNamespace

from turret_control.aiming import AimingController, TargetPixel
from turret_control.competition_geometry import with_competition_geometry
from turret_control.config import TurretConfig, with_recorded_mechanism_geometry
from turret_control.muzzle_camera import MuzzleIntrinsics, barrel_aim_pixel
from turret_control.muzzle_correction import (
    MuzzleCorrectionSettings,
    MuzzleShotCorrector,
    aim_error_rad,
    expected_hole_pixel,
    locate_hole,
)

INTRINSICS = MuzzleIntrinsics.from_raw(1280, 720, 532.7, 532.7, 637.4, 358.7)
SETTINGS = MuzzleCorrectionSettings(offset_x_m=-0.0665, offset_y_m=0.060)
DEG = math.radians(1.0)


def _hole(u, v, hole_id=None, size=40.0):
    return SimpleNamespace(u_px=u, v_px=v, size_px=size, hole_id=hole_id)


class ExpectedPixelTests(unittest.TestCase):
    def test_no_elevation_is_the_barrel_point(self):
        expected = expected_hole_pixel(INTRINSICS, SETTINGS, 5.0, 0.0)
        barrel = barrel_aim_pixel(INTRINSICS, 5.0, -0.0665, 0.060)
        self.assertAlmostEqual(expected[0], barrel[0], places=6)
        self.assertAlmostEqual(expected[1], barrel[1], places=6)

    def test_raised_barrel_sees_the_hole_lower(self):
        level = expected_hole_pixel(INTRINSICS, SETTINGS, 5.0, 0.0)
        raised = expected_hole_pixel(INTRINSICS, SETTINGS, 5.0, 2.0 * DEG)
        self.assertAlmostEqual(raised[0], level[0], places=6)
        self.assertAlmostEqual(raised[1] - level[1], 532.7 * 2.0 * DEG, delta=0.5)

    def test_boresight_offsets_shift_the_expectation(self):
        settings = MuzzleCorrectionSettings(offset_x_m=-0.0665, offset_y_m=0.060,
                                            boresight_yaw_rad=0.5 * DEG, boresight_pitch_rad=0.5 * DEG)
        base = expected_hole_pixel(INTRINSICS, SETTINGS, 5.0, 0.0)
        shifted = expected_hole_pixel(INTRINSICS, settings, 5.0, 0.0)
        self.assertGreater(shifted[0], base[0])  # camera axis right of barrel
        self.assertLess(shifted[1], base[1])     # camera axis above barrel


class AimErrorTests(unittest.TestCase):
    def test_signs(self):
        expected = (360.0, 640.0)
        right_yaw, _ = aim_error_rad(INTRINSICS, expected, (370.0, 640.0))
        left_yaw, _ = aim_error_rad(INTRINSICS, expected, (350.0, 640.0))
        _, up_pitch = aim_error_rad(INTRINSICS, expected, (360.0, 630.0))
        self.assertGreater(right_yaw, 0)   # hole further right: turn right
        self.assertLess(left_yaw, 0)
        self.assertGreater(up_pitch, 0)    # hole higher than expected: raise
        self.assertAlmostEqual(right_yaw, math.atan(10 / 532.7), delta=1e-3)


class LocateHoleTests(unittest.TestCase):
    def setUp(self):
        self.plan = [_hole(100.0 * c, 100.0 * r, f"H{c}{r}") for c in range(1, 4) for r in range(1, 4)]

    def test_uses_the_real_detection_near_the_labelled_hole(self):
        detections = [_hole(203.0, 198.0), _hole(305.0, 300.0)]
        self.assertEqual(locate_hole(self.plan, detections, "H22"), (203.0, 198.0))

    def test_missing_ambiguous_or_wrong_size_is_rejected(self):
        self.assertIsNone(locate_hole(self.plan, [], "H22"))
        self.assertIsNone(locate_hole(self.plan, [_hole(250.0, 200.0)], "H22"))  # beyond gate
        self.assertIsNone(locate_hole(self.plan, [_hole(201.0, 200.0), _hole(199.0, 202.0)], "H22"))
        self.assertIsNone(locate_hole(self.plan, [_hole(200.0, 200.0, size=10.0)], "H22"))
        self.assertIsNone(locate_hole(self.plan, [_hole(200.0, 200.0)], "H99"))
        self.assertIsNone(locate_hole(None, [_hole(200.0, 200.0)], "H22"))


class CorrectorTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.measured = None      # pixel the fake muzzle camera reports
        self.frame_ready = True
        self.applied = []

    def measure(self, hole_id, captured_after):
        if not self.frame_ready:
            return None
        return (self.measured is not None, self.measured)

    def corrector(self, mode):
        return MuzzleShotCorrector(mode, INTRINSICS, SETTINGS, self.measure,
                                   lambda hole, yaw, pitch: self.applied.append((hole, yaw, pitch)))

    def run_until_released(self, corrector, key=("r", 0, "H05"), limit=50, dt=0.05):
        holds = []
        for _ in range(limit):
            hold = corrector.step(self.now, key, "H05", True, 5.0, 0.0)
            holds.append(hold)
            self.now += dt
            if hold is None and corrector.record_for(key).verdict != "pending":
                break
        return holds

    def expected(self):
        return expected_hole_pixel(INTRINSICS, SETTINGS, 5.0, 0.0)

    def test_measure_mode_never_holds_and_records_the_error(self):
        corrector = self.corrector("measure")
        u, v = self.expected()
        self.measured = (u + 532.7 * math.tan(0.4 * DEG), v)
        holds = self.run_until_released(corrector)
        self.assertTrue(all(h is None for h in holds))
        record = corrector.last
        self.assertEqual(record.verdict, "measured")
        self.assertAlmostEqual(math.degrees(record.error_yaw_rad), 0.4, places=3)
        self.assertEqual(self.applied, [])

    def test_apply_mode_holds_corrects_resettles_then_releases(self):
        corrector = self.corrector("apply")
        u, v = self.expected()
        self.measured = (u + 532.7 * math.tan(0.4 * DEG), v)
        key = ("r", 0, "H05")
        self.assertEqual(corrector.step(0.0, key, "H05", True, 5.0, 0.0), "砲口相機確認中")
        hold = None
        for i in range(1, 10):  # settle delay, then the first measurement
            hold = corrector.step(0.05 * i, key, "H05", True, 5.0, 0.0)
            if self.applied:
                break
        self.assertEqual(hold, "砲口相機修正後重新到位")
        self.assertEqual(len(self.applied), 1)
        self.assertAlmostEqual(math.degrees(self.applied[0][1]), 0.4, places=3)
        # The corrected aim now puts the hole where it belongs.
        self.measured = self.expected()
        self.now = 1.0
        self.run_until_released(corrector, key)
        self.assertEqual(corrector.last.verdict, "corrected")
        self.assertIsNone(corrector.step(self.now, key, "H05", True, 5.0, 0.0))
        # The next ball at the same hole starts from the learned correction.
        nxt = ("r", 1, "H05")
        corrector.step(self.now, nxt, "H05", True, 5.0, 0.0)
        self.assertAlmostEqual(math.degrees(corrector.record_for(nxt).total_yaw_rad), 0.4, places=3)
        corrector.reset()
        self.assertEqual(self.applied[-1], ("H05", 0.0, 0.0))

    def test_small_error_is_verified_without_correction(self):
        corrector = self.corrector("apply")
        u, v = self.expected()
        self.measured = (u + 0.5, v - 0.5)
        self.run_until_released(corrector)
        self.assertEqual(corrector.last.verdict, "verified")
        self.assertEqual(self.applied, [])

    def test_doubtful_measurements_release_on_the_unchanged_aim(self):
        u, v = self.expected()
        for measured, verdict in ((None, "not_found"),
                                  ((u + 532.7 * math.tan(3 * DEG), v), "rejected")):
            with self.subTest(verdict=verdict):
                self.applied.clear()
                self.measured = measured
                corrector = self.corrector("apply")
                self.run_until_released(corrector)
                self.assertEqual(corrector.last.verdict, verdict)
                self.assertEqual(self.applied, [])

    def test_no_frame_times_out_instead_of_stalling(self):
        self.frame_ready = False
        corrector = self.corrector("apply")
        holds = self.run_until_released(corrector, limit=100)
        self.assertIsNone(holds[-1])
        self.assertEqual(corrector.last.verdict, "timeout")

    def test_unstable_aim_keeps_holding_without_measuring(self):
        corrector = self.corrector("apply")
        for i in range(100):
            self.assertEqual(corrector.step(i * 0.05, ("r", 0, "H05"), "H05", False, 5.0, 0.0),
                             "砲口相機確認中")
        self.assertEqual(corrector.last.measurements, 0)

    def test_off_mode_does_nothing(self):
        corrector = self.corrector("off")
        self.assertIsNone(corrector.step(0.0, ("r", 0, "H05"), "H05", True, 5.0, 0.0))
        self.assertIsNone(corrector.last)


class YawTrimTests(unittest.TestCase):
    def test_yaw_trim_turns_the_commanded_yaw(self):
        config = with_recorded_mechanism_geometry(with_competition_geometry(TurretConfig()))
        base = AimingController(config).update(TargetPixel(640.0, 300.0, 1.0, 6.0),
                                               config.yaw_zero_rad, config.pitch_zero_rad)
        trimmed = AimingController(config).update(
            TargetPixel(640.0, 300.0, 1.0, 6.0, yaw_trim_rad=0.3 * DEG),
            config.yaw_zero_rad, config.pitch_zero_rad)
        self.assertAlmostEqual(trimmed.target_yaw_rad - base.target_yaw_rad,
                               config.yaw_sign * 0.3 * DEG, places=9)
        self.assertAlmostEqual(trimmed.target_pitch_rad, base.target_pitch_rad, places=12)


if __name__ == "__main__":
    unittest.main()

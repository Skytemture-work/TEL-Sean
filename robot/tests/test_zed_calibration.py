import tempfile
import unittest
from pathlib import Path

from turret_control.config import TurretConfig
from turret_control.geometry import normalized_image_coordinates
from turret_control.zed_calibration import load_zed2_uvc_calibration


class ZedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "SN123.conf"
        self.config = TurretConfig()

    def write_calibration(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_raw_hd_values_are_used_for_angle_calculation(self):
        self.write_calibration("""[LEFT_CAM_HD]
fx=531.5
fy=529.25
cx=637.0
cy=358.0
k1=0
k2=0
p1=0
p2=0
k3=0
""")
        calibrated = load_zed2_uvc_calibration(self.config, self.path)
        self.assertEqual((calibrated.focal_length_x_px, calibrated.focal_length_y_px), (531.5, 529.25))
        self.assertEqual((calibrated.camera_center_x_px, calibrated.camera_center_y_px), (637.0, 358.0))
        x, y = normalized_image_coordinates(737.0, 308.0, calibrated)
        self.assertAlmostEqual(x, 100.0 / 531.5)
        self.assertAlmostEqual(y, 50.0 / 529.25)

    def test_separate_distortion_coefficients_follow_opencv_order(self):
        self.write_calibration("""[LEFT_CAM_HD]
fx=530
fy=532
cx=640
cy=360
[LEFT_DISTO]
k1=.1
k2=.2
k3=.3
k4=.4
k5=.5
k6=.6
p1=.01
p2=.02
""")
        calibrated = load_zed2_uvc_calibration(self.config, self.path)
        self.assertEqual(calibrated.camera_distortion_coeffs, (.1, .2, .01, .02, .3, .4, .5, .6))

    def test_rejects_missing_nonfinite_and_wrong_resolution(self):
        self.write_calibration("[LEFT_CAM_HD]\nfx=nan\n")
        with self.assertRaises(ValueError):
            load_zed2_uvc_calibration(self.config, self.path)
        self.write_calibration("""[LEFT_CAM_HD]
fx=nan
fy=530
cx=640
cy=360
k1=0
k2=0
p1=0
p2=0
k3=0
""")
        with self.assertRaises(ValueError):
            load_zed2_uvc_calibration(self.config, self.path)
        from dataclasses import replace
        with self.assertRaises(ValueError):
            load_zed2_uvc_calibration(replace(self.config, image_width_px=640), self.path)


if __name__ == "__main__":
    unittest.main()

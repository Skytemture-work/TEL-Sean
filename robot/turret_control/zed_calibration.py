"""Load the factory raw-left calibration for ZED 2 UVC images."""

from configparser import ConfigParser, Error as ConfigError
from dataclasses import replace
import math
from pathlib import Path

from .config import TurretConfig


def load_zed2_uvc_calibration(config: TurretConfig, path: Path) -> TurretConfig:
    """Apply one camera's HD720 raw-left factory parameters without scaling."""
    if (config.image_width_px, config.image_height_px) != (1280, 720):
        raise ValueError("ZED 2 HD 原廠標定需要 1280×720 左眼影像")
    parser = ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))
    try:
        with Path(path).open(encoding="utf-8-sig") as handle:
            parser.read_file(handle)
        camera = parser["LEFT_CAM_HD"]
        distortion = parser["LEFT_DISTO"] if parser.has_section("LEFT_DISTO") else camera
        fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))
        coeffs = tuple(float(distortion[key]) for key in ("k1", "k2", "p1", "p2", "k3"))
        extra = tuple(key in distortion for key in ("k4", "k5", "k6"))
        if any(extra) and not all(extra):
            raise ValueError("k4、k5、k6 必須全部提供")
        if all(extra):
            coeffs += tuple(float(distortion[key]) for key in ("k4", "k5", "k6"))
        if any(float(distortion.get(key, 0.0)) != 0.0 for key in ("s1", "s2", "s3", "s4")):
            raise ValueError("不支援非零 thin-prism 畸變係數")
    except (ConfigError, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"無法讀取 ZED 左眼原廠標定檔 {path}: {exc}") from exc
    if (not all(math.isfinite(value) for value in (fx, fy, cx, cy, *coeffs))
            or fx <= 0 or fy <= 0
            or not 0 <= cx < config.image_width_px
            or not 0 <= cy < config.image_height_px):
        raise ValueError("ZED 左眼標定參數無效或與 1280×720 解析度不符")
    return replace(config, focal_length_x_px=fx, focal_length_y_px=fy,
                   camera_principal_x_px=cx, camera_principal_y_px=cy,
                   camera_distortion_coeffs=coeffs)

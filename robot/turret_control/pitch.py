"""Conversions between physical muzzle elevation and vertical motor angle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class PitchCalibration:
    """Calibrated relationship between muzzle elevation and motor position.

    ``motor_zero_rad`` is the motor feedback angle when the muzzle is level.
    ``motor_rad_per_muzzle_rad`` supports a geared or linked mechanism; it is
    1.0 for a direct-drive axis.  ``motor_sign`` is 1 when increasing motor
    position raises the muzzle and -1 when it lowers it.
    """

    motor_zero_rad: float
    motor_rad_per_muzzle_rad: float = 1.0
    motor_sign: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.motor_zero_rad):
            raise ValueError("motor_zero_rad must be finite")
        if (
            not math.isfinite(self.motor_rad_per_muzzle_rad)
            or self.motor_rad_per_muzzle_rad <= 0.0
        ):
            raise ValueError("motor_rad_per_muzzle_rad must be finite and positive")
        if self.motor_sign not in (-1, 1):
            raise ValueError("motor_sign must be either 1 or -1")

    def motor_from_muzzle(self, muzzle_rad: float) -> float:
        """Convert a physical muzzle elevation to an absolute motor angle."""
        if not math.isfinite(muzzle_rad):
            raise ValueError("muzzle_rad must be finite")
        return (
            self.motor_zero_rad
            + self.motor_sign * self.motor_rad_per_muzzle_rad * muzzle_rad
        )

    def muzzle_from_motor(self, motor_rad: float) -> float:
        """Convert an absolute motor angle to physical muzzle elevation."""
        if not math.isfinite(motor_rad):
            raise ValueError("motor_rad must be finite")
        return (
            (motor_rad - self.motor_zero_rad)
            / (self.motor_sign * self.motor_rad_per_muzzle_rad)
        )


def load_pitch_calibration(path: Path) -> PitchCalibration:
    """Load the physical muzzle-to-motor calibration from a startup pose."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    vertical = data["vertical"]
    zero_elevation_deg = float(vertical.get("muzzle_elevation_deg_at_angle", 0.0))
    motor_rad_per_muzzle_rad = float(vertical.get("motor_rad_per_muzzle_rad", 1.0))
    motor_sign = int(vertical.get("motor_sign", 1))
    motor_angle_at_reference = float(vertical["angle_rad"])
    if "level_reference_angle_rad" in vertical:
        # An inclinometer-measured level point is the authoritative zero;
        # the separate home position must never redefine that reference.
        motor_zero_rad = float(vertical["level_reference_angle_rad"])
    else:
        motor_zero_rad = (
            motor_angle_at_reference
            - motor_sign * motor_rad_per_muzzle_rad * math.radians(zero_elevation_deg)
        )
    return PitchCalibration(
        motor_zero_rad=motor_zero_rad,
        motor_rad_per_muzzle_rad=motor_rad_per_muzzle_rad,
        motor_sign=motor_sign,
    )


def load_yaw_calibration(path: Path) -> PitchCalibration:
    """Load physical turret yaw mapped onto the horizontal motor encoder."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    horizontal = data["horizontal"]
    reference_yaw_deg = float(horizontal.get("turret_yaw_deg_at_angle", 0.0))
    motor_rad_per_yaw_rad = float(horizontal.get("motor_rad_per_yaw_rad", 1.0))
    motor_sign = int(horizontal.get("motor_sign", 1))
    motor_angle_at_reference = float(horizontal["angle_rad"])
    motor_zero_rad = (
        motor_angle_at_reference
        - motor_sign * motor_rad_per_yaw_rad * math.radians(reference_yaw_deg)
    )
    return PitchCalibration(
        motor_zero_rad=motor_zero_rad,
        motor_rad_per_muzzle_rad=motor_rad_per_yaw_rad,
        motor_sign=motor_sign,
    )


def _save_calibration_data(path: Path, data: dict) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(str(path), str(backup_path))
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path


def save_level_reference_calibration(path: Path, pitch_motor_rad: float) -> Path:
    """Record an inclinometer-confirmed level position without moving home."""
    if not math.isfinite(pitch_motor_rad):
        raise ValueError("水平 0° 馬達位置必須是有限數字")
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    vertical = data["vertical"]
    calibration = PitchCalibration(
        float(pitch_motor_rad), float(vertical.get("motor_rad_per_muzzle_rad", 1.0)),
        int(vertical.get("motor_sign", -1)),
    )
    home_elevation = calibration.muzzle_from_motor(float(vertical["angle_rad"]))
    if home_elevation <= 0:
        raise ValueError("記錄的 home 必須高於水平 0°")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    vertical.update({
        "level_reference_angle_rad": float(pitch_motor_rad),
        "level_reference_captured_at": timestamp,
        "muzzle_elevation_deg_at_angle": math.degrees(home_elevation),
    })
    data["captured_at"] = timestamp
    data["note"] = "Inclinometer-confirmed level zero; home motor position and pitch ratio were preserved."
    return _save_calibration_data(path, data)


def save_manual_origin_calibration(
    path: Path,
    yaw_motor_rad: float,
    pitch_motor_rad: float,
    pitch_reference_deg: float = 45.0,
) -> Path:
    """Record yaw zero and home without changing an existing pitch level zero.

    The motor's internal encoder zero is deliberately left untouched.  The
    previous JSON file is copied to ``.bak`` and the replacement is written
    atomically so a failed write cannot leave a partial calibration file.
    """
    values = (yaw_motor_rad, pitch_motor_rad, pitch_reference_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("manual calibration values must be finite")
    if not 0.0 <= pitch_reference_deg <= 45.0:
        raise ValueError("pitch_reference_deg must be between 0 and 45")

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    horizontal = data["horizontal"]
    horizontal.update(
        {
            "angle_rad": float(yaw_motor_rad),
            "angle_deg": math.degrees(yaw_motor_rad),
            "turret_yaw_deg_at_angle": 0.0,
            "motor_sign": -1,
        }
    )
    vertical = data["vertical"]
    if "level_reference_angle_rad" in vertical:
        level = float(vertical["level_reference_angle_rad"])
        ratio = float(vertical.get("motor_rad_per_muzzle_rad", 1.0))
        pitch_reference_deg = math.degrees(
            PitchCalibration(level, ratio, -1).muzzle_from_motor(float(pitch_motor_rad))
        )
    vertical.update(
        {
            "angle_rad": float(pitch_motor_rad),
            "angle_deg": math.degrees(pitch_motor_rad),
            "muzzle_elevation_deg_at_angle": float(pitch_reference_deg),
            "motor_sign": -1,
        }
    )
    data["captured_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    data["source"] = "manual mechanism origin alignment with DAMIAO feedback"
    data["note"] = (
        "Manual home: horizontal turret 0 deg and inferred muzzle elevation "
        f"{pitch_reference_deg:.2f} deg. Measured level reference and encoder zeros are preserved."
    )

    return _save_calibration_data(path, data)

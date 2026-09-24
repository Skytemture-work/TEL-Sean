"""Laptop-safe demo for the turret aiming layer.

Examples:
    python -m turret_control.run_demo --u 720 --v 360
    python -m turret_control.run_demo --u 640 --v 260 --confidence 0.95
"""

import argparse

from .aiming import TargetPixel
from .backends import MockMotorBackend
from .config import TurretConfig
from .controller import TurretController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=float, default=640.0, help="target x pixel")
    parser.add_argument("--v", type=float, default=360.0, help="target y pixel")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    config = TurretConfig()
    motors = MockMotorBackend()
    turret = TurretController(config, motors)
    target = TargetPixel(args.u, args.v, args.confidence)

    # Feed several identical frames to demonstrate the firing stability gate.
    for frame_number in range(config.stable_frames_required):
        command = turret.update(target)
        print(
            f"frame={frame_number + 1} "
            f"yaw_error={command.yaw_error_rad:.4f} rad "
            f"pitch_error={command.pitch_error_rad:.4f} rad "
            f"stable={command.stable} "
            f"fire_allowed={command.fire_allowed}"
        )

    turret.close()


if __name__ == "__main__":
    main()

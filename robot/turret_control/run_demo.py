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
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="range to target in metres; without it the parallax/drop "
             "corrections cannot be applied and firing stays blocked",
    )
    args = parser.parse_args()

    config = TurretConfig()
    motors = MockMotorBackend(motor_ids=[config.yaw_motor_id, config.pitch_motor_id])
    turret = TurretController(config, motors)
    target = TargetPixel(args.u, args.v, args.confidence, args.distance)

    # Feed several identical frames to demonstrate the firing stability gate.
    # The first tick has no motor feedback yet, and with a base-fixed camera
    # arrival is judged from feedback, so allow a couple of extra frames.
    for frame_number in range(config.stable_frames_required + 2):
        command = turret.update(target)
        servo = (
            "servo=n/a"
            if command.yaw_servo_error_rad is None
            else (
                f"servo_yaw={command.yaw_servo_error_rad:+.4f} "
                f"servo_pitch={command.pitch_servo_error_rad:+.4f}"
            )
        )
        print(
            f"frame={frame_number + 1} "
            f"yaw_error={command.yaw_error_rad:.4f} rad "
            f"pitch_error={command.pitch_error_rad:.4f} rad "
            f"{servo} "
            f"reachable={command.target_reachable} "
            f"stable={command.stable} "
            f"fire_allowed={command.fire_allowed}"
            + (f" blocked={command.blocked_reason}" if command.blocked_reason else "")
        )

    turret.close()


if __name__ == "__main__":
    main()

"""Interactive manual move for the two DAMIAO turret motors.

Yaw is relative to the current feedback position.  Pitch is the absolute
physical muzzle elevation and is limited to 0..30 degrees.

Examples::

    python3 -m turret_control.run_manual_move
    python3 -m turret_control.run_manual_move --yaw 180 --pitch 15
    python3 -m turret_control.run_manual_move --pitch 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

from .backends import DamiaoSerialBackend
from .limits import load_motor_limits
from .pitch import load_pitch_calibration, load_yaw_calibration
from .trajectory import smoothstep_sample


DEFAULT_STARTUP_POSE = Path(__file__).with_name("startup_pose.json")
MIN_MUZZLE_ELEVATION_DEG = 0.0
MAX_MUZZLE_ELEVATION_DEG = 30.0


def angle_or_prompt(value: Optional[float], label: str) -> float:
    if value is not None:
        return value
    while True:
        try:
            return float(input(f"{label} 旋轉角度（度，正負代表方向）：").strip())
        except ValueError:
            print("請輸入數字，例如 180、-90 或 0。")


def main() -> None:
    parser = argparse.ArgumentParser(description="輸入水平／垂直角度，讓兩顆馬達同步轉動")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--yaw", type=float, help="水平相對角度，單位度")
    parser.add_argument("--pitch", type=float, help="物理砲口絕對仰角，0 到 30 度")
    parser.add_argument("--startup-pose", type=Path, default=DEFAULT_STARTUP_POSE)
    parser.add_argument("--speed", type=float, default=0.30, help="最大速度 rad/s")
    parser.add_argument("--kp", type=float, default=30.0, help="MIT 位置增益")
    parser.add_argument("--kd", type=float, default=1.0, help="MIT 速度阻尼")
    parser.add_argument("--yaw-tau-ff", type=float, default=0.0, help="水平力矩前饋 Nm")
    parser.add_argument("--pitch-tau-ff", type=float, default=0.0, help="垂直力矩前饋 Nm")
    args = parser.parse_args()

    yaw_deg = angle_or_prompt(args.yaw, "水平")
    pitch_deg = angle_or_prompt(args.pitch, "垂直砲口目標")
    if args.speed <= 0:
        raise SystemExit("--speed 必須大於 0")
    if args.kp < 0 or args.kd <= 0:
        raise SystemExit("--kp 必須 >= 0，--kd 必須 > 0")
    if not math.isfinite(pitch_deg) or not (
        MIN_MUZZLE_ELEVATION_DEG
        <= pitch_deg
        <= MAX_MUZZLE_ELEVATION_DEG
    ):
        raise SystemExit("垂直砲口仰角必須在 0° 到 30° 之間")

    yaw_delta = math.radians(yaw_deg)
    pitch_calibration = load_pitch_calibration(args.startup_pose)
    pitch_target = pitch_calibration.motor_from_muzzle(math.radians(pitch_deg))
    motors = DamiaoSerialBackend(
        motor_ids=[0x03, 0x01],
        port=args.port,
        mode="mit",
        kp=args.kp,
        kd=args.kd,
        enable_on_start=False,
        position_limits=load_motor_limits(),
        yaw_calibration=load_yaw_calibration(args.startup_pose),
        pitch_calibration=pitch_calibration,
    )
    try:
        motors.refresh()
        yaw = motors.get_state(0x03)
        pitch = motors.get_state(0x01)
        if yaw is None or pitch is None:
            raise RuntimeError("讀不到兩顆馬達回饋")

        start = {0x03: yaw.position_rad, 0x01: pitch.position_rad}
        target = {0x03: start[0x03] + yaw_delta, 0x01: pitch_target}
        pitch_delta = target[0x01] - start[0x01]
        print(
            f"水平相對 {yaw_deg:+.1f}°，砲口目標 {pitch_deg:+.1f}°；"
            f"垂直馬達目標 {pitch_target:+.6f} rad；"
            f"速度上限 {args.speed:.2f} rad/s；"
            f"力矩前饋 yaw={args.yaw_tau_ff:+.2f} Nm，pitch={args.pitch_tau_ff:+.2f} Nm"
        )
        input("確認機構安全後按 Enter 開始，Ctrl+C 可停止：")

        motors.enable()
        # smoothstep's peak speed is 1.5 * distance / duration.  The 1.5
        # factor keeps the requested dq feed-forward within --speed.
        duration = max(
            1.5 * max(abs(yaw_delta), abs(pitch_delta)) / args.speed,
            0.25,
        )
        start_time = time.monotonic()
        next_tick = start_time
        while True:
            elapsed = time.monotonic() - start_time
            yaw_position, yaw_velocity = smoothstep_sample(
                start[0x03], target[0x03], elapsed, duration
            )
            pitch_position, pitch_velocity = smoothstep_sample(
                start[0x01], target[0x01], elapsed, duration
            )
            motors.send_targets(
                {
                    0x03: (
                        yaw_position,
                        args.speed,
                        yaw_velocity,
                        args.yaw_tau_ff,
                    ),
                    0x01: (
                        pitch_position,
                        args.speed,
                        pitch_velocity,
                        args.pitch_tau_ff,
                    ),
                }
            )
            motors.refresh()
            if elapsed >= duration:
                break
            next_tick += 0.005
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        for _ in range(40):
            motors.send_targets(
                {
                    0x03: (target[0x03], args.speed, 0.0, args.yaw_tau_ff),
                    0x01: (target[0x01], args.speed, 0.0, args.pitch_tau_ff),
                }
            )
            motors.refresh()
            time.sleep(0.005)
        print("完成，兩顆馬達維持目標位置。")
    except KeyboardInterrupt:
        print("\n使用者中止，安全停止。")
    finally:
        motors.close()
        print("兩顆馬達已失能。")


if __name__ == "__main__":
    main()

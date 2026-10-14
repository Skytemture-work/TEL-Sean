"""Interactive manual move for the two DAMIAO turret motors.

Positive angles are the controller's positive direction; negative angles move
the opposite way.  Angles are relative to the current feedback position.

Examples::

    python3 -m turret_control.run_manual_move
    python3 -m turret_control.run_manual_move --yaw 180 --pitch 90
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

from .backends import DamiaoSerialBackend


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
    parser.add_argument("--pitch", type=float, help="垂直相對角度，單位度")
    parser.add_argument("--speed", type=float, default=0.30, help="最大速度 rad/s")
    args = parser.parse_args()

    yaw_deg = angle_or_prompt(args.yaw, "水平")
    pitch_deg = angle_or_prompt(args.pitch, "垂直")
    if args.speed <= 0:
        raise SystemExit("--speed 必須大於 0")

    yaw_delta = math.radians(yaw_deg)
    pitch_delta = math.radians(pitch_deg)
    motors = DamiaoSerialBackend(
        motor_ids=[0x03, 0x01],
        port=args.port,
        mode="mit",
        kp=5.0,
        kd=1.0,
        enable_on_start=False,
    )
    try:
        motors.refresh()
        yaw = motors.get_state(0x03)
        pitch = motors.get_state(0x01)
        if yaw is None or pitch is None:
            raise RuntimeError("讀不到兩顆馬達回饋")

        start = {0x03: yaw.position_rad, 0x01: pitch.position_rad}
        target = {0x03: start[0x03] + yaw_delta, 0x01: start[0x01] + pitch_delta}
        print(
            f"水平 {yaw_deg:+.1f}°，垂直 {pitch_deg:+.1f}°；"
            f"速度上限 {args.speed:.2f} rad/s"
        )
        input("確認機構安全後按 Enter 開始，Ctrl+C 可停止：")

        motors.enable()
        duration = max(abs(yaw_delta), abs(pitch_delta)) / args.speed + 1.0
        start_time = time.monotonic()
        next_tick = start_time
        while time.monotonic() - start_time < duration:
            elapsed = time.monotonic() - start_time
            progress = min(elapsed / max(duration - 1.0, 0.1), 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            motors.set_position(0x03, start[0x03] + yaw_delta * smooth, args.speed)
            motors.set_position(0x01, start[0x01] + pitch_delta * smooth, args.speed)
            motors.refresh()
            next_tick += 0.005
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        for _ in range(40):
            motors.set_position(0x03, target[0x03], args.speed)
            motors.set_position(0x01, target[0x01], args.speed)
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

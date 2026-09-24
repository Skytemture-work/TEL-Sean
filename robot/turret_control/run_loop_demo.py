"""Fixed-rate control loop demo, mock motors only -- nothing moves.

Shows the two rates running side by side: a slow "detector" thread standing in
for YOLO, and the 200 Hz motor loop that never waits for it.

    python3 -m turret_control.run_loop_demo
    python3 -m turret_control.run_loop_demo --detector-hz 5 --drop-after 1.0
"""

from __future__ import annotations

import argparse
import time

from .aiming import TargetPixel
from .backends import MockMotorBackend
from .config import TurretConfig
from .control_loop import ControlLoop, DetectionThread, LatestDetection
from .controller import TurretController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--detector-hz", type=float, default=30.0,
                        help="stand-in for YOLO; far slower than the motor loop")
    parser.add_argument("--u", type=float, default=900.0)
    parser.add_argument("--v", type=float, default=200.0)
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--drop-after", type=float, default=None,
                        help="stop detecting after N seconds to show hold()")
    args = parser.parse_args()

    config = TurretConfig()
    motors = MockMotorBackend(
        motor_ids=[config.yaw_motor_id, config.pitch_motor_id], verbose=False
    )
    turret = TurretController(config, motors)
    slot = LatestDetection(max_age_s=0.2)

    started = time.monotonic()
    target = TargetPixel(args.u, args.v, 0.95, args.distance)

    def detector():
        time.sleep(1.0 / args.detector_hz)
        if args.drop_after is not None and time.monotonic() - started > args.drop_after:
            return None
        return target

    loop = ControlLoop(turret, slot, rate_hz=args.rate_hz)
    print(f"馬達迴圈 {args.rate_hz:.0f} Hz ／ 偵測 ~{args.detector_hz:.0f} Hz，"
          f"跑 {args.seconds:.1f} 秒")

    with DetectionThread(detector, slot):
        stats = loop.run(duration_s=args.seconds)

    print(f"\n迴圈實測      {stats.actual_hz:.1f} Hz（目標 {args.rate_hz:.0f}）")
    print(f"來不及的圈數  {stats.late_ticks}/{stats.ticks}")
    print(f"偵測發布次數  {slot.publish_count}  →  約 {slot.publish_count / stats.elapsed_s:.1f} Hz")
    print(f"沒有目標的圈  {stats.stale_ticks}/{stats.ticks}")
    print(f"可開火的圈    {stats.fire_ticks}/{stats.ticks}")
    print(f"送出的馬達幀  {len(motors.commands)}  "
          f"（每圈 {len(motors.commands) / stats.ticks:.1f} 幀，兩顆馬達）")

    turret.close()


if __name__ == "__main__":
    main()

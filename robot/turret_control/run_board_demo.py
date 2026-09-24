"""Full board engagement demo, mock motors only -- nothing moves, nothing fires.

Simulates the 3x4 board, builds the plan, and works through all 12 holes at a
real 200 Hz loop rate.

    python3 -m turret_control.run_board_demo
    python3 -m turret_control.run_board_demo --distance 4 --cooldown 0.3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .backends import MockMotorBackend
from .board import BoardTarget, build_plan
from .config import TurretConfig
from .controller import TurretController
from .engagement import BoardEngagement


def fake_board(config: TurretConfig, distance_m: float):
    """Three columns of four holes: small on top, large below."""
    focal = config.focal_length_x_px
    small = focal * config.small_hole_diameter_m / distance_m
    large = focal * config.large_hole_diameter_m / distance_m
    targets = []
    for u in (400.0, 640.0, 880.0):
        targets.append(BoardTarget(u, 150.0, small, small, 0.9))
        for v in (300.0, 450.0, 600.0):
            targets.append(BoardTarget(u, v, large, large, 0.9))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, default=6.0, help="board range in metres")
    parser.add_argument("--cooldown", type=float, default=0.5, help="seconds between shots")
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="append fired/skipped events as JSONL to this path",
    )
    args = parser.parse_args()

    config = TurretConfig()
    motors = MockMotorBackend(
        motor_ids=[config.yaw_motor_id, config.pitch_motor_id], verbose=False
    )
    turret = TurretController(config, motors)

    plan = build_plan(
        fake_board(config, args.distance),
        config.focal_length_x_px,
        config.small_hole_diameter_m,
        config.large_hole_diameter_m,
        config.engage_large_first,
        config.engage_left_to_right,
        config.engage_bottom_to_top,
    )
    print(f"靶板 {len(plan)} 個孔，接戰順序：\n{plan.describe()}\n")

    shots = []
    driver = BoardEngagement(
        config, turret, plan,
        fire=lambda target: shots.append((time.monotonic(), target)) or True,
        shot_cooldown_s=args.cooldown,
        event_log_path=args.log,
    )

    dt = 1.0 / args.rate_hz
    ticks = late = 0
    start = next_deadline = time.monotonic()
    while not driver.update().finished:
        ticks += 1
        next_deadline += dt
        slack = next_deadline - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        else:
            late += 1
            next_deadline = time.monotonic()
        if time.monotonic() - start > 60.0:
            print("逾時，中止")
            break

    elapsed = time.monotonic() - start
    print(f"打完 {len(shots)}/{len(plan)} 個孔，耗時 {elapsed:.2f} 秒")
    print(f"迴圈 {ticks / elapsed:.1f} Hz，來不及的圈數 {late}/{ticks}")
    print(f"送出的馬達幀 {len(motors.commands)}（每圈 {len(motors.commands)/max(ticks,1):.1f}）\n")
    print("射擊順序：")
    for index, (stamp, target) in enumerate(shots, 1):
        print(f"  {index:2d}. t={stamp - start:5.2f}s  {target.hole_id or 'H??'} "
              f"col={target.column} row={target.row} "
              f"v={target.v_px:5.1f}  Ø{target.diameter_m * 100:.0f}cm "
              f"range={target.distance_m:.2f}m")
    driver.close()
    turret.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify both DAMIAO motors on one USB-to-CAN bus.

The default mode is read-only: it probes both ESC IDs, reads each motor's
parameters and MST_ID, and samples the two feedback routes without enabling
the motors. Use --enable only after both motors are clamped safely; in that
mode the script holds each motor at its current position at 200 Hz.

Examples on AGX Xavier:

    # Probe and read feedback. No enable and no commanded motion.
    python3 04_dual_check.py

    # Also enable both motors and continuously hold their present positions.
    python3 04_dual_check.py --enable --duration 10

The two IDs currently measured on this robot are 0x03 (yaw) and 0x01
(pitch). The backend reads MST_ID from each motor instead of assuming that
the two feedback IDs are equal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# This script lives beside DM_CAN.py, while turret_control is its sibling.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turret_control.backends import DamiaoSerialBackend  # noqa: E402


DEFAULT_IDS = (0x03, 0x01)  # yaw, pitch
DEFAULT_RATE_HZ = 200.0


def parse_id(value: str) -> int:
    motor_id = int(value, 0)
    if not 0 < motor_id <= 0x7F:
        raise argparse.ArgumentTypeError("ESC_ID must be in 0x01..0x7F")
    return motor_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe and optionally hold two DAMIAO motors together."
    )
    parser.add_argument(
        "--ids",
        nargs=2,
        type=parse_id,
        default=DEFAULT_IDS,
        metavar=("YAW_ID", "PITCH_ID"),
        help="two ESC_ID values; default: 0x03 0x01",
    )
    parser.add_argument("--port", default=None, help="serial port; default: auto-detect")
    parser.add_argument("--mode", choices=("posvel", "mit"), default="posvel")
    parser.add_argument("--kp", type=float, default=30.0, help="MIT position kp")
    parser.add_argument("--kd", type=float, default=1.0, help="MIT position kd")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="seconds to sample/hold; use 0 to run until Ctrl+C",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="enable both motors and hold current positions; default is read-only",
    )
    return parser


def print_state(backend: DamiaoSerialBackend, motor_ids) -> None:
    states = []
    for motor_id in motor_ids:
        state = backend.get_state(motor_id)
        if state is None:
            states.append(f"0x{motor_id:02X}: NO FEEDBACK")
        else:
            states.append(
                f"0x{motor_id:02X}: "
                f"pos={state.position_rad:+.4f} rad "
                f"vel={state.velocity_rad_s:+.4f} rad/s "
                f"tau={state.torque_nm:+.3f} Nm"
            )
    print(" | ".join(states), flush=True)


def run(args: argparse.Namespace) -> int:
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be zero or positive")
    if args.enable and args.mode == "mit" and args.kd <= 0:
        raise ValueError("--kd must be positive in MIT mode")

    motor_ids = tuple(args.ids)
    print(f"ESC_ID: yaw=0x{motor_ids[0]:02X}, pitch=0x{motor_ids[1]:02X}")
    print("建立連線並讀取兩顆馬達參數...")

    backend = DamiaoSerialBackend(
        motor_ids=motor_ids,
        port=args.port,
        mode=args.mode,
        kp=args.kp,
        kd=args.kd,
        enable_on_start=False,
    )

    try:
        print(f"序列埠：{backend.port}")
        print(
            "MST_ID: "
            + ", ".join(
                f"0x{motor_id:02X}->0x{backend.master_ids[motor_id]:02X}"
                for motor_id in motor_ids
            )
        )
        print(
            f"映射範圍：PMAX={backend.pmax}, "
            f"VMAX={backend.vmax}, TMAX={backend.tmax}"
        )

        backend.refresh()
        print_state(backend, motor_ids)

        if args.enable:
            print("即將使能；只會保持目前位置，不會執行掃動。")
            backend.enable()
            print("兩顆馬達已使能，開始 200 Hz hold。Ctrl+C 可停止。")
        else:
            print("唯讀模式：不使能馬達，只讀取回饋。")

        period = 1.0 / args.rate
        start = time.monotonic()
        next_tick = start
        last_print = start - 1.0
        ticks = 0
        late_ticks = 0

        while args.duration == 0 or time.monotonic() - start < args.duration:
            now = time.monotonic()
            if args.enable:
                backend.refresh()
                backend.hold()
            elif now - last_print >= 0.5:
                backend.refresh()

            if now - last_print >= 0.5:
                print_state(backend, motor_ids)
                last_print = now

            ticks += 1
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late_ticks += 1
                next_tick = time.monotonic()

        elapsed = max(time.monotonic() - start, 1e-9)
        print(
            f"完成：{ticks} ticks，實測 {ticks / elapsed:.1f} Hz，"
            f"逾時 {late_ticks}/{ticks}"
        )
        return 0
    except KeyboardInterrupt:
        print("\n使用者中斷，進行安全收尾。")
        return 130
    finally:
        backend.close()
        print("已失能兩顆馬達並關閉序列埠。")


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"dual-check failed: {exc}") from exc


if __name__ == "__main__":
    main()

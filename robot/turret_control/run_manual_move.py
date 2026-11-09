"""Interactive manual move for the two DAMIAO turret motors.

Yaw is relative to the current feedback position.  Pitch is the absolute
physical muzzle elevation and is limited to 0..35 degrees in the normal
activity region.

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
MAX_MUZZLE_ELEVATION_DEG = 35.0


def angle_or_prompt(value: Optional[float], label: str) -> float:
    if value is not None:
        return value
    while True:
        try:
            return float(input(f"{label} 旋轉角度（度，正負代表方向）：").strip())
        except ValueError:
            print("請輸入數字，例如 180、-90 或 0。")


def stop_and_disable(motors) -> None:
    """僅在使用者停止或控制異常結束時失能並關閉連線。"""
    print("[SOFTWARE DISABLE] 現在開始執行程式失能指令。", flush=True)
    motors.close()
    print("兩顆馬達已失能，連線已關閉。", flush=True)


def read_enabled_start(motors):
    """使能後等待兩軸的新回授，避免以未更新的預設位置建立軌跡。"""
    requested_at = time.monotonic()
    while time.monotonic() - requested_at < 1.0:
        # 等待回授時持續送出啟動保持指令，避免通訊逾時。
        motors.hold()
        motors.refresh()
        now = time.monotonic()
        states = {i: motors.get_state(i) for i in (0x03, 0x01)}
        if all(
            s is not None and s.err == 1
            and s.feedback_at is not None
            and requested_at <= s.feedback_at <= now
            and now - s.feedback_at <= 0.1
            and math.isfinite(s.position_rad)
            for s in states.values()
        ):
            return {i: s.position_rad for i, s in states.items()}
        time.sleep(0.005)
    raise RuntimeError("使能後未收到兩軸有效的新回授，取消移動")


def hold_position_forever(motors, target, speed, kp, yaw_tau_ff, pitch_tau_ff,
                          stop_event=None, ready_event=None, update_targets=None,
                          yaw_kp=None, pitch_kp=None, yaw_kd=None, pitch_kd=None) -> None:
    """持續送出保持指令；不呼叫 close/disable，也沒有到時退出條件。"""
    motors.kp = kp
    motors.set_motor_gains({
        0x03: (kp if yaw_kp is None else yaw_kp, motors.kd if yaw_kd is None else yaw_kd),
        0x01: (kp if pitch_kp is None else pitch_kp, motors.kd if pitch_kd is None else pitch_kd),
    })
    targets = {
        0x03: (target[0x03], speed, 0.0, yaw_tau_ff),
        0x01: (target[0x01], speed, 0.0, pitch_tau_ff),
    }
    next_status = time.monotonic()
    while stop_event is None or not stop_event.is_set():
        # 先維持控制並收取回授，再檢查狀態；避免其他模組啟動後
        # 執行緒恢復時，直接拿上一輪快取的回授判定故障。
        motors.send_targets(targets)
        motors.refresh()
        if update_targets is not None:
            targets = update_targets(motors, targets)
        if ready_event is not None:
            ready_event.set()
        now = time.monotonic()
        if now >= next_status:
            for motor_id in (0x03, 0x01):
                state = motors.get_state(motor_id)
                if state is None:
                    print(f"[KEEP ENABLED] ID=0x{motor_id:02X} 無回授", flush=True)
                    continue
                age = (
                    f"{now - state.feedback_at:.3f}s"
                    if state.feedback_at is not None else "未知"
                )
                print(
                    f"[KEEP ENABLED] ID=0x{motor_id:02X} err={state.err} "
                    f"請求目標={targets[motor_id][0]:+.5f}rad "
                    f"實際={state.position_rad:+.5f}rad "
                    f"力矩={state.torque_nm:+.2f}Nm "
                    f"Kp={motors.gains_for(motor_id)[0]:g} "
                    f"Kd={motors.gains_for(motor_id)[1]:g} 回授距今={age}",
                    flush=True,
                )
            next_status = now + 1.0
        time.sleep(0.005)


def main_keep_enabled(argv=None, *, confirm=True, stop_event=None, ready_event=None,
                      on_shutdown=None, update_targets=None) -> None:
    parser = argparse.ArgumentParser(description="輸入水平／垂直角度，讓兩顆馬達同步轉動")
    # Auto-detect the DAMIAO USB-to-CAN board by VID/PID. /dev/ttyACM* order
    # changes with USB enumeration and can point at the UART/VESC instead.
    parser.add_argument("--port", default=None)
    parser.add_argument("--yaw", type=float, help="水平相對角度，單位度")
    parser.add_argument("--pitch", type=float, help="物理砲口絕對仰角，0 到 35 度")
    parser.add_argument("--startup-pose", type=Path, default=DEFAULT_STARTUP_POSE)
    parser.add_argument("--speed", type=float, default=0.30, help="最大速度 rad/s")
    parser.add_argument("--kp", type=float, default=30.0, help="到位後的 MIT 保持位置增益")
    parser.add_argument("--yaw-kp", type=float, default=None, help="水平軸保持 Kp；預設沿用 --kp")
    parser.add_argument("--pitch-kp", type=float, default=None, help="垂直軸保持 Kp；預設沿用 --kp")
    parser.add_argument("--yaw-kd", type=float, default=None, help="水平軸保持 Kd；預設沿用 --kd")
    parser.add_argument("--pitch-kd", type=float, default=None, help="垂直軸保持 Kd；預設沿用 --kd")
    parser.add_argument("--move-kp", type=float, default=20.0, help="移動時的 MIT 位置增益")
    parser.add_argument("--kd", type=float, default=1.0, help="MIT 速度阻尼")
    # 相容 manual.py；不論有無此參數，到位後一律持續保持。
    parser.add_argument("--hold", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--yaw-tau-ff", type=float, default=0.0, help="水平力矩前饋 Nm")
    parser.add_argument("--pitch-tau-ff", type=float, default=3.0, help="垂直支撐力矩 Nm，移動時漸增、保持時持續施加")
    args = parser.parse_args(argv)

    yaw_deg = angle_or_prompt(args.yaw, "水平")
    pitch_deg = angle_or_prompt(args.pitch, "垂直砲口目標")
    if args.speed <= 0:
        raise SystemExit("--speed 必須大於 0")
    if not all(math.isfinite(v) for v in (args.yaw_tau_ff, args.pitch_tau_ff)):
        raise SystemExit("力矩前饋必須是有限數值")
    hold_gains = (
        args.kp if args.yaw_kp is None else args.yaw_kp,
        args.kp if args.pitch_kp is None else args.pitch_kp,
        args.kd if args.yaw_kd is None else args.yaw_kd,
        args.kd if args.pitch_kd is None else args.pitch_kd,
    )
    if (not all(math.isfinite(v) and 0 <= v <= 500 for v in (args.kp, args.move_kp, *hold_gains[:2]))
            or not all(math.isfinite(v) and 0 < v <= 5 for v in (args.kd, *hold_gains[2:]))):
        raise SystemExit("--kp / --move-kp 必須在 0..500，--kd 必須在 (0,5]")
    if not math.isfinite(pitch_deg) or not (
        MIN_MUZZLE_ELEVATION_DEG
        <= pitch_deg
        <= MAX_MUZZLE_ELEVATION_DEG
    ):
        raise SystemExit("垂直砲口仰角必須在 0° 到 35° 之間")

    yaw_delta = math.radians(yaw_deg)
    pitch_calibration = load_pitch_calibration(args.startup_pose)
    pitch_target = pitch_calibration.motor_from_muzzle(math.radians(pitch_deg))
    motors = DamiaoSerialBackend(
        motor_ids=[0x03, 0x01],
        port=args.port,
        mode="mit",
        kp=args.move_kp,
        kd=args.kd,
        enable_on_start=False,
        position_limits=load_motor_limits(),
        yaw_calibration=load_yaw_calibration(args.startup_pose),
        pitch_calibration=pitch_calibration,
    )
    try:
        if max(abs(args.yaw_tau_ff), abs(args.pitch_tau_ff)) > motors.tmax:
            raise ValueError(f"力矩前饋不可超過馬達上限 {motors.tmax:g} Nm")
        print(
            f"水平相對 {yaw_deg:+.1f}°，砲口目標 {pitch_deg:+.1f}°；"
            f"垂直馬達目標 {pitch_target:+.6f} rad；"
            f"速度上限 {args.speed:.2f} rad/s；"
            f"力矩前饋 yaw={args.yaw_tau_ff:+.2f} Nm，pitch={args.pitch_tau_ff:+.2f} Nm"
        )
        if confirm:
            input("確認機構安全後按 Enter 開始，Ctrl+C 可停止：")

        motors.enable()
        start = read_enabled_start(motors)
        target = {0x03: start[0x03] + yaw_delta, 0x01: pitch_target}
        pitch_delta = target[0x01] - start[0x01]
        print(
            f"[START] 使用使能後新回授：yaw={start[0x03]:+.5f}rad "
            f"pitch={start[0x01]:+.5f}rad",
            flush=True,
        )
        # smoothstep's peak speed is 1.5 * distance / duration.  The 1.5
        # factor keeps the requested dq feed-forward within --speed.
        duration = max(
            1.5 * max(abs(yaw_delta), abs(pitch_delta)) / args.speed,
            1.0,  # 支撐力矩至少用一秒漸增，避免瞬間加入。
        )
        start_time = time.monotonic()
        next_tick = start_time
        next_move_status = start_time
        print(f"[MOVE] 預計軌跡時間 {duration:.2f}s", flush=True)
        while True:
            if stop_event is not None and stop_event.is_set():
                stop_and_disable(motors)
                return
            elapsed = time.monotonic() - start_time
            yaw_position, yaw_velocity = smoothstep_sample(
                start[0x03], target[0x03], elapsed, duration
            )
            pitch_position, pitch_velocity = smoothstep_sample(
                start[0x01], target[0x01], elapsed, duration
            )
            pitch_support, _ = smoothstep_sample(
                0.0, args.pitch_tau_ff, elapsed, duration
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
                        pitch_support,
                    ),
                }
            )
            motors.refresh()
            now = time.monotonic()
            if now >= next_move_status:
                for motor_id, requested in ((0x03, yaw_position), (0x01, pitch_position)):
                    state = motors.get_state(motor_id)
                    if state is None:
                        print(f"[MOVE] ID=0x{motor_id:02X} 無回授", flush=True)
                        continue
                    age = (
                        f"{now - state.feedback_at:.3f}s"
                        if state.feedback_at is not None else "未知"
                    )
                    print(
                        f"[MOVE] t={elapsed:.2f}/{duration:.2f}s "
                        f"ID=0x{motor_id:02X} err={state.err} "
                        f"請求目標={requested:+.5f}rad "
                        f"實際={state.position_rad:+.5f}rad "
                        f"力矩={state.torque_nm:+.2f}Nm "
                        f"Kp={motors.kp:g} 回授距今={age}",
                        flush=True,
                    )
                next_move_status = now + 1.0
            if elapsed >= duration:
                break
            next_tick += 0.005
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        print(
            "軌跡完成，保持增益 yaw Kp/Kd={:.1f}/{:.1f}、pitch Kp/Kd={:.1f}/{:.1f}；"
            "持續保持，按 Ctrl+C 才停止並失能。".format(
                hold_gains[0], hold_gains[2], hold_gains[1], hold_gains[3]
            ),
            flush=True,
        )
        hold_position_forever(
            motors, target, args.speed, args.kp, args.yaw_tau_ff, args.pitch_tau_ff,
            stop_event=stop_event, ready_event=ready_event,
            update_targets=update_targets,
            yaw_kp=hold_gains[0], pitch_kp=hold_gains[1],
            yaw_kd=hold_gains[2], pitch_kd=hold_gains[3],
        )
    except KeyboardInterrupt:
        print("\n使用者中止，安全停止。")
        stop_and_disable(motors)
    except BaseException as exc:
        print(f"控制發生錯誤，停止並失能：{exc}", flush=True)
        stop_and_disable(motors)
        raise
    else:
        try:
            if on_shutdown is not None:
                on_shutdown(motors)
        finally:
            stop_and_disable(motors)


def main() -> None:
    # manual.py 與命令列統一使用不會到位後自動失能的版本。
    main_keep_enabled()


if __name__ == "__main__":
    main()

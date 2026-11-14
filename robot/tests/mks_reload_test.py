#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 can0 上依序測試 MKS CAN ID 1、2，由終端機輸入各自的步數。"""

import time

import can
from mks_servo_can import MksServo
from mks_servo_can.mks_enums import (
    Direction,
    Enable,
    MksCommands,
    SuccessStatus,
    WorkMode,
)

CAN_INTERFACE = "can0"
MOTOR_IDS = (1, 2)
DEFAULT_PULSES_BY_ID = {1: 0, 2: 6670}
TEST_SPEED_RPM = 1500
ACCELERATION = 1
MOVE_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.1


def read_steps(motor_id, default_steps):
    """讀取單顆馬達的非負整數步數。"""
    while True:
        raw_steps = input(
            f"輸入 CAN ID {motor_id} 要移動的 steps（Enter 使用 {default_steps}）："
        ).strip()
        if not raw_steps:
            return default_steps

        try:
            steps = int(raw_steps)
        except ValueError:
            print("請輸入整數 steps。")
            continue

        if steps < 0:
            print("steps 不能是負數。")
            continue
        return steps


def set_enabled(servo, enabled):
    value = Enable.Enable.value if enabled else Enable.Disable.value
    return servo.set_generic_status(MksCommands.ENABLE_MOTOR_COMMAND, [value])


def test_motor(bus, notifier, motor_id, steps):
    servo = MksServo(bus, notifier, motor_id)
    servo.timeout = 1.0
    enabled = False

    try:
        mode_result = servo.set_work_mode(WorkMode.SrvFoc)
        print(f"ID {motor_id} 工作模式: {mode_result}", flush=True)
        if mode_result != SuccessStatus.Success:
            raise RuntimeError("設定工作模式失敗，未送移動命令")

        enable_result = set_enabled(servo, True)
        print(f"ID {motor_id} 啟用: {enable_result}", flush=True)
        if enable_result != SuccessStatus.Success:
            raise RuntimeError("啟用失敗，未送移動命令")
        enabled = True

        before = servo.read_encoder_value_addition()
        print(f"ID {motor_id} 移動前 encoder: {before}", flush=True)

        result = servo.run_motor_relative_motion_by_pulses(
            Direction.CW,
            TEST_SPEED_RPM,
            ACCELERATION,
            steps,
        )
        print(f"ID {motor_id} 移動命令回覆: {result}", flush=True)
        if result is None:
            raise RuntimeError("沒有收到移動命令回覆")

        deadline = time.monotonic() + MOVE_TIMEOUT_S
        running = True
        while time.monotonic() < deadline:
            running = servo.is_motor_running()
            if not running:
                break
            time.sleep(POLL_INTERVAL_S)

        if running:
            print(f"ID {motor_id} 逾時，送出停止命令", flush=True)
            servo.stop_motor_relative_motion_by_pulses(ACCELERATION)
            time.sleep(0.2)

        after = servo.read_encoder_value_addition()
        speed = servo.read_motor_speed()
        delta = None
        if before not in (None, -1) and after not in (None, -1):
            delta = after - before

        print(
            f"ID {motor_id} 完成：steps={steps} encoder_after={after} "
            f"delta={delta} speed={speed} RPM",
            flush=True,
        )
    finally:
        if enabled:
            try:
                result = set_enabled(servo, False)
                print(f"ID {motor_id} 停用: {result}", flush=True)
            except Exception as exc:
                print(f"ID {motor_id} 停用命令錯誤: {exc}", flush=True)


def main():
    print(
        f"MKS movement test: CAN={CAN_INTERFACE}, motors={MOTOR_IDS}, "
        f"speed={TEST_SPEED_RPM} RPM",
        flush=True,
    )
    input("確認機構周圍安全後按 Enter，開始輸入各馬達的 steps：")

    steps_by_id = {
        motor_id: read_steps(motor_id, DEFAULT_PULSES_BY_ID[motor_id])
        for motor_id in MOTOR_IDS
    }
    print(f"本次設定 steps：{steps_by_id}", flush=True)
    input("確認設定正確且機構周圍安全後按 Enter，開始測試：")

    bus = can.Bus(interface="socketcan", channel=CAN_INTERFACE)
    notifier = can.Notifier(bus, [])
    try:
        for index, motor_id in enumerate(MOTOR_IDS):
            steps = steps_by_id[motor_id]
            print(f"\n===== 現在測試 CAN ID {motor_id}，steps={steps} =====", flush=True)
            test_motor(bus, notifier, motor_id, steps)

            if index + 1 < len(MOTOR_IDS):
                next_motor_id = MOTOR_IDS[index + 1]
                input(f"請記下剛才動的實體馬達，按 Enter 再測 ID {next_motor_id}：")

        print("\n兩個 ID 測試完成。", flush=True)
    finally:
        notifier.stop()
        bus.shutdown()


if __name__ == "__main__":
    main()

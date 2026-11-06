#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, sequential MKS CAN-node identification test.

Runs each configured node on can0 separately so the operator can identify
which physical motor each CAN ID controls.
"""

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
TEST_PULSES_BY_ID = {1: 0, 2: 6670}
TEST_SPEED_RPM = 1500
ACCELERATION = 1
MOVE_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.1


def set_enabled(servo, enabled):
    value = Enable.Enable.value if enabled else Enable.Disable.value
    return servo.set_generic_status(
        MksCommands.ENABLE_MOTOR_COMMAND,
        [value],
    )


def test_motor(bus, notifier, motor_id):
    servo = MksServo(bus, notifier, motor_id)
    servo.timeout = 1.0
    enabled = False
    move_started = False

    try:
        mode_result = servo.set_work_mode(WorkMode.SrvFoc)
        print("ID {} 工作模式: {}".format(motor_id, mode_result), flush=True)
        if mode_result != SuccessStatus.Success:
            raise RuntimeError("設定工作模式失敗，未送移動命令")

        enable_result = set_enabled(servo, True)
        print("ID {} 啟用: {}".format(motor_id, enable_result), flush=True)
        if enable_result != SuccessStatus.Success:
            raise RuntimeError("啟用失敗，未送移動命令")
        enabled = True

        before = servo.read_encoder_value_addition()
        print("ID {} 移動前 encoder: {}".format(motor_id, before), flush=True)

        result = servo.run_motor_relative_motion_by_pulses(
            Direction.CW,
            TEST_SPEED_RPM,
            ACCELERATION,
            TEST_PULSES_BY_ID[motor_id],
        )
        print("ID {} 移動命令回覆: {}".format(motor_id, result), flush=True)
        if result is None:
            raise RuntimeError("沒有收到移動命令回覆")
        move_started = True

        deadline = time.monotonic() + MOVE_TIMEOUT_S
        running = True
        while time.monotonic() < deadline:
            running = servo.is_motor_running()
            if not running:
                break
            time.sleep(POLL_INTERVAL_S)

        if running:
            print("ID {} 逾時，送出停止命令".format(motor_id), flush=True)
            servo.stop_motor_relative_motion_by_pulses(ACCELERATION)
            time.sleep(0.2)

        after = servo.read_encoder_value_addition()
        speed = servo.read_motor_speed()
        delta = None
        if before is not None and after is not None and before != -1 and after != -1:
            delta = after - before
        print(
            "ID {} 完成：encoder_after={} delta={} speed={} RPM".format(
                motor_id, after, delta, speed
            ),
            flush=True,
        )
    finally:
        if enabled:
            try:
                print("ID {} 停用: {}".format(
                    motor_id, set_enabled(servo, False)
                ), flush=True)
            except Exception as exc:
                print("ID {} 停用命令錯誤: {}".format(motor_id, exc), flush=True)


def main():
    print(
        "MKS node identification: CAN={}, pulses by ID={}, speed={} RPM".format(
            CAN_INTERFACE, TEST_PULSES_BY_ID, TEST_SPEED_RPM
        ),
        flush=True,
    )
    input("確認機構周圍安全後按 Enter，開始測試 ID 1：")

    bus = can.Bus(interface="socketcan", channel=CAN_INTERFACE)
    notifier = can.Notifier(bus, [])
    try:
        for index, motor_id in enumerate(MOTOR_IDS):
            print("\n===== 現在測試 CAN ID {} =====".format(motor_id), flush=True)
            test_motor(bus, notifier, motor_id)
            if index + 1 < len(MOTOR_IDS):
                input(
                    "請記下剛才動的實體馬達；按 Enter 再測 ID {}：".format(
                        MOTOR_IDS[index + 1]
                    )
                )
        print("\n兩個 ID 測試完成。", flush=True)
    finally:
        notifier.stop()
        bus.shutdown()


if __name__ == "__main__":
    main()

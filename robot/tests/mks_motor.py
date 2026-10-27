#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import can

from mks_servo_can import MksServo
from mks_servo_can.mks_enums import (
    Enable,
    Direction,
    WorkMode,
    SuccessStatus,
)

# ===== 使用者設定 =====
CAN_INTERFACE = "can0"
MOTOR_CAN_ID = 1

MOVE_PULSES = 800       # 相對移動脈衝數
TARGET_SPEED_RPM = 30   # 先用低速測試
ACCELERATION = 2
POLL_INTERVAL = 0.3
MOVE_TIMEOUT = 30       # 最長等待秒數

def enable_motor_compat(servo, enabled=True):
    """相容目前安裝的套件 enable_motor 參數問題。"""
    method_globals = servo.enable_motor.__func__.__globals__
    commands = method_globals["MksCommands"]

    value = Enable.Enable.value if enabled else Enable.Disable.value

    return servo.set_generic_status(
        commands.ENABLE_MOTOR_COMMAND,
        [value]
    )

def main():
    bus = None
    notifier = None
    servo = None
    move_started = False

    try:
        bus = can.Bus(
            interface="socketcan",
            channel=CAN_INTERFACE
        )
        notifier = can.Notifier(bus, [])
        servo = MksServo(bus, notifier, MOTOR_CAN_ID)

        # 1. 設定工作模式
        result = servo.set_work_mode(WorkMode.SrvFoc)
        print("工作模式：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("設定工作模式失敗")

        time.sleep(0.5)

        # 2. 啟用馬達
        result = enable_motor_compat(servo, True)
        print("啟用結果：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("啟用馬達失敗")

        time.sleep(0.5)

        # 3. 讀取移動前回授
        encoder_before = servo.read_encoder_value_addition()
        speed_before = servo.read_motor_speed()

        print("\n===== 移動前 =====")
        print("累積編碼器值：", encoder_before)
        print("實際轉速 RPM：", speed_before)

        if encoder_before is None:
            raise RuntimeError("讀取初始編碼器值失敗")

        # 4. 執行有限脈衝的相對位置移動
        print("\n開始移動 {} 脈衝".format(MOVE_PULSES))

        result = servo.run_motor_relative_motion_by_pulses(
            Direction.CW,
            TARGET_SPEED_RPM,
            ACCELERATION,
            MOVE_PULSES
        )
        print("移動指令回傳：", result)

        if result is None:
            raise RuntimeError("沒有收到有效的移動指令回覆")

        move_started = True
        start_time = time.monotonic()

        # 5. 持續讀取回授，直到停止或逾時
        while True:
            elapsed = time.monotonic() - start_time

            encoder_now = servo.read_encoder_value_addition()
            speed_now = servo.read_motor_speed()
            running = servo.is_motor_running()

            if encoder_now is None:
                print("警告：本次編碼器讀取失敗")
            else:
                delta = encoder_now - encoder_before
                print(
                    "時間={:.1f}s | Encoder={} | 位移={} | "
                    "RPM={} | 運轉中={}".format(
                        elapsed,
                        encoder_now,
                        delta,
                        speed_now,
                        running
                    )
                )

            if not running:
                print("\n馬達回報已停止。")
                break

            if elapsed >= MOVE_TIMEOUT:
                print("\n逾時：嘗試停止相對位置移動。")
                servo.stop_motor_relative_motion_by_pulses(
                    ACCELERATION
                )
                break

            time.sleep(POLL_INTERVAL)

        # 6. 讀取最終回授
        time.sleep(0.2)

        encoder_after = servo.read_encoder_value_addition()
        speed_after = servo.read_motor_speed()
        status_after = servo.query_motor_status()

        print("\n===== 最終回授 =====")
        print("移動前 Encoder：", encoder_before)
        print("移動後 Encoder：", encoder_after)

        if encoder_after is not None:
            print("實際編碼器位移：", encoder_after - encoder_before)

        print("最終 RPM：", speed_after)
        print("馬達狀態：", status_after)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，嘗試停止馬達。")
        if servo is not None and move_started:
            try:
                servo.stop_motor_relative_motion_by_pulses(
                    ACCELERATION
                )
            except Exception as exc:
                print("停止指令錯誤：", exc)

    except Exception as exc:
        print("\n程式錯誤：", exc)

    finally:
        if notifier is not None:
            notifier.stop()
        if bus is not None:
            bus.shutdown()

        print("CAN 連線已關閉。")

if __name__ == "__main__":
    main()

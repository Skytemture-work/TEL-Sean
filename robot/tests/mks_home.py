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
    MksCommands,
)

# ==============================
# 使用者設定
# ==============================

CAN_INTERFACE = "can0"
MOTOR_CAN_ID = 1

# 歸位方向：先確認 CW 是否朝向你的原點
HOME_DIRECTION = Direction.CW

# 無限位歸位速度，RPM
HOME_SPEED_RPM = 30

# SERVO42D_CAN 歸位電流，mA
# 200 mA 是低電流起始測試值，不保證足以推動負載
HOME_CURRENT_MA = 200

# 歸位碰到阻力後，反向退回的馬達角度
# 0x0400 = 22.5 度；0x2000 = 180 度
REVERSE_ANGLE = 0x0400

POLL_INTERVAL = 0.3
HOME_TIMEOUT = 30.0

def enable_motor_compat(servo, enabled=True):
    """相容目前安裝版本的 enable_motor 參數型態。"""
    method_globals = servo.enable_motor.__func__.__globals__
    commands = method_globals["MksCommands"]

    value = Enable.Enable.value if enabled else Enable.Disable.value

    return servo.set_generic_status(
        commands.ENABLE_MOTOR_COMMAND,
        [value]
    )

def set_no_limit_home_parameters(servo):
    """設定 0x90 的 noLimit 模式，以及 0x94 的歸位電流和退回角度。"""

    # 0x90: 設定歸位參數
    # homeTrig: 0 = Low（noLimit 模式下不使用限位開關）
    # homeDir: 0 = CW, 1 = CCW
    # homeSpeed: 0~3000 RPM
    # EndLimit: 0 = Disable
    # mode: 0 = Limit switch, 1 = noLimit

    speed = HOME_SPEED_RPM
    direction_value = HOME_DIRECTION.value

    home_data = [
        0,
        direction_value,
        (speed >> 8) & 0x0F,
        speed & 0xFF,
        0,
        1,
    ]

    print("\n[1] 設定 noLimit 歸位模式...")
    result = servo.set_generic_status(
        MksCommands.SET_HOME_COMMAND,
        home_data
    )
    print("0x90 設定結果：", result)

    if result != SuccessStatus.Success:
        raise RuntimeError("0x90 歸位模式設定失敗")

    # 0x94: 4-byte 退回角度 + 2-byte 歸位電流
    reverse_angle = REVERSE_ANGLE
    current = HOME_CURRENT_MA

    home_current_data = [
        (reverse_angle >> 24) & 0xFF,
        (reverse_angle >> 16) & 0xFF,
        (reverse_angle >> 8) & 0xFF,
        reverse_angle & 0xFF,
        (current >> 8) & 0xFF,
        current & 0xFF,
    ]

    print("\n[2] 設定歸位電流與退回角度...")
    result = servo.set_generic_status(
        MksCommands.SET_NO_LIMIT_GO_HOME_COMMAND,
        home_current_data
    )
    print("0x94 設定結果：", result)

    if result != SuccessStatus.Success:
        raise RuntimeError("0x94 歸位電流設定失敗")

def main():
    bus = None
    notifier = None
    servo = None
    homing_started = False
    homing_completed = False

    try:
        print("=" * 50)
        print("MKS SERVO42D_CAN 無限位回零測試")
        print("CAN interface:", CAN_INTERFACE)
        print("Motor CAN ID:", MOTOR_CAN_ID)
        print("歸位速度:", HOME_SPEED_RPM, "RPM")
        print("歸位電流:", HOME_CURRENT_MA, "mA")
        print("退回角度參數: 0x{:04X}".format(REVERSE_ANGLE))
        print("=" * 50)

        # 1. 初始化 CAN
        bus = can.Bus(
            interface="socketcan",
            channel=CAN_INTERFACE
        )
        notifier = can.Notifier(bus, [])
        servo = MksServo(bus, notifier, MOTOR_CAN_ID)

        # 2. 設定工作模式
        print("\n[0] 設定工作模式 SrvFoc...")
        result = servo.set_work_mode(WorkMode.SrvFoc)
        print("工作模式設定結果：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("工作模式設定失敗")

        time.sleep(0.5)

        # 3. 啟用馬達
        print("\n啟用馬達...")
        result = enable_motor_compat(servo, True)
        print("啟用結果：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("馬達啟用失敗")

        time.sleep(0.5)

        # 4. 讀取初始編碼器值
        encoder_before = servo.read_encoder_value_addition()
        print("\n歸位前編碼器值：", encoder_before)

        # 5. 設定 noLimit 歸位參數
        set_no_limit_home_parameters(servo)

        # 6. 開始歸位
        print("\n[3] 開始執行回零指令 0x91...")
        response = servo.set_generic(
            MksCommands.GO_HOME_COMMAND,
            servo.GENERIC_RESPONSE_LENGTH
        )

        if response is None:
            raise RuntimeError("0x91 沒有收到有效回覆")

        # 0x91 回傳：0 失敗、1 開始、2 成功
        home_status = int(response[1])
        print("0x91 初始回覆狀態：", home_status)

        if home_status == 0:
            raise RuntimeError("驅動器拒絕回零指令")

        if home_status not in (1, 2):
            raise RuntimeError(
                "未知的回零狀態：{}".format(home_status)
            )

        homing_started = True

        if home_status == 2:
            homing_completed = True
            print("驅動器回報回零成功。")

        # 7. 監控馬達狀態與編碼器
        start_time = time.monotonic()
        seen_running = False
        last_encoder = encoder_before
        no_motion_since = start_time

        while not homing_completed:
            elapsed = time.monotonic() - start_time

            if elapsed >= HOME_TIMEOUT:
                raise TimeoutError(
                    "回零逾時，請檢查方向、電流與機構是否卡住"
                )

            running = servo.is_motor_running()
            encoder_now = servo.read_encoder_value_addition()

            if running:
                seen_running = True

            if encoder_now is not None:
                if encoder_now != last_encoder:
                    last_encoder = encoder_now
                    no_motion_since = time.monotonic()

            print(
                "時間={:.1f}s | 運轉中={} | Encoder={}".format(
                    elapsed, running, encoder_now
                )
            )

            # 只有確認曾經運轉，後來才停止，才視為完成候選
            if seen_running and not running:
                homing_completed = True
                break

            # 初始回覆為開始，但長時間沒有運轉或編碼器變化
            # 不直接當作成功，避免誤判
            if (
                not seen_running
                and elapsed > 2.0
                and not running
                and encoder_now == encoder_before
            ):
                raise RuntimeError(
                    "未偵測到馬達開始移動；停止判定，請檢查方向與驅動器設定"
                )

            time.sleep(POLL_INTERVAL)

        # 8. 讀取完成後回授
        time.sleep(0.2)

        encoder_after = servo.read_encoder_value_addition()
        speed_after = servo.read_motor_speed()
        status_after = servo.query_motor_status()

        print("\n" + "=" * 50)
        print("回零流程結束")
        print("歸位前 Encoder：", encoder_before)
        print("歸位後 Encoder：", encoder_after)
        print("最後轉速 RPM：", speed_after)
        print("最後馬達狀態：", status_after)

        if homing_completed:
            print(
                "注意：已觀察到停止，但請依驅動器回報與實際機構"
                "確認是否真正歸位成功。"
            )

    except KeyboardInterrupt:
        print("\n使用者中斷程式。")
        if servo is not None and homing_started:
            try:
                servo.emergency_stop_motor()
                print("已送出緊急停止指令。")
            except Exception as exc:
                print("緊急停止指令失敗：", exc)

    except Exception as exc:
        print("\n回零失敗：", exc)

        if servo is not None and homing_started:
            try:
                servo.emergency_stop_motor()
                print("已送出緊急停止指令。")
            except Exception as stop_exc:
                print("緊急停止指令失敗：", stop_exc)

    finally:
        if notifier is not None:
            notifier.stop()

        if bus is not None:
            bus.shutdown()

        print("CAN 連線已關閉。")

if __name__ == "__main__":
    main()

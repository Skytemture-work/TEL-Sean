
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MKS SERVO42D_CAN

流程：
起始位置 → 1700 脈波
按 Enter → 3000 脈波
按 Enter → 4200 脈波
自動執行無限位回零
"""

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

# ============================================================
# 使用者設定
# ============================================================

CAN_INTERFACE = "can0"
MOTOR_CAN_ID = 1

# 回零方向
HOME_DIRECTION = Direction.CW
HOME_SPEED_RPM = 30
HOME_CURRENT_MA = 200
REVERSE_ANGLE = 0x0400
HOME_TIMEOUT = 30.0

# 三段累積目標位置（脈波）
TARGET_POSITIONS = [1700, 3000, 4300]

# 前進／後退方向
# 自動設定成回零方向的反方向
MOVE_DIRECTION = (
    Direction.CCW
    if HOME_DIRECTION == Direction.CW
    else Direction.CW
)

MOVE_SPEED_RPM = 30
ACCELERATION = 2
MOVE_TIMEOUT = 30.0

POLL_INTERVAL = 0.3
PAUSE_BETWEEN = 0.5

# ============================================================
# 啟用馬達
# ============================================================

def enable_motor_compat(servo, enabled=True):
    method_globals = servo.enable_motor.__func__.__globals__
    commands = method_globals["MksCommands"]

    value = (
        Enable.Enable.value
        if enabled
        else Enable.Disable.value
    )

    return servo.set_generic_status(
        commands.ENABLE_MOTOR_COMMAND,
        [value]
    )

# ============================================================
# 設定無限位回零
# ============================================================

def set_no_limit_home_parameters(servo):

    print("\n設定無限位回零參數...")

    speed = HOME_SPEED_RPM

    home_data = [
        0,
        HOME_DIRECTION.value,
        (speed >> 8) & 0x0F,
        speed & 0xFF,
        0,
        1,  # noLimit 模式
    ]

    result = servo.set_generic_status(
        MksCommands.SET_HOME_COMMAND,
        home_data
    )

    print("0x90 設定結果：", result)

    if result != SuccessStatus.Success:
        raise RuntimeError("無限位回零模式設定失敗")

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

    result = servo.set_generic_status(
        MksCommands.SET_NO_LIMIT_GO_HOME_COMMAND,
        home_current_data
    )

    print("0x94 設定結果：", result)

    if result != SuccessStatus.Success:
        raise RuntimeError("無限位回零電流設定失敗")

    print("無限位回零參數設定完成。")

# ============================================================
# 相對移動指定脈波
# ============================================================

def move_relative(servo, pulses, target_position, state):

    print("\n" + "=" * 45)
    print("目標位置：{} 脈波".format(target_position))
    print("本段移動：{} 脈波".format(pulses))
    print("移動方向：", MOVE_DIRECTION)
    print("=" * 45)

    encoder_before = servo.read_encoder_value_addition()

    if encoder_before is None:
        raise RuntimeError("無法讀取移動前的編碼器值")

    print("移動前 Encoder：", encoder_before)

    # 在發送命令前設定階段，讓錯誤處理可以嘗試停止馬達
    state["phase"] = "move"

    result = servo.run_motor_relative_motion_by_pulses(
        MOVE_DIRECTION,
        MOVE_SPEED_RPM,
        ACCELERATION,
        pulses
    )

    print("移動指令回傳：", result)

    if result is None:
        raise RuntimeError("移動指令沒有收到有效回覆")

    start_time = time.monotonic()
    seen_running = False
    movement_seen = False
    stopped_since = None

    while True:

        elapsed = time.monotonic() - start_time

        if elapsed >= MOVE_TIMEOUT:
            raise TimeoutError(
                "移動到 {} 脈波位置逾時".format(target_position)
            )

        running = servo.is_motor_running()
        encoder_now = servo.read_encoder_value_addition()

        if encoder_now is not None and encoder_now != encoder_before:
            movement_seen = True

        if running:
            seen_running = True
            stopped_since = None

        elif seen_running or movement_seen:

            if stopped_since is None:
                stopped_since = time.monotonic()

            elif time.monotonic() - stopped_since >= 0.9:
                break

        elif elapsed > 2.0:
            raise RuntimeError(
                "未偵測到馬達移動，請檢查方向、驅動器及機構"
            )

        delta = (
            None
            if encoder_now is None
            else encoder_now - encoder_before
        )

        print(
            "時間={:.1f}s | 運轉中={} | Encoder={} | 位移={}".format(
                elapsed,
                running,
                encoder_now,
                delta
            )
        )

        time.sleep(POLL_INTERVAL)

    time.sleep(0.2)

    encoder_after = servo.read_encoder_value_addition()

    print("\n本段移動完成。")
    print("移動前 Encoder：", encoder_before)
    print("移動後 Encoder：", encoder_after)

    if encoder_after is not None:
        print("實際編碼器位移：", encoder_after - encoder_before)

    state["phase"] = None

# ============================================================
# 無限位回零
# ============================================================

def go_home(servo, state):

    print("\n" + "=" * 45)
    print("開始無限位回零")
    print("=" * 45)

    encoder_before = servo.read_encoder_value_addition()
    print("回零前 Encoder：", encoder_before)

    set_no_limit_home_parameters(servo)

    # 發送回零命令前就設定狀態，避免發生錯誤時無法停止
    state["phase"] = "home"

    response = servo.set_generic(
        MksCommands.GO_HOME_COMMAND,
        servo.GENERIC_RESPONSE_LENGTH
    )

    if response is None or len(response) < 2:
        raise RuntimeError("回零命令沒有收到有效回覆")

    home_status = int(response[1])
    print("0x91 初始回覆狀態：", home_status)

    # 0 = 失敗，1 = 開始，2 = 成功
    if home_status == 0:
        raise RuntimeError("驅動器拒絕回零命令")

    if home_status == 2:
        print("驅動器回報回零成功。")
        state["phase"] = None
        return

    if home_status != 1:
        raise RuntimeError(
            "未知的回零狀態：{}".format(home_status)
        )

    start_time = time.monotonic()
    seen_running = False
    movement_seen = False
    stopped_since = None

    while True:

        elapsed = time.monotonic() - start_time

        if elapsed >= HOME_TIMEOUT:
            raise TimeoutError("無限位回零逾時")

        running = servo.is_motor_running()
        encoder_now = servo.read_encoder_value_addition()

        if (
            encoder_before is not None
            and encoder_now is not None
            and encoder_now != encoder_before
        ):
            movement_seen = True

        if running:
            seen_running = True
            stopped_since = None

        elif seen_running or movement_seen:

            if stopped_since is None:
                stopped_since = time.monotonic()

            elif time.monotonic() - stopped_since >= 1.0:
                break

        elif elapsed > 2.0:
            raise RuntimeError(
                "未偵測到回零動作，請檢查方向與驅動器設定"
            )

        print(
            "時間={:.1f}s | 運轉中={} | Encoder={}".format(
                elapsed,
                running,
                encoder_now
            )
        )

        time.sleep(POLL_INTERVAL)

    time.sleep(0.3)

    encoder_after = servo.read_encoder_value_addition()
    speed_after = servo.read_motor_speed()
    status_after = servo.query_motor_status()

    print("\n回零動作已停止。")
    print("回零前 Encoder：", encoder_before)
    print("回零後 Encoder：", encoder_after)
    print("最後轉速 RPM：", speed_after)
    print("最後馬達狀態：", status_after)
    print("注意：馬達停止不一定代表驅動器已確認回零成功。")

    state["phase"] = None

# ============================================================
# 緊急停止
# ============================================================

def emergency_stop(servo, state):

    if servo is None or state["phase"] is None:
        return

    try:
        if state["phase"] == "move":
            servo.stop_motor_relative_motion_by_pulses(
                ACCELERATION
            )
        else:
            servo.emergency_stop_motor()

        print("已送出停止指令。")

    except Exception as exc:
        print("停止指令失敗：", exc)

# ============================================================
# 主程式
# ============================================================

def main():

    bus = None
    notifier = None
    servo = None

    state = {"phase": None}

    try:

        print("=" * 50)
        print("MKS SERVO42D_CAN 三段位置控制")
        print("目標位置：1700 → 3000 → 4200 → 回零")
        print("CAN：{} | Motor ID：{}".format(
            CAN_INTERFACE,
            MOTOR_CAN_ID
        ))
        print("=" * 50)

        bus = can.Bus(
            interface="socketcan",
            channel=CAN_INTERFACE
        )

        notifier = can.Notifier(bus, [])
        servo = MksServo(bus, notifier, MOTOR_CAN_ID)

        # 設定工作模式
        print("\n設定工作模式 SrvFoc...")

        result = servo.set_work_mode(WorkMode.SrvFoc)
        print("工作模式設定結果：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("工作模式設定失敗")

        time.sleep(0.5)

        # 啟用馬達
        print("啟用馬達...")

        result = enable_motor_compat(servo, True)
        print("啟用結果：", result)

        if result != SuccessStatus.Success:
            raise RuntimeError("馬達啟用失敗")

        time.sleep(0.5)

        # 記錄程式開始時的編碼器位置
        start_encoder = servo.read_encoder_value_addition()

        if start_encoder is None:
            raise RuntimeError("無法讀取初始 Encoder")

        print("\n起始 Encoder：", start_encoder)

        previous_target = 0

        # ----------------------------------------------------
        # 第一段：到 1700，等待 Enter
        # ----------------------------------------------------

        target = TARGET_POSITIONS[0]
        pulses = target - previous_target

        move_relative(
            servo,
            pulses,
            target,
            state
        )

        input("\n已到達第一段位置 1700。按 Enter 前往 3000...")
        previous_target = target

        # ----------------------------------------------------
        # 第二段：到 3000，等待 Enter
        # ----------------------------------------------------

        target = TARGET_POSITIONS[1]
        pulses = target - previous_target

        move_relative(
            servo,
            pulses,
            target,
            state
        )

        input("\n已到達第二段位置 3000。按 Enter 前往 4200...")
        previous_target = target

        # ----------------------------------------------------
        # 第三段：到 4200，然後自動回零
        # ----------------------------------------------------

        target = TARGET_POSITIONS[2]
        pulses = target - previous_target

        move_relative(
            servo,
            pulses,
            target,
            state
        )

        print("\n已到達第三段位置 4200。")
        time.sleep(PAUSE_BETWEEN)

        go_home(servo, state)

        print("\n全部流程完成。")

    except KeyboardInterrupt:
        print("\n使用者中斷程式。")
        emergency_stop(servo, state)

    except Exception as exc:
        print("\n流程失敗：", exc)
        emergency_stop(servo, state)

    finally:

        if notifier is not None:
            notifier.stop()

        if bus is not None:
            bus.shutdown()

        print("CAN 連線已關閉。")

if __name__ == "__main__":
    main()

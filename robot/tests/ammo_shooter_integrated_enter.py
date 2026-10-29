#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MKS SERVO42D_CAN

流程：
起始位置 → 1700 脈波
按 Enter → 3000 脈波
自動：無限位回零（朝原點退回）
自動：反向無限位回零（朝相反方向歸零）

移動階段若堵轉：硬停 → 無限位回零退回原點 → 中止。
"""

import time
from collections import deque

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

# 暫定介面分配：射擊機構 can1、彈藥倉 can0
SHOOTING_CAN_INTERFACE = "can1"
AMMO_CAN_INTERFACE = "can0"

# 每格彈藥倉控制
AMMO_MOTOR_CAN_ID = 1
AMMO_SLOT_PULSES = 6666
AMMO_MOVE_SPEED_RPM = 30
AMMO_MOVE_DIRECTION = Direction.CW
AMMO_MOVE_TIMEOUT = 60.0

SLOTS = 6
SHOTS_PER_SLOT = 3
MOTOR_CAN_ID = 1

# 回零方向
HOME_DIRECTION = Direction.CW
HOME_SPEED_RPM = 30
HOME_CURRENT_MA = 200
REVERSE_ANGLE = 0x0400
HOME_TIMEOUT = 30.0

# 兩段累積目標位置（脈波）
TARGET_POSITIONS = [1700, 3000]

# 前進／後退方向：自動設定成回零方向的反方向
MOVE_DIRECTION = (
    Direction.CCW
    if HOME_DIRECTION == Direction.CW
    else Direction.CW
)

MOVE_SPEED_RPM = 30
ACCELERATION = 2
MOVE_TIMEOUT = 30.0

POLL_INTERVAL = 0.3          # 回零階段的輪詢間隔
PAUSE_BETWEEN = 0.5

# ---------- 快速堵轉偵測（移動階段）----------
MOVE_POLL_INTERVAL = 0.05    # 移動階段輪詢間隔，越小反應越快
ENCODER_COUNTS_PER_REV = 16384   # 編碼器每圈計數 (0x4000)
STALL_WINDOW = 0.25          # 觀察視窗（秒）
STALL_PROGRESS_RATIO = 0.2   # 視窗內進度低於「預期的 20%」就判定堵轉
STALL_ARM_COUNTS = 300       # 起步後至少轉過這麼多 counts 才啟動偵測
STALL_START_GRACE = 1.0      # 指令送出後超過這麼久仍沒轉過 ARM_COUNTS → 視為卡住

# ---------- 到位驗證（驅動器自己保護停機時，也能被抓到）----------
# 每個脈波對應多少編碼器 counts。
# 預設假設細分 16 → 每圈 3200 脈波，編碼器每圈 16384 → 5.12。
# 每段結束會印出「實測 counts/pulse」，請用一次空載的正常移動校正這個值。
COUNTS_PER_PULSE = 5.12
MIN_COMPLETE_RATIO = 0.8     # 實際位移低於預期的 80% → 視為沒走完（撞牆/被保護停機）
STALL_SETTLE = 0.15          # 回報停止後，等這麼久再驗證位移
DEBUG_STALL = True           # 每次輪詢都印出，方便你回報實際數據


class StallDetected(RuntimeError):
    """移動階段偵測到堵轉（撞牆）時拋出，由主程式接手反向回原點。"""


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

def set_no_limit_home_parameters(servo, direction=None):

    print("\n設定無限位回零參數...")

    speed = HOME_SPEED_RPM

    home_data = [
        0,
        (direction if direction is not None else HOME_DIRECTION).value,
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
# 堵轉處理：立刻硬停，必要時釋放扭力
# ============================================================

def handle_stall(servo, state, message):
    # 先硬停，不走減速斜坡
    try:
        servo.emergency_stop_motor()
    except Exception as exc:
        print("硬停指令失敗：", exc)

    state["phase"] = None

    raise StallDetected(message)


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

    # 預期 STALL_WINDOW 內的前進量，依設定轉速換算
    expected_counts = (
        MOVE_SPEED_RPM / 60.0
        * ENCODER_COUNTS_PER_REV
        * STALL_WINDOW
    )
    stall_threshold = max(10, int(expected_counts * STALL_PROGRESS_RATIO))

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
    armed = False
    stopped_since = None
    samples = deque()          # (時間, encoder)
    last_print = 0.0

    while True:

        now = time.monotonic()
        elapsed = now - start_time

        if elapsed >= MOVE_TIMEOUT:
            raise TimeoutError(
                "移動到 {} 脈波位置逾時".format(target_position)
            )

        running = servo.is_motor_running()
        encoder_now = servo.read_encoder_value_addition()

        moved = 0

        if encoder_now is not None:
            moved = abs(encoder_now - encoder_before)

            if moved > 0:
                movement_seen = True

            if moved >= STALL_ARM_COUNTS:
                armed = True

            samples.append((now, encoder_now))
            while samples and now - samples[0][0] > STALL_WINDOW:
                samples.popleft()

        if running:
            seen_running = True
            stopped_since = None

            # 堵轉偵測：馬達仍在「運轉」，但編碼器幾乎沒前進
            if armed and len(samples) >= 2:
                span = samples[-1][0] - samples[0][0]

                if span >= STALL_WINDOW * 0.7:
                    progress = abs(samples[-1][1] - samples[0][1])

                    if progress < stall_threshold:
                        handle_stall(
                            servo, state,
                            "運轉中但編碼器停滯：已移動 {} counts，"
                            "最近 {:.2f}s 只前進 {}（門檻 {}）".format(
                                moved, STALL_WINDOW,
                                progress, stall_threshold
                            )
                        )

            # 起步後遲遲沒有轉過 ARM_COUNTS：一開始就被擋住
            if not armed and elapsed >= STALL_START_GRACE:
                handle_stall(
                    servo, state,
                    "起步 {:.1f}s 後仍只移動 {} counts，疑似一開始就被擋住"
                    .format(elapsed, moved)
                )

        elif seen_running or movement_seen:

            if stopped_since is None:
                stopped_since = now

            elif now - stopped_since >= STALL_SETTLE:
                break

        elif elapsed > 2.0:
            raise RuntimeError(
                "未偵測到馬達移動，請檢查方向、驅動器及機構"
            )

        # 輪詢列印（DEBUG_STALL 時每次都印）
        if DEBUG_STALL or elapsed - last_print >= 0.5:
            last_print = elapsed
            delta = (
                None
                if encoder_now is None
                else encoder_now - encoder_before
            )
            print(
                "時間={:.1f}s | 運轉中={} | Encoder={} | 位移={}".format(
                    elapsed, running, encoder_now, delta
                )
            )

        time.sleep(MOVE_POLL_INTERVAL)

    time.sleep(0.1)

    encoder_after = servo.read_encoder_value_addition()

    if encoder_after is None:
        raise RuntimeError("無法讀取移動後的編碼器值")

    actual = abs(encoder_after - encoder_before)
    expected_total = pulses * COUNTS_PER_PULSE

    print("\n本段移動結束。")
    print("移動前 Encoder：", encoder_before)
    print("移動後 Encoder：", encoder_after)
    print("實際編碼器位移：", actual)
    print("預期位移：{:.0f}（實測 counts/pulse = {:.3f}）".format(
        expected_total, actual / pulses
    ))

    # 到位驗證：停下來不等於走完。被擋住、或驅動器自己保護停機，
    # 位移都會明顯不足。
    if actual < expected_total * MIN_COMPLETE_RATIO:
        handle_stall(
            servo, state,
            "馬達已停止但沒走完：實際 {} counts，預期 {:.0f} counts"
            .format(actual, expected_total)
        )

    state["phase"] = None


# ============================================================
# 無限位回零
# ============================================================

def go_home(servo, state, direction=None, label="無限位回零"):

    print("\n" + "=" * 45)
    print("開始{}｜方向：{}".format(
        label, direction if direction is not None else HOME_DIRECTION
    ))
    print("=" * 45)

    encoder_before = servo.read_encoder_value_addition()
    print("回零前 Encoder：", encoder_before)

    set_no_limit_home_parameters(servo, direction)

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
        # 移動階段也改用硬停，不走減速斜坡
        servo.emergency_stop_motor()
        print("已送出停止指令。")

    except Exception as exc:
        print("停止指令失敗：", exc)


# ============================================================
# 主程式
# ============================================================

def run_sequence(servo, state):
    """依序執行兩段移動，接著兩次回零。移動中若堵轉會拋出 StallDetected。"""

    previous_target = 0

    # 第一段：到 1700，等待 Enter
    target = TARGET_POSITIONS[0]
    move_relative(servo, target - previous_target, target, state)
    input("\n已到達第一段位置 1700。按 Enter 前往 3000...")
    previous_target = target

    # 第二段：到 3000
    target = TARGET_POSITIONS[1]
    move_relative(servo, target - previous_target, target, state)
    print("\n已到達第二段位置 3000。")
    time.sleep(PAUSE_BETWEEN)

    # 第一次：反向無限位歸零
    go_home(servo, state, MOVE_DIRECTION, "反向無限位回零")
    time.sleep(PAUSE_BETWEEN)

    # 第二次：正向無限位歸零
    go_home(servo, state, HOME_DIRECTION, "正向無限位回零")

def recover_to_origin(servo, state):
    """堵轉後：確認已停止，重新啟用馬達，再以無限位回零反向退回原點。"""

    print("\n" + "!" * 45)
    print("保護觸發，馬達已硬停，立即反向回原點")
    print("!" * 45)

    time.sleep(0.3)

    # 保險起見重新啟用（若驅動器因保護而失能也能恢復）
    result = enable_motor_compat(servo, True)
    print("重新啟用結果：", result)

    if result != SuccessStatus.Success:
        raise RuntimeError("保護後重新啟用馬達失敗，無法回原點")

    time.sleep(0.3)

    # 堵轉後實際位置已不可信，用無限位回零比相對移動可靠
    go_home(servo, state)

    print("\n已因保護觸發而回到原點，流程中止。")



# ============================================================
# 彈藥倉移動：每次相對移動一格（6666 pulses）
# ============================================================

def rotate_ammo_one_slot(servo):
    print("\n" + "=" * 50)
    print("彈藥倉準備轉動一格：{} pulses".format(AMMO_SLOT_PULSES))
    print("方向：{}".format(AMMO_MOVE_DIRECTION))
    print("=" * 50)

    encoder_before = servo.read_encoder_value_addition()
    if encoder_before is None:
        raise RuntimeError("彈藥倉無法讀取移動前 Encoder")

    result = servo.run_motor_relative_motion_by_pulses(
        AMMO_MOVE_DIRECTION,
        AMMO_MOVE_SPEED_RPM,
        ACCELERATION,
        AMMO_SLOT_PULSES
    )
    print("彈藥倉移動指令回傳：", result)
    if result is None:
        raise RuntimeError("彈藥倉移動指令沒有收到有效回覆")

    start = time.monotonic()
    seen_running = False
    movement_seen = False
    stopped_since = None

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= AMMO_MOVE_TIMEOUT:
            try:
                servo.emergency_stop_motor()
            finally:
                raise TimeoutError("彈藥倉轉動逾時，已送出緊急停止")

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
            elif time.monotonic() - stopped_since >= STALL_SETTLE:
                break
        elif elapsed > 2.0:
            raise RuntimeError("彈藥倉未偵測到移動，請檢查 can0、馬達 ID 與機構")

        print(
            "彈藥倉 | 時間={:.1f}s | 運轉中={} | Encoder={}".format(
                elapsed, running, encoder_now
            )
        )
        time.sleep(MOVE_POLL_INTERVAL)

    time.sleep(0.2)
    encoder_after = servo.read_encoder_value_addition()
    if encoder_after is None:
        raise RuntimeError("彈藥倉無法讀取移動後 Encoder")

    print("彈藥倉移動完成，Encoder 位移：{}".format(
        encoder_after - encoder_before
    ))


# ============================================================
# 每顆球的射擊機構循環
# 1700 pulses -> 3000 pulses -> 反向回零 -> 正向回零
# ============================================================

def run_one_shot_cycle(servo, state, slot_number, shot_number):
    print("\n" + "#" * 55)
    print("彈藥格 {} / {}｜第 {} / {} 發".format(
        slot_number, SLOTS, shot_number, SHOTS_PER_SLOT
    ))
    print("#" * 55)

    # 第一段：相對移動 1700 pulses
    move_relative(servo, 1700, 1700, state)
    time.sleep(PAUSE_BETWEEN)

    # 第二段：從 1700 移動至累積目標 3000，因此本段是 1300 pulses
    move_relative(servo, 3000 - 1700, 3000, state)
    time.sleep(PAUSE_BETWEEN)

    # 先反向無限位回零，再正向無限位回零
    go_home(servo, state, MOVE_DIRECTION, "反向無限位回零")
    time.sleep(PAUSE_BETWEEN)
    go_home(servo, state, HOME_DIRECTION, "正向無限位回零")

    print("第 {} 格第 {} 發的動作流程完成。".format(
        slot_number, shot_number
    ))


def stop_motor_safely(servo, label):
    if servo is None:
        return
    try:
        servo.emergency_stop_motor()
        print("{}：已送出緊急停止。".format(label))
    except Exception as exc:
        print("{}：緊急停止指令失敗：{}".format(label, exc))


# ============================================================
# 主程式：每格 3 發，完成後轉一格，共 6 格
# ============================================================

def main():
    shooting_bus = None
    shooting_notifier = None
    ammo_bus = None
    ammo_notifier = None
    shooting_servo = None
    ammo_servo = None
    shooting_state = {"phase": None}

    try:
        print("=" * 60)
        print("六格彈藥倉自動循環")
        print("射擊機構：can1 / Motor ID {}".format(MOTOR_CAN_ID))
        print("彈藥倉：can0 / Motor ID {}".format(AMMO_MOTOR_CAN_ID))
        print("每格 {} 發；每格轉動 {} pulses；共 {} 格".format(
            SHOTS_PER_SLOT, AMMO_SLOT_PULSES, SLOTS
        ))
        print("=" * 60)
        print("注意：can0/can1 分配及兩顆馬達 ID 都是暫定值，請先確認實際接線。")

        # 建立射擊機構 CAN
        shooting_bus = can.Bus(
            interface="socketcan",
            channel=SHOOTING_CAN_INTERFACE
        )
        shooting_notifier = can.Notifier(shooting_bus, [])
        shooting_servo = MksServo(
            shooting_bus, shooting_notifier, MOTOR_CAN_ID
        )

        # 建立彈藥倉 CAN
        ammo_bus = can.Bus(
            interface="socketcan",
            channel=AMMO_CAN_INTERFACE
        )
        ammo_notifier = can.Notifier(ammo_bus, [])
        ammo_servo = MksServo(
            ammo_bus, ammo_notifier, AMMO_MOTOR_CAN_ID
        )

        # 設定兩顆馬達工作模式
        print("\n設定射擊機構工作模式...")
        result = shooting_servo.set_work_mode(WorkMode.SrvFoc)
        print("射擊機構工作模式設定：", result)
        if result != SuccessStatus.Success:
            raise RuntimeError("射擊機構工作模式設定失敗")

        print("\n設定彈藥倉工作模式...")
        result = ammo_servo.set_work_mode(WorkMode.SrvFoc)
        print("彈藥倉工作模式設定：", result)
        if result != SuccessStatus.Success:
            raise RuntimeError("彈藥倉工作模式設定失敗")

        time.sleep(0.5)

        # 啟用兩顆馬達
        result = enable_motor_compat(shooting_servo, True)
        print("射擊機構啟用結果：", result)
        if result != SuccessStatus.Success:
            raise RuntimeError("射擊機構啟用失敗")

        result = enable_motor_compat(ammo_servo, True)
        print("彈藥倉啟用結果：", result)
        if result != SuccessStatus.Success:
            raise RuntimeError("彈藥倉啟用失敗")

        time.sleep(0.5)

        if shooting_servo.read_encoder_value_addition() is None:
            raise RuntimeError("無法讀取射擊機構初始 Encoder")
        if ammo_servo.read_encoder_value_addition() is None:
            raise RuntimeError("無法讀取彈藥倉初始 Encoder")

        # Enter 觸發模式：每按一次 Enter，只執行一發完整射擊循環。
        # 每格完成 3 發後，前 5 格轉動彈藥倉 6666 pulses；
        # 第 6 格完成 3 發後直接結束，不再轉動彈藥倉。
        total_shots = 0
        for slot in range(1, SLOTS + 1):
            print("\\n" + "=" * 60)
            print("目前位置：第 {} / {} 格".format(slot, SLOTS))
            print("本格需要完成 {} 發；每按一次 Enter 執行 1 發。".format(
                SHOTS_PER_SLOT
            ))
            print("=" * 60)

            for shot in range(1, SHOTS_PER_SLOT + 1):
                input("\\n按 Enter 執行第 {} 格第 {} / {} 發（Ctrl+C 可中止）...".format(
                    slot, shot, SHOTS_PER_SLOT
                ))

                run_one_shot_cycle(
                    shooting_servo, shooting_state, slot, shot
                )
                total_shots += 1

                print("\\n第 {} 格進度：{}/{} 發完成。".format(
                    slot, shot, SHOTS_PER_SLOT
                ))

            print("\\n第 {} 格的 {} 發射擊循環已全部完成。".format(
                slot, SHOTS_PER_SLOT
            ))

            if slot < SLOTS:
                print("本格射擊成功，彈藥倉開始轉動 {} pulses。".format(
                    AMMO_SLOT_PULSES
                ))
                rotate_ammo_one_slot(ammo_servo)
                print("彈藥倉已轉到第 {} 格，等待 Enter 進行下一發。".format(
                    slot + 1
                ))
            else:
                print("第六格完成，不再轉動彈藥倉。")

        print("\\n六格流程全部完成，共完成 {} 次射擊循環。".format(
            total_shots
        ))

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止兩顆馬達。")
        stop_motor_safely(shooting_servo, "射擊機構")
        stop_motor_safely(ammo_servo, "彈藥倉")

    except StallDetected as exc:
        print("\n射擊機構偵測到堵轉：", exc)
        stop_motor_safely(shooting_servo, "射擊機構")
        stop_motor_safely(ammo_servo, "彈藥倉")
        print("流程中止；未完成的格位不會自動切換。")

    except Exception as exc:
        print("\n流程錯誤：", exc)
        stop_motor_safely(shooting_servo, "射擊機構")
        stop_motor_safely(ammo_servo, "彈藥倉")
        print("流程中止；請先確認機構狀態再重新執行。")

    finally:
        if shooting_notifier is not None:
            shooting_notifier.stop()
        if ammo_notifier is not None:
            ammo_notifier.stop()
        if shooting_bus is not None:
            shooting_bus.shutdown()
        if ammo_bus is not None:
            ammo_bus.shutdown()
        print("CAN 連線已關閉。")


if __name__ == "__main__":
    main()

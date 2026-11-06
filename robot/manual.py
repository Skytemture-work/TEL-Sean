# manual.py
# 手動模式：ESP32 遙控器 + MKS 射擊機構/彈藥倉 + 雙 VESC 飛輪馬達
#
# 封包格式由 robot/comm_uart.py 定義：
# E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED
#
# 暫定硬體分配：
#   can0：共用 MKS CAN bus；彈藥倉 ID 2、射擊機構 ID 1
#   彈藥倉每格移動 6666 pulses
#   VESC_LEFT_PORT / VESC_RIGHT_PORT：射擊雙馬達轉速控制（UART）
#
# 注意：
# 2. FIRE 每次由 0 變 1，只執行一階段：1700、再移動 1300、回零後 ID2 轉格。
# 3. 第三階段保留 ID1 反向／正向回零；完成後 ID2 正轉 AMMO_SLOT_PULSES。
# 5. 手動 VESC 目標轉速為 1000 + speed*10 RPM（speed 0..100）。
# 6. 請卸除彈藥進行低速測試。
#
import time
import struct
import threading
import math

import serial

import can
from config import DM_YAW_MIN_DEG, DM_YAW_MAX_DEG, DM_PITCH_MIN_DEG, DM_PITCH_MAX_DEG
from mks_servo_can import MksServo
from mks_servo_can.mks_enums import (
    Enable,
    Direction,
    WorkMode,
    SuccessStatus,
    MksCommands,
)


# -------------------- 可調整設定 --------------------
DM_STARTUP_CLI_TEST_ONLY = False
DM_STARTUP_TEST_PORT = "/dev/ttyACM0"
DM_STARTUP_TEST_YAW_DEG = 0.0
DM_STARTUP_TEST_PITCH_DEG = 20.0
DM_STARTUP_TEST_MOVE_KP = 20.0
DM_STARTUP_TEST_SPEED_RAD_S = 0.40
DM_STARTUP_TEST_KP = 30.0  # 與 run_manual_move 預設保持增益一致。
DM_STARTUP_TEST_KD = 1.0
DM_STARTUP_TEST_PITCH_TAU_FF = 3.0  # Nm，正向支撐；由 run_manual_move 漸增施加。
DM_UART_CENTER = 50
DM_UART_DEG_PER_UNIT = 1.0
DM_UART_YAW_SIGN = 1.0  # 遙控器左右方向反轉。
DM_UART_PITCH_SIGN = 1.0  # 遙控器上下方向反轉。
DM_POSITION_TOLERANCE_DEG = 2.5  # 實際角度到位容許誤差；負載下約有 1.14° 穩態誤差。
DM_FEEDBACK_MAX_AGE_S = 0.5
DM_FEEDBACK_RECOVERY_TIMEOUT_S = 1.0
AMMO_CAN_INTERFACE = "can0"
SHOOTING_CAN_INTERFACE = "can0"

# 雙 VESC 射擊馬達：請依實際裝置名稱調整。
# 建議用 /dev/serial/by-id/ 確認每個 USB 裝置的固定名稱。
VESC_LEFT_PORT = "/dev/ttyACM1"
VESC_RIGHT_PORT = "/dev/ttyACM2"
VESC_BAUDRATE = 921600
VESC_LEFT_POLE_PAIRS = 7
VESC_RIGHT_POLE_PAIRS = 7
VESC_MIN_RPM = 1000.0
VESC_RPM_PER_SPEED_UNIT = 10.0  # speed=100 對應 2000 RPM
VESC_LOOP_INTERVAL = 0.10
VESC_TELEMETRY_TIMEOUT = 0.5
VESC_MAX_MOTOR_CURRENT_A = 50.0
VESC_MAX_INPUT_VOLTAGE_V = 30.0
VESC_MIN_INPUT_VOLTAGE_V = 10.5
VESC_MAX_TRIM_ERPM = 100.0
VESC_MAX_SYNC_CORRECTION_ERPM = 50.0
VESC_SPEED_KP = 0.10
VESC_SPEED_KI = 0.02
VESC_SYNC_KP = 0.10

# VESC command IDs / packet format
VESC_COMM_GET_VALUES = 4
VESC_COMM_SET_RPM = 8
VESC_PACKET_SHORT = 2
VESC_PACKET_LONG = 3
VESC_PACKET_END = 3

AMMO_MOTOR_CAN_ID = 2
SHOOTING_MOTOR_CAN_ID = 1

AMMO_SLOT_PULSES = 6670
AMMO_SPEED_RPM = 1500
SHOOTING_SPEED_RPM = 10      # 初次整合先低速測試
ACCELERATION = 2

HOME_DIRECTION = Direction.CW
MOVE_DIRECTION = (
    Direction.CCW if HOME_DIRECTION == Direction.CW else Direction.CW
)
HOME_SPEED_RPM = 10
HOME_CURRENT_MA = 300
REVERSE_ANGLE = 0x0400

TARGET_POSITIONS = (1700, 3000)
SLOTS_PER_MAGAZINE = 6

MOVE_TIMEOUT = 30.0
HOME_TIMEOUT = 30.0
POLL_INTERVAL = 0.1
SETTLE_TIME = 0.2

# FIRE 指令若持續為 1，只在上升沿觸發一次，不會連續重複射擊。
# 射擊循環在背景執行緒處理；主迴圈仍可更新 VESC speed 與接收遙控器封包。
# ---------------------------------------------------


def normalize_vrb_speed(vrb_value):
    """Convert raw iBUS VRB (1000..2000) or ESP32 mapped speed (0..100) to 0..100."""
    value = int(vrb_value)
    if 1000 <= value <= 2000:
        value = round((value - 1000) / 10.0)
    return max(0, min(100, value))


# ==================== VESC 雙馬達控制 ====================

def vesc_crc16(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def vesc_make_packet(payload):
    length = len(payload)
    if length <= 255:
        packet = bytearray([VESC_PACKET_SHORT, length])
    elif length <= 65535:
        packet = bytearray([
            VESC_PACKET_LONG, (length >> 8) & 0xFF, length & 0xFF
        ])
    else:
        raise ValueError("VESC payload too large")
    packet.extend(payload)
    packet.extend(struct.pack(">H", vesc_crc16(payload)))
    packet.append(VESC_PACKET_END)
    return bytes(packet)


def vesc_send_command(ser, command, data=b""):
    payload = bytes([command]) + data
    ser.write(vesc_make_packet(payload))
    ser.flush()


def vesc_send_set_rpm(ser, erpm):
    vesc_send_command(ser, VESC_COMM_SET_RPM, struct.pack(">i", int(erpm)))


class VESCReader:
    """持續緩存 VESC UART 封包，避免一次讀取錯位後永久失同步。"""

    def __init__(self, ser):
        self.ser = ser
        self.buffer = bytearray()

    def read_packet(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while self.buffer:
                start = next(
                    (i for i, b in enumerate(self.buffer)
                     if b in (VESC_PACKET_SHORT, VESC_PACKET_LONG)),
                    None,
                )
                if start is None:
                    self.buffer.clear()
                    break
                if start:
                    del self.buffer[:start]

                if self.buffer[0] == VESC_PACKET_SHORT:
                    if len(self.buffer) < 2:
                        break
                    header_len = 2
                    payload_len = self.buffer[1]
                else:
                    if len(self.buffer) < 3:
                        break
                    header_len = 3
                    payload_len = (self.buffer[1] << 8) | self.buffer[2]

                if payload_len <= 0 or payload_len > 1024:
                    del self.buffer[0]
                    continue

                total_len = header_len + payload_len + 3
                if len(self.buffer) < total_len:
                    break

                packet = bytes(self.buffer[:total_len])
                if packet[-1] != VESC_PACKET_END:
                    del self.buffer[0]
                    continue

                p0 = header_len
                p1 = p0 + payload_len
                payload = packet[p0:p1]
                received_crc = struct.unpack(">H", packet[p1:p1 + 2])[0]
                if vesc_crc16(payload) == received_crc:
                    del self.buffer[:total_len]
                    return payload
                del self.buffer[0]

            chunk = self.ser.read(1)
            if chunk:
                self.buffer.extend(chunk)
        return None

    def get_values(self, timeout=VESC_TELEMETRY_TIMEOUT):
        vesc_send_command(self.ser, VESC_COMM_GET_VALUES)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = self.read_packet(max(0.01, deadline - time.monotonic()))
            if payload and payload[0] == VESC_COMM_GET_VALUES:
                return payload
        return None


def vesc_decode_values(payload):
    # 與使用者提供的舊版 VESC telemetry decoder 相同；需與韌體版本核對。
    if payload is None or len(payload) < 29:
        raise ValueError("VESC telemetry payload missing/too short")
    if payload[0] != VESC_COMM_GET_VALUES:
        raise ValueError("Unexpected VESC telemetry command ID")
    return {
        "motor_temp": struct.unpack_from(">h", payload, 3)[0] / 10.0,
        "motor_current": struct.unpack_from(">i", payload, 5)[0] / 100.0,
        "erpm": struct.unpack_from(">i", payload, 23)[0],
        "voltage": struct.unpack_from(">h", payload, 27)[0] / 10.0,
    }


class VESCDualMotorController:
    """
    背景控制左右 VESC 馬達。
    手動 speed 0..100 對應 1000..2000 RPM；None 代表安全停轉。
    保留使用者原始程式的 PI 調速與交叉同步修正。
    """

    def __init__(self):
        self._lock = threading.Lock()
        # VRB=0 對應最低目標 1000 RPM，啟動時先以最低轉速運作。
        self._speed_value = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._ser_left = None
        self._ser_right = None
        self.error = None

    def start(self):
        print("[VESC] 啟動控制執行緒，初始 VRB=0，目標 1000 RPM。", flush=True)
        self._thread = threading.Thread(
            target=self._run, name="VESC-Dual-Motor", daemon=True
        )
        self._thread.start()

    def set_speed(self, speed_value):
        if speed_value is not None:
            speed_value = max(0, min(100, int(speed_value)))
        with self._lock:
            previous = self._speed_value
            self._speed_value = speed_value
        if speed_value != previous:
            target_rpm = (
                "STOP" if speed_value is None
                else int(VESC_MIN_RPM + speed_value * VESC_RPM_PER_SPEED_UNIT)
            )
            print(
                "[VESC CMD] speed={} -> target={} RPM".format(
                    "STOP" if speed_value is None else speed_value, target_rpm
                ),
                flush=True,
            )

    def _get_speed_value(self):
        with self._lock:
            return self._speed_value

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _read_motor(self, reader):
        payload = reader.get_values()
        if payload is None:
            raise RuntimeError("VESC telemetry timeout")
        values = vesc_decode_values(payload)
        if not (-100000 <= values["erpm"] <= 100000):
            raise RuntimeError("Implausible VESC ERPM")
        if not (0.0 <= values["voltage"] <= VESC_MAX_INPUT_VOLTAGE_V):
            raise RuntimeError(
                "Unexpected VESC input voltage: {:.1f} V".format(values["voltage"])
            )
        if abs(values["motor_current"]) > VESC_MAX_MOTOR_CURRENT_A:
            raise RuntimeError(
                "VESC current limit exceeded: {:.2f} A".format(values["motor_current"])
            )
        if values["voltage"] < VESC_MIN_INPUT_VOLTAGE_V:
            raise RuntimeError(
                "VESC input voltage too low: {:.1f} V".format(values["voltage"])
            )
        return values

    def _stop_motors(self):
        for ser in (self._ser_left, self._ser_right):
            if ser is not None and ser.is_open:
                try:
                    vesc_send_set_rpm(ser, 0)
                except Exception:
                    pass
        time.sleep(0.05)
        for ser in (self._ser_left, self._ser_right):
            if ser is not None and ser.is_open:
                try:
                    vesc_send_set_rpm(ser, 0)
                except Exception:
                    pass

    def _run(self):
        pi_left_integral = 0.0
        pi_right_integral = 0.0
        last_time = time.monotonic()
        try:
            print(
                "[VESC] 開啟雙馬達序列埠：{} / {}".format(
                    VESC_LEFT_PORT, VESC_RIGHT_PORT
                )
            )
            self._ser_left = serial.Serial(
                VESC_LEFT_PORT, VESC_BAUDRATE, timeout=0.05, write_timeout=0.5
            )
            self._ser_right = serial.Serial(
                VESC_RIGHT_PORT, VESC_BAUDRATE, timeout=0.05, write_timeout=0.5
            )
            time.sleep(0.2)
            self._ser_left.reset_input_buffer()
            self._ser_right.reset_input_buffer()
            reader_left = VESCReader(self._ser_left)
            reader_right = VESCReader(self._ser_right)

            left = self._read_motor(reader_left)
            right = self._read_motor(reader_right)
            print(
                "[VESC] 初始回授 LEFT ERPM={} {:.1f}V, RIGHT ERPM={} {:.1f}V".format(
                    left["erpm"], left["voltage"], right["erpm"], right["voltage"]
                )
            )

            while not self._stop_event.is_set():
                now = time.monotonic()
                dt = self._clamp(now - last_time, 0.01, 0.5)
                last_time = now

                left = self._read_motor(reader_left)
                right = self._read_motor(reader_right)

                speed_value = self._get_speed_value()
                target_rpm = (
                    None if speed_value is None
                    else VESC_MIN_RPM + speed_value * VESC_RPM_PER_SPEED_UNIT
                )

                if target_rpm is None:
                    # 自動模式或關閉時清除積分並命令兩顆馬達停轉。
                    pi_left_integral = 0.0
                    pi_right_integral = 0.0
                    target_left = target_right = 0
                else:
                    rpm_left = left["erpm"] / VESC_LEFT_POLE_PAIRS
                    rpm_right = right["erpm"] / VESC_RIGHT_POLE_PAIRS
                    base_left = target_rpm * VESC_LEFT_POLE_PAIRS
                    base_right = target_rpm * VESC_RIGHT_POLE_PAIRS

                    error_left = target_rpm - rpm_left
                    error_right = target_rpm - rpm_right
                    pi_left_integral += error_left * dt
                    pi_right_integral += error_right * dt
                    trim_left = self._clamp(
                        VESC_SPEED_KP * error_left + VESC_SPEED_KI * pi_left_integral,
                        -VESC_MAX_TRIM_ERPM, VESC_MAX_TRIM_ERPM,
                    )
                    trim_right = self._clamp(
                        VESC_SPEED_KP * error_right + VESC_SPEED_KI * pi_right_integral,
                        -VESC_MAX_TRIM_ERPM, VESC_MAX_TRIM_ERPM,
                    )
                    sync = self._clamp(
                        VESC_SYNC_KP * (rpm_left - rpm_right),
                        -VESC_MAX_SYNC_CORRECTION_ERPM,
                        VESC_MAX_SYNC_CORRECTION_ERPM,
                    )
                    target_left = int(base_left + trim_left - sync)
                    target_right = int(base_right + trim_right + sync)

                vesc_send_set_rpm(self._ser_left, target_left)
                vesc_send_set_rpm(self._ser_right, target_right)
                print(
                    "[VESC] VRB={} -> target={} RPM | "
                    "L cmd={} ERPM actual={} ERPM ({:.1f} RPM) | "
                    "R cmd={} ERPM actual={} ERPM ({:.1f} RPM)".format(
                        "STOP" if speed_value is None else speed_value,
                        0 if target_rpm is None else int(target_rpm),
                        target_left,
                        left["erpm"],
                        left["erpm"] / VESC_LEFT_POLE_PAIRS,
                        target_right,
                        right["erpm"],
                        right["erpm"] / VESC_RIGHT_POLE_PAIRS,
                    ),
                    flush=True,
                )
                self._stop_event.wait(VESC_LOOP_INTERVAL)

        except Exception as exc:
            self.error = str(exc)
            print("[VESC ERROR] 停止雙馬達控制：{}".format(exc))
        finally:
            self._stop_motors()
            for ser in (self._ser_left, self._ser_right):
                if ser is not None and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass
            print("[VESC] 雙馬達控制執行緒已關閉。")

    def close(self, timeout=3.0):
        self.set_speed(None)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # 若執行緒仍在 telemetry timeout，盡可能送出停止命令。
        self._stop_motors()


def enable_motor_compat(servo, enabled=True):
    method_globals = servo.enable_motor.__func__.__globals__
    commands = method_globals["MksCommands"]
    value = Enable.Enable.value if enabled else Enable.Disable.value
    return servo.set_generic_status(commands.ENABLE_MOTOR_COMMAND, [value])


def dm_startup_arguments():
    return [
        "--port", DM_STARTUP_TEST_PORT,
        "--yaw", str(DM_STARTUP_TEST_YAW_DEG),
        "--pitch", str(DM_STARTUP_TEST_PITCH_DEG),
        "--speed", str(DM_STARTUP_TEST_SPEED_RAD_S),
        "--move-kp", str(DM_STARTUP_TEST_MOVE_KP),
        "--kp", str(DM_STARTUP_TEST_KP),
        "--kd", str(DM_STARTUP_TEST_KD),
        "--pitch-tau-ff", str(DM_STARTUP_TEST_PITCH_TAU_FF),
    ]


def run_dm_startup_test():
    from turret_control.run_manual_move import main_keep_enabled
    main_keep_enabled(dm_startup_arguments(), confirm=False)


def return_dm_before_shutdown(motors):
    """使用既有連線，回到校正後 yaw 0°、pitch 35° 再結束。"""
    from turret_control.run_manual_move import DEFAULT_STARTUP_POSE, read_enabled_start
    from turret_control.pitch import load_pitch_calibration, load_yaw_calibration
    from turret_control.trajectory import smoothstep_sample

    cal = {3: load_yaw_calibration(DEFAULT_STARTUP_POSE),
           1: load_pitch_calibration(DEFAULT_STARTUP_POSE)}
    destination = {3: cal[3].motor_from_muzzle(0.0),
                   1: cal[1].motor_from_muzzle(math.radians(35.0))}
    start = read_enabled_start(motors)
    speed = DM_STARTUP_TEST_SPEED_RAD_S
    duration = max(1.5 * max(abs(destination[i] - start[i]) for i in (3, 1)) / speed, 1.0)
    motors.kp = DM_STARTUP_TEST_KP
    began = time.monotonic()
    settled = None
    next_status = began
    print(f"[Manual DM] 關閉前回位 yaw=0° pitch=35°，軌跡約 {duration:.1f}s。", flush=True)
    while True:
        now = time.monotonic()
        elapsed = now - began
        targets = {}
        for i in (3, 1):
            q, dq = smoothstep_sample(start[i], destination[i], elapsed, duration)
            targets[i] = (q, speed, dq, DM_STARTUP_TEST_PITCH_TAU_FF if i == 1 else 0.0)
        motors.send_targets(targets)
        motors.refresh()
        states = {i: motors.get_state(i) for i in (3, 1)}
        for i, state in states.items():
            if (state is None or state.err != 1 or state.feedback_at is None
                    or time.monotonic() - state.feedback_at > 0.5
                    or not math.isfinite(state.position_rad)):
                raise RuntimeError(f"DM 回位時 0x{i:02X} 回授異常")
        arrived = elapsed >= duration and all(
            abs(math.degrees(cal[i].muzzle_from_motor(states[i].position_rad)
                             - cal[i].muzzle_from_motor(destination[i]))) <= DM_POSITION_TOLERANCE_DEG
            for i in (3, 1)
        )
        if now >= next_status:
            print(f"[Manual DM 回位] {elapsed:.1f}s " + " ".join(
                f"{name}={math.degrees(cal[i].muzzle_from_motor(states[i].position_rad)):+.1f}°"
                for i, name in ((3, "yaw"), (1, "pitch"))), flush=True)
            next_status = now + 1.0
        if arrived:
            if settled is None:
                settled = now
            if now - settled >= 0.05:
                print("[Manual DM] 已回位，準備失能。", flush=True)
                return
        else:
            settled = None
        if elapsed > duration + 10.0:
            raise TimeoutError("DM 關閉前回位逾時")
        time.sleep(0.005)


class DMStartupController:
    """DM 連線只由背景執行緒操作；UART 只提交一筆不可覆寫的目標。"""

    def __init__(self):
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = None
        self.target_lock = threading.Lock()
        self.yaw_min_deg, self.yaw_max_deg = DM_YAW_MIN_DEG, DM_YAW_MAX_DEG
        self.pitch_min_deg, self.pitch_max_deg = DM_PITCH_MIN_DEG, DM_PITCH_MAX_DEG
        self.yaw_deg = max(self.yaw_min_deg, min(self.yaw_max_deg, DM_STARTUP_TEST_YAW_DEG))
        self.pitch_deg = max(self.pitch_min_deg, min(self.pitch_max_deg, DM_STARTUP_TEST_PITCH_DEG))
        self.moving = True
        self._pending = None
        self._motion_start = None
        self._settled_at = None
        self._states = None
        self._feedback_wait_started = None
        self._last_busy_notice = None
        self._last_limit_notice = None
        self._next_record_print = 0.0
        self.thread = threading.Thread(target=self._run, name="Manual-DM", daemon=True)

    def update_from_uart(self, yaw, pitch):
        """新目標 = 已記錄角度 + (UART - 50) * 比例 * 方向；未到位時忽略。"""
        with self.target_lock:
            dy = (max(0, min(100, int(yaw))) - DM_UART_CENTER) * DM_UART_DEG_PER_UNIT * DM_UART_YAW_SIGN
            dp = (max(0, min(100, int(pitch))) - DM_UART_CENTER) * DM_UART_DEG_PER_UNIT * DM_UART_PITCH_SIGN
            if dy == 0 and dp == 0:
                return False
            if self.stop_event.is_set() or self.error is not None:
                return False
            now = time.monotonic()
            if self._states is None or any(
                    now - s.feedback_at > DM_FEEDBACK_MAX_AGE_S for s in self._states.values()):
                return False
            at_position = self._at_position(self._states)
            # 無論目前是否仍在移動鎖定，只要位於任一邊界，都統一檢查是否為向內退回。
            # 涵蓋 pitch 上/下限，以及 yaw 左/右限；不再只依賴移動逾時狀態。
            retreats = self._retreat_deltas(dy, dp, now)
            if self.moving and not retreats:
                if self._last_busy_notice is None or now - self._last_busy_notice >= 1.0:
                    detail = self._position_error_text(self._states) if self._states is not None else "等待有效回授"
                    print(f"[Manual DM 鎖定] 忽略新方向：{detail}；pitch增量={dp:+.2f}° "
                          f"軌跡完成={self._motion_start is not None and now - self._motion_start >= self._duration} "
                          f"位於邊界={self._target_at_any_limit()} "
                          f"待處理目標={self._pending is not None}；到位門檻 ±{DM_POSITION_TOLERANCE_DEG:g}°。", flush=True)
                    self._last_busy_notice = now
                return False
            if not at_position and not retreats:
                return False
            raw_yaw, raw_pitch = self.yaw_deg + dy, self.pitch_deg + dp
            anchors = {}
            if retreats:
                raw_yaw, raw_pitch = self.yaw_deg, self.pitch_deg
                for i, delta in retreats.items():
                    actual = math.degrees(self._cal[i].muzzle_from_motor(self._states[i].position_rad))
                    recorded = self.yaw_deg if i == 3 else self.pitch_deg
                    anchor = min(recorded, actual) if delta < 0 else max(recorded, actual)
                    anchors[i] = anchor
                    if i == 3:
                        raw_yaw = anchor + delta
                    else:
                        raw_pitch = anchor + delta
            if raw_pitch < DM_PITCH_MIN_DEG:
                raw_pitch = DM_PITCH_MIN_DEG
            if raw_pitch > DM_PITCH_MAX_DEG:
                raw_pitch = DM_PITCH_MAX_DEG
            if raw_yaw < DM_YAW_MIN_DEG:
                raw_yaw = DM_YAW_MIN_DEG
            if raw_yaw > DM_YAW_MAX_DEG:
                raw_yaw = DM_YAW_MAX_DEG
            requested = self._bounded_target(raw_yaw, raw_pitch, self._states)
            if requested is None:
                if retreats:
                    print("[Manual DM 退回拒絕] 工作空間沒有合法退回目標。", flush=True)
                return False
            yaw_deg = math.degrees(self._cal[3].muzzle_from_motor(requested[3]))
            pitch_deg = math.degrees(self._cal[1].muzzle_from_motor(requested[1]))
            if retreats:
                accepted = {3: yaw_deg, 1: pitch_deg}
                recorded = {3: self.yaw_deg, 1: self.pitch_deg}
                # 裁切後仍須確實向內；沒有退回要求的軸維持原目標。
                if any((accepted[i] - anchors[i]) * (1 if delta > 0 else -1) <= 1e-6
                       for i, delta in retreats.items()) or any(
                       abs(accepted[i] - recorded[i]) > 1e-6 for i in (3, 1) if i not in retreats):
                    print(f"[Manual DM 退回拒絕] 裁切後 yaw={yaw_deg:+.2f}° pitch={pitch_deg:+.2f}°，"
                          "無法保持其他軸並向邊界內退回。", flush=True)
                    return False
            if abs(yaw_deg - raw_yaw) > 1e-6 or abs(pitch_deg - raw_pitch) > 1e-6:
                if self._last_limit_notice is None or now - self._last_limit_notice >= 1.0:
                    print(f"[Manual DM 邊界裁切] 請求 yaw={raw_yaw:+.2f}° pitch={raw_pitch:+.2f}° "
                          f"→ 紀錄 yaw={yaw_deg:+.2f}° pitch={pitch_deg:+.2f}°", flush=True)
                    self._last_limit_notice = now
            if abs(yaw_deg - self.yaw_deg) < 1e-6 and abs(pitch_deg - self.pitch_deg) < 1e-6:
                return False
            self.yaw_deg, self.pitch_deg = yaw_deg, pitch_deg
            self._pending = requested
            self.moving = True
            if retreats:
                axes = ", ".join("yaw" if i == 3 else "pitch" for i in retreats)
                print(f"[Manual DM 邊界退回] {axes} 向內退回，其他軸保持原目標。", flush=True)
            print(f"[Manual DM 目標] yaw={yaw_deg:+.2f}° pitch={pitch_deg:+.2f}°；到位前鎖定。", flush=True)
            return True

    def _target_at_limit(self, motor_id, side):
        """side=-1 為下/左界，side=+1 為上/右界，包含工作空間裁切。"""
        from turret_control.limits import max_activity_pitch_for_yaw_deg
        if motor_id == 3:
            value, lower, upper = self.yaw_deg, self.yaw_min_deg, self.yaw_max_deg
        else:
            value, lower = self.pitch_deg, self.pitch_min_deg
            upper = min(self.pitch_max_deg, max_activity_pitch_for_yaw_deg(self.yaw_deg))
        if ((side < 0 and value <= lower + DM_POSITION_TOLERANCE_DEG)
                or (side > 0 and value >= upper - DM_POSITION_TOLERANCE_DEG)):
            return True
        if self._states is None or self._motion_start is None:
            return False
        yaw, pitch = self.yaw_deg, self.pitch_deg
        if motor_id == 3:
            yaw += side * DM_POSITION_TOLERANCE_DEG
        else:
            pitch += side * DM_POSITION_TOLERANCE_DEG
        probe = self._bounded_target(yaw, pitch, self._states)
        if probe is None:
            return False
        projected = math.degrees(self._cal[motor_id].muzzle_from_motor(probe[motor_id]))
        return (projected - value) * side <= 1e-6

    def _target_at_any_limit(self):
        return any(self._target_at_limit(i, side) for i in (3, 1) for side in (-1, 1))

    def _retreat_deltas(self, dy, dp, now):
        """軌跡完成且回授正常時，僅接受各邊界往內的增量，不排隊。"""
        if (self._motion_start is None or self._pending is not None or self._states is None
                or now - self._motion_start < self._duration
                or any(s.err != 1 or s.feedback_at is None
                       or now - s.feedback_at > DM_FEEDBACK_MAX_AGE_S
                       or not math.isfinite(s.position_rad) or not math.isfinite(s.velocity_rad_s)
                       for s in self._states.values())):
            return {}
        return {i: delta for i, delta in ((3, dy), (1, dp))
                if delta != 0 and self._target_at_limit(i, -1 if delta > 0 else 1)}

    def _pitch_target_at_upper_limit(self):
        return self._target_at_limit(1, 1)

    def _can_retreat_pitch(self, dp, now):
        return 1 in self._retreat_deltas(0, dp, now)

    def _bounded_target(self, yaw_deg, pitch_deg, states):
        """先限制實際角度，再限制馬達座標與工作空間，最後才寫入紀錄。"""
        from turret_control.limits import (
            clamp_motor_position, clamp_workspace_target, max_activity_pitch_for_yaw_deg,
        )
        yaw_deg = max(self.yaw_min_deg, min(self.yaw_max_deg, yaw_deg))
        pitch_max = min(self.pitch_max_deg, max_activity_pitch_for_yaw_deg(yaw_deg))
        if pitch_max < self.pitch_min_deg:
            return None
        pitch_deg = max(self.pitch_min_deg, min(pitch_max, pitch_deg))
        requested = {3: self._cal[3].motor_from_muzzle(math.radians(yaw_deg)),
                     1: self._cal[1].motor_from_muzzle(math.radians(pitch_deg))}
        for i in (3, 1):
            requested[i], _ = clamp_motor_position(requested[i], self._limits[i])
        if self._workspace is not None:
            safe, _ = clamp_workspace_target(
                self._workspace, states[3].position_rad, states[1].position_rad,
                requested[3], requested[1])
            requested[3], requested[1] = safe
        # 工作空間裁切後再次確認，不能讓紀錄超過任何單軸或聯合邊界。
        yaw_deg = math.degrees(self._cal[3].muzzle_from_motor(requested[3]))
        pitch_deg = math.degrees(self._cal[1].muzzle_from_motor(requested[1]))
        if (not self.yaw_min_deg - 1e-9 <= yaw_deg <= self.yaw_max_deg + 1e-9
                or not self.pitch_min_deg - 1e-9 <= pitch_deg <= self.pitch_max_deg + 1e-9
                or pitch_deg > max_activity_pitch_for_yaw_deg(yaw_deg) + 1e-9):
            return None
        return requested

    def _position_error_text(self, states):
        return " ".join(
            f"{name}誤差={math.degrees(self._cal[i].muzzle_from_motor(states[i].position_rad) - self._cal[i].muzzle_from_motor(self._destination[i])):+.2f}°"
            for i, name in ((3, "yaw"), (1, "pitch"))
        )

    def _load_recorded_angle_limits(self):
        """將 limits.py 的馬達座標邊界換算成紀錄角度，取設定與實測交集。"""
        from turret_control.run_manual_move import DEFAULT_STARTUP_POSE
        from turret_control.pitch import load_yaw_calibration, load_pitch_calibration
        from turret_control.limits import (
            load_motor_limits, load_workspace_limit,
            constrain_yaw_design_limit, constrain_pitch_activity_limit,
        )
        self._cal = {3: load_yaw_calibration(DEFAULT_STARTUP_POSE),
                     1: load_pitch_calibration(DEFAULT_STARTUP_POSE)}
        self._limits = load_motor_limits()
        configured = {3: (DM_YAW_MIN_DEG, DM_YAW_MAX_DEG),
                      1: (DM_PITCH_MIN_DEG, DM_PITCH_MAX_DEG)}
        constraints = {3: constrain_yaw_design_limit, 1: constrain_pitch_activity_limit}
        degrees = {}
        for i, (low_deg, high_deg) in configured.items():
            if not all(math.isfinite(v) for v in (low_deg, high_deg)) or low_deg >= high_deg:
                raise ValueError(f"DM 0x{i:02X} 角度邊界設定無效")
            low, high = constraints[i](self._limits[i], self._cal[i])
            configured_rad = sorted(self._cal[i].motor_from_muzzle(math.radians(v))
                                    for v in (low_deg, high_deg))
            low, high = max(low, configured_rad[0]), min(high, configured_rad[1])
            if low >= high:
                raise ValueError(f"DM 0x{i:02X} 設定範圍與既有機構邊界沒有交集")
            self._limits[i] = (low, high)
            degrees[i] = sorted(math.degrees(self._cal[i].muzzle_from_motor(v)) for v in (low, high))
        self.yaw_min_deg, self.yaw_max_deg = degrees[3]
        self.pitch_min_deg, self.pitch_max_deg = degrees[1]
        self._workspace = load_workspace_limit()
        print(f"[Manual DM 邊界] yaw=[{self.yaw_min_deg:+.2f}, {self.yaw_max_deg:+.2f}]° "
              f"pitch=[{self.pitch_min_deg:+.2f}, {self.pitch_max_deg:+.2f}]°", flush=True)

    def _at_position(self, states):
        """UART 接受與移動完成共用同一到位判定，避免解鎖後仍拒收。"""
        return all(
            abs(math.degrees(self._cal[i].muzzle_from_motor(states[i].position_rad)
                             - self._cal[i].muzzle_from_motor(self._destination[i])))
            <= DM_POSITION_TOLERANCE_DEG and abs(states[i].velocity_rad_s) <= 0.05
            for i in (3, 1)
        )

    def _update_targets(self, motors, targets):
        from turret_control.trajectory import smoothstep_sample
        now = time.monotonic()
        states = {i: motors.get_state(i) for i in (3, 1)}
        stale = []
        for i, state in states.items():
            if state is None or state.feedback_at is None:
                stale.append(f"0x{i:02X} 無回授")
                continue
            age = now - state.feedback_at
            if age > DM_FEEDBACK_MAX_AGE_S:
                stale.append(f"0x{i:02X} err={state.err} 回授距今={age:.3f}s")
                continue
            if state.err != 1 or not math.isfinite(state.position_rad):
                raise RuntimeError(
                    f"DM 定位時 0x{i:02X} err={state.err} "
                    f"位置={state.position_rad} 回授距今={age:.3f}s "
                    "(err: 0=失能, 1=使能, 8..14=故障)"
                )
        with self.target_lock:
            if stale:
                self._states = None  # 等待回授時禁止接受 UART 新目標。
                self._settled_at = None
                if self._feedback_wait_started is None:
                    self._feedback_wait_started = now
                    print("[Manual DM] 暫停更新目標、持續保持並等待新回授：" + "; ".join(stale), flush=True)
                if now - self._feedback_wait_started >= DM_FEEDBACK_RECOVERY_TIMEOUT_S:
                    raise TimeoutError("DM 等待新回授逾時：" + "; ".join(stale))
                return {i: (v[0], v[1], 0.0, v[3]) for i, v in targets.items()}
            if self._feedback_wait_started is not None:
                if self._motion_start is not None:
                    self._motion_start += now - self._feedback_wait_started
                self._feedback_wait_started = None
                print("[Manual DM] 新回授已恢復。", flush=True)
            if self._motion_start is None:
                self._load_recorded_angle_limits()
                self._destination = self._bounded_target(
                    math.degrees(self._cal[3].muzzle_from_motor(targets[3][0])),
                    math.degrees(self._cal[1].muzzle_from_motor(targets[1][0])), states)
                if self._destination is None:
                    raise RuntimeError("DM 啟動目標與角度/工作空間邊界沒有合法交集")
                self.yaw_deg = math.degrees(self._cal[3].muzzle_from_motor(self._destination[3]))
                self.pitch_deg = math.degrees(self._cal[1].muzzle_from_motor(self._destination[1]))
                self._motion_start = now
                self._duration = 0.0
                self._start_positions = dict(self._destination)
            self._states = states
            if self._pending is not None:
                self._destination = self._pending
                self._pending = None
                self._start_positions = {i: states[i].position_rad for i in (3, 1)}
                self._duration = max(1.5 * max(abs(self._destination[i] - self._start_positions[i])
                                             for i in (3, 1)) / DM_STARTUP_TEST_SPEED_RAD_S, 0.25)
                self._motion_start = now
                self._settled_at = None
            elapsed = now - self._motion_start
            result = {}
            for i in (3, 1):
                if self._duration > 0:
                    q, dq = smoothstep_sample(self._start_positions[i], self._destination[i], elapsed, self._duration)
                else:
                    q, dq = self._destination[i], 0.0
                result[i] = (q, DM_STARTUP_TEST_SPEED_RAD_S, dq,
                             DM_STARTUP_TEST_PITCH_TAU_FF if i == 1 else 0.0)
            # 插值過程也逐筆檢查，不能只限制最後的紀錄目標。
            bounded = self._bounded_target(
                math.degrees(self._cal[3].muzzle_from_motor(result[3][0])),
                math.degrees(self._cal[1].muzzle_from_motor(result[1][0])), states)
            if bounded is None:
                raise RuntimeError("DM 插值目標超出角度/工作空間邊界")
            for i in (3, 1):
                q, speed, dq, tau = result[i]
                result[i] = (bounded[i], speed, 0.0 if abs(bounded[i] - q) > 1e-9 else dq, tau)
            if now >= self._next_record_print:
                from turret_control.limits import max_activity_pitch_for_yaw_deg
                print(f"[Manual DM 紀錄] yaw_deg={self.yaw_deg:+.2f}° pitch_deg={self.pitch_deg:+.2f}° "
                      f"| yaw邊界=[{self.yaw_min_deg:+.2f},{self.yaw_max_deg:+.2f}]° "
                      f"pitch邊界=[{self.pitch_min_deg:+.2f},{self.pitch_max_deg:+.2f}]° "
                      f"此yaw的pitch上限={min(self.pitch_max_deg, max_activity_pitch_for_yaw_deg(self.yaw_deg)):.2f}° "
                      f"| {'移動鎖定' if self.moving else '可接受新目標'}", flush=True)
                print("[Manual DM 實際/送出] " + " ".join(
                    f"{name}實際={math.degrees(self._cal[i].muzzle_from_motor(states[i].position_rad)):+.2f}° "
                    f"命令={math.degrees(self._cal[i].muzzle_from_motor(result[i][0])):+.2f}°"
                    for i, name in ((3, "yaw"), (1, "pitch"))), flush=True)
                self._next_record_print = now + 1.0
            arrived = elapsed >= self._duration and self._at_position(states)
            if self.moving:
                if arrived:
                    if self._settled_at is None:
                        self._settled_at = now
                    if now - self._settled_at >= 0.05:
                        self.moving = False
                        self._last_busy_notice = None
                        print("[Manual DM] 已定位，可接受下一筆方向增量。", flush=True)
                else:
                    self._settled_at = None
                if self.moving and elapsed > self._duration + MOVE_TIMEOUT:
                    if self._target_at_any_limit():
                        if self._last_busy_notice is None or now - self._last_busy_notice >= 1.0:
                            print("[Manual DM 邊界等待] 保持位置，可輸入 yaw/pitch 向邊界內退回："
                                  + self._position_error_text(states), flush=True)
                            self._last_busy_notice = now
                    else:
                        raise TimeoutError("DM 未到定位點，停止接受新角度：" + self._position_error_text(states))
            return result

    def _run(self):
        try:
            from turret_control.run_manual_move import main_keep_enabled
            main_keep_enabled(
                dm_startup_arguments(), confirm=False,
                stop_event=self.stop_event, ready_event=self.ready_event,
                on_shutdown=return_dm_before_shutdown, update_targets=self._update_targets)
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready_event.set()

    def start(self):
        self.thread.start()
        while not self.ready_event.wait(0.05):
            pass
        self.check_health()
        print("[Manual DM] 背景保持已啟動，繼續初始化 MKS/VESC。", flush=True)

    def check_health(self):
        if self.error is not None:
            raise RuntimeError(f"DM 控制失敗：{self.error}") from self.error
        if self.thread.ident is not None and not self.thread.is_alive() and not self.stop_event.is_set():
            raise RuntimeError("DM 控制執行緒意外結束")

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()
        if self.error is not None:
            print(f"[Manual DM] 關閉時控制錯誤：{self.error}", flush=True)


class ManualController:
    def __init__(self, uart):
        if DM_STARTUP_CLI_TEST_ONLY:
            run_dm_startup_test()
            raise SystemExit(0)
        self.uart = uart
        self.dm_controller = DMStartupController()
        self.ammo_bus = None
        self.ammo_notifier = None
        self.shoot_bus = None
        self.shoot_notifier = None
        self.ammo_servo = None
        self.shoot_servo = None

        self.lock = threading.Lock()
        self.busy = False
        self.last_fire = 0
        self.current_slot = 1
        self.shot_stage = 0
        self.reload_pending = False
        self.shot_error = None
        self.current_mode = "M"
        self.last_yaw = 50
        self.last_pitch = 50
        self.last_speed = 0
        self.vesc_controller = VESCDualMotorController()
        self.last_vx = 0
        self.last_vy = 0
        self.last_wz = 0
        try:
            self.dm_controller.start()
            self._initialize_can()
            self.vesc_controller.start()
        except BaseException:
            self.close()
            raise

    def _initialize_can(self):
        print("[Manual] 初始化彈藥倉 CAN：{}".format(AMMO_CAN_INTERFACE))
        self.ammo_bus = can.Bus(
            interface="socketcan", channel=AMMO_CAN_INTERFACE
        )
        self.ammo_notifier = can.Notifier(self.ammo_bus, [])
        self.ammo_servo = MksServo(
            self.ammo_bus, self.ammo_notifier, AMMO_MOTOR_CAN_ID
        )

        if SHOOTING_CAN_INTERFACE == AMMO_CAN_INTERFACE:
            # Share one receiver on a shared CAN channel so MKS replies cannot
            # be consumed by a second notifier attached to the same interface.
            self.shoot_bus = self.ammo_bus
            self.shoot_notifier = self.ammo_notifier
        else:
            self.shoot_bus = can.Bus(
                interface="socketcan", channel=SHOOTING_CAN_INTERFACE
            )
            self.shoot_notifier = can.Notifier(self.shoot_bus, [])
        print("[Manual] 初始化射擊機構 CAN：{}".format(SHOOTING_CAN_INTERFACE))
        self.shoot_servo = MksServo(
            self.shoot_bus, self.shoot_notifier, SHOOTING_MOTOR_CAN_ID
        )

        for label, servo in (
            ("彈藥倉", self.ammo_servo),
            ("射擊機構", self.shoot_servo),
        ):
            result = servo.set_work_mode(WorkMode.SrvFoc)
            print("[Manual] {}工作模式：{}".format(label, result))
            if result != SuccessStatus.Success:
                raise RuntimeError("{}工作模式設定失敗".format(label))
            time.sleep(0.2)

            result = enable_motor_compat(servo, True)
            print("[Manual] {}啟用：{}".format(label, result))
            if result != SuccessStatus.Success:
                raise RuntimeError("{}馬達啟用失敗".format(label))
            time.sleep(0.2)

    def check_health(self):
        self.dm_controller.check_health()

    @property
    def dm_yaw_deg(self):
        return self.dm_controller.yaw_deg

    @property
    def dm_pitch_deg(self):
        return self.dm_controller.pitch_deg

    def on_mode_change(self, mode, fire=None):
        self.current_mode = mode
        # 記住切換當下的 FIRE 狀態，避免按住按鈕切入手動時誤觸發射擊。
        self.last_fire = 1 if fire else 0
        if mode != "M":
            # 切到自動模式時停止手動射擊雙 VESC 馬達。
            self.last_speed = 0
            if getattr(self, "vesc_controller", None) is not None:
                self.vesc_controller.set_speed(None)
            # 切換時透過既有 UART 協定送出中立方向值。
            self._send_turret_command(50, 50, 0, 0, mode="A")

    def handle_packet(self, packet):
        """處理一筆 ESP32 遙控器封包。"""
        self.last_yaw = self._clamp(packet.get("yaw", 50), 0, 100)
        self.last_pitch = self._clamp(packet.get("pitch", 50), 0, 100)
        raw_vrb = packet.get("speed", 0)
        self.last_speed = normalize_vrb_speed(raw_vrb)
        self.vesc_controller.set_speed(self.last_speed)
        print(
            "[Manual RX] mode={} | yaw={} pitch={} fire={} VRB raw={} normalized={} -> VESC target={} RPM".format(
                packet.get("mode", self.current_mode),
                self.last_yaw,
                self.last_pitch,
                packet.get("fire", 0),
                raw_vrb,
                self.last_speed,
                int(VESC_MIN_RPM + self.last_speed * VESC_RPM_PER_SPEED_UNIT),
            ),
            flush=True,
        )
        self.last_vx = self._clamp(packet.get("vx", 0), -100, 100)
        self.last_vy = self._clamp(packet.get("vy", 0), -100, 100)
        self.last_wz = self._clamp(packet.get("wz", 0), -100, 100)

        packet_mode = packet.get("mode", self.current_mode)
        if packet_mode != self.current_mode:
            self.on_mode_change(packet_mode, fire=packet.get("fire", 0))

        if self.current_mode == "M":
            self.dm_controller.update_from_uart(self.last_yaw, self.last_pitch)

        fire_now = 1 if packet.get("fire", 0) else 0
        fire_rising_edge = fire_now == 1 and self.last_fire == 0
        self.last_fire = fire_now

        if fire_rising_edge:
            self.request_shot()

    def request_shot(self):
        """每次 FIRE 上升沿只執行目前的一個階段，不排隊。"""
        with self.lock:
            if self.shot_error is not None:
                print("[Manual Error] 動作已因錯誤停止，不再重試：{}".format(self.shot_error), flush=True)
                return
            if self.busy:
                print("[Manual] 馬達正在運動，忽略此次射擊要求。")
                return
            if self.current_mode != "M":
                print("[Manual] 目前不是手動模式，拒絕射擊。")
                return
            self.busy = True

        worker = threading.Thread(target=self._shot_worker, daemon=True)
        worker.start()

    def _shot_worker(self):
        try:
            stage = self.shot_stage
            print("[Manual] FIRE：執行第 {}/3 階段，彈藥格位 {}。".format(
                stage + 1, self.current_slot
            ), flush=True)
            self._run_one_shot_cycle()
            if stage < 2:
                self.shot_stage = stage + 1
                print("[Manual] 本階段完成，放開 FIRE 後再按一次執行第 {}/3 階段。".format(
                    self.shot_stage + 1
                ), flush=True)
                return

            if self.current_mode != "M":
                print("[Manual] ID1 已回零；ID2 待轉格，切回手動後重新按 FIRE。", flush=True)
                return
            print("[Manual] ID1 回零完成，開始 ID2 轉格。", flush=True)
            self._rotate_ammo_one_slot()
            self.current_slot = self.current_slot % SLOTS_PER_MAGAZINE + 1
            self.reload_pending = False
            self.shot_stage = 0
            print("[Manual] ID2 轉格完成，目前格位 {}；下一次 FIRE 從第 1 階段開始。".format(
                self.current_slot
            ), flush=True)
        except Exception as exc:
            with self.lock:
                self.shot_error = "第 {}/3 階段失敗：{}".format(self.shot_stage + 1, exc)
            print("[Manual Error] {}；已停止，不再重試。".format(self.shot_error), flush=True)
            self._emergency_stop(self.shoot_servo, "射擊機構")
            self._emergency_stop(self.ammo_servo, "彈藥倉")
        finally:
            with self.lock:
                self.busy = False

    def _run_one_shot_cycle(self):
        """一次只執行一個 ID1 階段；第三階段成功後由 worker 觸發 ID2。"""
        if self.shot_stage == 0:
            self._move_relative(
                self.shoot_servo, TARGET_POSITIONS[0], MOVE_DIRECTION, SHOOTING_SPEED_RPM
            )
        elif self.shot_stage == 1:
            self._move_relative(
                self.shoot_servo, TARGET_POSITIONS[1] - TARGET_POSITIONS[0],
                MOVE_DIRECTION, SHOOTING_SPEED_RPM
            )
        elif self.shot_stage == 2:
            if not self.reload_pending:
                self._go_home(self.shoot_servo, MOVE_DIRECTION, "反向無限位回零")
                time.sleep(SETTLE_TIME)
                self._go_home(self.shoot_servo, HOME_DIRECTION, "正向無限位回零")
                self.reload_pending = True
        else:
            raise RuntimeError("未知射擊階段：{}".format(self.shot_stage))

    def _move_relative(self, servo, pulses, direction, speed):
        before = servo.read_encoder_value_addition()
        if before is None:
            raise RuntimeError("移動前 Encoder 讀取失敗")

        result = servo.run_motor_relative_motion_by_pulses(
            direction, speed, ACCELERATION, pulses
        )
        print("[Manual] 相對移動 {} pulses，回傳：{}".format(pulses, result))
        if result is None:
            raise RuntimeError("相對移動指令沒有收到有效回覆")

        start = time.monotonic()
        seen_running = False
        movement_seen = False
        stopped_at = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed > MOVE_TIMEOUT:
                self._emergency_stop(servo, "相對移動逾時")
                raise TimeoutError("相對位置移動逾時")

            running = servo.is_motor_running()
            now_encoder = servo.read_encoder_value_addition()
            if now_encoder is not None and now_encoder != before:
                movement_seen = True

            if running:
                seen_running = True
                stopped_at = None
            elif seen_running or movement_seen:
                if stopped_at is None:
                    stopped_at = time.monotonic()
                elif time.monotonic() - stopped_at >= SETTLE_TIME:
                    break
            elif elapsed > 2.0:
                raise RuntimeError("馬達未偵測到移動")

            time.sleep(POLL_INTERVAL)

        after = servo.read_encoder_value_addition()
        if after is None:
            raise RuntimeError("移動後 Encoder 讀取失敗")
        print("[Manual] Encoder 位移：{}".format(after - before))

    def _go_home(self, servo, direction, label):
        print("[Manual] 開始{}".format(label))
        home_data = [
            0,
            direction.value,
            (HOME_SPEED_RPM >> 8) & 0x0F,
            HOME_SPEED_RPM & 0xFF,
            0,
            1,  # noLimit 模式
        ]
        result = servo.set_generic_status(
            MksCommands.SET_HOME_COMMAND, home_data
        )
        if result != SuccessStatus.Success:
            raise RuntimeError("{}參數設定失敗".format(label))

        home_current_data = [
            (REVERSE_ANGLE >> 24) & 0xFF,
            (REVERSE_ANGLE >> 16) & 0xFF,
            (REVERSE_ANGLE >> 8) & 0xFF,
            REVERSE_ANGLE & 0xFF,
            (HOME_CURRENT_MA >> 8) & 0xFF,
            HOME_CURRENT_MA & 0xFF,
        ]
        result = servo.set_generic_status(
            MksCommands.SET_NO_LIMIT_GO_HOME_COMMAND, home_current_data
        )
        if result != SuccessStatus.Success:
            raise RuntimeError("{}電流參數設定失敗".format(label))

        response = servo.set_generic(
            MksCommands.GO_HOME_COMMAND, servo.GENERIC_RESPONSE_LENGTH
        )
        if response is None or len(response) < 2:
            raise RuntimeError("{}沒有收到有效回覆".format(label))

        status = int(response[1])
        print("[Manual] {}初始狀態：{}".format(label, status))
        if status == 0:
            raise RuntimeError("驅動器拒絕{}".format(label))
        if status == 2:
            return
        if status != 1:
            raise RuntimeError("{}回覆未知狀態 {}".format(label, status))

        start = time.monotonic()
        seen_running = False
        movement_seen = False
        before = servo.read_encoder_value_addition()
        stopped_at = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed > HOME_TIMEOUT:
                self._emergency_stop(servo, "回零逾時")
                raise TimeoutError("{}逾時".format(label))

            running = servo.is_motor_running()
            now_encoder = servo.read_encoder_value_addition()
            if before is not None and now_encoder is not None and now_encoder != before:
                movement_seen = True

            if running:
                seen_running = True
                stopped_at = None
            elif seen_running or movement_seen:
                if stopped_at is None:
                    stopped_at = time.monotonic()
                elif time.monotonic() - stopped_at >= 1.0:
                    break
            elif elapsed > 2.0:
                raise RuntimeError("{}未偵測到馬達移動".format(label))

            time.sleep(POLL_INTERVAL)

        # 驅動器停止不一定代表回零成功；此處只確認動作已結束。
        print("[Manual] {}動作停止；請確認機構實際原點。".format(label))

    def _rotate_ammo_one_slot(self):
        print(
            "[Manual] 發送彈藥倉命令：CAN={} ID={} direction=CW speed={} RPM pulses={}".format(
                AMMO_CAN_INTERFACE,
                AMMO_MOTOR_CAN_ID,
                AMMO_SPEED_RPM,
                AMMO_SLOT_PULSES,
            ),
            flush=True,
        )
        before = self.ammo_servo.read_encoder_value_addition()
        if before is None:
            raise RuntimeError("彈藥倉移動前 Encoder 讀取失敗")

        result = self.ammo_servo.run_motor_relative_motion_by_pulses(
            Direction.CW, AMMO_SPEED_RPM, ACCELERATION, AMMO_SLOT_PULSES
        )
        print("[Manual] 彈藥倉轉一格指令回傳：", result)
        if result is None:
            raise RuntimeError("彈藥倉移動指令沒有有效回覆")
        print("[Manual] 彈藥倉驅動器回覆狀態：{}".format(result), flush=True)

        start = time.monotonic()
        seen_running = False
        movement_seen = False
        stopped_at = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed > MOVE_TIMEOUT:
                self._emergency_stop(self.ammo_servo, "彈藥倉轉格逾時")
                raise TimeoutError("彈藥倉轉格逾時")

            running = self.ammo_servo.is_motor_running()
            now_encoder = self.ammo_servo.read_encoder_value_addition()
            if (
                before not in (None, -1)
                and now_encoder not in (None, -1)
                and now_encoder != before
            ):
                movement_seen = True

            if running:
                seen_running = True
                stopped_at = None
            elif seen_running or movement_seen:
                if stopped_at is None:
                    stopped_at = time.monotonic()
                elif time.monotonic() - stopped_at >= SETTLE_TIME:
                    break
            elif elapsed > 2.0:
                raise RuntimeError("彈藥倉未偵測到移動")

            time.sleep(POLL_INTERVAL)

        after = self.ammo_servo.read_encoder_value_addition()
        if before in (None, -1) or after in (None, -1):
            print(
                "[Manual] 彈藥倉 Encoder 回授無效；以馬達運轉狀態確認動作停止。"
            )
        else:
            print("[Manual] 彈藥倉 Encoder 位移：{}".format(after - before))

    def _send_turret_command(self, yaw, pitch, fire, speed, mode="M"):
        # 保留遙控器方向值，由 ESP32 端依現有 UART 協定執行砲台方向控制。
        self.uart.send_packet(
            mode=mode,
            vx=0,
            vy=0,
            wz=0,
            yaw=self._clamp(yaw, 0, 100),
            pitch=self._clamp(pitch, 0, 100),
            fire=fire,
            speed=speed,
        )

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, int(value)))

    @staticmethod
    def _emergency_stop(servo, label):
        if servo is None:
            return
        try:
            servo.emergency_stop_motor()
            print("[Manual] {}已送出緊急停止。".format(label))
        except Exception as exc:
            print("[Manual] {}緊急停止失敗：{}".format(label, exc))

    def close(self):
        self.current_mode = "A"
        # 先停止 VESC/MKS，再等待 DM 回位。
        if getattr(self, "vesc_controller", None) is not None:
            try:
                self.vesc_controller.close()
            except Exception as exc:
                print("[Manual] VESC 雙馬達關閉失敗：{}".format(exc))

        self._emergency_stop(self.shoot_servo, "射擊機構")
        self._emergency_stop(self.ammo_servo, "彈藥倉")

        try:
            if getattr(self, "dm_controller", None) is not None:
                self.dm_controller.close()
        finally:
            for notifier in {self.shoot_notifier, self.ammo_notifier}:
                if notifier is not None:
                    notifier.stop()
            for bus in {self.shoot_bus, self.ammo_bus}:
                if bus is not None:
                    bus.shutdown()



if __name__ == "__main__":
    if DM_STARTUP_CLI_TEST_ONLY:
        run_dm_startup_test()
    else:
        print("測試模式已關閉；請執行 main.py 啟動完整控制流程。")

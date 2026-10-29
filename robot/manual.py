# manual.py
# 手動模式：ESP32 遙控器 + DM 雙軸砲台 + MKS 射擊機構/彈藥倉 + 雙 VESC 飛輪馬達
#
# 封包格式由 robot/comm_uart.py 定義：
# E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED
#
# 暫定硬體分配：
#   can0：彈藥倉馬達，ID 1，每格 6666 pulses
#   can1：MKS 射擊機構馬達，ID 1
#   VESC_LEFT_PORT / VESC_RIGHT_PORT：射擊雙馬達轉速控制（UART）
#
# 注意：
# 1. ESP32 的 yaw/pitch (0~100) 由 Xavier 轉成 DM 目標角：Yaw -35~+35°、Pitch 0~45°。
# 2. FIRE 每次由 0 變 1，觸發一個射擊循環。
# 3. 每個射擊循環包含 1700 -> 3000 pulses -> 反向回零 -> 正向回零。
# 4. 每格累積 3 次循環後，才允許彈藥倉轉動一格。
# 5. VESC speed=0..100 線性映射至 0..VESC_MAX_RPM；speed=0 送出 0 ERPM。
# 6. 請先確認 DM 與兩個 VESC 使用不同的序列埠，並卸除彈藥進行低速測試。

import time
import struct
import threading
from pathlib import Path

import serial

# 使用專案已驗證的 DM 雙軸控制器（包含加減速、回授及水平安全聯鎖）。
from turret_control.run_vertical_panel import MotorWorker
from turret_control.pitch import load_pitch_calibration, load_yaw_calibration

import can
from mks_servo_can import MksServo
from mks_servo_can.mks_enums import (
    Enable,
    Direction,
    WorkMode,
    SuccessStatus,
    MksCommands,
)


# -------------------- 可調整設定 --------------------
AMMO_CAN_INTERFACE = "can0"
SHOOTING_CAN_INTERFACE = "can1"

# DM USB-to-CAN 調試板（與 Xavier UART / MKS CAN 分開）
DM_PORT = "/dev/ttyACM0"
DM_STARTUP_POSE = Path(__file__).resolve().parent / "turret_control" / "startup_pose.json"
DM_CONTROL_MODE = "posvel"
DM_SPEED_RAD_S = 0.06
DM_ACCELERATION_RAD_S2 = 0.05
DM_TORQUE_FF_NM = 0.0
DM_TORQUE_RAMP_S = 1.5
DM_KP = 50.0
DM_KD = 1.5
DM_MAX_TRACKING_ERROR_DEG = 10.0
DM_STALL_TIMEOUT_S = 2.0
DM_MIN_PITCH_DEG = 0.0
DM_MAX_PITCH_DEG = 45.0
DM_MIN_YAW_DEG = -35.0
DM_MAX_YAW_DEG = 35.0
DM_MAX_PITCH_FOR_YAW_DEG = 35.0

# 雙 VESC 射擊馬達：請依實際裝置名稱調整。
# 注意：DM_PORT 使用 /dev/ttyACM0，因此 VESC 不可再使用同一個序列埠。
# 建議用 /dev/serial/by-id/ 確認每個 USB 裝置的固定名稱。
VESC_LEFT_PORT = "/dev/ttyACM1"
VESC_RIGHT_PORT = "/dev/ttyACM2"
VESC_BAUDRATE = 921600
VESC_LEFT_POLE_PAIRS = 7
VESC_RIGHT_POLE_PAIRS = 7
VESC_MAX_RPM = 3000.0          # speed=100 對應的目標 RPM
VESC_LOOP_INTERVAL = 0.10
VESC_TELEMETRY_TIMEOUT = 0.5
VESC_MAX_MOTOR_CURRENT_A = 50.0
VESC_MAX_INPUT_VOLTAGE_V = 30.0
VESC_MIN_INPUT_VOLTAGE_V = 18.0
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

AMMO_MOTOR_CAN_ID = 1
SHOOTING_MOTOR_CAN_ID = 1

AMMO_SLOT_PULSES = 6666
AMMO_SPEED_RPM = 10          # 初次整合先低速測試
SHOOTING_SPEED_RPM = 10      # 初次整合先低速測試
ACCELERATION = 2

HOME_DIRECTION = Direction.CW
MOVE_DIRECTION = (
    Direction.CCW if HOME_DIRECTION == Direction.CW else Direction.CW
)
HOME_SPEED_RPM = 10
HOME_CURRENT_MA = 200
REVERSE_ANGLE = 0x0400

TARGET_POSITIONS = (1700, 3000)
SLOTS_PER_MAGAZINE = 6
SHOTS_PER_SLOT = 3

MOVE_TIMEOUT = 30.0
HOME_TIMEOUT = 30.0
POLL_INTERVAL = 0.1
SETTLE_TIME = 0.2

# FIRE 指令若持續為 1，只在上升沿觸發一次，不會連續重複射擊。
# 射擊循環在背景執行緒處理；主迴圈仍可更新 VESC speed 與接收遙控器封包。
# ---------------------------------------------------


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
    speed 0..100 線性映射為 0..VESC_MAX_RPM，0 代表送出 0 ERPM。
    保留使用者原始程式的 PI 調速與交叉同步修正。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._speed_percent = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._ser_left = None
        self._ser_right = None
        self.error = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="VESC-Dual-Motor", daemon=True
        )
        self._thread.start()

    def set_speed(self, speed_percent):
        speed_percent = max(0, min(100, int(speed_percent)))
        with self._lock:
            self._speed_percent = speed_percent

    def _get_speed_percent(self):
        with self._lock:
            return self._speed_percent

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

                speed_percent = self._get_speed_percent()
                target_rpm = VESC_MAX_RPM * speed_percent / 100.0

                if target_rpm <= 0:
                    # speed=0 時清除積分並命令兩顆馬達停轉。
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
                    "[VESC] speed={}%, target={} RPM | L={} ERPM, R={} ERPM".format(
                        speed_percent, int(target_rpm), left["erpm"], right["erpm"]
                    )
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
        self.set_speed(0)
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


class ManualController:
    def __init__(self, uart):
        self.uart = uart
        self.ammo_bus = None
        self.ammo_notifier = None
        self.shoot_bus = None
        self.shoot_notifier = None
        self.ammo_servo = None
        self.shoot_servo = None

        self.lock = threading.Lock()
        self.busy = False
        self.last_fire = 0
        self.shots_in_slot = 0
        self.current_slot = 1
        self.current_mode = "M"
        self.last_yaw = 50
        self.last_pitch = 50
        self.last_speed = 0
        self.vesc_controller = VESCDualMotorController()
        self.last_vx = 0
        self.last_vy = 0
        self.last_wz = 0
        self._initialize_can()
        self._initialize_dm_turret()
        self.vesc_controller.start()
        self._turret_enable_requested = False
        self._last_turret_command = None

    def _initialize_dm_turret(self):
        """啟動 DM 雙軸控制執行緒；啟動時不會立即使能馬達。"""
        print("[Manual] 初始化 DM 水平/垂直砲台，USB-CAN：{}".format(DM_PORT))
        pitch_calibration = load_pitch_calibration(DM_STARTUP_POSE)
        yaw_calibration = load_yaw_calibration(DM_STARTUP_POSE)
        self.turret_worker = MotorWorker(
            port=DM_PORT,
            pitch_calibration=pitch_calibration,
            speed_rad_s=DM_SPEED_RAD_S,
            acceleration_rad_s2=DM_ACCELERATION_RAD_S2,
            torque_ff_nm=DM_TORQUE_FF_NM,
            torque_ramp_s=DM_TORQUE_RAMP_S,
            kp=DM_KP,
            kd=DM_KD,
            max_tracking_error_deg=DM_MAX_TRACKING_ERROR_DEG,
            stall_timeout_s=DM_STALL_TIMEOUT_S,
            min_muzzle_deg=DM_MIN_PITCH_DEG,
            max_muzzle_deg=DM_MAX_PITCH_DEG,
            control_mode=DM_CONTROL_MODE,
            yaw_calibration=yaw_calibration,
            min_yaw_deg=DM_MIN_YAW_DEG,
            max_yaw_deg=DM_MAX_YAW_DEG,
            max_pitch_for_yaw_deg=DM_MAX_PITCH_FOR_YAW_DEG,
        )
        self.turret_worker.start()

    @staticmethod
    def _packet_axis_to_degrees(yaw_value, pitch_value):
        """ESP32 yaw/pitch 0..100 -> 水平角 -35..+35°、砲口仰角 0..45°。"""
        yaw_value = max(0, min(100, int(yaw_value)))
        pitch_value = max(0, min(100, int(pitch_value)))
        yaw_deg = DM_MIN_YAW_DEG + (DM_MAX_YAW_DEG - DM_MIN_YAW_DEG) * yaw_value / 100.0
        pitch_deg = DM_MIN_PITCH_DEG + (DM_MAX_PITCH_DEG - DM_MIN_PITCH_DEG) * pitch_value / 100.0
        return yaw_deg, pitch_deg

    def _update_dm_turret(self, yaw_value, pitch_value):
        """依 ESP32 回傳的手動搖桿值更新 DM 雙軸目標。"""
        snapshot = self.turret_worker.snapshot()
        if not snapshot.ready:
            if snapshot.error:
                print("[Manual] DM 控制器尚未就緒：{}".format(snapshot.error))
            return

        if not self._turret_enable_requested:
            self.turret_worker.request_enable()
            self._turret_enable_requested = True

        yaw_deg, pitch_deg = self._packet_axis_to_degrees(yaw_value, pitch_value)
        target = (round(yaw_deg, 2), round(pitch_deg, 2))
        if target != self._last_turret_command:
            self.turret_worker.request_aim(yaw_deg, pitch_deg)
            self._last_turret_command = target
            print(
                "[Manual] DM 目標：Yaw={:+.2f}° Pitch={:.2f}° (raw yaw={}, pitch={})".format(
                    yaw_deg, pitch_deg, yaw_value, pitch_value
                )
            )

    def _initialize_can(self):
        print("[Manual] 初始化彈藥倉 CAN：{}".format(AMMO_CAN_INTERFACE))
        self.ammo_bus = can.Bus(
            interface="socketcan", channel=AMMO_CAN_INTERFACE
        )
        self.ammo_notifier = can.Notifier(self.ammo_bus, [])
        self.ammo_servo = MksServo(
            self.ammo_bus, self.ammo_notifier, AMMO_MOTOR_CAN_ID
        )

        print("[Manual] 初始化射擊機構 CAN：{}".format(SHOOTING_CAN_INTERFACE))
        self.shoot_bus = can.Bus(
            interface="socketcan", channel=SHOOTING_CAN_INTERFACE
        )
        self.shoot_notifier = can.Notifier(self.shoot_bus, [])
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

    def on_mode_change(self, mode):
        self.current_mode = mode
        # 切換模式時清除按鈕狀態，避免回到手動時按住 FIRE 被當成新按下
        self.last_fire = 0
        if mode != "M":
            # 切到自動模式時停止手動射擊雙 VESC 馬達。
            self.last_speed = 0
            if getattr(self, "vesc_controller", None) is not None:
                self.vesc_controller.set_speed(0)
            # 自動模式不以手動封包更新 DM 目標；維持最後的目標姿態。
            self._send_turret_command(50, 50, 0, 0, mode="A")
            self._last_turret_command = None
        else:
            # 回到手動模式時，下一筆封包會重新要求使能並更新目標。
            self._turret_enable_requested = False
            self._last_turret_command = None

    def handle_packet(self, packet):
        """處理一筆 ESP32 遙控器封包。"""
        self.last_yaw = self._clamp(packet.get("yaw", 50), 0, 100)
        self.last_pitch = self._clamp(packet.get("pitch", 50), 0, 100)
        self.last_speed = self._clamp(packet.get("speed", 0), 0, 100)
        self.vesc_controller.set_speed(self.last_speed)
        self.last_vx = self._clamp(packet.get("vx", 0), -100, 100)
        self.last_vy = self._clamp(packet.get("vy", 0), -100, 100)
        self.last_wz = self._clamp(packet.get("wz", 0), -100, 100)

        packet_mode = packet.get("mode", self.current_mode)
        if packet_mode != self.current_mode:
            self.on_mode_change(packet_mode)

        # 手動模式：Xavier 直接控制 DM 水平/垂直馬達。
        # 封包來自 ESP32：E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED。
        if self.current_mode == "M":
            self._update_dm_turret(self.last_yaw, self.last_pitch)

        fire_now = 1 if packet.get("fire", 0) else 0
        fire_rising_edge = fire_now == 1 and self.last_fire == 0
        self.last_fire = fire_now

        if fire_rising_edge:
            self.request_shot()

    def request_shot(self):
        """按一次 FIRE 在背景執行一發射擊循環，主迴圈仍可讀取遙控器。"""
        with self.lock:
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
            print(
                "\\n[Manual] 第 {} 格，準備射擊第 {} / {} 發".format(
                    self.current_slot, self.shots_in_slot + 1, SHOTS_PER_SLOT
                )
            )

            self._run_one_shot_cycle()
            self.shots_in_slot += 1
            print(
                "[Manual] 射擊循環完成：本格計數 {}/{}".format(
                    self.shots_in_slot, SHOTS_PER_SLOT
                )
            )

            if self.shots_in_slot >= SHOTS_PER_SLOT:
                if self.current_mode != "M":
                    print("[Manual] 模式已切換，暫不轉動彈藥倉。")
                    return
                print("[Manual] 本格三次射擊循環完成，允許彈藥倉轉一格。")
                self._rotate_ammo_one_slot()
                self.shots_in_slot = 0
                self.current_slot += 1
                if self.current_slot > SLOTS_PER_MAGAZINE:
                    self.current_slot = 1
                print("[Manual] 彈藥倉目前格位：{}".format(self.current_slot))

        except Exception as exc:
            print("[Manual Error] 射擊/轉格流程失敗：{}".format(exc))
            self._emergency_stop(self.shoot_servo, "射擊機構")
            self._emergency_stop(self.ammo_servo, "彈藥倉")
            # 不增加射擊計數、不自動轉格
        finally:
            with self.lock:
                self.busy = False

    def _run_one_shot_cycle(self):
        # 第一段：從當前原點移動 1700 pulses
        self._move_relative(
            self.shoot_servo, 1700, MOVE_DIRECTION, SHOOTING_SPEED_RPM
        )
        time.sleep(SETTLE_TIME)

        # 第二段：由 1700 到累積目標 3000，因此本段移動 1300 pulses
        self._move_relative(
            self.shoot_servo, 3000 - 1700, MOVE_DIRECTION, SHOOTING_SPEED_RPM
        )
        time.sleep(SETTLE_TIME)

        # 沿用原本的兩向無限位回零流程
        self._go_home(self.shoot_servo, MOVE_DIRECTION, "反向無限位回零")
        time.sleep(SETTLE_TIME)
        self._go_home(self.shoot_servo, HOME_DIRECTION, "正向無限位回零")

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
        before = self.ammo_servo.read_encoder_value_addition()
        if before is None:
            raise RuntimeError("彈藥倉移動前 Encoder 讀取失敗")

        result = self.ammo_servo.run_motor_relative_motion_by_pulses(
            Direction.CW, AMMO_SPEED_RPM, ACCELERATION, AMMO_SLOT_PULSES
        )
        print("[Manual] 彈藥倉轉一格指令回傳：", result)
        if result is None:
            raise RuntimeError("彈藥倉移動指令沒有有效回覆")

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
                raise RuntimeError("彈藥倉未偵測到移動")

            time.sleep(POLL_INTERVAL)

        after = self.ammo_servo.read_encoder_value_addition()
        if after is None:
            raise RuntimeError("彈藥倉移動後 Encoder 讀取失敗")
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
        # 結束時先命令 VESC 雙馬達停轉，再停止 MKS 並關閉 CAN bus。
        if getattr(self, "vesc_controller", None) is not None:
            try:
                self.vesc_controller.close()
            except Exception as exc:
                print("[Manual] VESC 雙馬達關閉失敗：{}".format(exc))

        self._emergency_stop(self.shoot_servo, "射擊機構")
        self._emergency_stop(self.ammo_servo, "彈藥倉")

        if self.shoot_notifier is not None:
            self.shoot_notifier.stop()
        if self.ammo_notifier is not None:
            self.ammo_notifier.stop()
        if self.shoot_bus is not None:
            self.shoot_bus.shutdown()
        if self.ammo_bus is not None:
            self.ammo_bus.shutdown()

        # 關閉 DM worker 前會失能馬達；垂直軸可能下墜，請先支撐砲管。
        if getattr(self, "turret_worker", None) is not None:
            try:
                self.turret_worker.request_disable()
                time.sleep(0.2)
                self.turret_worker.request_shutdown()
                self.turret_worker.wait_closed(timeout=3.0)
            except Exception as exc:
                print("[Manual] DM 砲台關閉失敗：{}".format(exc))

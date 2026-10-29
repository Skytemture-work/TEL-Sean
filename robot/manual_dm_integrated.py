# manual.py
# 手動模式：ESP32 遙控器 + DM 雙軸砲台 + MKS 彈藥倉/射擊機構控制
#
# 封包格式由 robot/comm_uart.py 定義：
# E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED
#
# 暫定硬體分配：
#   can0：彈藥倉馬達，ID 1，每格 6666 pulses
#   can1：射擊機構馬達，ID 1
#
# 注意：
# 1. ESP32 的 yaw/pitch (0~100) 由 Xavier 轉成 DM 目標角：Yaw -35~+35°、Pitch 0~45°。
# 2. FIRE 每次由 0 變 1，觸發一個射擊循環。
# 3. 每個射擊循環包含 1700 -> 3000 pulses -> 反向回零 -> 正向回零。
# 4. 每格累積 3 次循環後，才允許彈藥倉轉動一格。
# 5. 這是依現有協定和程式做的暫定整合，請先卸除彈藥/球進行低速測試。

import time
import threading
from pathlib import Path

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
# 由於完整射擊流程是阻塞式，流程期間收到的遙控器封包不會被即時處理。
# ---------------------------------------------------


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
        self.last_vx = 0
        self.last_vy = 0
        self.last_wz = 0
        self._initialize_can()
        self._initialize_dm_turret()
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
        # 結束時停止馬達，關閉兩個 CAN bus
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

"""Computer-entered absolute angles and single-stage firing for the foam-ball robot.

Run with ``python3 -m turret_control.run_precision_fire``. Hardware startup,
flywheel control and the three-stage feed/reload sequence use ManualController.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time

from manual import (
    DM_POSITION_TOLERANCE_DEG,
    DM_STARTUP_TEST_SPEED_RAD_S,
    MOVE_TIMEOUT,
    ManualController,
    VESC_MIN_RPM,
    VESC_RPM_PER_SPEED_UNIT,
)


HELP = """電腦精確角度／發射控制：
  aim <yaw> <pitch>  設定實體絕對水平角、砲口仰角（度），等待回授到位
  speed <0..100>     飛輪速度，沿用 main.py 的 VRB 對應；0 為停轉
  fire               執行一次 FIRE 階段；三次完成後依 main.py 流程轉一格
  stop               停止飛輪，砲口保持位置
  status             顯示目標／回授角度、飛輪設定及目前發射階段
  help               顯示操作說明
  quit               停止並關閉所有控制器（也可按 Ctrl+C）
例如：aim 10 20 → speed 50 → fire → fire → fire
"""


class PrecisionSession:
    def __init__(self, controller):
        self.controller = controller
        self.speed = 0
        self.aimed = False

    def check_health(self):
        self.controller.check_health()
        if self.controller.shot_error is not None:
            raise RuntimeError(self.controller.shot_error)

    def wait_for_aim(self):
        status = self.controller.dm_controller.aim_status()
        distance = max(
            abs(status[axis] - status['actual_' + axis])
            if status['actual_' + axis] is not None
            else status[axis + '_limits'][1] - status[axis + '_limits'][0]
            for axis in ('yaw', 'pitch')
        )
        deadline = (time.monotonic() + 1.5 * math.radians(distance)
                    / DM_STARTUP_TEST_SPEED_RAD_S + MOVE_TIMEOUT + 2.0)
        while True:
            self.check_health()
            status = self.controller.dm_controller.aim_status()
            if status['ready']:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError("等待角度回授到位逾時，停止操作")
            time.sleep(0.05)

    def aim(self, yaw, pitch):
        self.check_health()
        if self.controller.busy:
            raise ValueError("發射機構運動中，請等待完成")
        if not self.controller.dm_controller.set_absolute_angles(yaw, pitch):
            raise ValueError("砲口尚未到位或回授未就緒，未接受新角度")
        self.aimed = False
        self.wait_for_aim()
        self.aimed = True
        self.show_status()

    def set_speed(self, value):
        value = int(value)
        if not 0 <= value <= 100:
            raise ValueError("speed 必須是 0..100 的整數")
        self.check_health()
        self.controller.vesc_controller.set_speed(value)
        self.speed = value
        rpm = 0 if value == 0 else VESC_MIN_RPM + value * VESC_RPM_PER_SPEED_UNIT
        print(f"[電腦控制] 飛輪 speed={value}，目標 {rpm:g} RPM", flush=True)

    def fire(self):
        self.check_health()
        if not self.aimed or not self.controller.dm_controller.aim_status()['ready']:
            raise ValueError("請先用 aim 設定角度並確認回授到位，再輸入 fire")
        if self.speed <= 0:
            raise ValueError("飛輪目前停轉，請先用 speed 設定速度")
        if not self.controller.request_shot(wait=True):
            raise ValueError("目前無法接受發射指令")
        self.check_health()
        self.show_status()

    def show_status(self):
        status = self.controller.dm_controller.aim_status()
        print(f"[角度目標] yaw={status['yaw']:+.3f}° pitch={status['pitch']:+.3f}°", flush=True)
        if status['actual_yaw'] is None or status['actual_pitch'] is None:
            print("[角度回授] 尚無有效的新回授", flush=True)
        else:
            print(f"[角度回授] yaw={status['actual_yaw']:+.3f}° "
                  f"pitch={status['actual_pitch']:+.3f}°；"
                  f"{'已到位' if status['ready'] else '尚未到位'} "
                  f"（沿用 main.py 到位門檻 ±{DM_POSITION_TOLERANCE_DEG:g}°）", flush=True)
        rpm = 0 if self.speed == 0 else VESC_MIN_RPM + self.speed * VESC_RPM_PER_SPEED_UNIT
        print(f"[發射] 飛輪 speed={self.speed}（目標 {rpm:g} RPM）；格位 {self.controller.current_slot}；"
              f"下一次 FIRE 為第 {self.controller.shot_stage + 1}/3 階段", flush=True)
        yaw_low, yaw_high = status['yaw_limits']
        pitch_low, pitch_high = status['pitch_limits']
        print(f"[角度邊界] yaw=[{yaw_low:+.2f}, {yaw_high:+.2f}]° "
              f"pitch=[{pitch_low:+.2f}, {pitch_high:+.2f}]°；"
              "仍受機構聯合工作空間限制", flush=True)

    def execute(self, line):
        parts = line.strip().split()
        if not parts:
            return True
        command, args = parts[0].lower(), parts[1:]
        if command == 'aim' and len(args) == 2:
            self.aim(float(args[0]), float(args[1]))
        elif command == 'speed' and len(args) == 1:
            self.set_speed(args[0])
        elif command == 'fire' and not args:
            self.fire()
        elif command == 'stop' and not args:
            self.set_speed(0)
        elif command == 'status' and not args:
            self.show_status()
        elif command == 'help' and not args:
            print(HELP, flush=True)
        elif command == 'quit' and not args:
            return False
        else:
            raise ValueError("指令格式不正確，輸入 help 查看操作方式")
        return True


def read_command(check_health):
    """等待終端輸入時持續檢查馬達狀態，故障不必等到按 Enter 才處理。"""
    print('precision> ', end='', flush=True)
    while True:
        check_health()
        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if readable:
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yaw', type=float, help='啟動後設定的實體絕對水平角（度），與 --pitch 一起使用')
    parser.add_argument('--pitch', type=float, help='啟動後設定的實體絕對仰角（度），與 --yaw 一起使用')
    args = parser.parse_args(argv)
    if (args.yaw is None) != (args.pitch is None):
        parser.error('--yaw 與 --pitch 必須一起指定')
    if args.yaw is not None and not all(math.isfinite(v) for v in (args.yaw, args.pitch)):
        parser.error('角度必須是有限數值')

    controller = None
    try:
        print("[啟動] 沿用 main.py 的 DM 啟動姿態與 MKS／雙 VESC 初始化；飛輪初始停轉。", flush=True)
        # All commands are local. No ESP32 or vehicle movement packets are sent.
        controller = ManualController(uart=None, log_status=False)
        session = PrecisionSession(controller)
        session.wait_for_aim()
        print(HELP, flush=True)
        if args.yaw is not None:
            session.aim(args.yaw, args.pitch)
        else:
            session.show_status()
        while True:
            line = read_command(session.check_health)
            session.check_health()
            try:
                if not session.execute(line):
                    break
            except ValueError as exc:
                print(f"[輸入未執行] {exc}", flush=True)
    except (KeyboardInterrupt, EOFError):
        print("\n[電腦控制] 結束操作，停止並關閉控制器。", flush=True)
    except Exception as exc:
        print(f"[電腦控制錯誤] {exc}", flush=True)
        raise SystemExit(1) from exc
    finally:
        if controller is not None:
            controller.close()


if __name__ == '__main__':
    main()

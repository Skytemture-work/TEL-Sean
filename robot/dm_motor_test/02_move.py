#!/usr/bin/env python3
"""
步驟二：讓 DM-J4340P-2EC 真的轉起來。

*** 執行前務必：馬達本體鎖在檯面/夾具上，輸出軸不要接任何負載，手離開轉子。***

用法:
    python3 02_move.py --mode vel     --vel 0.5          # 速度模式，等速 0.5 rad/s（預設）
    python3 02_move.py --mode posvel  --amp 1.57 --vel 1 # 位置速度模式，±90° 來回
    python3 02_move.py --mode mit     --amp 0.5          # MIT 模式，kp/kd 阻抗控制

Ctrl+C 會先失能馬達再退出。
"""
import argparse
import math
import time

import serial

from DM_CAN import Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable
from dm_port import find_port

MOTOR_TYPE = DM_Motor_Type.DM4340
BAUD = 921600
DT = 0.005  # 200 Hz。必須持續送命令，否則會觸發「通訊丟失防護」自動失能


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="不給就自動偵測 VID:PID 2e88:4603")
    ap.add_argument("--id", type=lambda s: int(s, 0), default=0x01, help="馬達 CAN ID (ESC_ID)")
    ap.add_argument("--mst", type=lambda s: int(s, 0), default=0x00, help="反饋 MST_ID")
    ap.add_argument("--mode", choices=["vel", "posvel", "mit"], default="vel")
    ap.add_argument("--vel", type=float, default=0.5, help="速度 rad/s（輸出軸，上限約 5.4）")
    ap.add_argument("--amp", type=float, default=1.57, help="位置振幅 rad（輸出軸）")
    ap.add_argument("--period", type=float, default=6.0, help="來回週期 秒")
    ap.add_argument("--kp", type=float, default=30.0, help="MIT 模式 kp [0,500]")
    ap.add_argument("--kd", type=float, default=1.0, help="MIT 模式 kd [0,5]，位置控制時不可為 0")
    ap.add_argument("--duration", type=float, default=15.0, help="運轉秒數")
    ap.add_argument("--set-zero", action="store_true", help="開始前把目前位置設為 0 位")
    args = ap.parse_args()

    port = find_port(args.port)
    print(f"使用序列埠：{port}")
    dev = serial.Serial(port, BAUD, timeout=0.5)
    mc = MotorControl(dev)
    m = Motor(MOTOR_TYPE, args.id, args.mst)
    mc.addMotor(m)

    # 把函式庫的映射範圍對齊馬達實際暫存器，否則位置/速度/扭矩全部解碼錯
    pmax = mc.read_motor_param(m, DM_variable.PMAX)
    vmax = mc.read_motor_param(m, DM_variable.VMAX)
    tmax = mc.read_motor_param(m, DM_variable.TMAX)
    if None in (pmax, vmax, tmax):
        print(f"讀不到 0x{args.id:02X} 的 PMAX/VMAX/TMAX，先跑 01_check.py 確認 ID")
        dev.close()
        return
    mc.change_limit_param(MOTOR_TYPE, pmax, vmax, tmax)
    print(f"映射範圍已對齊: PMAX={pmax} VMAX={vmax} TMAX={tmax}")

    if abs(args.vel) > vmax:
        print(f"--vel {args.vel} 超過 VMAX {vmax}，中止")
        dev.close()
        return
    if abs(args.vel) > 5.4:
        print(f"⚠ {args.vel} rad/s 已超過 24V 版空載最高轉速 (52rpm≈5.4rad/s)，馬達會跟不上")

    mode_map = {"vel": Control_Type.VEL, "posvel": Control_Type.POS_VEL, "mit": Control_Type.MIT}
    if not mc.switchControlMode(m, mode_map[args.mode]):
        print(f"切換到 {args.mode} 模式失敗，先跑 01_check.py 確認 ID 對不對")
        dev.close()
        return
    print(f"已切到 {args.mode} 模式")

    if args.set_zero:
        mc.disable(m)          # 設零點必須在失能狀態
        mc.set_zero_position(m)
        print("已將目前位置設為 0 位")

    mc.enable(m)
    print("已使能（綠燈常亮）。Ctrl+C 停止。")
    time.sleep(0.1)

    t0 = time.time()
    next_t = t0
    late = 0
    n = 0
    try:
        while time.time() - t0 < args.duration:
            t = time.time() - t0
            n += 1
            s = math.sin(2 * math.pi * t / args.period)

            if args.mode == "vel":
                mc.control_Vel(m, args.vel * s)
            elif args.mode == "posvel":
                mc.control_Pos_Vel(m, args.amp * s, abs(args.vel))
            else:  # mit
                mc.controlMIT(m, args.kp, args.kd, args.amp * s, 0.0, 0.0)

            print(f"\rt={t:5.2f}s  pos={m.getPosition():+7.3f} rad  "
                  f"vel={m.getVelocity():+6.3f} rad/s  tau={m.getTorque():+6.2f} Nm",
                  end="", flush=True)

            # 補償式排程：睡到「下一個固定時刻」而不是固定睡 DT。
            # 直接 sleep(DT) 會把 USB 來回(~1ms)和 print 的耗時疊加上去，
            # 實測變成 5.87ms/圈（170Hz）而不是 200Hz。
            next_t += DT
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1        # 這圈已經來不及，不補睡，直接進下一圈
                next_t = time.time()
    except KeyboardInterrupt:
        print("\n使用者中斷")
    finally:
        elapsed = time.time() - t0   # 要在收尾斜坡之前抓，否則會低估迴圈頻率
        # 位置類模式先把命令收回目前位置，避免失能瞬間掉落
        if args.mode in ("posvel", "mit"):
            here = m.getPosition()
            for _ in range(20):
                if args.mode == "posvel":
                    mc.control_Pos_Vel(m, here, 0.5)
                else:
                    mc.controlMIT(m, args.kp, args.kd, here, 0.0, 0.0)
                time.sleep(DT)
        else:
            for _ in range(20):
                mc.control_Vel(m, 0.0)
                time.sleep(DT)
        mc.disable(m)
        dev.close()
        if n:
            print(f"\n迴圈實測 {n / elapsed:.1f} Hz（目標 {1 / DT:.0f} Hz），"
                  f"來不及的圈數 {late}/{n}")
        print("已失能並關閉序列埠（紅燈常亮）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
步驟一：只讀參數，不會讓馬達轉。
用途：確認 USB-to-CAN 板、CAN 接線、馬達 ID 都正常。

用法:
    python3 01_check.py              # 掃描 CAN ID 0x01~0x08
    python3 01_check.py --id 0x01    # 只查指定 ID
"""
import argparse
import serial

from DM_CAN import Motor, MotorControl, DM_Motor_Type, DM_variable
from dm_port import find_port

BAUD = 921600  # USB-to-CAN 調試板的虛擬序列埠速率，不是 CAN 波特率


def probe(mc, slave_id):
    """回傳 dict，讀不到就回 None。"""
    mc.motors_map.clear()  # 掃描時只留當前這顆，避免回覆被歸到別的 ID
    m = Motor(DM_Motor_Type.DM4340, slave_id, 0x00)
    mc.addMotor(m)
    gr = mc.read_motor_param(m, DM_variable.Gr)
    if gr is None:
        return None
    return {
        "motor": m,
        "Gr": gr,
        "MST_ID": mc.read_motor_param(m, DM_variable.MST_ID),
        "CTRL_MODE": mc.read_motor_param(m, DM_variable.CTRL_MODE),
        "PMAX": mc.read_motor_param(m, DM_variable.PMAX),
        "VMAX": mc.read_motor_param(m, DM_variable.VMAX),
        "TMAX": mc.read_motor_param(m, DM_variable.TMAX),
        "sw_ver": mc.read_motor_param(m, DM_variable.sw_ver),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="不給就自動偵測 VID:PID 2e88:4603")
    ap.add_argument("--id", type=lambda s: int(s, 0), default=None)
    args = ap.parse_args()

    port = find_port(args.port)
    print(f"使用序列埠：{port}")
    dev = serial.Serial(port, BAUD, timeout=0.5)
    mc = MotorControl(dev)

    ids = [args.id] if args.id is not None else range(0x01, 0x09)
    found = []
    for sid in ids:
        info = probe(mc, sid)
        if info is None:
            print(f"  CAN ID 0x{sid:02X} : 無回應")
            continue
        found.append(sid)
        print(f"\n>>> 找到馬達 CAN ID = 0x{sid:02X}")
        print(f"    軟體版本 sw_ver : {info['sw_ver']}")
        print(f"    減速比    Gr     : {info['Gr']}      (DM-J4340P-2EC 應為 40.0)")
        print(f"    反饋 MST_ID     : 0x{int(info['MST_ID']):02X}")
        print(f"    控制模式 CTRL_MODE: {info['CTRL_MODE']}  (1=MIT 2=位置速度 3=速度 4=力位混控)")
        print(f"    PMAX / VMAX / TMAX: {info['PMAX']} / {info['VMAX']} / {info['TMAX']}")
        if abs(info["PMAX"] - 12.5) > 0.01 or abs(info["VMAX"] - 8.0) > 0.01 or abs(info["TMAX"] - 28.0) > 0.01:
            print("    ⚠ 與函式庫 DM4340 內建的 [12.5, 8, 28] 不同，")
            print("      02_move.py 需要用 change_limit_param() 對齊，否則位置/速度解碼會錯。")

    dev.close()
    if not found:
        print("\n沒有找到任何馬達。檢查：24V 電源有沒有開、CAN_H/CAN_L 有沒有接反、")
        print("USB-to-CAN 板是不是這個 port、馬達燈號是否為紅燈常亮（失能待命）。")
    else:
        print(f"\n完成，找到 {len(found)} 顆：{[hex(i) for i in found]}")


if __name__ == "__main__":
    main()

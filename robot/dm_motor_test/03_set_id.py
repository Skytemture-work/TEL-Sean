#!/usr/bin/env python3
"""
步驟三：修改馬達的 CAN ID（ESC_ID）與回饋 ID（MST_ID）。

*** 匯流排上必須只有一顆馬達。***
兩顆達妙馬達出廠都是 ESC_ID = 0x01，同時掛上去會互相對撞，也沒辦法分別定址。
所以第二顆到貨後要先單獨接上，改好 ID 再跟第一顆併線。

用法:
    python3 03_set_id.py                                # 只讀，不改任何東西
    python3 03_set_id.py --new-id 0x02                  # 只改 RAM，掉電還原（可先試）
    python3 03_set_id.py --new-id 0x02 --save           # 改完寫入 flash（永久）
    python3 03_set_id.py --new-id 0x02 --new-mst 0x12 --save

⚠ flash 只有約 1 萬次擦寫壽命，--save 不要反覆跑。
⚠ 改 ID 後馬達立刻用新 ID 回應，函式庫的回傳值會誤報失敗，所以本程式一律重新掃描驗證。
"""
import argparse
import sys
import time

import serial

from DM_CAN import Motor, MotorControl, DM_Motor_Type, DM_variable
from dm_port import find_port

BAUD = 921600
SCAN_RANGE = range(0x01, 0x11)


def scan(mc, ids=SCAN_RANGE):
    """回傳 {id: {欄位}}，掃描匯流排上所有有回應的馬達。"""
    found = {}
    for sid in ids:
        mc.motors_map.clear()
        m = Motor(DM_Motor_Type.DM4340, sid, 0x00)
        mc.addMotor(m)
        gr = mc.read_motor_param(m, DM_variable.Gr)
        if gr is None:
            continue
        found[sid] = {
            "Gr": gr,
            "ESC_ID": mc.read_motor_param(m, DM_variable.ESC_ID),
            "MST_ID": mc.read_motor_param(m, DM_variable.MST_ID),
            "CTRL_MODE": mc.read_motor_param(m, DM_variable.CTRL_MODE),
            "sw_ver": mc.read_motor_param(m, DM_variable.sw_ver),
        }
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="不給就自動偵測 VID:PID 2e88:4603")
    ap.add_argument("--new-id", type=lambda s: int(s, 0), default=None,
                    help="新的 ESC_ID，例如 0x02。範圍 0x01~0x0F")
    ap.add_argument("--new-mst", type=lambda s: int(s, 0), default=None,
                    help="新的 MST_ID（選用）。不給就不動")
    ap.add_argument("--save", action="store_true",
                    help="寫入 flash 永久保留。不加只改 RAM，掉電還原")
    ap.add_argument("--yes", action="store_true", help="跳過確認提示")
    args = ap.parse_args()

    if args.new_id is not None and not 0x01 <= args.new_id <= 0x0F:
        print(f"--new-id 0x{args.new_id:02X} 超出範圍。"
              "回饋幀用 data[0] 的低 4 位帶 ID，所以只能是 0x01~0x0F。")
        return 1

    port = find_port(args.port)
    print(f"使用序列埠：{port}\n")
    dev = serial.Serial(port, BAUD, timeout=0.5)
    mc = MotorControl(dev)

    print(f"掃描 0x{SCAN_RANGE[0]:02X}~0x{SCAN_RANGE[-1]:02X} ...")
    found = scan(mc)

    if not found:
        print("\n❌ 沒有找到任何馬達。檢查 24V 電源、CAN_H/CAN_L、接頭有沒有壓好。")
        dev.close()
        return 1

    for sid, info in found.items():
        print(f"  0x{sid:02X}  Gr={info['Gr']}  ESC_ID={int(info['ESC_ID'])}  "
              f"MST_ID=0x{int(info['MST_ID']):02X}  CTRL_MODE={int(info['CTRL_MODE'])}  "
              f"sw_ver={info['sw_ver']}")

    if len(found) > 1:
        print(f"\n❌ 匯流排上有 {len(found)} 顆馬達，拒絕修改。")
        print("   改 ID 必須一次只接一顆，否則沒辦法確定改到的是哪一顆。")
        dev.close()
        return 1

    old_id = next(iter(found))

    if args.new_id is None and args.new_mst is None:
        print("\n（唯讀模式，沒有修改任何東西。加 --new-id 才會改。）")
        dev.close()
        return 0

    new_id = args.new_id if args.new_id is not None else old_id
    new_mst = args.new_mst if args.new_mst is not None else int(found[old_id]["MST_ID"])

    print(f"\n預計修改：ESC_ID 0x{old_id:02X} → 0x{new_id:02X}"
          f"   MST_ID 0x{int(found[old_id]['MST_ID']):02X} → 0x{new_mst:02X}")
    print(f"flash 寫入：{'是（永久保留）' if args.save else '否（只改 RAM，掉電還原）'}")

    if args.save and not args.yes:
        print("\n⚠ flash 約 1 萬次擦寫壽命。確定要寫入嗎？")
        if input("  輸入 yes 繼續：").strip().lower() != "yes":
            print("已取消。")
            dev.close()
            return 0

    mc.motors_map.clear()
    motor = Motor(DM_Motor_Type.DM4340, old_id, 0x00)
    mc.addMotor(motor)

    # 先改 MST_ID（ID 還沒變，定址還是舊的）
    if args.new_mst is not None and new_mst != int(found[old_id]["MST_ID"]):
        mc.change_motor_param(motor, DM_variable.MST_ID, new_mst)
        time.sleep(0.05)

    # 再改 ESC_ID。改完馬達立刻用新 ID 回應，回傳值不可信，靠重掃驗證。
    if new_id != old_id:
        mc.change_motor_param(motor, DM_variable.ESC_ID, new_id)
        time.sleep(0.1)

    print("\n重新掃描驗證 ...")
    mc.motors_map.clear()
    after = scan(mc)

    if new_id not in after:
        print(f"❌ 改完之後 0x{new_id:02X} 沒有回應。實際掃到：{[hex(i) for i in after]}")
        print("   馬達可能還在舊 ID，或參數沒寫進去。斷電重上再跑一次唯讀模式確認。")
        dev.close()
        return 1

    info = after[new_id]
    print(f"  ✅ 0x{new_id:02X}  ESC_ID={int(info['ESC_ID'])}  "
          f"MST_ID=0x{int(info['MST_ID']):02X}  Gr={info['Gr']}")

    if args.save:
        mc.motors_map.clear()
        saved_motor = Motor(DM_Motor_Type.DM4340, new_id, 0x00)
        mc.addMotor(saved_motor)
        mc.save_motor_param(saved_motor)   # 內部會先 disable
        time.sleep(0.2)
        print("  ✅ 已寫入 flash，掉電後保留。")
        print("     建議斷電重上，再跑一次唯讀模式確認真的存進去了。")
    else:
        print("  ⚠ 只改了 RAM，掉電會還原。確認沒問題後加 --save 才會永久保留。")

    dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

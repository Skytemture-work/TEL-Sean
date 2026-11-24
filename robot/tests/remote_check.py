"""遙控器按鈕檢查：讀 ESP32 封包，每個開關／搖桿一變就印出它代表的意義。

用法（先停掉 main.py，它會佔用序列埠）：
    cd ~/robot/TEL-Sean/robot
    python3 tests/remote_check.py            # 一直跑，Ctrl+C 結束
    python3 tests/remote_check.py --log /tmp/remote.log   # 同時寫檔

注意：手動模式（SWA=M）下 ESP32 會直接用搖桿驅動底盤。測搖桿時把油門（CH3）
拉到最低，或把車架高讓輪子離地。本程式只讀封包，不送任何底盤指令。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from comm_uart import UARTController  # noqa: E402

STICK_STEP = 10  # 搖桿值變化超過這麼多才印，避免洗版


def describe(field, value, packet):
    mode = packet["mode"]
    if field == "mode":
        return {"A": "SWA → 自動模式 A", "M": "SWA → 手動模式 M",
                "X": "ESP32 沒收到遙控器訊號 (X)"}.get(value, f"模式 {value!r}")
    if field == "auto_start":
        return ("VRA 轉到底（≥1800）→ auto_start=1：自動模式發車"
                if value else "VRA 未轉到底 → auto_start=0")
    if field == "fire":
        return (f"SWD = {value}：自動模式選{'下' if value else '上'}半場"
                f"；手動模式{'由 0 變 1 會射一個階段' if value else '（放開）'}")
    if field == "reload":
        return f"SWC = {value}：RELOAD（裝填完成／急停後出發，由 0 撥到 1 才算）"
    if field == "vin":
        return f"SWB = {value}：下半場{'小孔模式' if value else '全打大孔（預設）'}（發車時鎖定）"
    if field == "speed":
        return (f"VRB 飛輪速度 = {value}（0~100，手動模式才有效）" if mode == "M"
                else f"speed = {value}（自動模式為 AGX 指令回傳，與 VRB 無關）")
    if field in ("vx", "vy", "wz"):
        names = {"vx": "前後（右搖桿 CH2）", "vy": "左右平移（右搖桿 CH1）", "wz": "旋轉（左搖桿 CH4）"}
        note = "" if mode == "M" else "（自動模式為 AGX 指令回傳）"
        return f"底盤{names[field]} = {value}{note}"
    if field in ("yaw", "pitch"):
        names = {"yaw": "砲台水平（CH1）", "pitch": "砲台仰角（CH2）"}
        note = "" if mode == "M" else "（自動模式不使用）"
        return f"{names[field]} = {value}（0~100，50 置中）{note}"
    return f"{field} = {value}"


# yaw/pitch are no longer used: the turret aims by itself in manual mode too.
FIELDS = ("mode", "auto_start", "fire", "reload", "vin", "speed", "vx", "vy", "wz")
STICKS = {"speed", "vx", "vy", "wz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None, help="同時寫入這個檔案")
    args = parser.parse_args()
    log = args.log.open("a", encoding="utf-8") if args.log else None

    def out(text):
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    if uart.ser is None or not uart.ser.is_open:
        out("無法開啟 ESP32 序列埠（main.py 是否還在跑？）")
        return 1
    out("開始監看遙控器。依序撥動每個開關、轉旋鈕、推搖桿；Ctrl+C 結束。")
    last = None
    last_rx = time.monotonic()
    gap_reported = False
    has_vin = None
    try:
        while True:
            packet = uart.read_esp32_packet()
            now = time.monotonic()
            if packet is None:
                if now - last_rx > 0.5 and not gap_reported:
                    out("⚠ ESP32 封包中斷（急停斷電或線鬆了？）")
                    gap_reported = True
                time.sleep(0.01)
                continue
            if gap_reported:
                out(f"ESP32 封包恢復（中斷 {now - last_rx:.1f} 秒）")
                gap_reported = False
            last_rx = now
            fields = len(packet["raw"].split(","))
            if has_vin is None or has_vin != (fields >= 12):
                has_vin = fields >= 12
                out("封包有 VIN 欄位（新韌體）" if has_vin else "⚠ 封包沒有 VIN 欄位：ESP32 還是舊韌體，SWB 讀不到")
            if last is None:
                out("目前狀態：" + " | ".join(describe(f, packet.get(f, 0), packet) for f in FIELDS))
            else:
                for f in FIELDS:
                    old, new = last.get(f, 0), packet.get(f, 0)
                    if old == new:
                        continue
                    if f in STICKS and abs(new - old) < STICK_STEP:
                        continue
                    out(describe(f, new, packet))
                    last[f] = new
                packet = {**last, "raw": packet["raw"]}
            last = dict(packet)
            time.sleep(0.01)
    except KeyboardInterrupt:
        out("結束監看。")
        return 0
    finally:
        uart.close()
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())

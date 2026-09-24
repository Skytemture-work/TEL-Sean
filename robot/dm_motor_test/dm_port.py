"""
自動找出達妙 USB-to-CAN 調試板的序列埠。

板子是 HDSC CDC Device，VID:PID = 2e88:4603。用 VID:PID 認而不是認
/dev/ttyACM0，換到 AGX（Xavier 本身還有內建 ACM 除錯口會佔號碼）
或插拔順序改變時都不會找錯。
"""
from serial.tools import list_ports

DAMIAO_VID = 0x2E88
DAMIAO_PID = 0x4603
SYMLINK = "/dev/damiao"  # 由 99-damiao.rules 建立，有的話優先用


def find_port(explicit=None):
    """回傳序列埠路徑。找不到就 raise RuntimeError。"""
    if explicit:
        return explicit

    hits = [p.device for p in list_ports.comports()
            if p.vid == DAMIAO_VID and p.pid == DAMIAO_PID]

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(
            f"找到 {len(hits)} 塊達妙調試板 {hits}，請用 --port 指定")

    ports = list_ports.comports()
    detail = "\n".join(f"  {p.device}  {p.description}" for p in ports) or "  (無)"
    raise RuntimeError(
        "找不到達妙 USB-to-CAN 調試板 (VID:PID 2e88:4603)。\n"
        f"目前系統上的序列埠：\n{detail}\n"
        "檢查：USB 線有沒有插好、lsusb 看得到 2e88:4603 嗎？")

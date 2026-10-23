
#!/usr/bin/env python3
import struct
import time
import can

# =========================
# CAN 設定
# =========================
CAN_INTERFACE = "socketcan"
CAN_CHANNEL = "can0"
CAN_BITRATE = 500000       # 必須與控制器設定一致
VESC_ID = 1                # 請改成 MKS 實際設定的 ID

# VESC CAN Packet ID
CAN_PACKET_SET_RPM = 3
CAN_PACKET_STATUS = 9

# 初次測試請使用低速，確認馬達與控制方向
MAX_TEST_ERPM = 1000


def send_erpm(bus, erpm):
    """傳送目標電氣轉速 ERPM。"""
    can_id = (CAN_PACKET_SET_RPM << 8) | VESC_ID

    msg = can.Message(
        arbitration_id=can_id,
        is_extended_id=True,
        data=struct.pack(">i", int(erpm))
    )
    bus.send(msg)
    print(f"[TX] Target ERPM = {erpm}")


def receive_status(bus):
    """讀取 VESC STATUS 回報。"""
    msg = bus.recv(timeout=0.1)

    if msg is None or not msg.is_extended_id:
        return

    packet_id = (msg.arbitration_id >> 8) & 0xFF
    controller_id = msg.arbitration_id & 0xFF

    if controller_id != VESC_ID:
        return

    if packet_id != CAN_PACKET_STATUS or len(msg.data) < 8:
        return

    # VESC STATUS:
    # 4 bytes ERPM, 2 bytes current, 2 bytes duty
    erpm = struct.unpack(">i", bytes(msg.data[0:4]))[0]
    current = struct.unpack(">h", bytes(msg.data[4:6]))[0] / 10.0
    duty = struct.unpack(">h", bytes(msg.data[6:8]))[0] / 1000.0

    print(
        f"[RX] ERPM={erpm:>8} | "
        f"Current={current:>6.1f} A | "
        f"Duty={duty * 100:>6.1f}%"
    )


def main():
    bus = can.Bus(
        interface=CAN_INTERFACE,
        channel=CAN_CHANNEL,
        bitrate=CAN_BITRATE
    )

    print("MKS CAN Motor Controller")
    print("輸入目標 ERPM，例如 500；輸入 0 停止。")

    try:
        while True:
            text = input("\nTarget ERPM > ").strip()

            try:
                target = int(text)
            except ValueError:
                print("請輸入整數 ERPM。")
                continue

            if abs(target) > MAX_TEST_ERPM:
                print(f"測試階段請限制在 ±{MAX_TEST_ERPM} ERPM。")
                continue

            send_erpm(bus, target)

            # 接收約 2 秒的狀態回報
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                receive_status(bus)

    except KeyboardInterrupt:
        print("\n停止命令已送出。")
        try:
            send_erpm(bus, 0)
        except can.CanError as exc:
            print(f"傳送停止命令失敗：{exc}")

    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()

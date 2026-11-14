# main.py
import time
import config
from comm_uart import UARTController
from vision_target import VisionTracker
from lidar_avoid import LidarAvoidance
from manual import (
    ManualController,
    VESC_MIN_RPM,
    VESC_RPM_PER_SPEED_UNIT,
    normalize_vrb_speed,
)


def main():
    print("====================================================")
    print("TEL-Sean 主控系統：手動 / 自動模式")
    print("====================================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    vision = VisionTracker()
    lidar = LidarAvoidance()

    # 手動模式控制器：接收遙控器封包，管理砲台指令、射擊與彈藥倉格位
    manual = ManualController(uart=uart)

    try:
        vision.start()
        lidar.start()
        time.sleep(1.0)

        # 預設手動模式；收到 ESP32 封包後依封包中的 mode 切換
        current_mode = "M"
        print("[System] 已啟動，預設手動模式，等待 ESP32 遙控器封包...")

        while True:
            manual.check_health()
            packet = uart.read_esp32_packet()

            if packet is not None:
                packet_mode = packet.get("mode", "?")
                raw_speed = packet.get("speed")
                packet_speed = normalize_vrb_speed(raw_speed) if raw_speed is not None else 0
                vesc_target = (
                    "STOP"
                    if packet_mode != "M" or raw_speed is None
                    else "{} RPM".format(
                        int(VESC_MIN_RPM + packet_speed * VESC_RPM_PER_SPEED_UNIT)
                    )
                )
                print(
                    "[ESP32 RX] raw={!r} | mode={} vx={} vy={} wz={} "
                    "yaw={} pitch={} fire={} speed={} reload={} normalized_speed={} | VESC target={}".format(
                        packet.get("raw", ""),
                        packet_mode,
                        packet.get("vx", 0),
                        packet.get("vy", 0),
                        packet.get("wz", 0),
                        packet.get("yaw", 50),
                        packet.get("pitch", 50),
                        packet.get("fire", 0),
                        packet.get("speed", 0),
                        packet.get("reload"),
                        packet_speed,
                        vesc_target,
                    ),
                    flush=True,
                )
                new_mode = packet.get("mode", current_mode)
                if new_mode not in ("M", "A"):
                    new_mode = current_mode

                if new_mode != current_mode:
                    print(
                        "\n[Mode Switch] 正在切換至：{}".format(
                            "自動模式" if new_mode == "A" else "手動模式"
                        )
                    )
                    if new_mode == "M":
                        vision.set_active(False)
                        lidar.set_active(False)

                    # 手動切自動時，等待 ID1 回零及 DM 回位、失能完成。
                    manual.on_mode_change(
                        new_mode, fire=packet.get("fire", 0), reload=packet.get("reload")
                    )
                    if new_mode == "A":
                        vision.set_active(True)
                        lidar.set_active(True)
                    current_mode = new_mode

                if current_mode == "M":
                    # 手動模式：使用遙控器 yaw/pitch 控制砲台方向，
                    # fire 按鈕每次「由放開變按下」觸發一發射擊循環。
                    manual.handle_packet(packet)
                    time.sleep(0.02)
                    continue

            else:
                uart.report_if_silent()

            if current_mode == "A":
                # 自動模式維持原有視覺追蹤 + LiDAR 避障流程
                (raw_vx, raw_vy, raw_wz), (yaw, pitch), detected = (
                    vision.get_vision_data()
                )

                if detected:
                    final_vx, final_vy, final_wz, is_evading = (
                        lidar.apply_priority_override(raw_vx, raw_vy, raw_wz)
                    )

                    if is_evading:
                        fire_cmd, fire_speed = 0, 0
                    else:
                        fire_cmd = 1 if (
                            abs(raw_vx) < 15 and abs(raw_vy) < 15
                        ) else 0
                        fire_speed = 80 if fire_cmd else 0
                else:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    yaw, pitch = 50, 50
                    fire_cmd, fire_speed = 0, 0

                uart.send_packet(
                    mode="A",
                    vx=final_vx,
                    vy=final_vy,
                    wz=final_wz,
                    yaw=yaw,
                    pitch=pitch,
                    fire=fire_cmd,
                    speed=fire_speed,
                )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[System] 使用者中斷，關閉系統...")
    finally:
        try:
            manual.close()
        except Exception as exc:
            print("[Manual] 關閉控制器時發生錯誤：", exc)
        uart.close()


if __name__ == "__main__":
    main()

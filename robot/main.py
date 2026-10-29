# main.py
import time
import config
from comm_uart import UARTController
from vision_target import VisionTracker
from lidar_avoid import LidarAvoidance
from manual import ManualController


def main():
    print("====================================================")
    print("TEL-Sean 主控系統：手動 / 自動模式")
    print("====================================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    vision = VisionTracker()
    lidar = LidarAvoidance()

    # 手動模式控制器：接收遙控器封包，管理砲台指令、射擊與彈藥倉格位
    manual = ManualController(uart=uart)

    vision.start()
    lidar.start()
    time.sleep(1.0)

    # 預設手動模式；收到 ESP32 封包後依封包中的 mode 切換
    current_mode = "M"
    print("[System] 已啟動，預設手動模式，等待 ESP32 遙控器封包...")

    try:
        while True:
            packet = uart.read_esp32_packet()

            if packet is not None:
                new_mode = packet.get("mode", current_mode)
                if new_mode not in ("M", "A"):
                    new_mode = current_mode

                if new_mode != current_mode:
                    print(
                        "\n[Mode Switch] 切換至：{}".format(
                            "自動模式" if new_mode == "A" else "手動模式"
                        )
                    )
                    current_mode = new_mode
                    is_auto = current_mode == "A"
                    vision.set_active(is_auto)
                    lidar.set_active(is_auto)

                    # 切換模式時清除手動按鈕邊緣狀態，避免按住按鈕時誤觸發
                    manual.on_mode_change(current_mode)

                if current_mode == "M":
                    # 手動模式：使用遙控器 yaw/pitch 控制砲台方向，
                    # fire 按鈕每次「由放開變按下」觸發一發射擊循環。
                    manual.handle_packet(packet)
                    time.sleep(0.02)
                    continue

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

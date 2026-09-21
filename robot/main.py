# main.py
import time
import config
from comm_uart import UARTController
from vision_target import VisionTracker
from lidar_avoid import LidarAvoidance

def main():
    print("====================================================")
    print(" 競賽極速主控系統 (優先級: 避障 > 視覺 | 支援無縫手動切換) ")
    print("====================================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    vision = VisionTracker()
    lidar = LidarAvoidance()

    vision.start()
    lidar.start()
    time.sleep(1.0)
    
    current_mode = 'M'
    print("[System] 系統啟動，等待 ESP32 通訊...")

    try:
        while True:
            # 1. 讀取 ESP32 遙控器模式狀態
            esp32_packet = uart.read_esp32_packet()
            
            if esp32_packet is not None:
                new_mode = esp32_packet.get('mode', 'M')
                
                # 狀態切換事件
                if new_mode != current_mode:
                    print(f"\n[Mode Switch] 切換至: {'自動(Auto)' if new_mode=='A' else '手動(Manual)'} 模式")
                    current_mode = new_mode
                    # 控制子執行緒休眠/喚醒，節省 Xavier 算力
                    is_auto = (current_mode == 'A')
                    vision.set_active(is_auto)
                    lidar.set_active(is_auto)

            # ==========================================
            # 分流控制邏輯
            # ==========================================
            if current_mode == 'M':
                # 手動模式：Xavier 閉嘴不送資料，ESP32 掌握全局
                time.sleep(0.05) 
                continue

            elif current_mode == 'A':
                # 自動模式：Xavier 接管
                (raw_vx, raw_vy, raw_wz), (yaw, pitch), detected = vision.get_vision_data()

                if detected:
                    # [最高優先級] 避障覆蓋
                    final_vx, final_vy, final_wz, is_evading = lidar.apply_priority_override(
                        raw_vx, raw_vy, raw_wz
                    )

                    if is_evading:
                        fire_cmd, fire_speed = 0, 0  # 閃避時強制鎖定射擊
                    else:
                        fire_cmd = 1 if (abs(raw_vx) < 15 and abs(raw_vy) < 15) else 0
                        fire_speed = 80 if fire_cmd else 0
                else:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    yaw, pitch = 50, 50
                    fire_cmd, fire_speed = 0, 0

                # 傳送指令給 ESP32 (即使底盤閃避中，yaw/pitch 仍會持續鎖定標靶)
                uart.send_packet(
                    mode='A',
                    vx=final_vx,
                    vy=final_vy,
                    wz=final_wz,
                    yaw=yaw,
                    pitch=pitch,
                    fire=fire_cmd,
                    speed=fire_speed
                )

            time.sleep(0.02) # 50Hz 控制週期

    except KeyboardInterrupt:
        print("\n[System] 使用者中斷，關閉系統...")
    finally:
        uart.close()

if __name__ == '__main__':
    main()

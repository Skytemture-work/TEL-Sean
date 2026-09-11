import time
import sys
import os

# 將上一層目錄加入系統路徑，以利匯入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from comm_uart import UARTController

def main():
    print("========================================")
    print(" UART 單元測試 (Xavier <-> ESP32)")
    print(" 請嘗試操作遙控器 SW2，觀察模式切換")
    print("========================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    
    if uart.ser is None:
        print("❌ UART 初始化失敗，請檢查硬體腳位 (TX/RX) 與權限 (sudo chmod 777 /dev/ttyTHS1)")
        return

    print("✅ UART 開啟成功，開始監聽資料...\n")

    try:
        while True:
            # 測試讀取 ESP32 封包
            packet = uart.read_esp32_packet()
            
            if packet is not None:
                mode_str = "自動 (A)" if packet['mode'] == 'A' else "手動 (M)"
                print(f"[收到 ESP32 狀態] 模式: {mode_str} | 底盤: VX={packet['vx']}, VY={packet['vy']} | 砲台: YAW={packet['yaw']}")

            # 測試發送虛擬封包給 ESP32 (模擬自動模式)
            # 你可以用示波器或 ESP32 端的 Serial Monitor 查看是否有收到這組資料
            uart.send_packet(mode='A', vx=10, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0)
            
            time.sleep(0.1) # 10Hz 測試頻率

    except KeyboardInterrupt:
        print("\n測試結束，關閉 UART")
        uart.close()

if __name__ == '__main__':
    main()
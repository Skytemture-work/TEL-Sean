import time
import sys
import os
import threading

# 將上一層目錄加入系統路徑，以利匯入 config 與 comm_uart 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from comm_uart import UARTController

# 全域運行標記
running = True

def receive_thread(uart):
    """背景執行的接收線程：持續監聽從 RX 傳進來的資料 (例如 MobaXterm 傳來的訊息)"""
    global running
    while running:
        try:
            # 讀取 ESP32/MobaXterm 發送過來的封包 (例如: E,A,0,0,0,50,50,0,0\n)
            packet = uart.read_esp32_packet()
            
            if packet is not None:
                mode_str = "自動 (A)" if packet['mode'] == 'A' else "手動 (M)"
                print(f"\n📥 [收到電腦/ESP32 回傳] 模式: {mode_str} | 底盤: VX={packet['vx']}, VY={packet['vy']} | 砲台: YAW={packet['yaw']}")
                print("請輸入要傳送的訊息 > ", end="", flush=True)  # 重新印出提示字元
        except Exception as e:
            pass
        time.sleep(0.02) # 50Hz 讀取頻率

def main():
    global running
    print("========================================")
    print(" 雙向 UART 測試 (Xavier <-> 電腦/MobaXterm)")
    print("========================================")

    # 初始化 UART 控制器
    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    
    if uart.ser is None:
        print("❌ UART contest 初始化失敗，請檢查硬體腳位與權限")
        return

    print("✅ UART 開啟成功！")
    print("💡 提示 1：你可以在鍵盤輸入任何文字後按 Enter，透過 TX 送給 MobaXterm")
    print("💡 提示 2：輸入 'auto' 可發送一次標準 Auto 測試封包 (X,A,10,0,0...)")
    print("💡 提示 3：在 MobaXterm 打字 Enter 送出 (如 E,A,0,0,0,50,50,0,0)，這裡會即時印出")
    print("💡 提示 4：輸入 'exit' 或按 Ctrl+C 可結束程式\n")

    # 啟動背景接收 Thread
    t_rx = threading.Thread(target=receive_thread, args=(uart,), daemon=True)
    t_rx.start()

    try:
        while True:
            # 主線程負責等待使用者在鍵盤輸入內容並傳送 (TX)
            user_input = input("請輸入要傳送的訊息 > ")

            if user_input.strip().lower() == 'exit':
                print("結束測試程式。")
                break
            
            # 快捷指令：輸入 auto 發送標準測試封包
            elif user_input.strip().lower() == 'auto':
                uart.send_packet(mode='A', vx=10, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0)
                print("📤 [已送出 Auto 封包] -> X,A,10,0,0,50,50,0,0")
            
            # 自訂字串傳送
            else:
                msg_to_send = user_input + "\n"
                uart.ser.write(msg_to_send.encode('utf-8'))
                print(f"📤 [已透過 TX 送出] -> {user_input}")

    except KeyboardInterrupt:
        print("\n使用者中斷，關閉 UART")
    finally:
        running = False
        uart.close()

if __name__ == '__main__':
    main()

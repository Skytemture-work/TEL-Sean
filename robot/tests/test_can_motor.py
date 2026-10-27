import time
import serial

# 請根據您的作業系統修改正確的串口路徑（如 Windows 的 'COM3' 或 Linux 的 '/dev/ttyACM0'）
serial_port = '/dev/ttyACM0'


def main():
    # 這是實機腳本，不應在 unittest 收集檔案時連接或轉動馬達。
    from pyvesc import VESC

    try:
        # 初始化 VESC 連線
        with VESC(serial_port=serial_port, baudrate=921600) as motor:
            print("成功連接至 MKS MINI V6.7 Pro！")

            # 範例 1：設定馬達轉速 (RPM)
            # 注意：此處的 RPM 通常為轉子電轉速 (ERPM) = 機械轉速 * 馬達極對數
            print("設定轉速為 3000 ERPM...")
            motor.set_rpm(500)
            time.sleep(3)

            # 範例 2：讀取馬達即時數據
            print("\n開始讀取傳感器數據：")
            for _ in range(5):
                # 請求即時狀態
                data = motor.get_measurements()
                if data:
                    print(f"輸入電壓: {data.v_in:.2f} V")
                    print(f"馬達電流: {data.avg_motor_current:.2f} A")
                    print(f"當前 ERPM: {data.rpm}")
                    print(f"MOSFET 溫度: {data.temp_mos1:.1f} °C")
                    print("-" * 30)
                time.sleep(1)

            # 範例 3：煞車並釋放馬達
            print("馬達煞車...")
            motor.set_rpm(0)

    except serial.SerialException:
        print(f"無法打開串口 {serial_port}，請檢查設備連線或埠號是否正確。")
    except Exception as e:
        print(f"發生錯誤: {e}")


if __name__ == "__main__":
    main()

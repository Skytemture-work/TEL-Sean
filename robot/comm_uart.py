# comm_uart.py
import serial
import time

class UARTController:
    def __init__(self, port, baudrate):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
            print(f"[UART] 成功開啟通訊埠: {port}")
        except Exception as e:
            print(f"[UART Error] 無法開啟串口: {e}")
            self.ser = None

    def read_esp32_packet(self):
        """讀取並解析 ESP32 -> Xavier 的封包 (E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n)"""
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                
                # 檢查標頭是否為 ESP32 的 'E'
                if len(parts) >= 9 and parts[0] == 'E':
                    return {
                        'mode': parts[1],      # 'M' (Manual) 或 'A' (Auto)
                        'vx': int(parts[2]),
                        'vy': int(parts[3]),
                        'wz': int(parts[4]),
                        'yaw': int(parts[5]),
                        'pitch': int(parts[6]),
                        'fire': int(parts[7]),
                        'speed': int(parts[8])
                    }
        except Exception:
            pass
        return None

    def send_packet(self, mode='A', vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0):
        """發送 Xavier -> ESP32 封包"""
        if self.ser is None or not self.ser.is_open:
            return

        # 數值限幅保護 (-100 ~ 100)
        vx = max(-100, min(100, int(vx)))
        vy = max(-100, min(100, int(vy)))
        wz = max(-100, min(100, int(wz)))
        yaw = max(0, min(100, int(yaw)))
        pitch = max(0, min(100, int(pitch)))
        fire = 1 if fire else 0
        speed = max(0, min(100, int(speed)))

        packet = f"X,{mode},{vx},{vy},{wz},{yaw},{pitch},{fire},{speed}\n"
        try:
            self.ser.write(packet.encode('utf-8'))
        except Exception as e:
            print(f"[UART Error] 發送失敗: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.send_packet('A', 0, 0, 0, 50, 50, 0, 0) # 安全煞車
            self.ser.close()
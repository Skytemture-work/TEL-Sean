# comm_uart.py
import serial
import time

class UARTController:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self._rx_buffer = bytearray()
        self._rx_bytes_total = 0
        self._ignored_lines_total = 0
        self._last_valid_packet_at = None
        self._last_status_at = 0.0
        try:
            self.ser = serial.Serial(
                port, baudrate, bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                timeout=0.05, write_timeout=0.10,
                xonxoff=False, rtscts=False, dsrdtr=False,
            )
            self.ser.reset_input_buffer()
            print(f"[UART] 成功開啟通訊埠: {port} @ {baudrate}", flush=True)
        except Exception as e:
            print(f"[UART Error] 無法開啟串口: {e}")
            self.ser = None

    def read_esp32_packet(self):
        """只回傳目前最新的完整有效封包；丟棄舊完整行，保留未收完的尾端。"""
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            waiting = self.ser.in_waiting
            if waiting:
                chunk = self.ser.read(waiting)
                self._rx_buffer.extend(chunk)
                self._rx_bytes_total += len(chunk)

            newline = self._rx_buffer.rfind(b"\n")
            if newline < 0:
                if len(self._rx_buffer) > 4096:
                    print("[UART RX INVALID] line exceeded 4096 bytes; clearing buffer", flush=True)
                    self._rx_buffer.clear()
                return None

            complete_lines = bytes(self._rx_buffer[:newline]).split(b"\n")
            del self._rx_buffer[:newline + 1]
            # Search newest first: once a valid packet is found, all older
            # complete lines have already been removed and cannot be replayed.
            for raw_line in reversed(complete_lines):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Recover "TX->Xavier: E,..." and ignore debug-only lines.
                if not line.startswith("E,"):
                    packet_start = line.find("E,")
                    if packet_start >= 0:
                        line = line[packet_start:]
                    else:
                        self._ignored_lines_total += 1
                        continue

                parts = line.split(',')
                # ESP32 -> Xavier: E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED,RELOAD[,AUTO_START]
                if len(parts) not in (10, 11) or parts[0] != 'E' or parts[1] not in ('M', 'A'):
                    print(
                        "[UART RX INVALID] expected E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED,RELOAD[,AUTO_START]; got={!r}".format(line),
                        flush=True,
                    )
                    continue

                try:
                    packet = {
                        'raw': line,
                        'mode': parts[1],
                        'vx': int(parts[2]),
                        'vy': int(parts[3]),
                        'wz': int(parts[4]),
                        'yaw': int(parts[5]),
                        'pitch': int(parts[6]),
                        'fire': int(parts[7]),
                        'speed': int(parts[8]),
                        'reload': int(parts[9]),
                        'auto_start': int(parts[10]) if len(parts) == 11 else 0,
                    }
                except ValueError:
                    print("[UART RX INVALID] non-numeric ESP32 packet: {!r}".format(line), flush=True)
                    continue

                self._last_valid_packet_at = time.monotonic()
                return packet
        except serial.SerialException as exc:
            print("[UART ERROR] {}: {}".format(self.port, exc), flush=True)
        return None

    def report_if_silent(self, interval=3.0):
        """Periodically report when no valid ESP32 packet reaches this UART."""
        now = time.monotonic()
        if now - self._last_status_at < interval:
            return
        self._last_status_at = now
        if self._last_valid_packet_at is None:
            print(
                "[UART WAIT] no valid ESP32 packet on {} @ {}; received {} bytes, ignored {} debug lines".format(
                    self.port, self.baudrate, self._rx_bytes_total,
                    self._ignored_lines_total
                ),
                flush=True,
            )
        elif now - self._last_valid_packet_at >= interval:
            print(
                "[UART LOST] last valid ESP32 packet was {:.1f}s ago; received {} bytes total".format(
                    now - self._last_valid_packet_at, self._rx_bytes_total
                ),
                flush=True,
            )

    def send_packet(self, mode='A', vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0, *, trace=False):
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
            payload = packet.encode('ascii')
            written = self.ser.write(payload)
            if trace and written == len(payload):
                print(f"[ESP32 TX] {packet.rstrip()}", flush=True)
            elif written != len(payload):
                print(f"[UART Error] ESP32 TX short write: {written}/{len(payload)} bytes", flush=True)
        except Exception as e:
            print(f"[UART Error] 發送失敗: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.send_packet('A', 0, 0, 0, 50, 50, 0, 0) # 安全煞車
            self.ser.close()

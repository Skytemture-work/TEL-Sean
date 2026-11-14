"""Send alternating chassis commands to the ESP32 at 50 Hz."""

from itertools import cycle
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from comm_uart import UARTController


def main():
    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    if uart.ser is None or not uart.ser.is_open:
        return 1

    try:
        started_at = time.monotonic()
        last_command = None
        for vy in cycle((20, 0)):
            deadline = time.monotonic() + 10.0
            last_echo = None
            while time.monotonic() < deadline:
                # Raise VX by 5 every second and hold at 100.
                vx = min(100, 40 + 5 * int(time.monotonic() - started_at))
                command = (vx, vy)
                uart.send_packet(mode="A", vx=vx, vy=vy, trace=command != last_command)
                last_command = command
                packet = uart.read_esp32_packet()
                if packet is not None:
                    echo = tuple(packet[key] for key in ("mode", "vx", "vy", "wz"))
                    if echo != last_echo:
                        print(f"[ESP32 RX] {packet['raw']}", flush=True)
                        last_echo = echo
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n停止測試，送出底盤歸零封包。", flush=True)
    finally:
        uart.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

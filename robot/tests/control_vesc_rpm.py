
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import struct
import serial

# ==================================================
# 1. User configuration
# ==================================================

PORT = "/dev/ttyACM0"
BAUDRATE = 921600

# Motor settings
TARGET_RPM = 800
POLE_PAIRS = 7              # 必須確認實際極對數

# Safety settings
RUN_DURATION = 5.0
LOOP_INTERVAL = 0.10
MAX_MOTOR_CURRENT_A = 50.0   # 軟體保護門檻，需依硬體規格調整
TELEMETRY_TIMEOUT = 0.5

# False = 只讀取 VESC 資料，不驅動馬達
# True  = 啟用 RPM 控制（請先完成 VESC 馬達設定）
ENABLE_MOTOR_COMMAND = True

# VESC command IDs
COMM_GET_VALUES = 4
COMM_SET_RPM = 8

# VESC packet markers
PACKET_SHORT = 2
PACKET_LONG = 3
PACKET_END = 3

# ==================================================
# 2. CRC-16/CCITT
# ==================================================

def crc16(data):
    crc = 0

    for byte in data:
        crc ^= byte << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc

# ==================================================
# 3. VESC packet creation
# ==================================================

def make_packet(payload):
    length = len(payload)

    if length <= 0xFF:
        packet = bytearray([PACKET_SHORT, length])
    elif length <= 0xFFFF:
        packet = bytearray([
            PACKET_LONG,
            (length >> 8) & 0xFF,
            length & 0xFF
        ])
    else:
        raise ValueError("VESC payload too large")

    packet.extend(payload)
    packet.extend(struct.pack(">H", crc16(payload)))
    packet.append(PACKET_END)

    return bytes(packet)

def send_command(ser, command, data=b""):
    payload = bytes([command]) + data
    ser.write(make_packet(payload))
    ser.flush()

def send_set_rpm(ser, rpm_erpm):
    data = struct.pack(">i", int(rpm_erpm))
    send_command(ser, COMM_SET_RPM, data)

def request_values(ser):
    send_command(ser, COMM_GET_VALUES)

# ==================================================
# 4. Persistent packet reader
# ==================================================

class VESCReader:
    def __init__(self, ser):
        self.ser = ser
        self.buffer = bytearray()

    def read_packet(self, timeout=0.5):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            # Parse all complete packets already in the buffer.
            while self.buffer:
                start = next(
                    (
                        i for i, value in enumerate(self.buffer)
                        if value in (PACKET_SHORT, PACKET_LONG)
                    ),
                    None
                )

                if start is None:
                    self.buffer.clear()
                    break

                if start > 0:
                    del self.buffer[:start]

                if self.buffer[0] == PACKET_SHORT:
                    if len(self.buffer) < 2:
                        break

                    header_len = 2
                    payload_len = self.buffer[1]

                else:
                    if len(self.buffer) < 3:
                        break

                    header_len = 3
                    payload_len = (
                        (self.buffer[1] << 8) |
                        self.buffer[2]
                    )

                # Avoid unreasonable lengths if the stream is misaligned.
                if payload_len <= 0 or payload_len > 1024:
                    del self.buffer[0]
                    continue

                total_len = header_len + payload_len + 3

                if len(self.buffer) < total_len:
                    break

                packet = bytes(self.buffer[:total_len])

                if packet[-1] != PACKET_END:
                    del self.buffer[0]
                    continue

                payload_start = header_len
                payload_end = payload_start + payload_len
                payload = packet[payload_start:payload_end]

                received_crc = struct.unpack(
                    ">H", packet[payload_end:payload_end + 2]
                )[0]

                if crc16(payload) == received_crc:
                    del self.buffer[:total_len]
                    return payload

                # Invalid CRC: discard one byte and resynchronize.
                del self.buffer[0]

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            chunk = self.ser.read(1)

            if chunk:
                self.buffer.extend(chunk)

        return None

    def get_values(self, timeout=TELEMETRY_TIMEOUT):
        request_values(self.ser)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            payload = self.read_packet(
                timeout=min(0.1, remaining)
            )

            if payload is None:
                continue

            if payload[0] == COMM_GET_VALUES:
                return payload

        return None

# ==================================================
# 5. Decode COMM_GET_VALUES
# ==================================================

def decode_values(payload):
    """
    Legacy VESC COMM_GET_VALUES layout.

    Verify these offsets against the installed firmware.
    This function deliberately rejects short responses.
    """

    if payload is None:
        raise ValueError("No payload")

    if payload[0] != COMM_GET_VALUES:
        raise ValueError("Unexpected command ID")

    if len(payload) < 29:
        raise ValueError(
            f"Payload too short: {len(payload)} bytes"
        )

    mos_temp = struct.unpack_from(">h", payload, 1)[0] / 10.0
    motor_temp = struct.unpack_from(">h", payload, 3)[0] / 10.0
    motor_current = struct.unpack_from(">i", payload, 5)[0] / 100.0
    input_current = struct.unpack_from(">i", payload, 9)[0] / 100.0
    duty = struct.unpack_from(">h", payload, 21)[0] / 1000.0
    erpm = struct.unpack_from(">i", payload, 23)[0]
    voltage = struct.unpack_from(">h", payload, 27)[0] / 10.0

    return {
        "mos_temp": mos_temp,
        "motor_temp": motor_temp,
        "motor_current": motor_current,
        "input_current": input_current,
        "duty": duty,
        "erpm": erpm,
        "voltage": voltage,
    }

# ==================================================
# 6. Main
# ==================================================

def main():
    ser = None
    reader = None
    motor_command_enabled = False

    target_erpm = TARGET_RPM * POLE_PAIRS

    print("=" * 55)
    print("VESC USB Monitor / Speed Controller")
    print("=" * 55)
    print(f"Port: {PORT}")
    print(f"Baud rate: {BAUDRATE}")
    print(f"Motor command enabled: {ENABLE_MOTOR_COMMAND}")
    print(f"Target RPM: {TARGET_RPM}")
    print(f"Assumed pole pairs: {POLE_PAIRS}")
    print(f"Target ERPM: {target_erpm}")
    print(f"Maximum monitored motor current: "
          f"{MAX_MOTOR_CURRENT_A:.1f} A")
    print()

    if ENABLE_MOTOR_COMMAND:
        print("WARNING: This mode can make the motor rotate.")
        print("Confirm motor detection, current limits, and")
        print("physical power-disconnect capability first.")

        if input("Type ARM MOTOR to continue: ") != "ARM MOTOR":
            print("Cancelled.")
            return

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            timeout=0.05,
            write_timeout=0.5
        )

        time.sleep(0.2)
        ser.reset_input_buffer()

        reader = VESCReader(ser)

        # First confirm telemetry without sending any RPM command.
        initial_payload = reader.get_values()

        if initial_payload is None:
            print("ERROR: No valid GET_VALUES response.")
            print("Motor was not commanded.")
            return

        try:
            values = decode_values(initial_payload)
        except ValueError as exc:
            print(f"ERROR: Cannot decode telemetry: {exc}")
            print("Motor was not commanded.")
            return

        print("\nInitial VESC telemetry")
        print(f"  MOS temperature : {values['mos_temp']:.1f} C")
        print(f"  Motor temperature: {values['motor_temp']:.1f} C")
        print(f"  Motor current   : {values['motor_current']:.2f} A")
        print(f"  Input current   : {values['input_current']:.2f} A")
        print(f"  Duty cycle      : {values['duty']:.3f}")
        print(f"  ERPM            : {values['erpm']}")
        print(f"  Input voltage   : {values['voltage']:.1f} V")

        if not ENABLE_MOTOR_COMMAND:
            print("\nMONITOR MODE: no RPM command was sent.")
            print("Check the telemetry before enabling motor control.")
            return

        if abs(values["motor_current"]) >= MAX_MOTOR_CURRENT_A:
            print("\nSAFETY STOP: initial current exceeds threshold.")
            print("No RPM command was sent.")
            return

        motor_command_enabled = True
        start_time = time.monotonic()
        next_update = start_time

        print("\nStarting speed control...")

        while time.monotonic() - start_time < RUN_DURATION:
            send_set_rpm(ser, target_erpm)

            payload = reader.get_values()

            if payload is None:
                print("SAFETY STOP: telemetry timeout.")
                break

            try:
                values = decode_values(payload)
            except ValueError as exc:
                print(f"SAFETY STOP: invalid telemetry: {exc}")
                break

            current = values["motor_current"]

            actual_rpm = values["erpm"] / POLE_PAIRS
            error = TARGET_RPM - actual_rpm

            print(
                f"Target: {TARGET_RPM:6.1f} RPM | "
                f"Actual: {actual_rpm:8.1f} RPM | "
                f"Error: {error:8.1f} RPM | "
                f"Current: {current:7.2f} A | "
                f"Voltage: {values['voltage']:5.1f} V | "
                f"Duty: {values['duty']:.3f}"
            )

            if abs(current) >= MAX_MOTOR_CURRENT_A:
                print(
                    "SAFETY STOP: motor current exceeded "
                    f"{MAX_MOTOR_CURRENT_A:.1f} A."
                )
                break

            next_update += LOOP_INTERVAL
            time.sleep(max(0.0, next_update - time.monotonic()))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except (serial.SerialException, OSError) as exc:
        print(f"Serial communication error: {exc}")

    finally:
        if ser is not None and ser.is_open:
            if motor_command_enabled:
                try:
                    # Best-effort stop command.
                    send_set_rpm(ser, 0)
                    time.sleep(0.1)
                    send_set_rpm(ser, 0)
                    ser.flush()
                except Exception as exc:
                    print(f"Could not send zero-RPM command: {exc}")
                    print("Disconnect motor main power if necessary.")

            ser.close()

        print("Serial port closed.")

if __name__ == "__main__":
    main()

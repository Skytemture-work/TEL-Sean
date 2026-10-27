
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import struct
import threading
import serial

# ============================================================
# 1. User configuration
# ============================================================

PORT_LEFT = "/dev/ttyACM0"
PORT_RIGHT = "/dev/ttyACM1"
BAUDRATE = 921600

# Mechanical target speed
TARGET_RPM = 3000.0

# Must be verified for both motors.
POLE_PAIRS_LEFT = 7
POLE_PAIRS_RIGHT = 7

# Control loop
LOOP_INTERVAL = 0.10       # 10 Hz
TELEMETRY_TIMEOUT = 0.5

# Conservative software limits.
# Set these only after confirming hardware specifications.
MAX_MOTOR_CURRENT_A = 50.0
MAX_INPUT_VOLTAGE_V = 30.0
MIN_INPUT_VOLTAGE_V = 18.0

# Speed trim limits, in ERPM.
# The base target remains TARGET_RPM * pole_pairs.
MAX_TRIM_ERPM = 100.0
MAX_SYNC_CORRECTION_ERPM = 50.0

# Outer-loop PI gains.
# Start small. Tune only after verifying telemetry.
SPEED_KP = 0.10
SPEED_KI = 0.02

# Cross-coupled speed synchronization gain.
SYNC_KP = 0.10

# Safety: keep False until telemetry is verified.
ENABLE_MOTOR_COMMAND = True

# VESC command IDs
COMM_GET_VALUES = 4
COMM_SET_RPM = 8

PACKET_SHORT = 2
PACKET_LONG = 3
PACKET_END = 3

# ============================================================
# 2. VESC packet / CRC
# ============================================================

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

def make_packet(payload):
    length = len(payload)

    if length <= 255:
        packet = bytearray([PACKET_SHORT, length])
    elif length <= 65535:
        packet = bytearray([
            PACKET_LONG,
            (length >> 8) & 0xFF,
            length & 0xFF
        ])
    else:
        raise ValueError("Payload too large")

    packet.extend(payload)
    packet.extend(struct.pack(">H", crc16(payload)))
    packet.append(PACKET_END)

    return bytes(packet)

def send_command(ser, command, data=b""):
    payload = bytes([command]) + data
    ser.write(make_packet(payload))
    ser.flush()

def send_set_rpm(ser, erpm):
    send_command(
        ser,
        COMM_SET_RPM,
        struct.pack(">i", int(erpm))
    )

def request_values(ser):
    send_command(ser, COMM_GET_VALUES)

# ============================================================
# 3. Persistent packet reader
# ============================================================

class VESCReader:
    def __init__(self, ser):
        self.ser = ser
        self.buffer = bytearray()

    def read_packet(self, timeout):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            while self.buffer:
                start = next(
                    (
                        i for i, b in enumerate(self.buffer)
                        if b in (PACKET_SHORT, PACKET_LONG)
                    ),
                    None
                )

                if start is None:
                    self.buffer.clear()
                    break

                if start:
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
                        (self.buffer[1] << 8) | self.buffer[2]
                    )

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

                p0 = header_len
                p1 = p0 + payload_len
                payload = packet[p0:p1]

                received_crc = struct.unpack(
                    ">H", packet[p1:p1 + 2]
                )[0]

                if crc16(payload) == received_crc:
                    del self.buffer[:total_len]
                    return payload

                del self.buffer[0]

            chunk = self.ser.read(1)
            if chunk:
                self.buffer.extend(chunk)

        return None

    def get_values(self, timeout=TELEMETRY_TIMEOUT):
        request_values(self.ser)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            payload = self.read_packet(
                max(0.01, deadline - time.monotonic())
            )

            if payload and payload[0] == COMM_GET_VALUES:
                return payload

        return None

# ============================================================
# 4. Legacy telemetry decoder
# IMPORTANT: verify against your VESC firmware.
# ============================================================

def decode_values(payload):
    if payload is None or len(payload) < 29:
        raise ValueError("Missing or short telemetry payload")

    if payload[0] != COMM_GET_VALUES:
        raise ValueError("Unexpected telemetry command ID")

    return {
        "mos_temp": struct.unpack_from(">h", payload, 1)[0] / 10.0,
        "motor_temp": struct.unpack_from(">h", payload, 3)[0] / 10.0,
        "motor_current": struct.unpack_from(">i", payload, 5)[0] / 100.0,
        "input_current": struct.unpack_from(">i", payload, 9)[0] / 100.0,
        "duty": struct.unpack_from(">h", payload, 21)[0] / 1000.0,
        "erpm": struct.unpack_from(">i", payload, 23)[0],
        "voltage": struct.unpack_from(">h", payload, 27)[0] / 10.0,
    }

def read_motor(reader):
    payload = reader.get_values()
    if payload is None:
        raise RuntimeError("VESC telemetry timeout")

    values = decode_values(payload)

    if not (-100000 <= values["erpm"] <= 100000):
        raise RuntimeError("Implausible ERPM reading")

    if not (0.0 <= values["voltage"] <= MAX_INPUT_VOLTAGE_V):
        raise RuntimeError(
            f"Unexpected input voltage: {values['voltage']:.1f} V"
        )

    if abs(values["motor_current"]) > MAX_MOTOR_CURRENT_A:
        raise RuntimeError(
            f"Motor current limit exceeded: "
            f"{values['motor_current']:.2f} A"
        )

    if values["voltage"] < MIN_INPUT_VOLTAGE_V:
        raise RuntimeError(
            f"Input voltage too low: {values['voltage']:.1f} V"
        )

    return values

# ============================================================
# 5. Dual motor controller
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))

class PIController:
    def __init__(self, kp, ki, output_limit):
        self.kp = kp
        self.ki = ki
        self.output_limit = output_limit
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt

        output = self.kp * error + self.ki * self.integral

        output = clamp(
            output,
            -self.output_limit,
            self.output_limit
        )

        return output

    def reset(self):
        self.integral = 0.0

def stop_motors(ser_left, ser_right):
    # Best-effort stop command; not a hardware emergency stop.
    for ser in (ser_left, ser_right):
        if ser is not None and ser.is_open:
            try:
                send_set_rpm(ser, 0)
            except Exception:
                pass

    time.sleep(0.1)

    for ser in (ser_left, ser_right):
        if ser is not None and ser.is_open:
            try:
                send_set_rpm(ser, 0)
            except Exception:
                pass

def main():
    ser_left = None
    ser_right = None
    command_active = False

    base_left = TARGET_RPM * POLE_PAIRS_LEFT
    base_right = TARGET_RPM * POLE_PAIRS_RIGHT

    pi_left = PIController(
        SPEED_KP, SPEED_KI, MAX_TRIM_ERPM
    )
    pi_right = PIController(
        SPEED_KP, SPEED_KI, MAX_TRIM_ERPM
    )

    try:
        print("Opening both VESC serial ports...")

        ser_left = serial.Serial(
            PORT_LEFT,
            BAUDRATE,
            timeout=0.05,
            write_timeout=0.5
        )
        ser_right = serial.Serial(
            PORT_RIGHT,
            BAUDRATE,
            timeout=0.05,
            write_timeout=0.5
        )

        time.sleep(0.2)
        ser_left.reset_input_buffer()
        ser_right.reset_input_buffer()

        reader_left = VESCReader(ser_left)
        reader_right = VESCReader(ser_right)

        print("Reading initial telemetry...")

        left = read_motor(reader_left)
        right = read_motor(reader_right)

        print(
            f"LEFT  ERPM={left['erpm']}, "
            f"V={left['voltage']:.1f} V, "
            f"I={left['motor_current']:.2f} A"
        )
        print(
            f"RIGHT ERPM={right['erpm']}, "
            f"V={right['voltage']:.1f} V, "
            f"I={right['motor_current']:.2f} A"
        )

        print("\nVerify these values match VESC Tool.")
        print("No RPM commands have been sent yet.")

        if not ENABLE_MOTOR_COMMAND:
            print(
                "\nMONITOR ONLY. Set ENABLE_MOTOR_COMMAND=True "
                "only after validating telemetry and safety limits."
            )
            return

        command_active = True
        last_time = time.monotonic()

        print("\nRunning continuously. Press Ctrl+C to stop.")

        while True:
            now = time.monotonic()
            dt = clamp(now - last_time, 0.01, 0.5)
            last_time = now

            # Read both VESCs before updating either target.
            left = read_motor(reader_left)
            right = read_motor(reader_right)

            rpm_left = left["erpm"] / POLE_PAIRS_LEFT
            rpm_right = right["erpm"] / POLE_PAIRS_RIGHT

            # Individual speed tracking.
            error_left = TARGET_RPM - rpm_left
            error_right = TARGET_RPM - rpm_right

            trim_left = pi_left.update(error_left, dt)
            trim_right = pi_right.update(error_right, dt)

            # Cross-coupled synchronization:
            # if left is faster, reduce left target and raise right target.
            sync_error = rpm_left - rpm_right

            sync = clamp(
                SYNC_KP * sync_error,
                -MAX_SYNC_CORRECTION_ERPM,
                MAX_SYNC_CORRECTION_ERPM
            )

            target_left = base_left + trim_left - sync
            target_right = base_right + trim_right + sync

            # Keep each target close to the requested base speed.
            target_left = clamp(
                target_left,
                base_left - MAX_TRIM_ERPM - MAX_SYNC_CORRECTION_ERPM,
                base_left + MAX_TRIM_ERPM + MAX_SYNC_CORRECTION_ERPM
            )
            target_right = clamp(
                target_right,
                base_right - MAX_TRIM_ERPM - MAX_SYNC_CORRECTION_ERPM,
                base_right + MAX_TRIM_ERPM + MAX_SYNC_CORRECTION_ERPM
            )

            send_set_rpm(ser_left, target_left)
            send_set_rpm(ser_right, target_right)

            print(
                f"L target={target_left:7.1f} ERPM "
                f"actual={left['erpm']:7d} "
                f"rpm={rpm_left:7.1f} | "
                f"R target={target_right:7.1f} ERPM "
                f"actual={right['erpm']:7d} "
                f"rpm={rpm_right:7.1f} | "
                f"diff={rpm_left-rpm_right:7.1f} RPM | "
                f"I=({left['motor_current']:.1f},"
                f"{right['motor_current']:.1f}) A"
            )

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping both motors...")

    except Exception as exc:
        print(f"\nCONTROL STOP: {exc}")

    finally:
        if command_active:
            stop_motors(ser_left, ser_right)

        for ser in (ser_left, ser_right):
            if ser is not None and ser.is_open:
                ser.close()

        print("Both serial ports closed.")

if __name__ == "__main__":
    main()

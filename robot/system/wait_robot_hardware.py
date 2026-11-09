#!/usr/bin/env python3
"""Wait for configured CAN/serial devices without opening or driving motors."""

import argparse
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import manual
from dm_motor_test.dm_port import find_port


def check_hardware():
    manual.validate_mks_can_interfaces()
    dm_port = find_port(manual.DM_STARTUP_TEST_PORT)
    devices = {
        "ESP32 UART": config.SERIAL_PORT,
        "DM USB-to-CAN": dm_port,
        "LEFT VESC": manual.VESC_LEFT_PORT,
        "RIGHT VESC": manual.VESC_RIGHT_PORT,
    }
    physical_ports = set()
    for label, port in devices.items():
        if not os.path.exists(port):
            raise RuntimeError("{} 尚未出現：{}".format(label, port))
        if not os.access(port, os.R_OK | os.W_OK):
            raise RuntimeError("{} 尚無讀寫權限：{}".format(label, port))
        physical = os.path.realpath(port)
        if physical in physical_ports:
            raise RuntimeError("序列埠設定重複：{} -> {}".format(label, physical))
        physical_ports.add(physical)
    return devices


def main():
    parser = argparse.ArgumentParser(description="等待 CAN 和序列裝置就緒；只檢查，不使能馬達")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必須是大於 0 的有限數值")
    deadline = time.monotonic() + args.timeout
    last_error = None
    while True:
        try:
            devices = check_hardware()
        except (RuntimeError, OSError) as exc:
            message = str(exc)
            if message != last_error:
                print("[Startup] 等待硬體：{}".format(message), flush=True)
                last_error = message
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit("[Startup] 等待逾時：{}".format(message))
            time.sleep(min(0.5, remaining))
        else:
            print("[Startup] CAN 與序列裝置就緒：{}".format(devices), flush=True)
            return


if __name__ == "__main__":
    main()

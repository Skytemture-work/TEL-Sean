#!/usr/bin/env python3
"""Record a manually aligned ammo slot without commanding motor motion."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import can
from mks_servo_can import MksServo
from mks_servo_can.mks_enums import MotorStatus

from manual import AMMO_CAN_INTERFACE, AMMO_MOTOR_CAN_ID


DEFAULT_OUTPUT = Path(__file__).resolve().with_name("ammo_slot1_calibration.json")


def read_stopped_encoder(servo, *, samples=3, interval=0.3, max_drift=20):
    """Read stable accumulated encoder counts without changing driver state."""
    readings = []
    for index in range(samples):
        status = servo.query_motor_status()
        if status != MotorStatus.MotorStop:
            raise RuntimeError("儲存馬達未確認停止，狀態：{}".format(status))
        value = servo.read_encoder_value_addition()
        if value is None:
            raise RuntimeError("儲存馬達編碼器讀取失敗")
        readings.append(int(value))
        if index + 1 < samples:
            time.sleep(interval)
    if max(readings) - min(readings) > max_drift:
        raise RuntimeError("編碼器讀值不穩定，請確認儲存盤靜止：{}".format(readings))
    return readings


def main():
    parser = argparse.ArgumentParser(description="記錄已手動對準的儲存盤第一格；不會轉動馬達。")
    parser.add_argument("--confirm-aligned", action="store_true", help="確認第一格已對準且主控程式已關閉")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="校準資料檔案")
    args = parser.parse_args()
    if not args.confirm_aligned:
        parser.error("請先對準第一格並關閉主控，完成後加上 --confirm-aligned")
    if args.output.exists():
        parser.error("校準檔已存在，為保留既有資料，請先移走或使用其他 --output 路徑")

    bus = can.Bus(interface="socketcan", channel=AMMO_CAN_INTERFACE)
    notifier = None
    try:
        notifier = can.Notifier(bus, [])
        servo = MksServo(bus, notifier, AMMO_MOTOR_CAN_ID)
        servo.timeout = 0.8
        readings = read_stopped_encoder(servo)
    finally:
        if notifier is not None:
            notifier.stop()
        bus.shutdown()

    data = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "can_interface": AMMO_CAN_INTERFACE,
        "motor_can_id": AMMO_MOTOR_CAN_ID,
        "slot": 1,
        "encoder_counts": int(statistics.median(readings)),
        "encoder_samples": readings,
        "reference": "operator_aligned_slot_1",
        "restart_validity": "unverified; requires physical reference or re-confirmation",
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("第一格已記錄：encoder={} counts，樣本={}，檔案={}".format(
        data["encoder_counts"], readings, args.output
    ), flush=True)
    print("這是馬達編碼器基準；斷電、重啟或機構移動後需重新確認。", flush=True)


if __name__ == "__main__":
    main()

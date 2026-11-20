import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comm_uart import UARTController  # noqa: E402


class _FakeSerial:
    is_open = True

    def __init__(self, data: bytes):
        self._data = data

    @property
    def in_waiting(self):
        return len(self._data)

    def read(self, n):
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def reset_input_buffer(self):
        pass


def _read(line: str):
    with patch("comm_uart.serial.Serial", side_effect=OSError("no port")):
        uart = UARTController("/dev/null", 115200)
    uart.ser = _FakeSerial((line + "\n").encode())
    return uart.read_esp32_packet()


class Esp32PacketTests(unittest.TestCase):
    def test_vin_field_is_the_last_one(self):
        packet = _read("E,A,0,0,0,50,50,1,0,0,1,1")
        self.assertEqual(packet["vin"], 1)
        self.assertEqual(packet["auto_start"], 1)
        self.assertEqual(packet["fire"], 1)

    def test_older_firmware_without_vin_reads_zero(self):
        self.assertEqual(_read("E,A,0,0,0,50,50,1,0,0,1")["vin"], 0)
        packet = _read("E,M,0,0,0,50,50,0,0,0")
        self.assertEqual(packet["vin"], 0)
        self.assertEqual(packet["auto_start"], 0)

    def test_too_many_fields_is_rejected(self):
        self.assertIsNone(_read("E,A,0,0,0,50,50,1,0,0,1,1,9"))


if __name__ == "__main__":
    unittest.main()

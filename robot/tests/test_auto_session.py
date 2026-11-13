import threading
import time
import unittest
from unittest import mock

from turret_control import auto_session


class AutoAimSessionTests(unittest.TestCase):
    def test_stop_waits_for_main_to_finish_its_shutdown(self):
        calls = {}
        shutdown_done = threading.Event()

        def fake_main(argv, *, telemetry_callback=None, stop_event=None):
            calls["argv"] = argv
            stop_event.wait(5)
            time.sleep(0.2)  # simulated home + disable + release
            shutdown_done.set()

        with mock.patch.object(auto_session.run_board_real, "main", fake_main):
            session = auto_session.AutoAimSession(["--panel-fire", "--no-score-mode"])
            session.start()
            time.sleep(0.05)
            self.assertTrue(session.running)
            with self.assertRaises(RuntimeError):
                session.start()
            self.assertTrue(session.stop(timeout=5))
        self.assertTrue(shutdown_done.is_set())  # stop() returned only after cleanup
        self.assertFalse(session.running)
        self.assertIn("--headless", calls["argv"])
        self.assertIsNone(session.error)

    def test_startup_failure_is_reported_not_raised_in_host(self):
        def failing_main(argv, *, telemetry_callback=None, stop_event=None):
            raise SystemExit("安全停止：機構幾何尚未核准")

        with mock.patch.object(auto_session.run_board_real, "main", failing_main):
            session = auto_session.AutoAimSession()
            session.start()
            self.assertTrue(session.stop(timeout=5))
        self.assertIsInstance(session.error, SystemExit)


if __name__ == "__main__":
    unittest.main()

import threading
import time
import unittest
from unittest import mock

from turret_control import auto_session


class AutoAimSessionTests(unittest.TestCase):
    def test_stop_waits_for_main_to_finish_its_shutdown(self):
        calls = {}
        shutdown_done = threading.Event()

        def fake_main(argv, *, telemetry_callback=None, stop_event=None, reload_event=None):
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
        def failing_main(argv, *, telemetry_callback=None, stop_event=None, reload_event=None):
            raise SystemExit("安全停止：機構幾何尚未核准")

        with mock.patch.object(auto_session.run_board_real, "main", failing_main):
            session = auto_session.AutoAimSession()
            session.start()
            self.assertTrue(session.stop(timeout=5))
        self.assertIsInstance(session.error, SystemExit)


    def test_confirm_reload_only_while_the_match_waits_for_it(self):
        seen = {}
        waiting = threading.Event()

        def fake_main(argv, *, telemetry_callback=None, stop_event=None, reload_event=None):
            seen["event"] = reload_event
            telemetry_callback({"match": {"reload_wait": False}})
            waiting.wait(5)
            telemetry_callback({"match": {"reload_wait": True}})
            stop_event.wait(5)

        user_telemetry = []
        with mock.patch.object(auto_session.run_board_real, "main", fake_main):
            session = auto_session.AutoAimSession(["--match-fire"],
                                                  telemetry_callback=user_telemetry.append)
            session.start()
            time.sleep(0.05)
            self.assertFalse(session.reload_waiting)
            self.assertFalse(session.confirm_reload())  # not waiting: ignored
            self.assertFalse(seen["event"].is_set())
            waiting.set()
            time.sleep(0.05)
            self.assertTrue(session.reload_waiting)
            self.assertTrue(session.confirm_reload())
            self.assertTrue(seen["event"].is_set())
            self.assertEqual(session.match_state, {"reload_wait": True})
            self.assertTrue(session.stop(timeout=5))
        self.assertEqual(len(user_telemetry), 2)  # host callback still receives data
        self.assertFalse(session.reload_waiting)


class ExternalReloadTests(unittest.TestCase):
    def test_press_counts_only_while_waiting_and_is_never_stored(self):
        from turret_control.run_board_real import _take_external_reload
        event = threading.Event()
        self.assertFalse(_take_external_reload(None, True))
        event.set()
        self.assertFalse(_take_external_reload(event, False))  # early press discarded
        self.assertFalse(event.is_set())
        self.assertFalse(_take_external_reload(event, True))
        event.set()
        self.assertTrue(_take_external_reload(event, True))
        self.assertFalse(_take_external_reload(event, True))  # consumed once

if __name__ == "__main__":
    unittest.main()

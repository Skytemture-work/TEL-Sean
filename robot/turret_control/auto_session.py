"""Run the board auto-aim pipeline as a start/stop session for a host program.

Intended use from ``main.py`` (not wired in yet)::

    from turret_control.auto_session import AutoAimSession

    manual.close()                      # manual side returns to home and frees
                                        # the DM port, can0, VESC ports
    session = AutoAimSession(["--panel-fire", "--no-score-mode"])
    session.start()
    ...
    session.stop()                      # launcher off -> home -> disable -> camera released
    manual = ManualController(uart=uart)  # manual side takes over from home

Hand-over rule: exactly one owner per device (ZED, DM USB-CAN port, can0,
VESC ports).  Whoever releases the turret first returns to home; whoever takes
over starts from home.  ``stop()`` only returns after the auto side has fully
shut down, so the caller can reopen the devices right after it.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Sequence

from . import run_board_real


class AutoAimSession:
    def __init__(
        self,
        argv: Sequence[str] = ("--panel-fire", "--no-score-mode"),
        *,
        telemetry_callback: Optional[Callable[[dict], None]] = None,
    ):
        argv = list(argv)
        # An embedded session never owns an OpenCV window.
        if "--headless" not in argv:
            argv.append("--headless")
        self.argv = argv
        self._telemetry_callback = telemetry_callback
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("自動瞄準已在執行")
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="auto-aim", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        try:
            run_board_real.main(
                self.argv,
                telemetry_callback=self._telemetry_callback,
                stop_event=self._stop,
            )
        except BaseException as exc:  # includes SystemExit from argparse/startup checks
            self.error = exc
            print(f"自動瞄準結束（錯誤）：{exc!r}", flush=True)

    def stop(self, timeout: float = 30.0) -> bool:
        """Request shutdown and wait for home + disable + device release.

        Returns True once the session thread has exited.  The home return alone
        may take up to ~20 s, so keep the timeout generous.  If this returns
        False the devices may still be owned by the auto side: do NOT reopen
        them.
        """
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

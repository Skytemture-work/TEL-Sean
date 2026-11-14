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

Match mode (``["--match-fire", "--match-phase", "upper"]``): when the upper
half has fired its first magazine it holds and waits for a reload.  Call
``session.confirm_reload()`` (e.g. on the remote's RELOAD button in auto mode)
once the magazine is refilled -- same as the dashboard "裝填完成" button.
``session.reload_waiting`` tells whether it is waiting.  The session also ends
by itself when the match is over (``session.running`` becomes False).

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
        self._reload = threading.Event()
        self._match_state: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("自動瞄準已在執行")
        self._stop.clear()
        self._reload.clear()
        self._match_state = None
        self.error = None
        self._thread = threading.Thread(target=self._run, name="auto-aim", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        try:
            run_board_real.main(
                self.argv,
                telemetry_callback=self._on_telemetry,
                stop_event=self._stop,
                reload_event=self._reload,
            )
        except BaseException as exc:  # includes SystemExit from argparse/startup checks
            self.error = exc
            print(f"自動瞄準結束（錯誤）：{exc!r}", flush=True)

    def _on_telemetry(self, telemetry: dict) -> None:
        self._match_state = telemetry.get("match")
        if self._telemetry_callback is not None:
            self._telemetry_callback(telemetry)

    @property
    def match_state(self) -> Optional[dict]:
        """Latest match telemetry (phase, balls_remaining, reload_wait, ...)."""
        return self._match_state

    @property
    def reload_waiting(self) -> bool:
        """True while the match holds for the team to refill the magazine."""
        state = self._match_state
        return bool(self.running and state and state.get("reload_wait"))

    def confirm_reload(self) -> bool:
        """Magazine refilled: resume (flywheels on, 2-slot turn, then fire).

        Returns False and does nothing when no reload is awaited -- an early
        or accidental press is never remembered for later.
        """
        if not self.reload_waiting:
            return False
        self._reload.set()
        return True

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

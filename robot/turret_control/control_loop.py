"""Fixed-rate control loop.

Two rates, deliberately decoupled:

* The **motor loop** runs at a fixed 200 Hz and sends a frame *every* tick,
  target or not.  The motors auto-disable when frames stop arriving, and the
  vertical axis has no brake, so a stalled loop drops the launcher.
* **Detection** runs in its own thread at whatever rate YOLO manages (tens of
  Hz on this AGX).  The motor loop reads the newest result and never waits for
  it.

The loop sleeps until a fixed next deadline rather than sleeping a fixed
interval.  ``time.sleep(dt)`` adds the work time on top of the interval: on
this AGX the USB CDC round trip plus printing turned a nominal 200 Hz loop into
a measured 170 Hz.  Deadline scheduling measured 200.0 Hz with 0 late ticks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .aiming import AimCommand, TargetPixel
from .controller import TurretController

DEFAULT_RATE_HZ = 200.0
DEFAULT_MAX_AGE_S = 0.20


@dataclass
class LoopStats:
    ticks: int = 0
    late_ticks: int = 0
    elapsed_s: float = 0.0
    stale_ticks: int = 0
    fire_ticks: int = 0

    @property
    def actual_hz(self) -> float:
        return self.ticks / self.elapsed_s if self.elapsed_s > 0 else 0.0


class LatestDetection:
    """Thread-safe slot holding the newest detection.

    Reading a stale detection is worse than reading none: the turret would keep
    aiming confidently at where the target was a second ago.  Anything older
    than ``max_age_s`` reads back as "no target".
    """

    def __init__(self, max_age_s: float = DEFAULT_MAX_AGE_S):
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._target: Optional[TargetPixel] = None
        self._stamp: float = 0.0
        self.publish_count = 0

    def publish(self, target: Optional[TargetPixel]) -> None:
        with self._lock:
            self._target = target
            self._stamp = time.monotonic()
            self.publish_count += 1

    def get(self, now: Optional[float] = None) -> Optional[TargetPixel]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._target is None:
                return None
            if now - self._stamp > self.max_age_s:
                return None
            return self._target

    def age_s(self, now: Optional[float] = None) -> Optional[float]:
        now = time.monotonic() if now is None else now
        with self._lock:
            return None if self._stamp == 0.0 else now - self._stamp


class DetectionThread:
    """Runs ``detector()`` in a loop, publishing whatever it returns."""

    def __init__(self, detector: Callable[[], Optional[TargetPixel]], slot: LatestDetection):
        self.detector = detector
        self.slot = slot
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.slot.publish(self.detector())
            except BaseException as exc:  # noqa: BLE001 - surfaced via .error
                # A crashed detector must not leave a stale target behind.
                self.slot.publish(None)
                self.error = exc
                return

    def start(self) -> "DetectionThread":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "DetectionThread":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class ControlLoop:
    """Drive a :class:`TurretController` at a fixed rate."""

    def __init__(
        self,
        turret: TurretController,
        slot: LatestDetection,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.turret = turret
        self.slot = slot
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.stats = LoopStats()

    def run(
        self,
        duration_s: Optional[float] = None,
        on_tick: Optional[Callable[[AimCommand, LoopStats], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> LoopStats:
        """Tick until ``duration_s`` elapses, ``should_stop()`` is true, or Ctrl+C.

        ``duration_s=None`` runs forever.  The caller still owns closing the
        turret; this only stops ticking.
        """
        stats = LoopStats()
        start = time.monotonic()
        next_deadline = start

        try:
            while True:
                now = time.monotonic()
                if duration_s is not None and now - start >= duration_s:
                    break
                if should_stop is not None and should_stop():
                    break

                target = self.slot.get(now)
                if target is None:
                    stats.stale_ticks += 1

                command = self.turret.update(target)
                stats.ticks += 1
                if command.fire_allowed:
                    stats.fire_ticks += 1

                if on_tick is not None:
                    on_tick(command, stats)

                next_deadline += self.dt
                slack = next_deadline - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    stats.late_ticks += 1
                    next_deadline = time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            stats.elapsed_s = time.monotonic() - start
            self.stats = stats

        return stats

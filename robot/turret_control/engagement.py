"""Drives an EngagementPlan through the turret, one hole at a time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .aiming import AimCommand
from .board import BoardTarget, EngagementPlan
from .config import TurretConfig
from .controller import TurretController

# The launcher needs time to reload and the turret to settle.  Without this the
# 200 Hz loop would "fire" every tick for as long as the gate stayed open.
DEFAULT_SHOT_COOLDOWN_S = 1.0
# Give up on a hole that will not come together, so one unreachable target
# cannot stall the whole board.
DEFAULT_TARGET_TIMEOUT_S = 6.0


@dataclass
class EngagementStatus:
    command: AimCommand
    target: Optional[BoardTarget]
    index: int
    remaining: int
    fired: bool = False
    skipped: bool = False
    finished: bool = False


class BoardEngagement:
    """Aim at the current hole, fire once, advance.

    ``fire`` is called when the aiming gate opens.  It should trigger the real
    launcher and return True if the shot actually went out; returning False
    keeps the plan on the same hole.  It is never called more often than
    ``shot_cooldown_s``.
    """

    def __init__(
        self,
        config: TurretConfig,
        turret: TurretController,
        plan: Optional[EngagementPlan] = None,
        fire: Optional[Callable[[BoardTarget], bool]] = None,
        shot_cooldown_s: float = DEFAULT_SHOT_COOLDOWN_S,
        target_timeout_s: Optional[float] = DEFAULT_TARGET_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        event_log_path: Optional[Union[str, Path]] = None,
    ):
        if shot_cooldown_s < 0:
            raise ValueError("shot_cooldown_s must not be negative")
        self.config = config
        self.turret = turret
        self.plan = plan
        self.fire = fire
        self.shot_cooldown_s = shot_cooldown_s
        self.target_timeout_s = target_timeout_s
        self._clock = clock
        self._last_shot_at: Optional[float] = None
        self._target_started_at: Optional[float] = None
        self.shots = 0
        self.events: List[Dict[str, Any]] = []
        self.event_log_path = Path(event_log_path) if event_log_path is not None else None
        self._event_log = None
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_log = self.event_log_path.open("a", encoding="utf-8")

    def set_plan(self, plan: EngagementPlan) -> None:
        self.plan = plan
        self._target_started_at = self._clock()

    def update(self) -> EngagementStatus:
        """One tick.  Always drives the motors, target or not."""
        now = self._clock()

        if self.plan is None or self.plan.finished:
            # Nothing left to shoot, but the motors still must be held: they
            # auto-disable without frames and the vertical axis has no brake.
            command = self.turret.update(None)
            return EngagementStatus(
                command=command,
                target=None,
                index=self.plan.index if self.plan else 0,
                remaining=0,
                finished=True,
            )

        if self._target_started_at is None:
            self._target_started_at = now

        target = self.plan.current()
        command = self.turret.update(target.to_pixel())

        status = EngagementStatus(
            command=command,
            target=target,
            index=self.plan.index,
            remaining=self.plan.remaining,
        )

        if command.fire_allowed and self._cooldown_expired(now):
            fired = True if self.fire is None else bool(self.fire(target))
            if fired:
                plan_index = self.plan.index
                self.shots += 1
                self._last_shot_at = now
                self.plan.mark_engaged()
                self._target_started_at = now
                self._record_event(
                    "fired",
                    target,
                    plan_index,
                    shot_number=self.shots,
                )
                status.fired = True
                status.finished = self.plan.finished
                return status

        if self._timed_out(now):
            plan_index = self.plan.index
            skipped_target = target
            self.plan.skip()
            self._target_started_at = now
            self._record_event(
                "skipped",
                skipped_target,
                plan_index,
                reason=command.blocked_reason or "target timeout",
            )
            status.skipped = True
            status.finished = self.plan.finished

        return status

    def _cooldown_expired(self, now: float) -> bool:
        if self._last_shot_at is None:
            return True
        return now - self._last_shot_at >= self.shot_cooldown_s

    def _timed_out(self, now: float) -> bool:
        if self.target_timeout_s is None or self._target_started_at is None:
            return False
        return now - self._target_started_at >= self.target_timeout_s

    def _record_event(
        self,
        event: str,
        target: BoardTarget,
        plan_index: int,
        shot_number: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Record a board event in memory and, when configured, as JSONL."""
        record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "engagement_order": plan_index + 1,
            "hole_id": target.hole_id,
            "column": target.column,
            "row_from_bottom": target.row,
            "u_px": target.u_px,
            "v_px": target.v_px,
            "diameter_m": target.diameter_m,
            "distance_m": target.distance_m,
            "confidence": target.confidence,
        }
        if shot_number is not None:
            record["shot_number"] = shot_number
        if reason is not None:
            record["reason"] = reason

        self.events.append(record)
        if self._event_log is not None:
            json.dump(record, self._event_log, ensure_ascii=False)
            self._event_log.write("\n")
            self._event_log.flush()

    def close(self) -> None:
        """Flush and close the optional persistent event log."""
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = None

"""Match flow for --match-fire, one control tick at a time.

The remote switches the robot to auto mode at the whistle, so the match
starts as soon as the board is seen (no waiting for the light):

* lower half: the 18-ball serpentine plan, no bonus hole;
* upper half: shoot the lit bonus hole ``UPPER_BONUS_BALLS`` times, only
  within ``UPPER_BONUS_WINDOW_S`` of the start (the light is off afterwards).
  If the light is not confirmed within ``light_wait_s`` the shots start on
  ``UPPER_FALLBACK_BONUS_ID`` (H11) and switch to the real hole as soon as it
  is confirmed.  Then the serpentine grid pairs.  When the magazine is empty
  while the plan continues, hold the turret until the team confirms the
  reload, turn the magazine two slots, then continue.

The sequencer owns the plan it hands to the engagement driver and the ball
count; the caller feeds it the vision/launcher state and fired shots.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .board import EngagementPlan
from .scoring_strategy import (
    UPPER_BONUS_BALLS,
    UPPER_BONUS_WINDOW_S,
    UPPER_FALLBACK_BONUS_ID,
    UPPER_PHASE,
    append_reallocated_grid_shots,
    build_max_score_strategy,
)

SMALL_HOLE_IDS = ("H10", "H11", "H12")


class MatchSequencer:
    def __init__(self, phase: str, duration_s: float, balls_loaded: int,
                 set_plan: Callable[[EngagementPlan], None],
                 log: Callable[[str], None] = lambda message: print(message, flush=True),
                 *, light_wait_s: float = 5.0,
                 start_reload_rotation: Callable[[], bool] = lambda: True):
        self.phase = phase
        self.duration_s = duration_s
        self.magazine_size = balls_loaded
        self.balls_remaining = balls_loaded
        self._set_plan = set_plan
        self._log = log
        self.plan: Optional[EngagementPlan] = None
        self.started_at: Optional[float] = None
        self.waiting_light = False
        self.bonus_hole_id: Optional[str] = None
        self.bonus_is_fallback = False
        self.light_wait_s = light_wait_s
        self._start_reload_rotation = start_reload_rotation
        self.bonus_closed = phase != UPPER_PHASE
        self.reload_wait = False
        self.reload_balls = 0
        self.stopped: Optional[str] = None
        self._board_plan = None

    # ---- inputs -----------------------------------------------------------
    def record_fired(self) -> None:
        self.balls_remaining -= 1

    @property
    def hold_reason(self) -> Optional[str]:
        if self.reload_wait:
            return "等待裝填：完成後在儀表板按「裝填完成」"
        return None

    def elapsed(self, now: float) -> Optional[float]:
        return None if self.started_at is None else now - self.started_at

    # ---- one tick ---------------------------------------------------------
    def step(self, now: float, board_plan, lit_hole_id: Optional[str], launcher_idle: bool,
             mechanism_busy: bool, reload_confirmed: bool) -> None:
        if self.stopped is not None:
            return
        if board_plan is not None:
            self._board_plan = board_plan
        if self.started_at is None:
            if self._board_plan is None or not launcher_idle:
                return
            self.started_at = now
            if self.phase == UPPER_PHASE:
                self.waiting_light = True
                self._hand_over(EngagementPlan([]))
                self._log(f"開賽（遙控器切自動即開始）：上半場，時限 {self.duration_s:g} 秒。"
                          f"先找亮燈加分孔射 {UPPER_BONUS_BALLS} 顆（限開賽後 "
                          f"{UPPER_BONUS_WINDOW_S:g} 秒內），再射九宮格。")
            else:
                self._build(None, f"開賽（遙控器切自動即開始）：下半場，時限 "
                                  f"{self.duration_s:g} 秒，不射加分孔。")

        elapsed = now - self.started_at
        late = elapsed >= UPPER_BONUS_WINDOW_S
        if self.waiting_light:
            if late:
                self.waiting_light = False
                self._build(None, f"開賽已滿 {UPPER_BONUS_WINDOW_S:g} 秒，不射小孔，直接射九宮格。")
            elif lit_hole_id is not None:
                self.waiting_light = False
                self.bonus_hole_id = lit_hole_id
                self._build(lit_hole_id, f"亮燈加分孔 {lit_hole_id}：先射 {UPPER_BONUS_BALLS} 顆。")
            elif elapsed >= self.light_wait_s:
                self.waiting_light = False
                self.bonus_hole_id = UPPER_FALLBACK_BONUS_ID
                self.bonus_is_fallback = True
                self._build(UPPER_FALLBACK_BONUS_ID,
                            f"{self.light_wait_s:g} 秒內未確認亮燈：先射 {UPPER_FALLBACK_BONUS_ID}；"
                            "確認到亮燈後剩餘加分球改射亮燈孔。")
        elif (self.bonus_is_fallback and lit_hole_id is not None and not late
              and lit_hole_id != self.bonus_hole_id and launcher_idle):
            self._retarget_bonus(lit_hole_id)

        if not self.bonus_closed and not self.waiting_light and late and launcher_idle:
            self.bonus_closed = True
            self._close_bonus_window()

        if now - self.started_at >= self.duration_s:
            self._stop("比賽時間到，停止發射", launcher_idle)
            return

        if self.balls_remaining <= 0 and not self.reload_wait:
            plan_goes_on = self.plan is not None and not self.plan.finished and not self.waiting_light
            if not plan_goes_on:
                self._stop("已裝彈藥全部射出，停止發射", launcher_idle)
                return
            if launcher_idle:
                self.reload_wait = True
                self.reload_balls = min(self.magazine_size, self.plan.remaining)
                self._log(f"彈倉已射完：請裝填 {self.reload_balls} 顆，完成後在儀表板按「裝填完成」。"
                          "接下來：" + " ".join(self.remaining_ids()))

        if self.reload_wait and reload_confirmed:
            if mechanism_busy or not launcher_idle:
                self._log("彈倉仍在轉格，請等轉完再按「裝填完成」。")
            elif not self._start_reload_rotation():
                self._log("無法開始裝填轉格（機構忙碌或故障），請檢查後再按「裝填完成」。")
            else:
                self.reload_wait = False
                self.balls_remaining = self.reload_balls
                self._log(f"裝填完成：已裝 {self.balls_remaining} 顆；彈倉轉 2 格後繼續射擊。")

    def remaining_ids(self) -> List[str]:
        if self.plan is None:
            return []
        return [t.hole_id or "?" for t in self.plan.targets[self.plan.index:]]

    # ---- helpers ----------------------------------------------------------
    def _hand_over(self, plan: EngagementPlan) -> None:
        self.plan = plan
        self._set_plan(plan)

    def _build(self, bonus_hole_id, message: str) -> None:
        try:
            strategy = build_max_score_strategy(self._board_plan, bonus_hole_id, self.phase)
        except ValueError as exc:
            self._log(f"無法建立比賽射擊計畫：{exc}")
            self._hand_over(EngagementPlan([]))
            return
        self._log(message)
        self._log("射擊順序：" + " ".join(strategy.hole_ids))
        self._hand_over(strategy.plan)

    def _retarget_bonus(self, lit_hole_id: str) -> None:
        """The light was confirmed after the H11 fallback started."""
        plan = self.plan
        target = {t.hole_id: t for t in self._board_plan.targets}.get(lit_hole_id)
        if plan is None or target is None:
            return
        changed = 0
        for index in range(plan.index, len(plan.targets)):
            if plan.targets[index].hole_id in SMALL_HOLE_IDS:
                plan.targets[index] = target
                changed += 1
        self.bonus_hole_id = lit_hole_id
        self.bonus_is_fallback = False
        if changed:
            self._set_plan(plan)
        self._log(f"確認亮燈 {lit_hole_id}：剩餘 {changed} 顆加分球改射 {lit_hole_id}。")

    def _close_bonus_window(self) -> None:
        """Light is off: no ball may go to a small hole any more."""
        plan = self.plan
        skipped = 0
        while plan is not None and plan.current() is not None and plan.current().hole_id in SMALL_HOLE_IDS:
            plan.skip()
            skipped += 1
        if not skipped:
            return
        try:
            replacements = append_reallocated_grid_shots(plan, self._board_plan, self.phase, skipped)
        except ValueError as exc:
            replacements = ()
            self._log(f"加分球重新配置失敗：{exc}")
        self._set_plan(plan)
        self._log(f"開賽已滿 {UPPER_BONUS_WINDOW_S:g} 秒：略過 {skipped} 顆小孔，改投九宮格："
                  + (" ".join(replacements) or "無可用格"))

    def _stop(self, reason: str, launcher_idle: bool) -> None:
        if not launcher_idle:
            return  # let the running stage finish first
        self.stopped = reason
        self.reload_wait = False
        self.waiting_light = False
        self._log(f"{reason}（剩餘 {self.balls_remaining} 顆）。")
        self._hand_over(EngagementPlan([]))

    def telemetry(self, now: float, round_phase=None) -> dict:
        elapsed = self.elapsed(now)
        return {
            "phase": self.phase,
            "round_phase": round_phase,
            "bonus_hole_id": self.bonus_hole_id,
            "elapsed_s": None if elapsed is None else round(elapsed, 1),
            "remaining_s": None if elapsed is None else round(max(0.0, self.duration_s - elapsed), 1),
            "balls_remaining": self.balls_remaining,
            "stopped": self.stopped,
            "waiting_light": self.waiting_light,
            "reload_wait": self.reload_wait,
            "reload_balls": self.reload_balls if self.reload_wait else None,
        }

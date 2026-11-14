"""One panel click authorizes one shared MKS firing stage, never a queued shot."""

from __future__ import annotations

import time


PANEL_FLYWHEEL_SPEED = 100
FIRE_REQUEST_TTL_S = 1.0
# A stage may start only when both flywheels are within this fraction of the
# target and the reading is fresh (the VESC thread polls at 10 Hz).
FLYWHEEL_READY_FRACTION = 0.03
FLYWHEEL_READING_MAX_AGE_S = 0.5


class SharedAimMonitor:
    """Observe the existing aiming worker without opening or owning its port."""

    def __init__(self, motors):
        self.motors = motors

    def start(self, timeout_s=1.0):
        # The aiming worker can publish "enabled" just before its first full
        # feedback snapshot. Wait briefly for both axes, without sending a move.
        deadline = time.monotonic() + timeout_s
        while True:
            self.motors.refresh()
            if all(self.motors.get_state(i) is not None for i in (3, 1)):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("自動瞄準馬達回授無效，停止發射控制")
            time.sleep(0.01)

    def check_health(self):
        self.motors.refresh()
        if any(self.motors.get_state(i) is None for i in (3, 1)):
            raise RuntimeError("自動瞄準馬達回授無效，停止發射控制")

    def close(self):
        # SharedMotorBackend remains owned by the aiming pipeline.
        pass


class PanelLauncher:
    def __init__(self, motors, *, controller_factory=None, clock=time.monotonic,
                 push_rpm=None, push_acceleration=None, push_home_rpm=None):
        from manual import ManualController, VESC_MIN_RPM, VESC_RPM_PER_SPEED_UNIT

        self._clock = clock
        self._authorization = None
        self._pending_target = None
        self._pending_plan = None
        self._pending_index = None
        self._pending_revision = None
        self._pending_seq = None
        # Test/diagnostic record of the most recently *started* stage.
        self.shots_started = 0
        self.last_shot = None
        self._last_started_at = None
        self._last_released_at = None
        self.message = "等待瞄準到位後，點擊「射擊」執行一個階段"
        self.target_rpm = VESC_MIN_RPM + PANEL_FLYWHEEL_SPEED * VESC_RPM_PER_SPEED_UNIT
        factory = controller_factory or ManualController
        self.controller = factory(uart=None, log_status=False, dm_controller=SharedAimMonitor(motors))
        # Pusher (MKS ID1) overrides for the auto pipeline only; None keeps the
        # manual.py defaults so manual mode is unaffected.
        if push_rpm is not None:
            self.controller.shooting_speed_rpm = int(push_rpm)
        if push_acceleration is not None:
            self.controller.shooting_acceleration = int(push_acceleration)
        if push_home_rpm is not None:
            self.controller.home_speed_rpm = int(push_home_rpm)
        try:
            self.controller.vesc_controller.set_speed(PANEL_FLYWHEEL_SPEED)
            self.check_health()
        except BaseException:
            self.controller.close()
            raise

    @property
    def pending(self):
        return self._pending_target is not None

    @property
    def busy(self):
        """A stage is in progress and the muzzle must hold its pose.

        Once ID1 has finished (released), the muzzle is free even though the
        controller stays busy while ID2 rotates the magazine.
        """
        return self.pending and self.controller.busy and not self._pending_shot_released()

    @property
    def mechanism_busy(self):
        """Launcher hardware still moving (e.g. magazine rotating after shot 3).

        The muzzle may already aim at the next hole, but no new shot may start.
        """
        return bool(self.controller.busy)

    def shot_released(self):
        """The pending stage has released its ball (ID1 done); complete it now."""
        return self.pending and self._pending_shot_released()

    @property
    def last_released_at(self):
        """Clock time the previous stage released its ball (None before any)."""
        return self._last_released_at

    def flywheel_rpm(self):
        """(left, right) RPM when the reading is fresh, else None."""
        rpm = getattr(getattr(self.controller, "vesc_controller", None), "last_rpm", None)
        if not (isinstance(rpm, tuple) and len(rpm) == 3):
            return None
        left, right, read_at = rpm
        if not 0 <= time.monotonic() - read_at <= FLYWHEEL_READING_MAX_AGE_S:
            return None
        return left, right

    @property
    def flywheel_ready(self):
        rpm = self.flywheel_rpm()
        if rpm is None:
            return False
        from manual import flywheel_wheel_targets

        # 上下輪各自有目標（下旋轉速差），分別檢查是否到位。
        return all(abs(value - goal) <= FLYWHEEL_READY_FRACTION * goal
                   for value, goal in zip(rpm, flywheel_wheel_targets(self.target_rpm)))

    def _pending_shot_released(self):
        released = getattr(self.controller, "shot_released_seq", None)
        return (isinstance(released, int) and isinstance(self._pending_seq, int)
                and released >= self._pending_seq)

    def check_health(self):
        self.controller.check_health()
        if self.controller.shot_error is not None:
            raise RuntimeError(self.controller.shot_error)

    def arm(self, action, plan, plan_revision):
        """Authorization lives for this one control tick and this exact target."""
        self._authorization = None
        if self.pending and (plan is not self._pending_plan or plan_revision != self._pending_revision
                             or plan.index != self._pending_index or plan.current() is None
                             or plan.current().hole_id != self._pending_target.hole_id):
            raise RuntimeError("發射期間孔位計畫已改變，停止操作")
        if action is None:
            return
        if not self.pending and self.mechanism_busy:
            self.message = "彈藥倉轉格中，完成後再點擊「射擊」"
            return
        if not self.pending and not self.flywheel_ready:
            rpm = self.flywheel_rpm()
            self.message = ("飛輪尚未達到轉速（目標 {:.0f} RPM，目前 {}），請稍候".format(
                self.target_rpm, "無讀值" if rpm is None else "{:.0f}/{:.0f}".format(*rpm)))
            return
        revision, index, hole_id, issued_at = action
        target = plan.current() if plan is not None else None
        if (self.pending or revision != plan_revision or target is None
                or plan.index != index or target.hole_id != hole_id
                or not 0 <= self._clock() - issued_at <= FIRE_REQUEST_TTL_S):
            self.message = "射擊請求已失效，請確認目前孔位後重新點擊"
            return
        self._authorization = (plan, index, target, revision)

    def finish_tick(self):
        if self._authorization is not None:
            self.message = "射擊未執行：瞄準或冷卻條件尚未成立，請重新點擊"
        self._authorization = None

    def fire_ready(self):
        return self.pending or self._authorization is not None

    def fire(self, target):
        """Return success only after the background stage has actually completed."""
        self.check_health()
        if self.pending:
            # PnP refreshes replace BoardTarget instances every frame. Identity
            # is the plan revision + index + hole, not a target object's address.
            if (target.hole_id != self._pending_target.hole_id or self._pending_plan.current() is not target
                    or self._pending_plan.index != self._pending_index):
                raise RuntimeError("發射期間孔位計畫已改變，停止操作")
            released = self._pending_shot_released()
            if self.controller.busy and not released:
                return False
            self._pending_target = self._pending_plan = self._pending_index = self._pending_revision = None
            self._pending_seq = None
            self._last_released_at = self._clock()
            if self.controller.busy:
                # Shot 3 is out and ID1 is home; ID2 keeps rotating while the
                # turret already moves to the next hole.
                self.message = "已發射，彈藥倉轉格中；砲口先轉向下一孔，轉格完成後才可射擊"
            else:
                self.message = "本次發射階段完成，切換下一孔；再次發射仍需點擊「射擊」"
            return True
        authorization = self._authorization
        self._authorization = None
        if authorization is None:
            return False
        plan, index, authorized_target, revision = authorization
        if target is not authorized_target or plan.current() is not target or plan.index != index:
            self.message = "孔位已改變，射擊未執行"
            return False
        request_seq = getattr(self.controller, "shot_request_seq", None)
        slot = getattr(self.controller, "current_slot", None)
        stage = getattr(self.controller, "shot_stage", None)
        rpm = getattr(getattr(self.controller, "vesc_controller", None), "last_rpm", None)
        if not self.controller.request_shot():
            self.message = "發射機構未接受指令，請檢查目前狀態"
            return False
        now = self._clock()
        self.shots_started += 1
        released = self._last_released_at
        previous = self._last_started_at
        self.last_shot = {
            "prev_mechanism_s": (round(released - previous, 2)
                                 if released is not None and previous is not None
                                 and released >= previous else None),
            "wait_after_release_s": round(now - released, 2) if released is not None else None,
            "started_at": now,
            "hole_id": target.hole_id,
            "plan_index": index,
            "slot": slot if isinstance(slot, int) else None,
            "stage": stage + 1 if isinstance(stage, int) else None,
            "rpm": rpm if isinstance(rpm, tuple) and len(rpm) == 3 else None,
        }
        self._last_started_at = now
        self._pending_target, self._pending_plan, self._pending_index = target, plan, index
        self._pending_revision = revision
        self._pending_seq = request_seq + 1 if isinstance(request_seq, int) else None
        self.message = "正在執行發射階段，砲口保持位置；完成前不接受下一次射擊"
        return False

    def snapshot(self):
        return {
            "speed": PANEL_FLYWHEEL_SPEED,
            "target_rpm": self.target_rpm,
            "busy": bool(self.busy),
            "magazine_rotating": bool(self.mechanism_busy and not self.pending),
            "pending": self.pending,
            "slot": self.controller.current_slot,
            "next_stage": self.controller.shot_stage + 1,
            "flywheel_rpm": self.flywheel_rpm(),
            "flywheel_ready": bool(self.flywheel_ready),
            "message": self.message,
        }

    def close(self):
        self._authorization = None
        self.controller.close()


class AutoFireTrigger:
    """Replace the operator's click with an automatic one (``--auto-fire``).

    It only *creates the same click a person would make*; the click then goes
    through ``PanelLauncher.arm`` and the engagement gate exactly like a panel
    click, so every existing check (pose ready, aim reached and stable,
    cooldown, same plan/hole, magazine not rotating) still applies on the very
    tick the shot is taken.  On top of that the shot must have been ready
    continuously for ``settle_s`` on the same hole.
    """

    def __init__(self, settle_s=0.5, *, clock=time.monotonic):
        if not settle_s >= 0:
            raise ValueError("settle_s must be >= 0")
        self.settle_s = float(settle_s)
        self._clock = clock
        self._key = None
        self._since = None
        self.state = "等待瞄準到位"

    def _reset(self, state):
        self._key = None
        self._since = None
        self.state = state

    def take(self, plan, plan_revision, ready):
        """Return a click tuple once ready has held for ``settle_s``, else None."""
        target = plan.current() if plan is not None else None
        if target is None:
            self._reset("沒有待射孔位")
            return None
        if not ready:
            self._reset("等待瞄準到位")
            return None
        now = self._clock()
        key = (plan_revision, plan.index, target.hole_id)
        if key != self._key:
            self._key, self._since = key, now
        held = now - self._since
        if held < self.settle_s:
            self.state = f"{target.hole_id} 瞄準穩定中 {held:.1f}/{self.settle_s:.1f} s"
            return None
        self.state = f"{target.hole_id} 自動發射"
        # Fire once per settled hole; a new click needs a fresh settle period.
        self._key = self._since = None
        return (plan_revision, plan.index, target.hole_id, now)

    def snapshot(self):
        return {"enabled": True, "settle_s": self.settle_s, "state": self.state}

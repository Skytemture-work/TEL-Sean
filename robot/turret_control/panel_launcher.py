"""One panel click authorizes one shared MKS firing stage, never a queued shot."""

from __future__ import annotations

import time


PANEL_FLYWHEEL_SPEED = 100
FIRE_REQUEST_TTL_S = 1.0


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
    def __init__(self, motors, *, controller_factory=None, clock=time.monotonic):
        from manual import ManualController, VESC_MIN_RPM, VESC_RPM_PER_SPEED_UNIT

        self._clock = clock
        self._authorization = None
        self._pending_target = None
        self._pending_plan = None
        self._pending_index = None
        self._pending_revision = None
        self.message = "等待瞄準到位後，點擊「射擊」執行一個階段"
        self.target_rpm = VESC_MIN_RPM + PANEL_FLYWHEEL_SPEED * VESC_RPM_PER_SPEED_UNIT
        factory = controller_factory or ManualController
        self.controller = factory(uart=None, log_status=False, dm_controller=SharedAimMonitor(motors))
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
        return self.pending and self.controller.busy

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
            if self.controller.busy:
                return False
            self._pending_target = self._pending_plan = self._pending_index = self._pending_revision = None
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
        if not self.controller.request_shot():
            self.message = "發射機構未接受指令，請檢查目前狀態"
            return False
        self._pending_target, self._pending_plan, self._pending_index = target, plan, index
        self._pending_revision = revision
        self.message = "正在執行發射階段，砲口保持位置；完成前不接受下一次射擊"
        return False

    def snapshot(self):
        return {
            "speed": PANEL_FLYWHEEL_SPEED,
            "target_rpm": self.target_rpm,
            "busy": bool(self.busy),
            "pending": self.pending,
            "slot": self.controller.current_slot,
            "next_stage": self.controller.shot_stage + 1,
            "message": self.message,
        }

    def close(self):
        self._authorization = None
        self.controller.close()

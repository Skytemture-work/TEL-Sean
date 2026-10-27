"""Low-speed, one-cycle physical verification of the calibrated workspace."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable, Tuple

from .backends import DamiaoSerialBackend
from .limits import (
    DEFAULT_LIMITS_PATH,
    WorkspaceLimit,
    load_motor_limits,
    load_workspace_limit,
    workspace_base_contains,
)
from .pitch import load_pitch_calibration, load_yaw_calibration
DEFAULT_STARTUP_POSE = Path(__file__).with_name("startup_pose.json")
YAW_MOTOR_ID = 0x03
PITCH_MOTOR_ID = 0x01


def _boundary_point(
    workspace: WorkspaceLimit,
    vertex: Tuple[float, float],
    backoff_rad: float,
) -> Tuple[float, float]:
    """Approach a polygon vertex from the safe anchor, then back off slightly."""

    start = workspace.anchor
    safe_fraction = 0.0
    unsafe_fraction = 1.0
    for step in range(1, 401):
        fraction = step / 400.0
        candidate = (
            start[0] + (vertex[0] - start[0]) * fraction,
            start[1] + (vertex[1] - start[1]) * fraction,
        )
        if workspace_base_contains(workspace, *candidate):
            safe_fraction = fraction
        else:
            unsafe_fraction = fraction
            break
    for _ in range(40):
        fraction = (safe_fraction + unsafe_fraction) / 2.0
        candidate = (
            start[0] + (vertex[0] - start[0]) * fraction,
            start[1] + (vertex[1] - start[1]) * fraction,
        )
        if workspace_base_contains(workspace, *candidate):
            safe_fraction = fraction
        else:
            unsafe_fraction = fraction
    distance = math.hypot(vertex[0] - start[0], vertex[1] - start[1])
    if distance > 0.0:
        safe_fraction = max(0.0, safe_fraction - backoff_rad / distance)
    return (
        start[0] + (vertex[0] - start[0]) * safe_fraction,
        start[1] + (vertex[1] - start[1]) * safe_fraction,
    )


def _build_boundary_path(
    workspace: WorkspaceLimit, backoff_motor_deg: float
) -> Tuple[Tuple[float, float], ...]:
    if workspace.home_corridor is None:
        raise ValueError("workspace does not contain the calibrated 45-degree home corridor")
    backoff_rad = math.radians(backoff_motor_deg)
    points = tuple(
        _boundary_point(workspace, vertex, backoff_rad)
        for vertex in workspace.vertices
    )
    if len(points) < 3:
        raise ValueError("workspace boundary needs at least three waypoints")
    # Check each complete edge densely before any motor can be enabled.
    for start, end in zip(points, points[1:] + points[:1]):
        for step in range(101):
            fraction = step / 100.0
            point = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            if not workspace_base_contains(workspace, *point):
                raise ValueError("generated boundary edge leaves the safe inset workspace")
    return points


def _feedback_pair(motors: DamiaoSerialBackend) -> Tuple[float, float]:
    yaw = None
    pitch = None
    for _ in range(8):
        motors.refresh()
        yaw = motors.get_state(YAW_MOTOR_ID)
        pitch = motors.get_state(PITCH_MOTOR_ID)
        # DAMIAO uses status nibble 0 while disabled and 1 while enabled;
        # actual protection faults are 0x8..0xE.
        if (
            yaw is not None
            and pitch is not None
            and yaw.err is not None
            and pitch.err is not None
            and yaw.err < 0x8
            and pitch.err < 0x8
        ):
            return float(yaw.position_rad), float(pitch.position_rad)
        time.sleep(0.02)
    raise RuntimeError(
        f"motor feedback error: yaw={None if yaw is None else yaw.err}, "
        f"pitch={None if pitch is None else pitch.err}"
    )


def _smootherstep_sample(
    start: float, target: float, elapsed: float, duration: float
) -> Tuple[float, float]:
    """Quintic interpolation with zero velocity and acceleration at both ends."""

    if elapsed <= 0.0:
        return start, 0.0
    if elapsed >= duration:
        return target, 0.0
    u = elapsed / duration
    position_fraction = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
    velocity_fraction = (30.0 * u ** 2 - 60.0 * u ** 3 + 30.0 * u ** 4) / duration
    delta = target - start
    return start + delta * position_fraction, delta * velocity_fraction


def _move_to(
    motors: DamiaoSerialBackend,
    target: Tuple[float, float],
    speed: float,
    label: str,
) -> None:
    start = _feedback_pair(motors)
    distance = max(abs(target[0] - start[0]), abs(target[1] - start[1]))
    duration = max(1.875 * distance / speed, 0.6)
    print(f"移動：{label}（預計 {duration:.1f} 秒）", flush=True)
    trajectory_elapsed = 0.0
    last_tick = time.monotonic()
    next_tick = last_tick
    lag_started = None
    tracking_gate = math.radians(1.0)
    while trajectory_elapsed < duration:
        yaw, yaw_velocity = _smootherstep_sample(
            start[0], target[0], trajectory_elapsed, duration
        )
        pitch, pitch_velocity = _smootherstep_sample(
            start[1], target[1], trajectory_elapsed, duration
        )
        motors.send_targets(
            {
                YAW_MOTOR_ID: (yaw, speed, yaw_velocity, 0.0),
                PITCH_MOTOR_ID: (pitch, speed, pitch_velocity, 0.0),
            }
        )
        actual = _feedback_pair(motors)
        now = time.monotonic()
        lag = max(abs(actual[0] - yaw), abs(actual[1] - pitch))
        if lag <= tracking_gate:
            trajectory_elapsed = min(
                duration, trajectory_elapsed + min(now - last_tick, 0.02)
            )
            lag_started = None
        else:
            if lag_started is None:
                lag_started = now
            elif now - lag_started >= 5.0:
                raise RuntimeError(
                    f"{label} 追蹤落後超過5秒（{math.degrees(lag):.2f}°），停止"
                )
        last_tick = now
        next_tick += 0.01
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    for _ in range(20):
        motors.send_targets(
            {
                YAW_MOTOR_ID: (target[0], speed, 0.0, 0.0),
                PITCH_MOTOR_ID: (target[1], speed, 0.0, 0.0),
            }
        )
        _feedback_pair(motors)
        time.sleep(0.01)
    tolerance = math.radians(0.5)
    settle_deadline = time.monotonic() + 8.0
    while True:
        motors.send_targets(
            {
                YAW_MOTOR_ID: (target[0], speed, 0.0, 0.0),
                PITCH_MOTOR_ID: (target[1], speed, 0.0, 0.0),
            }
        )
        actual = _feedback_pair(motors)
        yaw_error = abs(actual[0] - target[0])
        pitch_error = abs(actual[1] - target[1])
        if yaw_error <= tolerance and pitch_error <= tolerance:
            print(
                f"到位：{label}（誤差 yaw {math.degrees(yaw_error):.2f}° / "
                f"pitch {math.degrees(pitch_error):.2f}°）",
                flush=True,
            )
            return
        if time.monotonic() >= settle_deadline:
            raise RuntimeError(
                f"{label} 未在 8 秒內到位：yaw 誤差 {math.degrees(yaw_error):.2f}°，"
                f"pitch 誤差 {math.degrees(pitch_error):.2f}°"
            )
        time.sleep(0.01)


def _physical_label(
    point: Tuple[float, float], yaw_calibration, pitch_calibration
) -> str:
    yaw = math.degrees(yaw_calibration.muzzle_from_motor(point[0]))
    pitch = math.degrees(pitch_calibration.muzzle_from_motor(point[1]))
    return f"yaw {yaw:+.2f}° / 仰角 {pitch:.2f}°"


def main() -> None:
    parser = argparse.ArgumentParser(description="低速走完 home 與安全工作區邊界一次")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS_PATH)
    parser.add_argument("--startup-pose", type=Path, default=DEFAULT_STARTUP_POSE)
    parser.add_argument(
        "--mode",
        choices=("posvel", "mit"),
        default="posvel",
        help="預設使用較保守的馬達內建位置速度模式",
    )
    parser.add_argument("--speed", type=float, default=0.08, help="最大速度 rad/s，預設 0.08")
    parser.add_argument(
        "--boundary-backoff-motor-deg",
        type=float,
        default=0.15,
        help="在既有 0.5 度內縮之外再退讓的馬達角度",
    )
    parser.add_argument("--dry-run", action="store_true", help="只驗證並顯示路徑")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.02 <= args.speed <= 0.25:
        raise SystemExit("--speed 必須介於 0.02 到 0.25 rad/s")
    if (
        not math.isfinite(args.boundary_backoff_motor_deg)
        or args.boundary_backoff_motor_deg < 0.1
    ):
        raise SystemExit("--boundary-backoff-motor-deg 必須至少 0.1 度")

    workspace = load_workspace_limit(args.limits)
    if workspace is None or workspace.home_corridor is None:
        raise SystemExit("找不到多邊形工作區或 home 通道")
    yaw_calibration = load_yaw_calibration(args.startup_pose)
    pitch_calibration = load_pitch_calibration(args.startup_pose)
    boundary = _build_boundary_path(workspace, args.boundary_backoff_motor_deg)
    home = (
        workspace.home_corridor.yaw_center_rad,
        workspace.home_corridor.home_pitch_rad,
    )

    print("已離線驗證完整巡邊路徑：")
    print(f"  控制模式：{args.mode}；速度上限 {args.speed:.3f} rad/s")
    print(f"  home：{_physical_label(home, yaw_calibration, pitch_calibration)}")
    print(f"  安全停放點：{_physical_label(workspace.anchor, yaw_calibration, pitch_calibration)}")
    for index, point in enumerate(boundary, 1):
        print(f"  邊界 {index}：{_physical_label(point, yaw_calibration, pitch_calibration)}")
    if args.dry_run:
        print("dry-run 完成；未開啟串列埠，也未使能馬達。")
        return

    motors = DamiaoSerialBackend(
        motor_ids=[YAW_MOTOR_ID, PITCH_MOTOR_ID],
        port=args.port,
        mode=args.mode,
        kp=30.0,
        kd=1.0,
        enable_on_start=False,
        position_limits=load_motor_limits(args.limits),
        workspace=workspace,
    )
    try:
        current = _feedback_pair(motors)
        print(f"目前位置：{_physical_label(current, yaw_calibration, pitch_calibration)}")
        if input(
            "確認砲口周圍無人、無工具／線材干涉、發射系統已斷電；"
            "輸入 MOVE 才會使能："
        ).strip().upper() != "MOVE":
            print("已取消，馬達未使能。")
            return

        motors.enable()
        # Explicit staging makes the 45-degree home interlock unambiguous.
        centered_low = (workspace.home_corridor.yaw_center_rad, current[1])
        _move_to(motors, centered_low, args.speed, "低姿態水平置中")
        _move_to(motors, home, args.speed, "到 45° home")
        _move_to(motors, workspace.anchor, args.speed, "由 home 垂直下降至活動區")
        _move_to(motors, boundary[0], args.speed, "進入巡邊起點")
        for index, point in enumerate(boundary[1:] + boundary[:1], 2):
            _move_to(motors, point, args.speed, f"沿安全邊界 {index - 1}/{len(boundary)}")
        _move_to(motors, workspace.anchor, args.speed, "回安全停放點")
        print("home 與完整邊界巡航已完成。")
    except KeyboardInterrupt:
        print("\n使用者中止；正在失能馬達。", flush=True)
    finally:
        motors.close()
        print("兩顆馬達已失能；請支撐垂直機構。")


if __name__ == "__main__":
    main()

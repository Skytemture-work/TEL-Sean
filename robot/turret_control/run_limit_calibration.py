"""Manually capture a non-rectangular safe workspace and home/reference pose.

The program never commands a position.  It disables both motors, asks the
operator to move the mechanism by hand, and records DAMIAO feedback when the
operator traces one closed path around the safe outline.  The captured
yaw/pitch polygon is later enforced by the real motor backend for every
position command.

Run on the AGX with the mechanism supported, for example::

    python3 -m turret_control.run_limit_calibration

Use ``--skip-origin`` when only the travel limits should be recaptured.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .backends import DamiaoSerialBackend
from .limits import (
    DEFAULT_LIMITS_PATH,
    load_axis_limits,
    load_hard_motor_limits,
    load_workspace_limit,
    save_workspace_limits,
)
from .pitch import (
    PitchCalibration,
    load_pitch_calibration,
    load_yaw_calibration,
    save_manual_origin_calibration,
)


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_STARTUP_POSE = Path(__file__).with_name("startup_pose.json")
YAW_MOTOR_ID = 0x03
PITCH_MOTOR_ID = 0x01
MIN_MUZZLE_ELEVATION_DEG = 0.0
MAX_MUZZLE_ELEVATION_DEG = 45.0


def _save_recovery_trace(path: Path, samples: list, duration_s: float, sample_hz: float) -> None:
    """Immediately preserve raw feedback before polygon validation or SAVE."""

    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    serializable_samples = []
    for sample in samples:
        converted = {
            "elapsed_s": float(sample["elapsed_s"]),
            "yaw_rad": float(sample["yaw_rad"]),
            "pitch_rad": float(sample["pitch_rad"]),
        }
        serializable_samples.append(converted)
    data = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "automatic raw recovery trace before workspace validation",
        "units": {"motor_position": "rad", "elapsed": "s"},
        "duration_s": float(duration_s),
        "sample_hz": float(sample_hz),
        "yaw_motor_id": f"0x{YAW_MOTOR_ID:02X}",
        "pitch_motor_id": f"0x{PITCH_MOTOR_ID:02X}",
        "samples": serializable_samples,
    }
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_recovery_trace(path: Path) -> list:
    """Load a previously preserved raw trace without changing its samples."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) < 20:
        raise ValueError("recovery trace needs at least 20 samples")
    converted = []
    for sample in samples:
        converted.append(
            {
                "elapsed_s": float(sample["elapsed_s"]),
                "yaw_rad": float(sample["yaw_rad"]),
                "pitch_rad": float(sample["pitch_rad"]),
            }
        )
    if not all(
        math.isfinite(value)
        for sample in converted
        for value in sample.values()
    ):
        raise ValueError("recovery trace contains a non-finite value")
    return converted


def _hard_limits_for_capture(
    limits_path: Path,
    startup_pose_path: Path,
    proposed_origin_yaw: object,
    proposed_origin_pitch: object,
    home_pitch_deg: float,
) -> tuple:
    """Intersect retained motor limits with physical yaw/elevation guards."""

    hard_limits = load_hard_motor_limits(limits_path)
    stored = load_pitch_calibration(startup_pose_path)
    if proposed_origin_pitch is None:
        calibration = stored
    else:
        # save_manual_origin_calibration stores motor_sign=-1 while retaining
        # the existing mechanism ratio.  Build the same calibration before any
        # file is changed so workspace validation uses the proposed home.
        motor_sign = -1
        motor_zero = float(proposed_origin_pitch) - (
            motor_sign
            * stored.motor_rad_per_muzzle_rad
            * math.radians(home_pitch_deg)
        )
        calibration = PitchCalibration(
            motor_zero_rad=motor_zero,
            motor_rad_per_muzzle_rad=stored.motor_rad_per_muzzle_rad,
            motor_sign=motor_sign,
        )

    physical_pitch_bounds = sorted(
        (
            calibration.motor_from_muzzle(math.radians(MIN_MUZZLE_ELEVATION_DEG)),
            calibration.motor_from_muzzle(math.radians(MAX_MUZZLE_ELEVATION_DEG)),
        )
    )
    if PITCH_MOTOR_ID in hard_limits:
        previous = hard_limits[PITCH_MOTOR_ID]
        physical_pitch_bounds = [
            max(physical_pitch_bounds[0], previous[0]),
            min(physical_pitch_bounds[1], previous[1]),
        ]
    if physical_pitch_bounds[0] >= physical_pitch_bounds[1]:
        raise ValueError("previous pitch limits do not overlap physical elevation 0..45 degrees")
    hard_limits[PITCH_MOTOR_ID] = (
        physical_pitch_bounds[0],
        physical_pitch_bounds[1],
    )

    stored_yaw = load_yaw_calibration(startup_pose_path)
    if proposed_origin_yaw is None:
        yaw_calibration = stored_yaw
    else:
        yaw_calibration = PitchCalibration(
            motor_zero_rad=float(proposed_origin_yaw),
            motor_rad_per_muzzle_rad=stored_yaw.motor_rad_per_muzzle_rad,
            motor_sign=-1,
        )
    physical_yaw_bounds = sorted(
        (
            yaw_calibration.motor_from_muzzle(math.radians(-35.0)),
            yaw_calibration.motor_from_muzzle(math.radians(35.0)),
        )
    )
    if YAW_MOTOR_ID in hard_limits:
        previous = hard_limits[YAW_MOTOR_ID]
        physical_yaw_bounds = [
            max(physical_yaw_bounds[0], previous[0]),
            min(physical_yaw_bounds[1], previous[1]),
        ]
    if physical_yaw_bounds[0] >= physical_yaw_bounds[1]:
        raise ValueError("previous yaw limits do not overlap physical yaw -35..35 degrees")
    hard_limits[YAW_MOTOR_ID] = (
        physical_yaw_bounds[0],
        physical_yaw_bounds[1],
    )
    return hard_limits, calibration, yaw_calibration


def _read_position(motors: DamiaoSerialBackend, motor_id: int) -> float:
    """Refresh a few times and return a finite feedback position."""

    for _ in range(5):
        motors.refresh()
        state = motors.get_state(motor_id)
        if (
            state is not None
            and state.err is not None
            and math.isfinite(state.position_rad)
        ):
            return state.position_rad
        time.sleep(0.03)
    raise RuntimeError(f"讀不到馬達 0x{motor_id:02X} 的位置回饋")


def _capture(
    motors: DamiaoSerialBackend,
    motor_id: int,
    instruction: str,
) -> float:
    input(f"\n{instruction}\n確認位置安全後按 Enter 記錄：")
    position = _read_position(motors, motor_id)
    print(f"已記錄 motor 0x{motor_id:02X}: {position:+.9f} rad ({math.degrees(position):+.3f}°)")
    return position


def _show_current(motors: DamiaoSerialBackend) -> None:
    yaw = _read_position(motors, YAW_MOTOR_ID)
    pitch = _read_position(motors, PITCH_MOTOR_ID)
    print(
        "目前回饋："
        f"水平 {yaw:+.9f} rad ({math.degrees(yaw):+.3f}°), "
        f"垂直 {pitch:+.9f} rad ({math.degrees(pitch):+.3f}°)"
    )


def _read_pair(motors: DamiaoSerialBackend) -> tuple:
    """Read one simultaneous yaw/pitch feedback pair, rejecting stale data."""

    motors.refresh()
    yaw = motors.get_state(YAW_MOTOR_ID)
    pitch = motors.get_state(PITCH_MOTOR_ID)
    if (
        yaw is None
        or pitch is None
        or yaw.err is None
        or pitch.err is None
        or not math.isfinite(yaw.position_rad)
        or not math.isfinite(pitch.position_rad)
    ):
        raise RuntimeError("讀不到兩顆馬達的即時位置回饋")
    return yaw.position_rad, pitch.position_rad


def _read_stable_pair(motors: DamiaoSerialBackend) -> tuple:
    """Discard stale post-prompt feedback and average the latest stable reads."""

    readings = []
    for _ in range(8):
        try:
            yaw, pitch = _read_pair(motors)
            readings.append((float(yaw), float(pitch)))
        except RuntimeError:
            pass
        time.sleep(0.03)
    if len(readings) < 3:
        raise RuntimeError("無法取得足夠的兩軸最新位置回饋")
    latest = readings[-3:]
    return (
        sum(value[0] for value in latest) / len(latest),
        sum(value[1] for value in latest) / len(latest),
    )


def _record_workspace(
    motors: DamiaoSerialBackend,
    duration_s: float,
    sample_hz: float,
    closure_motor_deg: float,
    closure_extra_s: float,
) -> list:
    """Collect an outline, continuing after the timer until it closes."""

    samples = []
    started_at = time.monotonic()
    minimum_deadline = started_at + duration_s
    final_deadline = minimum_deadline + closure_extra_s
    next_sample_at = started_at
    last_remaining = None
    extending = False
    while True:
        now = time.monotonic()
        if now < next_sample_at:
            time.sleep(min(next_sample_at - now, 0.02))
            continue
        yaw, pitch = _read_pair(motors)
        samples.append(
            {
                "elapsed_s": float(now - started_at),
                "yaw_rad": float(yaw),
                "pitch_rad": float(pitch),
            }
        )
        closure_gap_deg = math.degrees(
            math.hypot(
                samples[0]["yaw_rad"] - yaw,
                samples[0]["pitch_rad"] - pitch,
            )
        )
        if now < minimum_deadline:
            remaining = max(0, int(math.ceil(minimum_deadline - now)))
            if remaining != last_remaining:
                print(
                    f"量測中：剩餘 {remaining:2d} 秒，樣本 {len(samples)}",
                    flush=True,
                )
                last_remaining = remaining
        elif closure_gap_deg <= closure_motor_deg:
            print(
                f"輪廓已閉合：起終點距離 {closure_gap_deg:.2f} 馬達度。",
                flush=True,
            )
            break
        else:
            if not extending:
                print(
                    "30 秒已到，但尚未回到起點；請繼續沿原方向回到起點，"
                    "程式會保持錄製。",
                    flush=True,
                )
                extending = True
            remaining = max(0, int(math.ceil(final_deadline - now)))
            if remaining != last_remaining:
                print(
                    f"等待閉合：距起點 {closure_gap_deg:.2f} 馬達度；"
                    f"最多再錄 {remaining:2d} 秒",
                    flush=True,
                )
                last_remaining = remaining
            if now >= final_deadline:
                print("延長錄製時間已到；原始樣本仍會保存。", flush=True)
                break
        next_sample_at += 1.0 / sample_hz
        if next_sample_at < now:
            next_sample_at = now + 1.0 / sample_hz
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="手動描繪炮塔非矩形水平／垂直安全工作區"
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--startup-pose", type=Path, default=DEFAULT_STARTUP_POSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_LIMITS_PATH)
    parser.add_argument(
        "--margin-motor-deg",
        type=float,
        default=0.5,
        help="從兩端再向內保留的馬達角度安全裕量，預設 0.5 度",
    )
    parser.add_argument(
        "--home-pitch-deg",
        type=float,
        default=45.0,
        help="手動 home 參考點的砲口仰角，預設 45 度",
    )
    parser.add_argument(
        "--skip-origin",
        action="store_true",
        help="不更新 startup_pose.json，只記錄活動範圍",
    )
    parser.add_argument(
        "--recover-trace",
        action="store_true",
        help="使用 output 同目錄的 *_trace_latest.json，不重新描邊",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="手動描邊的收集秒數，預設 30 秒",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        help="描邊取樣頻率，預設 20 Hz",
    )
    parser.add_argument(
        "--simplify-motor-deg",
        type=float,
        default=0.25,
        help="儲存多邊形前可省略的輪廓細節，預設 0.25 馬達度",
    )
    parser.add_argument(
        "--closure-motor-deg",
        type=float,
        default=2.0,
        help="描邊終點與起點最大容許距離，預設 2 馬達度",
    )
    parser.add_argument(
        "--closure-extra-s",
        type=float,
        default=30.0,
        help="30 秒後尚未閉合時繼續錄製的最長時間，預設再 30 秒",
    )
    args = parser.parse_args()

    if not math.isfinite(args.margin_motor_deg) or args.margin_motor_deg < 0.0:
        raise SystemExit("--margin-motor-deg 必須是大於等於 0 的有限數字")
    if not math.isfinite(args.home_pitch_deg) or not 0.0 <= args.home_pitch_deg <= 45.0:
        raise SystemExit("--home-pitch-deg 必須介於 0 到 45 度")
    if not math.isfinite(args.duration_s) or args.duration_s < 5.0:
        raise SystemExit("--duration-s 必須至少 5 秒")
    if not math.isfinite(args.sample_hz) or not 2.0 <= args.sample_hz <= 50.0:
        raise SystemExit("--sample-hz 必須介於 2 到 50 Hz")
    if not math.isfinite(args.simplify_motor_deg) or args.simplify_motor_deg < 0.0:
        raise SystemExit("--simplify-motor-deg 必須是大於等於 0 的有限數字")
    if not math.isfinite(args.closure_motor_deg) or args.closure_motor_deg <= 0.0:
        raise SystemExit("--closure-motor-deg 必須是大於 0 的有限數字")
    if not math.isfinite(args.closure_extra_s) or args.closure_extra_s < 5.0:
        raise SystemExit("--closure-extra-s 必須至少 5 秒")
    if args.recover_trace and not args.skip_origin:
        raise SystemExit("--recover-trace 必須搭配 --skip-origin，避免改寫 home 原點")

    motors = None
    try:
        # No position command is sent by this program.  Starting disabled is
        # essential because the operator must be able to move the mechanism by
        # hand.  The existing backend also probes both motors and their MST_IDs.
        motors = DamiaoSerialBackend(
            motor_ids=[YAW_MOTOR_ID, PITCH_MOTOR_ID],
            port=args.port,
            mode="posvel",
            enable_on_start=False,
        )
        motors.disable()
        time.sleep(0.2)
        print("兩顆馬達已送出失能命令；程式不會送出任何位置目標。")
        print("請支撐垂直機構，避免失能後因重力下墜。")
        _show_current(motors)
        if args.recover_trace:
            if input("輸入 RECOVER 使用已保存的描邊資料，其他內容取消：").strip().upper() != "RECOVER":
                print("已取消。")
                return
        else:
            start_word = input(
                "輸入 RECORD 準備校正（此時尚未開始 30 秒描邊）；"
                "也可輸入 TRACE 繼續："
            ).strip().upper()
            if start_word not in ("RECORD", "TRACE"):
                print("未開始量測。")
                return

        origin_yaw = None
        origin_pitch = None
        if not args.skip_origin:
            print("\n第一部分：校正 home／原點。這會處理回 home 點的偏差。")
            origin_yaw = _capture(
                motors,
                YAW_MOTOR_ID,
                "請手動把水平軸放到理想 home／正前方 0°。",
            )
            origin_pitch = _capture(
                motors,
                PITCH_MOTOR_ID,
                f"請手動把垂直軸放到 home 仰角 {args.home_pitch_deg:.1f}°。",
            )

        hard_motor_limits, pitch_calibration, yaw_calibration = _hard_limits_for_capture(
            args.output,
            args.startup_pose,
            origin_yaw,
            origin_pitch,
            args.home_pitch_deg,
        )
        pitch_hard_min, pitch_hard_max = hard_motor_limits[PITCH_MOTOR_ID]
        physical_hard_bounds = sorted(
            math.degrees(pitch_calibration.muzzle_from_motor(value))
            for value in (pitch_hard_min, pitch_hard_max)
        )
        yaw_hard_min, yaw_hard_max = hard_motor_limits[YAW_MOTOR_ID]
        physical_yaw_hard_bounds = sorted(
            math.degrees(yaw_calibration.muzzle_from_motor(value))
            for value in (yaw_hard_min, yaw_hard_max)
        )
        print(
            "保留舊硬限制並套用物理仰角限制："
            f"pitch {physical_hard_bounds[0]:.2f}° .. {physical_hard_bounds[1]:.2f}°；"
            f"yaw {physical_yaw_hard_bounds[0]:.2f}° .. "
            f"{physical_yaw_hard_bounds[1]:.2f}°。"
            "新輪廓只能再縮小，不能放寬。"
        )

        recovery_path = args.output.with_name(
            args.output.stem + "_trace_latest.json"
        )
        if args.recover_trace:
            samples = _load_recovery_trace(recovery_path)
            print(f"已載入原始描邊資料：{recovery_path}")
        else:
            print("\n第二部分：描繪非矩形安全工作區。")
            print(
                "請先把砲口放到安全外框的任一點；開始後沿著『安全外框內側』"
                "以單一方向繞一整圈，最後回到開始位置。不要碰硬限位，也不要隨意"
                "在工作區內來回穿越，否則輪廓會自交而被拒絕。"
            )
            if input(
                f"輸入 TRACE 後開始 {args.duration_s:.0f} 秒描邊，其他內容取消："
            ).strip().upper() != "TRACE":
                print("未開始描邊。")
                return
            samples = _record_workspace(
                motors,
                args.duration_s,
                args.sample_hz,
                args.closure_motor_deg,
                args.closure_extra_s,
            )
        print(f"已收集 {len(samples)} 個 yaw/pitch 樣本。")
        negative_elevation_samples = sum(
            math.degrees(
                pitch_calibration.muzzle_from_motor(float(sample["pitch_rad"]))
            ) < MIN_MUZZLE_ELEVATION_DEG
            for sample in samples
        )
        outside_hard_limits = 0
        for sample in samples:
            yaw = float(sample["yaw_rad"])
            pitch = float(sample["pitch_rad"])
            yaw_bounds = hard_motor_limits.get(YAW_MOTOR_ID)
            pitch_bounds = hard_motor_limits[PITCH_MOTOR_ID]
            if (
                (yaw_bounds is not None and not yaw_bounds[0] <= yaw <= yaw_bounds[1])
                or not pitch_bounds[0] <= pitch <= pitch_bounds[1]
            ):
                outside_hard_limits += 1
        if outside_hard_limits:
            print(
                f"注意：{outside_hard_limits} 個樣本超出舊硬限制，"
                "建立輪廓時會被截在舊限制內。"
            )
        if negative_elevation_samples:
            print(
                f"注意：{negative_elevation_samples} 個樣本的換算仰角低於 0°，"
                "原始值會保留，但不會加入允許工作區。"
            )
        if not args.recover_trace:
            _save_recovery_trace(
                recovery_path,
                samples,
                duration_s=args.duration_s,
                sample_hz=args.sample_hz,
            )
            print(f"原始樣本已立即保存：{recovery_path}")
        save_options = {
            "samples": samples,
            "yaw_motor_id": YAW_MOTOR_ID,
            "pitch_motor_id": PITCH_MOTOR_ID,
            "margin_motor_deg": args.margin_motor_deg,
            "simplify_motor_deg": args.simplify_motor_deg,
            "closure_motor_deg": args.closure_motor_deg,
            "hard_motor_limits": hard_motor_limits,
            "yaw_symmetry_center_rad": yaw_calibration.motor_zero_rad,
            "home_position": {
                "yaw_rad": yaw_calibration.motor_from_muzzle(0.0),
                "pitch_rad": pitch_calibration.motor_from_muzzle(
                    math.radians(args.home_pitch_deg)
                ),
            },
        }
        while True:
            input(
                "請手動把砲口移到輪廓內、遠離邊界的安全停放點，"
                "靜止後按 Enter 記錄："
            )
            anchor_yaw, anchor_pitch = _read_stable_pair(motors)
            physical_yaw = math.degrees(yaw_calibration.muzzle_from_motor(anchor_yaw))
            physical_pitch = math.degrees(
                pitch_calibration.muzzle_from_motor(anchor_pitch)
            )
            print(
                "安全停放點回饋："
                f"yaw {physical_yaw:+.2f}°，仰角 {physical_pitch:.2f}°；"
                f"motor ({anchor_yaw:+.6f}, {anchor_pitch:+.6f}) rad"
            )
            try:
                with tempfile.TemporaryDirectory() as directory:
                    save_workspace_limits(
                        Path(directory) / "mechanism_limits.json",
                        anchor_position={
                            "yaw_rad": anchor_yaw,
                            "pitch_rad": anchor_pitch,
                        },
                        **save_options,
                    )
            except ValueError as exc:
                print(f"此停放點未通過安全檢查：{exc}")
                if input("按 Enter 重新放置並記錄；輸入 Q 取消：").strip().upper() == "Q":
                    print(f"已取消；原始樣本仍保存在：{recovery_path}")
                    return
                continue
            print("安全停放點已通過內縮、對稱與 home 通道檢查。")
            break
        closure_gap_deg = math.degrees(
            math.hypot(
                samples[0]["yaw_rad"] - samples[-1]["yaw_rad"],
                samples[0]["pitch_rad"] - samples[-1]["pitch_rad"],
            )
        )
        print(
            "\n將寫入多邊形工作區："
            f"樣本 {len(samples)}；起終點距離 {closure_gap_deg:.2f} 馬達度；"
            f"內縮裕量 {args.margin_motor_deg:.2f} 馬達度"
        )
        if input("輸入 SAVE 寫入設定檔，其他內容取消：").strip().upper() != "SAVE":
            print("已取消，未寫入設定檔。")
            return

        try:
            backup = save_workspace_limits(
                args.output,
                anchor_position={"yaw_rad": anchor_yaw, "pitch_rad": anchor_pitch},
                **save_options,
            )
        except ValueError as exc:
            print(f"工作區驗證失敗，未覆蓋限制檔：{exc}")
            print(f"原始樣本仍保存在：{recovery_path}")
            return
        print(f"安全活動範圍已寫入：{args.output}")
        if backup.exists():
            print(f"舊範圍備份：{backup}")
        if origin_yaw is not None and origin_pitch is not None:
            backup = save_manual_origin_calibration(
                args.startup_pose,
                yaw_motor_rad=origin_yaw,
                pitch_motor_rad=origin_pitch,
                pitch_reference_deg=args.home_pitch_deg,
            )
            print(f"home 原點已更新；舊檔備份：{backup}")
        loaded = load_axis_limits(args.output)
        for axis in ("yaw", "pitch"):
            limit = loaded[axis]
            print(
                f"  {axis}: motor 0x{limit.motor_id:02X}, "
                f"{limit.min_rad:+.6f} .. {limit.max_rad:+.6f} rad"
            )
        workspace = load_workspace_limit(args.output)
        if workspace is not None:
            print(f"  非矩形工作區：{len(workspace.vertices)} 個頂點，已套用")
        print("下次啟動實機控制程式時，所有位置命令都會套用這些限制。")
    finally:
        if motors is not None:
            # Motors are already disabled; do not send a final hold frame
            # while the operator may still be holding the mechanism by hand.
            motors.close(send_hold=False)


if __name__ == "__main__":
    main()

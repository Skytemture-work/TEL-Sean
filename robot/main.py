# main.py
import signal
import subprocess
import sys
import time
from pathlib import Path
import config
from comm_uart import UARTController
from vision_target import VisionTracker
from lidar_avoid import LidarAvoidance
from manual import (
    ManualController,
    VESC_MIN_RPM,
    VESC_RPM_PER_SPEED_UNIT,
    normalize_vrb_speed,
)


VISION_WARMUP_SECONDS = 3.0
AUTO_STATUS_INTERVAL_SECONDS = 1.0
POSITIVE_VY_IS_LEFT = True
AUTO_TARGET_DISTANCE_M = 2.60
AUTO_FORWARD_VX = 40
VISION_CENTER_VY = 5
VISION_LOST_TIMEOUT_SECONDS = 3.0
OCCLUSION_STRAFE_VY = 5


AUTO_FIRE_WEB_PORT = 8765
AIM_SCRIPT = Path(__file__).resolve().parent / "turret_control" / "run_board_real.py"


# The aiming process consumes this event only while waiting for a refill.
AIM_ENTRY = """
import sys
import threading
from turret_control.run_board_real import main

reload_event = threading.Event()

def read_reload_commands():
    for line in sys.stdin:
        if line.strip() == "RELOAD":
            reload_event.set()

threading.Thread(target=read_reload_commands, daemon=True).start()
main(sys.argv[1:], reload_event=reload_event)
"""


def stop_aim_process(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    process.wait(timeout=10)


def main():
    print("====================================================")
    print("TEL-Sean 主控系統：手動 / 自動模式")
    print("====================================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    vision = None
    aim_process = None
    lidar = LidarAvoidance()

    # 手動模式控制器：接收遙控器封包，管理砲台指令、射擊與彈藥倉格位
    manual = None
    try:
        print("[System] 正在初始化手動控制器（DM / MKS CAN / VESC）", flush=True)
        manual = ManualController(uart=uart)
        print("[System] 手動控制器初始化完成", flush=True)
        lidar.start()
        time.sleep(1.0)

        # 預設手動模式；收到 ESP32 封包後依封包中的 mode 切換
        current_mode = "M"
        awaiting_phase = False
        last_auto_reload = None
        vision_ready_at = 0.0
        next_auto_status_at = 0.0
        last_received_mode = None
        auto_complete = False
        auto_started = False
        center_only = False
        vision_detected_once = False
        last_target_side = 0
        occlusion_strafe_side = 0
        vision_lost_since = None
        held_vy = 0
        vy_hold_until = 0.0
        print("[System] 已啟動，預設手動模式，等待 ESP32 遙控器封包...")

        while True:
            if current_mode == "M" and manual is not None:
                manual.check_health()
            packet = uart.read_esp32_packet()

            if packet is not None:
                packet_mode = packet.get("mode", "?")
                if packet_mode != last_received_mode:
                    print(f"[ESP32 MODE] 收到 {packet_mode!r}，目前模式 {current_mode}", flush=True)
                    last_received_mode = packet_mode
                # raw_speed = packet.get("speed")
                # packet_speed = normalize_vrb_speed(raw_speed) if raw_speed is not None else 0
                # vesc_target = (
                #     "STOP"
                #     if packet_mode != "M" or raw_speed is None
                #     else "{} RPM".format(
                #         int(VESC_MIN_RPM + packet_speed * VESC_RPM_PER_SPEED_UNIT)
                #     )
                # )
                # print(
                #     "[ESP32 RX] raw={!r} | mode={} vx={} vy={} wz={} "
                #     "yaw={} pitch={} fire={} speed={} reload={} normalized_speed={} | VESC target={}".format(
                #         packet.get("raw", ""),
                #         packet_mode,
                #         packet.get("vx", 0),
                #         packet.get("vy", 0),
                #         packet.get("wz", 0),
                #         packet.get("yaw", 50),
                #         packet.get("pitch", 50),
                #         packet.get("fire", 0),
                #         packet.get("speed", 0),
                #         packet.get("reload"),
                #         packet_speed,
                #         vesc_target,
                #     ),
                #     flush=True,
                # )
                new_mode = packet.get("mode", current_mode)
                if new_mode not in ("M", "A"):
                    new_mode = current_mode

                if new_mode != current_mode:
                    print(
                        "\n[Mode Switch] 正在切換至：{}".format(
                            "自動模式" if new_mode == "A" else "手動模式"
                        )
                    )
                    auto_complete = False
                    auto_started = False
                    awaiting_phase = False
                    center_only = False
                    vision_detected_once = False
                    last_target_side = 0
                    occlusion_strafe_side = 0
                    vision_lost_since = None
                    held_vy = 0
                    vy_hold_until = 0.0
                    if new_mode == "M":
                        try:
                            stop_aim_process(aim_process)
                        except subprocess.TimeoutExpired:
                            print("[Aim] 瞄準程式尚未完成關閉，暫緩切回手動。", flush=True)
                            time.sleep(0.02)
                            continue
                        aim_process = None
                        if vision is not None:
                            vision.stop()
                            vision = None
                        lidar.set_active(False)

                    # 手動切自動時，等待 ID1 回零及 DM 回位、失能完成。
                    print("[Mode Switch] 等待手動控制器交接", flush=True)
                    if manual is None:
                        manual = ManualController(uart=uart)
                    manual.on_mode_change(
                        new_mode, fire=packet.get("fire", 0), reload=packet.get("reload")
                    )
                    print("[Mode Switch] 手動控制器交接完成", flush=True)
                    if new_mode == "A":
                        uart.send_packet(
                            mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                        )
                        vision = VisionTracker()
                        vision.start()
                        vision.set_active(True)
                        vision_ready_at = time.monotonic() + VISION_WARMUP_SECONDS
                        print(f"[Vision] 相機與模型初始化等待 {VISION_WARMUP_SECONDS:g} 秒", flush=True)
                        lidar.set_active(True)
                        next_auto_status_at = 0.0
                    current_mode = new_mode
                    print(f"[Mode Switch] 已切換至 {current_mode}", flush=True)

                if current_mode == "M":
                    # 手動模式：使用遙控器 yaw/pitch 控制砲台方向，
                    # fire 按鈕每次「由放開變按下」觸發一發射擊循環。
                    manual.handle_packet(packet)
                    time.sleep(0.02)
                    continue

            else:
                uart.report_if_silent()

            if current_mode == "A" and not auto_started:
                # 僅使用暖機完成後讀到的 A 模式封包。
                if (
                    time.monotonic() >= vision_ready_at
                    and packet is not None
                    and packet.get("mode") == "A"
                    and packet.get("auto_start", 0) > 0
                ):
                    auto_started = True
                    print("[Auto Drive] 收到 VRA 啟動，開始自動定位", flush=True)
                else:
                    uart.send_packet(
                        mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                    )
                    now = time.monotonic()
                    if now >= vision_ready_at and now >= next_auto_status_at:
                        next_auto_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                        print("[Auto Drive] 相機暖機完成，等待 VRA 啟動", flush=True)
                    time.sleep(0.02)
                    continue

            if awaiting_phase and packet is not None and packet.get("mode") == "A":
                awaiting_phase = False
                match_phase = "lower" if packet.get("fire") == 1 else "upper"
                try:
                    manual.close()
                    manual = None
                    aim_process = subprocess.Popen(
                        [
                            sys.executable, "-u", "-c", AIM_ENTRY,
                            "--match-fire", "--match-phase", match_phase,
                            "--web-port", str(AUTO_FIRE_WEB_PORT),
                            "--headless",
                        ],
                        cwd=str(AIM_SCRIPT.parent.parent),
                        stdin=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                    last_auto_reload = packet.get("reload")
                    print(f"[Aim] 已啟動比賽全自動（半場={match_phase}，PID={aim_process.pid}）", flush=True)
                except OSError as exc:
                    print(f"[Aim] 啟動失敗：{exc}", flush=True)

            if (
                current_mode == "A" and packet is not None and packet.get("mode") == "A"
                and aim_process is not None and aim_process.poll() is None
            ):
                reload_value = packet.get("reload")
                if reload_value in (0, 1):
                    if last_auto_reload == 0 and reload_value == 1:
                        try:
                            aim_process.stdin.write("RELOAD\n")
                            aim_process.stdin.flush()
                            print("[Aim] 已轉送遙控器裝填確認", flush=True)
                        except BrokenPipeError:
                            print("[Aim] 子程序已關閉裝填通道", flush=True)
                    last_auto_reload = reload_value

            if current_mode == "A" and not auto_complete:
                # 自動模式維持原有視覺追蹤 + LiDAR 避障流程
                now = time.monotonic()
                vision_warming_up = now < vision_ready_at
                fresh_scan, front_distance_m = lidar.get_front_obstacle()
                distance_text = (
                    "無有效距離" if front_distance_m is None
                    else f"前方 {front_distance_m * 100:.1f} cm"
                )

                raw_vx, raw_vy, raw_wz = 0, 0, 0
                yaw, pitch = 50, 50
                detected = False
                if not vision_warming_up:
                    (raw_vx, raw_vy, raw_wz), (yaw, pitch), detected = (
                        vision.get_vision_data()
                    )
                if not detected:
                    yaw, pitch = 50, 50

                if not vision_warming_up:
                    if vision_lost_since is not None and (
                        now - vision_lost_since >= VISION_LOST_TIMEOUT_SECONDS
                    ):
                        center_only = True
                    if detected:
                        vision_lost_since = None
                    elif vision_lost_since is None:
                        vision_lost_since = now

                vision_detected_once = vision_detected_once or detected
                centered = detected and abs(raw_vy) < VISION_CENTER_VY
                if detected and not center_only:
                    last_target_side = int(raw_vy > 0) - int(raw_vy < 0)

                final_vx, final_vy, final_wz = 0, 0, 0
                fire_cmd, fire_speed = 0, 0

                lidar_allows_motion = (
                    fresh_scan and front_distance_m is not None
                    and front_distance_m > lidar.safe_dist_m
                )
                if vision_warming_up or not lidar_allows_motion:
                    held_vy = 0
                    vy_hold_until = 0.0
                elif detected:
                    held_vy = 0 if centered else raw_vy
                    # 等差縮放：|vy|=21 時達到 2 秒上限。
                    hold_seconds = min(2.0, (abs(raw_vy) + 3) / 12.0)
                    vy_hold_until = now + hold_seconds if held_vy else 0.0
                holding_vy = not detected and held_vy != 0 and now < vy_hold_until

                arrived = False
                if not fresh_scan:
                    lidar_status = "等待有效前方掃描"
                elif front_distance_m is None:
                    lidar_status = "偵測範圍內無前方回波"
                elif front_distance_m <= lidar.safe_dist_m:
                    lidar_status = "偵測到障礙，停止移動"
                else:
                    arrived = front_distance_m < AUTO_TARGET_DISTANCE_M
                    lidar_status = (
                        "抵達目標距離" if arrived else "尚未抵達目標距離"
                    )
                    if not vision_warming_up:
                        final_vx = 0 if arrived or center_only else AUTO_FORWARD_VX
                        if detected:
                            final_vy = 0 if centered else raw_vy
                            final_wz = 0 if center_only else raw_wz
                        elif holding_vy:
                            final_vy = held_vy

                target_would_reverse = (
                    center_only and detected and not centered
                    and raw_vy * occlusion_strafe_side < 0
                )
                if target_would_reverse:
                    auto_complete = True
                elif center_only:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    front_clear = fresh_scan and (
                        front_distance_m is None or front_distance_m > lidar.safe_dist_m
                    )
                    auto_complete = front_clear and vision_detected_once
                    if front_clear:
                        lidar_status = "前方障礙已解除"
                        if not vision_detected_once:
                            lidar_status += "，等待鏡頭首次偵測靶面"
                    elif fresh_scan and last_target_side:
                        final_vy = last_target_side * OCCLUSION_STRAFE_VY
                        lidar_status = f"前方有障礙，沿最後靶面方向橫移（vy={final_vy}）"
                    elif fresh_scan:
                        lidar_status = "前方有障礙，無可用的靶面方向"
                else:
                    auto_complete = centered and arrived
                if auto_complete:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    yaw, pitch = 50, 50
                    fire_cmd, fire_speed = 0, 0

                # 自動模式的底盤前後方向與 Xavier 的正負號相反。
                tx_vx = -final_vx
                uart.send_packet(
                    mode="A",
                    vx=tx_vx,
                    vy=final_vy,
                    wz=final_wz,
                    yaw=yaw,
                    pitch=pitch,
                    fire=fire_cmd,
                    speed=fire_speed,
                )

                if center_only and final_vy:
                    occlusion_strafe_side = last_target_side

                if auto_complete:
                    lidar.set_active(False)
                    if target_would_reverse:
                        reason = "重新偵測到靶面，但修正方向會往回，定位完成"
                    elif center_only:
                        reason = "遮擋後前方障礙已解除"
                    else:
                        reason = "已置中且抵達目標距離"
                    print(f"[Auto Drive] {distance_text}，{reason}，底盤已送出停止，等待釋放 ZED", flush=True)
                    camera_released = vision.stop()
                    vision = None
                    if not camera_released:
                        print("[Aim] ZED 未確認釋放，未啟動瞄準", flush=True)
                    elif not AIM_SCRIPT.is_file():
                        print(f"[Aim] 找不到瞄準程式：{AIM_SCRIPT}", flush=True)
                    else:
                        awaiting_phase = True
                        print("[Aim] 定位完成，等待下一筆 A 模式封包選擇半場", flush=True)

                if not auto_complete and now >= next_auto_status_at:
                    next_auto_status_at = now + AUTO_STATUS_INTERVAL_SECONDS

                    if vision_warming_up:
                        remaining = vision_ready_at - now
                        vision_status = f"初始化等待中，剩 {remaining:.0f} 秒"
                    elif holding_vy:
                        remaining = vy_hold_until - now
                        vision_status = f"暫失目標，延續橫移（vy={held_vy}，剩 {remaining:.1f} 秒）"
                    elif not detected:
                        vision_status = "未偵測到目標"
                    elif centered:
                        vision_status = f"已置中（vy={raw_vy}）"
                    else:
                        move_left = (raw_vy > 0) == POSITIVE_VY_IS_LEFT
                        direction = "左" if move_left else "右"
                        vision_status = f"偵測到目標，需往{direction}修正（vy={raw_vy}）"

                    if center_only:
                        vision_status += "，遮擋逾時，依最後靶面方向避障"

                    print(
                        f"[LiDAR] {distance_text}，{lidar_status} | [鏡頭] {vision_status} | "
                        f"[送出] vx={tx_vx} vy={final_vy} wz={final_wz}",
                        flush=True,
                    )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[System] 使用者中斷，關閉系統...")
    finally:
        try:
            stop_aim_process(aim_process)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[Aim] 關閉時發生錯誤：{exc}", flush=True)
        if vision is not None:
            vision.stop()
        try:
            if manual is not None:
                manual.close()
        except Exception as exc:
            print("[Manual] 關閉控制器時發生錯誤：", exc)
        uart.close()


if __name__ == "__main__":
    main()

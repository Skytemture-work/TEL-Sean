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


AUTO_STATUS_INTERVAL_SECONDS = 1.0
AUTO_TARGET_DISTANCE_M = 2.60
AUTO_FORWARD_VX = 40
OCCLUSION_STRAFE_VY = 5
MAX_STRAFE_SECONDS = 1.0
VISION_WAIT_SECONDS = 10.0
NORMAL_STRAFE_DURATION_SCALE = 1.5
OCCLUSION_BRAKE_SECONDS = 0.15


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
        next_auto_status_at = 0.0
        last_received_mode = None
        auto_complete = False
        auto_started = False
        locked_vy = None
        vy_recorded_before_release = False
        strafe_until = None
        vision_wait_until = None
        positioning_moved = False
        brake_until = None
        avoid_until = None
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
                    locked_vy = None
                    vy_recorded_before_release = False
                    strafe_until = None
                    vision_wait_until = None
                    positioning_moved = False
                    brake_until = None
                    avoid_until = None
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
                        print("[Vision] 正在準備相機與模型", flush=True)
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

            if current_mode == "A" and not auto_complete:
                now = time.monotonic()
                if packet is not None and packet.get("mode") == "A":
                    swa_enabled = packet.get("auto_start") == 1
                    if not swa_enabled:
                        if auto_started:
                            print("[Auto Drive] SWA 關閉，暫停定位", flush=True)
                            locked_vy = None
                        auto_started = False
                        vy_recorded_before_release = False
                        strafe_until = None
                        vision_wait_until = None
                        positioning_moved = False
                        brake_until = None
                        avoid_until = None
                    elif not auto_started:
                        auto_started = True
                        vy_recorded_before_release = locked_vy is not None
                        next_auto_status_at = 0.0
                        print(
                            f"[Auto Drive] SWA 已放行，開始自動定位，Vision vy={locked_vy}",
                            flush=True,
                        )

                if not auto_started:
                    (_, preview_vy, _), _, detected = vision.get_vision_data()
                    if detected:
                        locked_vy = preview_vy
                    uart.send_packet(
                        mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                    )
                    if now >= next_auto_status_at:
                        next_auto_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                        if locked_vy is None:
                            print("[Auto Drive] 等待 auto_start=1", flush=True)
                        else:
                            print(
                                f"[Auto Drive] 等待 auto_start=1｜放行前已偵測到靶面，保留 vy={locked_vy}",
                                flush=True,
                            )
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
                fresh_scan, front_distance_m = lidar.get_front_obstacle()
                distance_text = (
                    "無有效距離" if front_distance_m is None
                    else f"前方 {front_distance_m * 100:.1f} cm"
                )

                yaw, pitch = 50, 50
                final_vx, final_vy, final_wz = 0, 0, 0
                fire_cmd, fire_speed = 0, 0
                arrived = False

                valid_distance = fresh_scan and front_distance_m is not None
                blocked = valid_distance and front_distance_m <= lidar.safe_dist_m
                auto_complete = False
                completion_reason = "已抵達目標距離並完成橫移"

                if blocked and positioning_moved and brake_until is None:
                    brake_until = now + OCCLUSION_BRAKE_SECONDS
                    print("[Auto Drive] 移動中遇到遮擋，緊急停止", flush=True)

                if brake_until is not None:
                    lidar_status = "遮擋後緊急停止"
                    if not vy_recorded_before_release:
                        if now >= brake_until:
                            auto_complete = True
                            completion_reason = "遮擋後停止，放行前未記錄 vy，跳過橫移"
                    elif not valid_distance:
                        avoid_until = None
                    elif now >= brake_until:
                        if avoid_until is not None and now >= avoid_until:
                            auto_complete = True
                            completion_reason = "遮擋後短距離橫移完成"
                        else:
                            if avoid_until is None:
                                avoid_until = now + MAX_STRAFE_SECONDS
                            final_vy = -OCCLUSION_STRAFE_VY if (locked_vy or 0) < 0 else OCCLUSION_STRAFE_VY
                            lidar_status = "遮擋後短距離橫移"
                elif (
                    vision_wait_until is not None
                    and locked_vy is None
                    and now >= vision_wait_until
                ):
                    auto_complete = True
                    completion_reason = "LiDAR 已到位，等待 Vision 10 秒逾時，定位完成"
                elif not fresh_scan:
                    lidar_status = "等待有效前方掃描"
                elif front_distance_m is None:
                    lidar_status = "偵測範圍內無前方回波"
                elif blocked:
                    lidar_status = "起步前有障礙，等待前方清空"
                else:
                    arrived = (
                        vision_wait_until is not None
                        or front_distance_m < AUTO_TARGET_DISTANCE_M
                    )
                    if arrived and locked_vy is None:
                        if vision_wait_until is None:
                            vision_wait_until = now + VISION_WAIT_SECONDS
                            next_auto_status_at = 0.0
                        (_, raw_vy, _), _, detected = vision.get_vision_data()
                        if detected:
                            locked_vy = raw_vy
                    lidar_status = "抵達目標距離" if arrived else "尚未抵達目標距離"
                    final_vx = 0 if arrived else AUTO_FORWARD_VX
                    if locked_vy is not None and strafe_until is None:
                        duration = 0.0
                        if locked_vy:
                            duration = NORMAL_STRAFE_DURATION_SCALE * min(
                                MAX_STRAFE_SECONDS, (abs(locked_vy) + 3) / 12.0
                            )
                        strafe_until = now + duration
                        next_auto_status_at = 0.0
                        print(
                            f"[Auto Drive] Vision vy={locked_vy}，正常橫移指令 "
                            f"vy={locked_vy}，持續 {duration:.2f} 秒",
                            flush=True,
                        )
                    if strafe_until is not None:
                        if now < strafe_until:
                            final_vy = max(-100, min(100, locked_vy))
                            if not arrived:
                                final_vx = abs(final_vy)
                        else:
                            auto_complete = arrived
                    positioning_moved = positioning_moved or bool(final_vx or final_vy)

                if auto_complete:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    yaw, pitch = 50, 50
                    fire_cmd, fire_speed = 0, 0

                # 自動模式的底盤前後方向與 Xavier 的正負號相反。
                tx_vx = -final_vx
                tx_vy = final_vy
                uart.send_packet(
                    mode="A",
                    vx=tx_vx,
                    vy=tx_vy,
                    wz=final_wz,
                    yaw=yaw,
                    pitch=pitch,
                    fire=fire_cmd,
                    speed=fire_speed,
                )

                if auto_complete:
                    lidar.set_active(False)
                    print(
                        f"[Auto Drive] {distance_text}，{completion_reason}，底盤已送出停止，等待釋放 ZED",
                        flush=True,
                    )
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

                    if locked_vy is None:
                        vision_status = "等待首次偵測靶面" if vision_wait_until is not None else "尚未偵測到靶面"
                    else:
                        vision_status = f"已鎖定 vy={locked_vy}"

                    print(
                        f"[LiDAR] {distance_text}，{lidar_status} | [鏡頭] {vision_status} | "
                        f"[送出] vx={tx_vx} vy={tx_vy} wz={final_wz}",
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

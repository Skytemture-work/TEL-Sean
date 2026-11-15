# main.py
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
import config
from comm_uart import UARTController
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
OCCLUSION_STRAFE_VY = 15
MAX_STRAFE_SECONDS = 1.0
VISION_WAIT_SECONDS = 10.0
NORMAL_STRAFE_DURATION_SCALE = 1.5
OCCLUSION_BRAKE_SECONDS = 0.15


AUTO_FIRE_WEB_PORT = 8765
AIM_SCRIPT = Path(__file__).resolve().parent / "turret_control" / "run_board_real.py"


# Keep the existing stdin channel for handoff and reload commands.
AIM_ENTRY = """
import queue
import socket
import sys
import threading
from turret_control.run_board_real import main

reload_event = threading.Event()
handoff = queue.Queue()
vision_socket = socket.socket(fileno=int(sys.argv[1]))
vision_socket.setblocking(False)

def publish_vy(captured_at, vy):
    try:
        vision_socket.send(f"{captured_at} {vy}".encode("ascii"))
    except BlockingIOError:
        pass

def next_command(timeout):
    try:
        return handoff.get(timeout=timeout)
    except queue.Empty:
        return ""

def read_commands():
    for line in sys.stdin:
        command = line.strip()
        if command == "RELOAD":
            reload_event.set()
        elif command in ("PREPARE_MOTORS", "VISION_ON", "VISION_OFF", "START upper", "START lower"):
            handoff.put(command)
    handoff.put(None)

threading.Thread(target=read_commands, daemon=True).start()
with vision_socket:
    main(sys.argv[2:], reload_event=reload_event, next_handoff_command=next_command,
         positioning_callback=publish_vy)
"""


def start_aim_process():
    parent_socket, child_socket = socket.socketpair(type=socket.SOCK_DGRAM)
    parent_socket.setblocking(False)
    try:
        with child_socket:
            process = subprocess.Popen(
                [
                    sys.executable, "-u", "-c", AIM_ENTRY, str(child_socket.fileno()),
                    "--match-fire", "--web-port", str(AUTO_FIRE_WEB_PORT), "--headless",
                ],
                cwd=str(AIM_SCRIPT.parent.parent),
                stdin=subprocess.PIPE,
                text=True,
                start_new_session=True,
                pass_fds=(child_socket.fileno(),),
            )
    except BaseException:
        parent_socket.close()
        raise
    return process, parent_socket


def read_positioning_vy(vision_socket, valid_after, latest):
    if vision_socket is None:
        return None
    while True:
        try:
            captured_at, vy = vision_socket.recv(128).decode("ascii").split()
        except BlockingIOError:
            break
        latest = float(captured_at), int(vy)
    if latest is not None:
        captured_at, _ = latest
        if captured_at >= valid_after and time.monotonic() - captured_at <= config.AUTO_VISION_MAX_AGE_S:
            return latest
    return None


def send_aim_command(process, command):
    if process is None:
        return False
    exit_code = process.poll()
    if exit_code is not None:
        print(f"[Aim] 子程序已結束（exit={exit_code}），無法送出 {command}", flush=True)
        return False
    try:
        process.stdin.write(command + "\n")
        process.stdin.flush()
        return True
    except OSError as exc:
        print(f"[Aim] 無法送出 {command}：{exc}", flush=True)
        return False


def prewarm_aim_process():
    """Start the aim subprocess ahead of time (boot / back in manual).

    It loads YOLO and opens both ZEDs, but touches no motor until
    PREPARE_MOTORS.  Positioning inference stays off (VISION_OFF) until the
    switch to auto sends VISION_ON, so manual mode does not load the GPU.
    """
    try:
        process, vision_socket = start_aim_process()
    except OSError as exc:
        print(f"[Aim] 無法啟動預熱子程序：{exc}", flush=True)
        return None, None
    send_aim_command(process, "VISION_OFF")
    print(f"[Aim] 開始背景預熱（PID={process.pid}，不碰馬達）", flush=True)
    return process, vision_socket


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
    vision_socket = None
    aim_process = None
    lidar = None

    # 手動模式控制器：接收遙控器封包，管理砲台指令、射擊與彈藥倉格位
    manual = None
    try:
        print("[System] 正在初始化手動控制器（DM / MKS CAN / VESC）", flush=True)
        manual = ManualController(uart=uart)
        print("[System] 手動控制器初始化完成", flush=True)

        # 預設手動模式；收到 ESP32 封包後依封包中的 mode 切換
        current_mode = "M"
        awaiting_phase = False
        aim_started = False
        aim_prepared = False
        last_auto_reload = None
        next_auto_status_at = 0.0
        last_received_mode = None
        auto_complete = False
        auto_started = False
        locked_vy = None
        latest_vy = None
        vision_valid_after = 0.0
        vy_recorded_before_release = False
        strafe_until = None
        vision_wait_until = None
        positioning_moved = False
        brake_until = None
        avoid_until = None
        aim_exit_reported = False
        # Warm the aim subprocess up while still in manual mode, so the
        # switch to auto finds the model loaded and both ZEDs open.
        aim_process, vision_socket = prewarm_aim_process()
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
                    if new_mode == "M":
                        try:
                            stop_aim_process(aim_process)
                        except subprocess.TimeoutExpired:
                            print("[Aim] 瞄準程式尚未完成關閉，暫緩切回手動。", flush=True)
                            time.sleep(0.02)
                            continue
                        aim_process = None
                        aim_started = False
                        aim_prepared = False
                        if vision_socket is not None:
                            vision_socket.close()
                            vision_socket = None
                        if lidar is not None:
                            lidar.stop()

                    auto_complete = False
                    auto_started = False
                    awaiting_phase = False
                    locked_vy = None
                    latest_vy = None
                    vision_valid_after = time.monotonic()
                    vy_recorded_before_release = False
                    strafe_until = None
                    vision_wait_until = None
                    positioning_moved = False
                    brake_until = None
                    avoid_until = None

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
                        if aim_process is not None and aim_process.poll() is None:
                            print(f"[Aim] 使用已預熱的子程序（PID={aim_process.pid}）", flush=True)
                        else:
                            if vision_socket is not None:
                                vision_socket.close()
                            aim_process, vision_socket = prewarm_aim_process()
                        aim_exit_reported = False
                        if aim_process is None:
                            auto_complete = True
                        else:
                            send_aim_command(aim_process, "VISION_ON")
                        if not auto_complete:
                            if lidar is not None:
                                lidar.join()
                            lidar = LidarAvoidance()
                            lidar.start()
                        next_auto_status_at = 0.0
                    current_mode = new_mode
                    print(f"[Mode Switch] 已切換至 {current_mode}", flush=True)
                    if current_mode == "M" and aim_process is None:
                        # Warm the next auto session up while driving manually.
                        aim_process, vision_socket = prewarm_aim_process()

                if current_mode == "M":
                    # 手動模式：使用遙控器 yaw/pitch 控制砲台方向，
                    # fire 按鈕每次「由放開變按下」觸發一發射擊循環。
                    manual.handle_packet(packet)
                    time.sleep(0.02)
                    continue

            else:
                uart.report_if_silent()

            # 自動瞄準子程序意外結束（例如機構故障）時，不要默默停在原地：
            # 停止底盤與定位、釋放 ZED，並提示撥回手動模式。
            if (current_mode == "A" and aim_process is not None and not aim_exit_reported
                    and aim_process.poll() is not None):
                aim_exit_reported = True
                auto_complete = True
                awaiting_phase = False
                if lidar is not None:
                    lidar.stop()
                uart.send_packet(
                    mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                )
                if aim_process.returncode == 0:
                    print("[Aim] 自動瞄準子程序已結束，底盤已送出停止；撥回手動模式即可繼續手動操作。",
                          flush=True)
                else:
                    print(
                        f"[Aim ERROR] 自動瞄準子程序異常結束（exit={aim_process.returncode}），"
                        "自動定位與射擊已停止，底盤已送出停止。請看上方錯誤訊息；"
                        "撥回手動模式即可繼續手動操作。",
                        flush=True,
                    )

            if current_mode == "A" and not auto_complete:
                now = time.monotonic()
                latest_vy = read_positioning_vy(vision_socket, vision_valid_after, latest_vy)
                if packet is not None and packet.get("mode") == "A":
                    swa_enabled = packet.get("auto_start") == 1
                    if not swa_enabled:
                        if auto_started:
                            print("[Auto Drive] SWA 關閉，暫停定位", flush=True)
                            locked_vy = None
                            latest_vy = None
                            vision_valid_after = time.monotonic()
                            send_aim_command(aim_process, "VISION_ON")
                        auto_started = False
                        vy_recorded_before_release = False
                        strafe_until = None
                        vision_wait_until = None
                        positioning_moved = False
                        brake_until = None
                        avoid_until = None
                    elif not auto_started:
                        locked_vy = None if latest_vy is None else latest_vy[1]
                        send_aim_command(aim_process, "VISION_OFF")
                        if not aim_prepared and aim_process is not None and aim_process.poll() is None:
                            if manual is not None:
                                manual.close()
                                manual = None
                            aim_prepared = send_aim_command(aim_process, "PREPARE_MOTORS")
                            if aim_prepared:
                                print("[Aim] SWA 已放行，背景準備 DM／VESC", flush=True)
                        auto_started = True
                        vy_recorded_before_release = locked_vy is not None
                        next_auto_status_at = 0.0
                        print(
                            f"[Auto Drive] SWA 已放行，開始自動定位，Vision vy={locked_vy}",
                            flush=True,
                        )

                if not auto_started:
                    locked_vy = None if latest_vy is None else latest_vy[1]
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

            # SWA already authorized motor preparation for this auto session.
            # The next A packet supplies only the half selection.
            if awaiting_phase and packet is not None and packet.get("mode") == "A":
                awaiting_phase = False
                match_phase = "lower" if packet.get("fire") == 1 else "upper"
                if aim_prepared:
                    last_auto_reload = packet.get("reload")
                    aim_started = send_aim_command(aim_process, f"START {match_phase}")
                    if aim_started:
                        print(f"[Aim] 已送出正式投球及半場設定（{match_phase}）", flush=True)
                else:
                    print("[Aim] 預熱子程序已結束，未交接機構", flush=True)

            if (
                current_mode == "A" and packet is not None and packet.get("mode") == "A"
                and aim_started
            ):
                reload_value = packet.get("reload")
                if reload_value in (0, 1):
                    if last_auto_reload == 0 and reload_value == 1:
                        if send_aim_command(aim_process, "RELOAD"):
                            print("[Aim] 已轉送遙控器裝填確認", flush=True)
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
                positioning_succeeded = False
                completion_reason = "已抵達目標距離並完成橫移"

                if blocked and positioning_moved and brake_until is None:
                    brake_until = now + OCCLUSION_BRAKE_SECONDS
                    print("[Auto Drive] 移動中遇到遮擋，緊急停止", flush=True)

                if brake_until is not None:
                    lidar_status = "遮擋後緊急停止"
                    if now >= brake_until:
                        if not vy_recorded_before_release or not locked_vy:
                            auto_complete = True
                            completion_reason = "遮擋後停止，放行前沒有有效橫移方向"
                        elif not valid_distance:
                            # 距離不明時停車，等待下一筆有效回波。
                            lidar_status = "遮擋後等待有效 LiDAR 距離"
                        elif not blocked:
                            auto_complete = True
                            completion_reason = "LiDAR 前方已清空，停止避障橫移"
                        elif avoid_until is not None and now >= avoid_until:
                            lidar_status = "避障橫移達時限，停車等待前方清空"
                        else:
                            if avoid_until is None:
                                avoid_until = now + MAX_STRAFE_SECONDS
                            final_vy = -OCCLUSION_STRAFE_VY if locked_vy < 0 else OCCLUSION_STRAFE_VY
                            lidar_status = "遮擋後橫移，等待 LiDAR 前方清空"
                elif (
                    vision_wait_until is not None
                    and locked_vy is None
                    and now >= vision_wait_until
                ):
                    auto_complete = True
                    completion_reason = "等待視覺橫移方向 10 秒逾時"
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
                            vision_valid_after = time.monotonic()
                            latest_vy = None
                            send_aim_command(aim_process, "VISION_ON")
                        if latest_vy is not None:
                            locked_vy = latest_vy[1]
                            send_aim_command(aim_process, "VISION_OFF")
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
                            positioning_succeeded = front_distance_m < AUTO_TARGET_DISTANCE_M
                            if auto_complete and not positioning_succeeded:
                                completion_reason = "橫移後未能確認仍在目標距離內"
                    positioning_moved = positioning_moved or bool(final_vx or final_vy)

                if auto_complete:
                    final_vx, final_vy, final_wz = 0, 0, 0
                    yaw, pitch = 50, 50
                    fire_cmd, fire_speed = 0, 0

                # 自動模式的底盤前後方向與 Xavier 的正負號相反。
                tx_vx = -final_vx
                tx_vy = final_vy
                if vy_recorded_before_release and brake_until is None:
                    tx_vy = max(-100, min(100, round(final_vy * 1.5)))
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
                    lidar.stop()
                    print("[LiDAR] 定位結束，關閉點雲、IMU 與心跳", flush=True)
                    print(
                        f"[Auto Drive] {distance_text}，{completion_reason}，底盤已送出停止",
                        flush=True,
                    )
                    if not positioning_succeeded:
                        print("[Aim] 定位未成功，取消本輪投球；請切回 M 模式後重試", flush=True)
                        if aim_process is not None:
                            aim_process.stdin.close()
                    elif aim_prepared and send_aim_command(aim_process, "VISION_OFF"):
                        awaiting_phase = True
                        print("[Aim] 沿用 ZED，等待下一筆 A 模式封包選擇半場", flush=True)
                    else:
                        print("[Aim] 投球子程序未完成機構交接", flush=True)

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
        if lidar is not None:
            lidar.stop()
            if lidar.ident is not None:
                lidar.join()
        try:
            stop_aim_process(aim_process)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[Aim] 關閉時發生錯誤：{exc}", flush=True)
        if vision_socket is not None:
            vision_socket.close()
        try:
            if manual is not None:
                manual.close()
        except Exception as exc:
            print("[Manual] 關閉控制器時發生錯誤：", exc)
        uart.close()


if __name__ == "__main__":
    main()

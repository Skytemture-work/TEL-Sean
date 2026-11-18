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


STARTUP_MODE_TIMEOUT_S = 1.0
AUTO_STATUS_INTERVAL_SECONDS = 1.0
AUTO_TARGET_DISTANCE_M = 2.80
AUTO_FORWARD_VX = 40
OCCLUSION_STRAFE_VY = 15
MAX_STRAFE_SECONDS = 1.0
VISION_WAIT_SECONDS = 10.0
# After a reload the robot sits in the start zone again; the remote's reload
# confirmation is the departure.  Wait this long for a fresh vy first.
REPOSITION_VY_WAIT_S = 3.0
# Leaving auto needs M continuously for this long: one stray M packet (a
# receiver glitch, the ESP32 rebooting after the e-stop) must not end the
# match and throw away its state.
MODE_TO_MANUAL_CONFIRM_S = 0.5
# Team rule: from the reload until the shooting program ends, auto mode is
# locked (the e-stop reboots the ESP32, which may report M for a moment).
# Emergency exit: hold SWA at M this long.
MODE_LOCK_ESCAPE_S = 3.0
# The ESP32 reboots during the upper-half reload.  For this long after the
# reload is confirmed, an M in the packet's mode field is ignored (treated as
# the current mode); every other field -- reload included -- is still read.
# 'X' (no receiver) is left alone: it already keeps the mode, and its fields
# are placeholders (reload reads 0) that must never count as a switch edge.
RELOAD_MODE_MASK_S = 10.0
NORMAL_STRAFE_DURATION_SCALE = 1.5
# Upper half: after the first magazine (RELOAD_WAIT) drive back to the start
# zone by replaying the departure's chassis commands backwards (open loop).
RETURN_TO_START_ENABLED = True
# The ESP32 reports every ~20 ms.  Silence this long, or any packet that is not
# mode A (the e-stop reboots it: silence, then 'X'), aborts the return so the
# chassis never resumes driving after the power comes back.
RETURN_PACKET_TIMEOUT_S = 0.3
# The ESP32 stops the chassis 500 ms after the last command: a longer gap in
# the recording did not move the robot any further.
ESP32_COMMAND_TIMEOUT_S = 0.5
OCCLUSION_BRAKE_SECONDS = 0.15


AUTO_FIRE_WEB_PORT = 8765
AIM_SCRIPT = Path(__file__).resolve().parent / "turret_control" / "run_board_real.py"


class _SafeStream:
    """stdout/stderr that never raises: with `| tee`, Ctrl+C also kills tee,
    and a print hitting the broken pipe must not abort the homing shutdown."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except (BrokenPipeError, OSError, ValueError):
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


# Keep the existing stdin channel for handoff and reload commands.
AIM_ENTRY = """
import queue
import socket
import sys
import threading


class _SafeStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except (BrokenPipeError, OSError, ValueError):
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


# A dead `| tee` must not break the homing shutdown halfway.
sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)
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

def publish_event(name):
    try:
        vision_socket.send(f"EVT {name}".encode("ascii"))
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
        elif command in ("PREPARE_MOTORS", "VISION_ON", "VISION_OFF", "START upper", "START lower",
                         "POSITIONED", "PHASE upper", "PHASE lower") or command.startswith("DEPART "):
            handoff.put(command)
    handoff.put(None)

threading.Thread(target=read_commands, daemon=True).start()
with vision_socket:
    main(sys.argv[2:], reload_event=reload_event, next_handoff_command=next_command,
         positioning_callback=publish_vy, event_callback=publish_event)
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


def read_positioning_vy(vision_socket, valid_after, latest, events=None):
    """Newest fresh (captured_at, vy); aim events ("EVT name") go to ``events``."""
    if vision_socket is None:
        return None
    while True:
        try:
            parts = vision_socket.recv(128).decode("ascii").split()
        except BlockingIOError:
            break
        if len(parts) == 2 and parts[0] == "EVT":
            if events is not None:
                events.append(parts[1])
            continue
        if len(parts) != 2:
            continue
        latest = float(parts[0]), int(parts[1])
    if latest is not None:
        captured_at, _ = latest
        if captured_at >= valid_after and time.monotonic() - captured_at <= config.AUTO_VISION_MAX_AGE_S:
            return latest
    return None


def build_return_path(samples, max_gap_s=ESP32_COMMAND_TIMEOUT_S):
    """Reverse a departure recording into return segments.

    ``samples`` are ``(time, vx, vy, wz)`` as sent to the ESP32, in order.
    Each command lasted until the next sample (at most ``max_gap_s``).  Equal
    neighbours merge, stops are dropped, and the result runs backwards with
    every component negated: ``[(duration_s, vx, vy, wz), ...]``.
    """
    segments = []
    for (t0, vx, vy, wz), (t1, *_rest) in zip(samples, samples[1:]):
        duration = min(max(0.0, t1 - t0), max_gap_s)
        if duration <= 0.0 or (vx, vy, wz) == (0, 0, 0):
            continue
        if segments and segments[-1][1:] == (vx, vy, wz):
            segments[-1] = (segments[-1][0] + duration, vx, vy, wz)
        else:
            segments.append((duration, vx, vy, wz))
    return [(duration, -vx, -vy, -wz) for duration, vx, vy, wz in reversed(segments)]


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


AIM_SHUTDOWN_TIMEOUT_S = 45.0  # pusher home + turret home (<= 20 s) + cameras


def stop_aim_process(process):
    """Ask the aim subprocess to shut down and wait for it to home everything.

    SIGINT runs its normal shutdown: flywheels off, ID1 pusher home, both DM
    axes home along the safe path, then disable.  Wait long enough for that.
    """
    if process is None or process.poll() is not None:
        return
    if not getattr(process, "tel_stop_sent", False):
        # Only once: a second SIGINT would interrupt the homing in progress.
        print("[Aim] 通知自動瞄準子程序結束：停飛輪、推球回原點、雙軸回 home…", flush=True)
        process.send_signal(signal.SIGINT)
        process.tel_stop_sent = True
    process.wait(timeout=AIM_SHUTDOWN_TIMEOUT_S)
    print(f"[Aim] 子程序已結束（exit={process.returncode}）", flush=True)


def main():
    sys.stdout = _SafeStream(sys.stdout)
    sys.stderr = _SafeStream(sys.stderr)
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
        if uart.ser is None or not uart.ser.is_open:
            raise RuntimeError("Xavier UART 未開啟，停止啟動")
        print("[System] 開機檢查 DM、MKS CAN 與雙 VESC", flush=True)
        manual = ManualController(uart=uart)
        manual.check_health()
        current_mode = None
        startup_mode_deadline = time.monotonic() + STARTUP_MODE_TIMEOUT_S
        awaiting_phase = False
        aim_started = False
        aim_prepared = False
        model_turn_started = False
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
        repositioning = False
        reposition_release_at = 0.0
        manual_request_since = None
        mode_locked = False
        mode_mask_until = 0.0
        release_phase = None  # half switch latched at the VRA release
        depart_at = None  # time.monotonic() of the VRA release: the match clock
        drive_log = []  # departure commands (time, vx, vy, wz) for the return
        return_path = None  # active return: [(duration, vx, vy, wz), ...]
        return_requested = False  # the return runs once per match
        return_arrived = False
        return_started_at = 0.0
        last_a_packet_at = 0.0
        print("[System] 已啟動，等待 ESP32 第一筆有效封包...", flush=True)

        while True:
            if current_mode == "M" and manual is not None:
                manual.check_health()
            packet = uart.read_esp32_packet()
            raw_packet_mode = packet.get("mode") if packet is not None else None
            if raw_packet_mode == "A":
                last_a_packet_at = time.monotonic()

            if (packet is not None and current_mode == "A"
                    and time.monotonic() < mode_mask_until
                    and packet.get("mode") == "M"):
                # ESP32 rebooting after the reload: keep auto, read the rest.
                packet["mode"] = current_mode
            default_to_auto = (
                current_mode is None
                and raw_packet_mode not in ("M", "A")
                and time.monotonic() >= startup_mode_deadline
            )
            if packet is not None or default_to_auto:
                mode_packet = packet if packet is not None else {}
                packet_mode = mode_packet.get("mode", "?")
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
                if default_to_auto:
                    new_mode = "A"
                    print("[System] 等待 ESP32 模式逾時，預設 A，等待放行。", flush=True)
                else:
                    new_mode = mode_packet.get("mode", current_mode)
                    if new_mode not in ("M", "A"):
                        new_mode = current_mode  # 'X' = ESP32 has no remote signal
                if current_mode == "A" and new_mode == "M":
                    mode_now = time.monotonic()
                    confirm_s = MODE_LOCK_ESCAPE_S if mode_locked else MODE_TO_MANUAL_CONFIRM_S
                    if manual_request_since is None:
                        manual_request_since = mode_now
                        print(("[Mode Switch] 自動模式鎖定中（裝填後到射擊結束）：SWA 保持 M "
                               if mode_locked else "[Mode Switch] 收到 M，持續 ")
                              + f"{confirm_s:g} 秒才切回手動", flush=True)
                    if mode_now - manual_request_since < confirm_s:
                        new_mode = current_mode
                    elif (model_turn_started and manual is not None
                          and not manual.wait_signal_turn(timeout=0)):
                        # The magazine still owns the MKS hardware.
                        new_mode = current_mode
                else:
                    manual_request_since = None

                if new_mode != current_mode or (new_mode == "M" and manual is None):
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

                    model_turn_started = False
                    auto_complete = False
                    auto_started = False
                    awaiting_phase = False
                    repositioning = False
                    mode_locked = False
                    mode_mask_until = 0.0
                    release_phase = None
                    depart_at = None
                    drive_log = []
                    return_path = None
                    return_requested = False
                    return_arrived = False
                    locked_vy = None
                    latest_vy = None
                    vision_valid_after = time.monotonic()
                    vy_recorded_before_release = False
                    strafe_until = None
                    vision_wait_until = None
                    positioning_moved = False
                    brake_until = None
                    avoid_until = None

                    if new_mode == "M" and manual is None:
                        print("[System] 正在初始化手動控制器（DM / MKS CAN / VESC）", flush=True)
                        manual = ManualController(uart=uart)
                        print("[System] 手動控制器初始化完成", flush=True)
                    if new_mode == "A":
                        uart.send_packet(
                            mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                        )
                        try:
                            aim_process, vision_socket = start_aim_process()
                        except OSError as exc:
                            print(f"[Aim] 無法啟動自動瞄準子程序：{exc}", flush=True)
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
                    initial_mode = current_mode is None
                    if initial_mode:
                        # Keep MODEL_READY queued until the startup indicator finishes.
                        print("[System] 設備初始化成功，彈藥倉轉兩格", flush=True)
                        manual.turn_startup_indicator()
                    if manual is not None:
                        manual.on_mode_change(
                            new_mode, fire=mode_packet.get("fire", 0), reload=mode_packet.get("reload")
                        )

                    current_mode = new_mode
                    print(f"[Mode Switch] 已切換至 {current_mode}", flush=True)
                    # Handoff can block on motors. Read a new packet before acting.
                    continue

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
                if mode_locked:
                    mode_locked = False
                    print("[Mode Switch] 射擊程式已結束，解除自動模式鎖定", flush=True)
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

            if current_mode == "A" and auto_complete and aim_started and not repositioning:
                aim_events = []
                latest_vy = read_positioning_vy(vision_socket, vision_valid_after, latest_vy,
                                                aim_events)
                if "RELOAD_WAIT" in aim_events and not mode_locked:
                    mode_locked = True
                    print("[Mode Switch] 進入裝填：自動模式鎖定到射擊程式結束"
                          f"（緊急時 SWA 保持 M {MODE_LOCK_ESCAPE_S:g} 秒可切回手動）", flush=True)
                    if last_auto_reload == 1:
                        # Only 0 -> 1 confirms.  Flipped down and up again while
                        # the ESP32 reboots, both edges would be missed.
                        print("[Aim] RELOAD 開關目前在 1：請先撥回 0，裝填好再撥到 1 出發",
                              flush=True)
                    if return_arrived:
                        print("[Return] 機構已停妥且車已在出發區：可以按急停裝填", flush=True)
                if (("RETURN_START" in aim_events or "RELOAD_WAIT" in aim_events)
                        and not return_requested):
                    # The last ball is out: drive back while the pusher homes
                    # and the flywheels stop.
                    return_requested = True
                    path = build_return_path(drive_log) if RETURN_TO_START_ENABLED else []
                    if path:
                        return_path = path
                        return_started_at = time.monotonic()
                        print(f"[Return] 依發車軌跡反向退回出發區（開環，{len(path)} 段，約 "
                              f"{sum(seg[0] for seg in path):.1f} 秒），推球回原點與飛輪停轉同時進行；"
                              "到達前請勿靠近車子。按急停、VRA 關閉或 ESP32 中斷都會立即停止回程。",
                              flush=True)
                    else:
                        print("[Return] 沒有可用的發車軌跡，請人工把車移回出發區裝填", flush=True)
                if "REPOSITION" in aim_events:
                    if return_path is not None:
                        return_path = None
                        uart.send_packet(mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50,
                                         fire=0, speed=0)
                        print("[Return] 已確認裝填：結束回程", flush=True)
                    # Reload done in the start zone: the reload confirmation is
                    # the departure.  Same positioning as at the match start.
                    repositioning = True
                    mode_mask_until = time.monotonic() + RELOAD_MODE_MASK_S
                    print(f"[Mode Switch] 裝填完成：{RELOAD_MODE_MASK_S:g} 秒內忽略封包模式"
                          "（ESP32 重啟），reload 等其他欄位照常讀取", flush=True)
                    reposition_release_at = time.monotonic() + REPOSITION_VY_WAIT_S
                    auto_complete = False
                    auto_started = False
                    locked_vy = None
                    latest_vy = None
                    vision_valid_after = time.monotonic()
                    vy_recorded_before_release = False
                    strafe_until = None
                    vision_wait_until = None
                    positioning_moved = False
                    brake_until = None
                    avoid_until = None
                    if lidar is not None:
                        lidar.join()
                    lidar = LidarAvoidance()
                    lidar.start()
                    next_auto_status_at = 0.0
                    print(f"[Auto Drive] 裝填完成：重新出發定位（先等視覺 vy，最多 "
                          f"{REPOSITION_VY_WAIT_S:g} 秒）", flush=True)

            if current_mode == "A" and not auto_complete:
                now = time.monotonic()
                pre_events = []
                latest_vy = read_positioning_vy(vision_socket, vision_valid_after, latest_vy,
                                                pre_events)
                if "MODEL_READY" in pre_events and manual is not None and not model_turn_started:
                    model_turn_started = manual.signal_model_ready()
                    if not model_turn_started:
                        raise RuntimeError("無法啟動 YOLO 就緒提示轉格")
                model_turn_ready = manual is None or (
                    model_turn_started and manual.wait_signal_turn(timeout=0)
                )
                if packet is not None and packet.get("mode") == "A":
                    # A re-departure after the reload needs no VRA release.
                    swa_enabled = packet.get("auto_start") == 1 or repositioning
                    reposition_hold = (repositioning and latest_vy is None
                                       and now < reposition_release_at)
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
                    elif not auto_started and not reposition_hold and model_turn_ready:
                        if manual is not None:
                            manual.close()
                            manual = None
                            # Housekeeping can block. Recheck VRA before driving or timing the match.
                            continue
                        # Keep the last seen vy: a single stale sample at the
                        # moment of release must not drop the strafe direction.
                        if latest_vy is not None:
                            locked_vy = latest_vy[1]
                        send_aim_command(aim_process, "VISION_OFF")
                        if not repositioning:
                            # Latch the half here: pre-aim and START share it.
                            release_phase = "lower" if packet.get("fire") == 1 else "upper"
                        if depart_at is None and not repositioning:
                            # The match clock runs from the first VRA release
                            # (the departure); both processes share CLOCK_MONOTONIC.
                            depart_at = now
                            send_aim_command(aim_process, f"DEPART {depart_at:.3f}")
                            print("[Match] VRA 發車：比賽計時開始", flush=True)
                        if not aim_prepared and aim_process is not None and aim_process.poll() is None:
                            # The half switch is already set on release: tell the aim
                            # process now, so the turret pre-aims at the right first
                            # hole (upper H11, lower H01) while the chassis drives.
                            send_aim_command(aim_process, f"PHASE {release_phase}")
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
                    # Sticky like the old VisionTracker: only a fresh sample
                    # updates it; SWA off / a mode switch clears it.
                    if latest_vy is not None:
                        locked_vy = latest_vy[1]
                    uart.send_packet(
                        mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0
                    )
                    if now >= next_auto_status_at:
                        next_auto_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                        if not model_turn_ready:
                            print("[Auto Drive] 等待 YOLO 就緒與提示轉格完成", flush=True)
                        elif locked_vy is None:
                            print("[Auto Drive] 等待 auto_start=1", flush=True)
                        else:
                            print(
                                f"[Auto Drive] 等待 auto_start=1｜放行前已偵測到靶面，保留 vy={locked_vy}",
                                flush=True,
                            )
                    time.sleep(0.02)
                    continue

            # SWA already authorized motor preparation for this auto session.
            # The half was latched at the SWA release (SWD moved during
            # positioning is ignored); the next A packet sets the reload baseline.
            if awaiting_phase and packet is not None and packet.get("mode") == "A":
                awaiting_phase = False
                match_phase = release_phase or (
                    "lower" if packet.get("fire") == 1 else "upper")
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
                    if mode_locked and reload_value != last_auto_reload:
                        print(f"[Aim] RELOAD 開關 {last_auto_reload} → {reload_value}", flush=True)
                    if last_auto_reload == 0 and reload_value == 1:
                        if send_aim_command(aim_process, "RELOAD"):
                            print("[Aim] 已轉送遙控器裝填確認", flush=True)
                    last_auto_reload = reload_value

            if current_mode == "A" and return_path is not None:
                now = time.monotonic()
                abort = None
                if raw_packet_mode is not None and raw_packet_mode != "A":
                    abort = f"ESP32 回報模式 {raw_packet_mode!r}（急停或重啟）"
                elif packet is not None and packet.get("auto_start") == 0:
                    abort = "VRA 已關閉"
                elif now - last_a_packet_at > RETURN_PACKET_TIMEOUT_S:
                    abort = "ESP32 沒有回應（可能已按急停）"
                elapsed = now - return_started_at
                command = None
                for duration, seg_vx, seg_vy, seg_wz in return_path:
                    if elapsed < duration:
                        command = (seg_vx, seg_vy, seg_wz)
                        break
                    elapsed -= duration
                if abort is not None or command is None:
                    return_path = None
                    uart.send_packet(mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50,
                                     fire=0, speed=0)
                    if abort is not None:
                        print(f"[Return] 中止回程：{abort}；底盤停止，請人工移回出發區"
                              + ("" if mode_locked else "；推球回原點、飛輪停轉可能還沒完成，"
                                 "等 [Aim] 機構停妥後再按急停"), flush=True)
                    else:
                        return_arrived = True
                        print("[Return] 已退回出發區（開環，位置可能有誤差）："
                              + ("可以按急停裝填" if mode_locked
                                 else "等 [Aim] 機構停妥（飛輪停、推球回原點）後再按急停"),
                              flush=True)
                else:
                    uart.send_packet(mode="A", vx=command[0], vy=command[1], wz=command[2],
                                     yaw=50, pitch=50, fire=0, speed=0)

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

                if (
                    brake_until is not None
                    and now >= brake_until
                    and vy_recorded_before_release
                    and locked_vy
                    and valid_distance
                    and not blocked
                ):
                    brake_until = None
                    avoid_until = None
                    # 避障後只沿 X 軸續行，原本的視覺橫移不再送出。
                    strafe_until = now
                    next_auto_status_at = 0.0
                    print("[Auto Drive] LiDAR 前方已清空，恢復目標距離定位", flush=True)

                if brake_until is not None:
                    lidar_status = "遮擋後緊急停止"
                    if now >= brake_until:
                        if not vy_recorded_before_release or not locked_vy:
                            auto_complete = True
                            completion_reason = "遮擋後停止，放行前沒有有效橫移方向"
                        elif not valid_distance:
                            # 距離不明時停車，等待下一筆有效回波。
                            lidar_status = "遮擋後等待有效 LiDAR 距離"
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
                if not repositioning:
                    # The match departure (not the re-departure after the
                    # reload): replayed backwards to return for the reload.
                    drive_log.append((now, tx_vx, tx_vy, final_wz))

                if auto_complete:
                    lidar.stop()
                    print("[LiDAR] 定位結束，關閉點雲、IMU 與心跳", flush=True)
                    print(
                        f"[Auto Drive] {distance_text}，{completion_reason}，底盤已送出停止",
                        flush=True,
                    )
                    if not positioning_succeeded:
                        # Never give up the round here: a cancelled round scores
                        # zero, while the turret (+/-28..34.5 deg) still reaches
                        # most holes from an imperfect stop, and holes it cannot
                        # reach are skipped after 3 s by the aim process.
                        print(f"[Aim] 定位不完整（{completion_reason}），仍照常開始投球；"
                              "打不到的孔會自動跳過", flush=True)
                    if repositioning:
                        repositioning = False
                        send_aim_command(aim_process, "VISION_OFF")
                        if send_aim_command(aim_process, "POSITIONED"):
                            print("[Aim] 重新定位完成，通知繼續射擊", flush=True)
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

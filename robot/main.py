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
from manual import ManualController


AUTO_STATUS_INTERVAL_SECONDS = 1.0
AUTO_TARGET_DISTANCE_M = 2.80
AUTO_FORWARD_VX = 40
# 移動中遇到遮擋：急停後直線後退，直到前方淨空 OCCLUSION_CLEARANCE_M（車尾需有空間）。
OCCLUSION_REVERSE_VX = 15
OCCLUSION_REVERSE_TIMEOUT_S = 3.0
OCCLUSION_CLEARANCE_M = 1.0
MAX_STRAFE_SECONDS = 1.0
VISION_WAIT_SECONDS = 10.0
# After a reload the robot sits in the start zone again; the remote's reload
# confirmation is the departure.  Wait this long for a fresh vy first.
REPOSITION_VY_WAIT_S = 3.0
# The remote's SWA only decides who drives the chassis: in M the driver (the
# ESP32 drives it directly), in A this program.  The turret always aims and
# fires automatically and the match keeps running across switches.
# Leaving auto needs M continuously for this long: one stray M packet (a
# receiver glitch) must not stop an automatic drive.
MODE_TO_MANUAL_CONFIRM_S = 0.5
# The ESP32 reboots after an emergency stop and may report M for a moment.
# For RELOAD_MODE_MASK_S after a re-departure from the start zone, switching
# to manual needs SWA held at M for MODE_LOCK_ESCAPE_S instead.
RELOAD_MODE_MASK_S = 10.0
MODE_LOCK_ESCAPE_S = 3.0
# After a gap in the ESP32 packets (it reboots after an e-stop) its reload
# field reads 0 until the receiver links.  With the switch left at 1 that
# would look like a 0 -> 1 confirmation the moment the remote links, and the
# robot would set off in the middle of a repair.  For this long after the
# packets come back the reload field only sets the baseline.
ESP32_GAP_S = 0.5
RELOAD_ARM_AFTER_RECONNECT_S = 3.0
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


class _DriveRun:
    """One automatic positioning drive from the start zone.

    ``kind`` "initial" is the VRA departure; "reposition" a re-departure after
    the reload or an emergency stop (the reload switch is the departure).
    """

    def __init__(self, kind, now):
        self.kind = kind
        self.started = False  # VRA released / the vy wait after a re-departure is over
        self.release_at = now + REPOSITION_VY_WAIT_S
        self.locked_vy = None
        self.vy_recorded_before_release = False
        self.strafe_until = None
        self.vision_wait_until = None
        self.moved = False
        self.brake_until = None
        self.recovery_hold_reason = None  # after the reverse: stay stopped, why

    def release(self, vy):
        self.started = True
        self.locked_vy = vy
        self.vy_recorded_before_release = vy is not None

    def pause(self):
        self.__init__(self.kind, time.monotonic())


def _lidar_running(lidar):
    return lidar is not None and lidar.is_alive() and not lidar._stop_requested.is_set()


def _restart_lidar(lidar):
    if lidar is not None:
        lidar.stop()
        if lidar.ident is not None:
            lidar.join(timeout=2.0)
    lidar = LidarAvoidance()
    lidar.start()
    return lidar


def main():
    sys.stdout = _SafeStream(sys.stdout)
    sys.stderr = _SafeStream(sys.stderr)
    print("====================================================")
    print("TEL-Sean 主控系統：自動發車；手動模式只接手底盤，砲塔一律自動瞄準射擊")
    print("====================================================")

    uart = UARTController(config.SERIAL_PORT, config.BAUDRATE)
    vision_socket = None
    aim_process = None
    lidar = None
    # 開機檢查與彈藥倉提示轉格用；發車時關閉，之後機構全由自動瞄準子程序負責。
    manual = None
    try:
        if uart.ser is None or not uart.ser.is_open:
            raise RuntimeError("Xavier UART 未開啟，停止啟動")
        print("[System] 開機檢查 DM、MKS CAN 與雙 VESC", flush=True)
        manual = ManualController(uart=uart)
        manual.check_health()
        # The aim process loads the model while the magazine turns.
        try:
            aim_process, vision_socket = start_aim_process()
        except OSError as exc:
            print(f"[Aim] 無法啟動自動瞄準子程序：{exc}", flush=True)
        send_aim_command(aim_process, "VISION_ON")
        print("[System] 設備初始化成功，彈藥倉轉兩格", flush=True)
        manual.turn_startup_indicator()
        # The turret is never driven by the remote any more: hand the startup
        # controller over at once (VESC off, ID1 home, DM home and released).
        manual.on_mode_change("A")
        lidar = LidarAvoidance()
        lidar.start()

        current_mode = None  # remote SWA (debounced): who drives the chassis
        last_received_mode = None
        manual_request_since = None
        mode_mask_until = 0.0
        model_ready = False
        model_turn_started = False
        signal_error_reported = False
        departed = False  # VRA released: the match runs, the aim process owns the turret
        aim_prepared = False
        aim_started = False  # START sent
        aim_exit_reported = False
        release_phase = None  # half switch latched at the VRA release
        depart_at = None  # time.monotonic() of the VRA release: the match clock
        run = None  # active automatic positioning drive (_DriveRun)
        estop = False  # the aim process reported a power loss; wait for REPOSITION
        reload_waiting = False
        last_reload = None
        latest_vy = None
        pre_vy = None  # sticky vy seen before the departure
        vision_valid_after = 0.0
        next_status_at = 0.0
        drive_log = []  # departure commands (time, vx, vy, wz) for the return
        drive_log_valid = False  # no manual driving since the departure
        return_path = None  # active return: [(duration, vx, vy, wz), ...]
        return_requested = False  # the return runs once per match
        return_arrived = False
        return_started_at = 0.0
        last_a_packet_at = 0.0
        last_packet_at = None
        reload_armed_at = 0.0
        reload_arm_announced = True

        def stop_chassis():
            uart.send_packet(mode="A", vx=0, vy=0, wz=0, yaw=50, pitch=50, fire=0, speed=0)

        def finish_positioning():
            """The chassis is at the shooting spot, or the driver has it: shoot."""
            nonlocal aim_started
            send_aim_command(aim_process, "VISION_OFF")
            if aim_started:
                if send_aim_command(aim_process, "POSITIONED"):
                    print("[Aim] 定位結束，通知繼續射擊", flush=True)
            elif aim_prepared:
                aim_started = send_aim_command(aim_process, f"START {release_phase}")
                if aim_started:
                    print(f"[Aim] 已送出正式投球及半場設定（{release_phase}）", flush=True)
            else:
                print("[Aim] 投球子程序未完成機構交接", flush=True)

        def end_run():
            nonlocal run
            run = None
            if lidar is not None:
                lidar.stop()
            stop_chassis()

        def switch_mode(new_mode):
            nonlocal current_mode, manual_request_since, return_path
            manual_request_since = None
            current_mode = new_mode
            print("\n[Mode Switch] 已切換至{}".format("自動模式" if new_mode == "A" else "手動模式"),
                  flush=True)
            stop_chassis()
            if new_mode == "M":
                if return_path is not None:
                    return_path = None
                    print("[Return] 手動接手：中止自動回程，請自行開回出發區", flush=True)
                if run is not None:
                    end_run()
                    print("[Mode Switch] 手動接手底盤：結束自動定位，砲塔照常自動瞄準射擊", flush=True)
                    finish_positioning()
                elif departed:
                    print("[Mode Switch] 手動控制底盤，砲塔照常自動瞄準射擊", flush=True)
                else:
                    print("[Mode Switch] 手動控制底盤；發車請切回自動模式並轉 VRA", flush=True)
            elif departed:
                print("[Mode Switch] 自動模式：底盤停住；下一次出發（裝填或急停後 RELOAD）才自動定位",
                      flush=True)

        def signal_turn_done():
            nonlocal signal_error_reported
            if manual is None or not model_turn_started:
                return True
            try:
                return manual.wait_signal_turn(timeout=0)
            except RuntimeError as exc:
                if not signal_error_reported:
                    signal_error_reported = True
                    print(f"[System] YOLO 就緒提示轉格失敗（{exc}），不影響發車", flush=True)
                return True

        print("[System] 已啟動，等待 ESP32 封包與 YOLO 就緒...", flush=True)

        while True:
            now = time.monotonic()
            packet = uart.read_esp32_packet()
            raw_mode = packet.get("mode") if packet is not None else None
            if raw_mode == "A":
                last_a_packet_at = now
            if packet is not None:
                if last_packet_at is not None and now - last_packet_at > ESP32_GAP_S:
                    reload_armed_at = now + RELOAD_ARM_AFTER_RECONNECT_S
                    reload_arm_announced = False
                    print(f"[ESP32] 封包中斷 {now - last_packet_at:.1f} 秒後恢復（可能是急停重啟）："
                          f"{RELOAD_ARM_AFTER_RECONNECT_S:g} 秒內 RELOAD 開關只記錄位置、不算出發",
                          flush=True)
                last_packet_at = now
            if packet is None:
                uart.report_if_silent()
            elif raw_mode != last_received_mode:
                print(f"[ESP32 MODE] 收到 {raw_mode!r}，目前模式 {current_mode}", flush=True)
                last_received_mode = raw_mode

            # ---- mode: only who drives the chassis ('X' = no remote, keep) ----
            if raw_mode in ("M", "A"):
                if current_mode is None:
                    current_mode = raw_mode
                    print("[Mode Switch] 目前為{}".format(
                        "自動模式" if raw_mode == "A" else "手動模式（發車請切回自動模式並轉 VRA）"),
                        flush=True)
                elif raw_mode == current_mode:
                    manual_request_since = None
                elif raw_mode == "A":
                    switch_mode("A")
                else:
                    confirm_s = (MODE_LOCK_ESCAPE_S if now < mode_mask_until
                                 else MODE_TO_MANUAL_CONFIRM_S)
                    if manual_request_since is None:
                        manual_request_since = now
                        print(f"[Mode Switch] 收到 M，持續 {confirm_s:g} 秒才切到手動", flush=True)
                    elif now - manual_request_since >= confirm_s:
                        switch_mode("M")
            if (departed and current_mode == "M" and raw_mode == "M"
                    and (packet.get("vx") or packet.get("vy") or packet.get("wz"))):
                # The driver moved the robot: the departure can no longer be replayed.
                drive_log_valid = False

            # ---- events from the aim process ----
            events = []
            latest_vy = read_positioning_vy(vision_socket, vision_valid_after, latest_vy, events)
            for event in events:
                if event == "MODEL_READY":
                    model_ready = True
                    if manual is not None and not model_turn_started:
                        model_turn_started = manual.signal_model_ready()
                        if not model_turn_started:
                            print("[System] YOLO 已就緒，但提示轉格未執行", flush=True)
                elif event == "POWER_LOST":
                    estop = True
                    reload_waiting = False
                    if run is not None:
                        end_run()
                        print("[E-STOP] 中止自動定位", flush=True)
                    if return_path is not None:
                        return_path = None
                        print("[E-STOP] 中止自動回程", flush=True)
                    stop_chassis()
                    print("[E-STOP] 馬達斷電：底盤停止。車拉回出發區、修好後，遙控器 RELOAD "
                          "撥 0 再撥 1，就從出發區重新出發，接著射下一孔。", flush=True)
                    if last_reload == 1:
                        print("[E-STOP] RELOAD 開關目前在 1：請先撥回 0，修好再撥到 1 出發", flush=True)
                elif event in ("RETURN_START", "RELOAD_WAIT"):
                    if event == "RELOAD_WAIT":
                        reload_waiting = True
                        if last_reload == 1:
                            # Only 0 -> 1 confirms.  Flipped down and up again while
                            # the ESP32 reboots, both edges would be missed.
                            print("[Aim] RELOAD 開關目前在 1：請先撥回 0，裝填好再撥到 1 出發",
                                  flush=True)
                        if return_arrived:
                            print("[Return] 機構已停妥且車已在出發區：可以按急停裝填", flush=True)
                    if not return_requested and run is None:
                        # The last ball is out: drive back while the pusher homes
                        # and the flywheels stop.
                        return_requested = True
                        path = (build_return_path(drive_log)
                                if RETURN_TO_START_ENABLED and drive_log_valid else [])
                        if current_mode != "A":
                            print("[Return] 手動模式：請自行把車開回出發區裝填", flush=True)
                        elif path:
                            return_path = path
                            return_started_at = now
                            print(f"[Return] 依發車軌跡反向退回出發區（開環，{len(path)} 段，約 "
                                  f"{sum(seg[0] for seg in path):.1f} 秒），推球回原點與飛輪停轉同時進行；"
                                  "到達前請勿靠近車子。按急停、VRA 關閉、切手動或 ESP32 中斷都會立即停止回程。",
                                  flush=True)
                        else:
                            print("[Return] 沒有可用的發車軌跡（或途中手動開過車），請把車移回出發區裝填",
                                  flush=True)
                elif event == "REPOSITION":
                    # Reload confirmed / power back after an e-stop: the robot is in
                    # the start zone and the reload switch is the departure.
                    estop = False
                    reload_waiting = False
                    if return_path is not None:
                        return_path = None
                        stop_chassis()
                        print("[Return] 已確認出發：結束回程", flush=True)
                    mode_mask_until = now + RELOAD_MODE_MASK_S
                    if current_mode == "A":
                        run = _DriveRun("reposition", now)
                        latest_vy = None
                        vision_valid_after = now
                        send_aim_command(aim_process, "VISION_ON")
                        lidar = _restart_lidar(lidar)
                        next_status_at = 0.0
                        print(f"[Auto Drive] 從出發區重新出發定位（先等視覺 vy，最多 "
                              f"{REPOSITION_VY_WAIT_S:g} 秒）；{RELOAD_MODE_MASK_S:g} 秒內切手動需 SWA "
                              f"保持 M {MODE_LOCK_ESCAPE_S:g} 秒", flush=True)
                    else:
                        print("[Mode] 手動模式：由駕駛開到定點，砲塔照常自動瞄準射擊", flush=True)
                        finish_positioning()

            # ---- the aim process ended (match over, or a fault it cannot recover) ----
            if aim_process is not None and not aim_exit_reported and aim_process.poll() is not None:
                aim_exit_reported = True
                estop = False
                return_path = None
                if run is not None:
                    end_run()
                stop_chassis()
                if aim_process.returncode == 0:
                    print("[Aim] 自動瞄準子程序已結束，底盤已停止；手動模式仍可開車。", flush=True)
                else:
                    print(f"[Aim ERROR] 自動瞄準子程序異常結束（exit={aim_process.returncode}），"
                          "定位與射擊已停止，底盤已停止。請看上方錯誤訊息；手動模式仍可開車。",
                          flush=True)

            # ---- reload switch: forwarded in both modes once the match runs ----
            if departed and packet is not None and raw_mode in ("M", "A"):
                reload_value = packet.get("reload")
                if reload_value in (0, 1):
                    armed = now >= reload_armed_at
                    if (estop or reload_waiting) and reload_value != last_reload:
                        print(f"[Aim] RELOAD 開關 {last_reload} → {reload_value}"
                              + ("" if armed else "（ESP32 剛恢復，只記錄位置）"), flush=True)
                    if armed and not reload_arm_announced:
                        reload_arm_announced = True
                        if (estop or reload_waiting) and reload_value == 1:
                            print("[Aim] RELOAD 開關目前在 1：請先撥回 0，好了再撥到 1 出發",
                                  flush=True)
                    if armed and last_reload == 0 and reload_value == 1:
                        if send_aim_command(aim_process, "RELOAD"):
                            print("[Aim] 已轉送遙控器 RELOAD", flush=True)
                    last_reload = reload_value

            # ---- before the departure: wait for VRA in auto mode ----
            if not departed:
                if latest_vy is not None:
                    pre_vy = latest_vy[1]  # sticky: only a fresh sample updates it
                stop_chassis()
                aim_alive = aim_process is not None and aim_process.poll() is None
                ready = model_ready and signal_turn_done() and aim_alive
                if (ready and current_mode == "A" and raw_mode == "A"
                        and packet.get("auto_start") == 1):
                    if manual is not None:
                        manual.close()
                        manual = None
                        # Housekeeping can block. Recheck VRA before driving or timing the match.
                        continue
                    if latest_vy is not None:
                        pre_vy = latest_vy[1]
                    send_aim_command(aim_process, "VISION_OFF")
                    # Latch the half here: pre-aim and START share it.
                    release_phase = "lower" if packet.get("fire") == 1 else "upper"
                    # The match clock runs from the VRA release (the departure);
                    # both processes share CLOCK_MONOTONIC.
                    depart_at = now
                    send_aim_command(aim_process, f"DEPART {depart_at:.3f}")
                    # Tell the half now, so the turret pre-aims at the right first
                    # hole (upper H11, lower H01) while the chassis drives.
                    send_aim_command(aim_process, f"PHASE {release_phase}")
                    aim_prepared = send_aim_command(aim_process, "PREPARE_MOTORS")
                    departed = True
                    last_reload = packet.get("reload")
                    drive_log = []
                    drive_log_valid = True
                    run = _DriveRun("initial", now)
                    run.release(pre_vy)
                    if not _lidar_running(lidar):
                        lidar = _restart_lidar(lidar)
                    next_status_at = 0.0
                    print("[Match] VRA 發車：比賽計時開始", flush=True)
                    print(f"[Auto Drive] VRA 已放行，開始自動定位，Vision vy={pre_vy}", flush=True)
                elif current_mode == "A" and now >= next_status_at:
                    next_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                    if not aim_alive:
                        print("[Auto Drive] 自動瞄準子程序未執行，無法發車", flush=True)
                    elif not ready:
                        print("[Auto Drive] 等待 YOLO 就緒與提示轉格完成", flush=True)
                    elif pre_vy is None:
                        print("[Auto Drive] 等待 auto_start=1", flush=True)
                    else:
                        print(f"[Auto Drive] 等待 auto_start=1｜放行前已偵測到靶面，保留 vy={pre_vy}",
                              flush=True)
                time.sleep(0.02)
                continue

            # ---- automatic return to the start zone for the reload ----
            if return_path is not None and current_mode == "A":
                abort = None
                if raw_mode is not None and raw_mode != "A":
                    abort = f"ESP32 回報模式 {raw_mode!r}（急停或重啟）"
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
                    stop_chassis()
                    if abort is not None:
                        print(f"[Return] 中止回程：{abort}；底盤停止，請人工移回出發區；"
                              "等 [Aim] 機構停妥（飛輪停、推球回原點）後再按急停", flush=True)
                    else:
                        return_arrived = True
                        print("[Return] 已退回出發區（開環，位置可能有誤差）："
                              + ("可以按急停裝填" if reload_waiting
                                 else "等 [Aim] 機構停妥（飛輪停、推球回原點）後再按急停"),
                              flush=True)
                else:
                    uart.send_packet(mode="A", vx=command[0], vy=command[1], wz=command[2],
                                     yaw=50, pitch=50, fire=0, speed=0)
                time.sleep(0.02)
                continue

            # ---- automatic positioning (auto mode only; manual = the driver) ----
            if run is None or current_mode != "A" or estop:
                stop_chassis()
                time.sleep(0.02)
                continue

            if run.kind == "initial" and raw_mode == "A":
                if packet.get("auto_start") != 1 and run.started:
                    print("[Auto Drive] VRA 關閉，暫停定位", flush=True)
                    run.pause()
                    latest_vy = None
                    vision_valid_after = now
                    send_aim_command(aim_process, "VISION_ON")
                elif packet.get("auto_start") == 1 and not run.started:
                    vy = latest_vy[1] if latest_vy is not None else run.locked_vy
                    send_aim_command(aim_process, "VISION_OFF")
                    run.release(vy)
                    next_status_at = 0.0
                    print(f"[Auto Drive] VRA 已放行，繼續自動定位，Vision vy={vy}", flush=True)
            elif run.kind == "reposition" and not run.started and (
                    latest_vy is not None or now >= run.release_at):
                vy = latest_vy[1] if latest_vy is not None else None
                send_aim_command(aim_process, "VISION_OFF")
                run.release(vy)
                next_status_at = 0.0
                print(f"[Auto Drive] 重新出發，開始自動定位，Vision vy={vy}", flush=True)

            if not run.started:
                if run.kind == "initial" and latest_vy is not None:
                    run.locked_vy = latest_vy[1]  # sticky while paused
                stop_chassis()
                if now >= next_status_at:
                    next_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                    print("[Auto Drive] 等待 VRA 放行" if run.kind == "initial"
                          else "[Auto Drive] 等待視覺 vy", flush=True)
                time.sleep(0.02)
                continue

            # 自動模式維持原有視覺追蹤 + LiDAR 避障流程
            fresh_scan, front_distance_m = lidar.get_front_obstacle()
            distance_text = (
                "無有效距離" if front_distance_m is None
                else f"前方 {front_distance_m * 100:.1f} cm"
            )

            final_vx, final_vy, final_wz = 0, 0, 0
            arrived = False

            valid_distance = fresh_scan and front_distance_m is not None
            blocked = valid_distance and front_distance_m <= lidar.safe_dist_m
            complete = False
            positioning_succeeded = False
            completion_reason = "已抵達目標距離並完成橫移"
            lidar_status = ""

            if blocked and run.moved and run.brake_until is None:
                run.brake_until = now + OCCLUSION_BRAKE_SECONDS
                print("[Auto Drive] 移動中遇到遮擋，緊急停止", flush=True)

            if run.brake_until is not None:
                lidar_status = "遮擋後緊急停止"
                if now >= run.brake_until:
                    rear_range_m = lidar.get_rear_range()
                    if run.recovery_hold_reason is None:
                        if (valid_distance and front_distance_m + config.LIDAR_TO_FRONT_M
                                >= OCCLUSION_CLEARANCE_M):
                            run.recovery_hold_reason = "前方達 100 cm 淨空，保持停車"
                        elif now - run.brake_until >= OCCLUSION_REVERSE_TIMEOUT_S:
                            run.recovery_hold_reason = "後退逾時，保持停車"
                        elif (now - last_a_packet_at > RETURN_PACKET_TIMEOUT_S
                              or raw_mode not in (None, "A")):
                            run.recovery_hold_reason = "ESP32 通訊中斷或模式異常，保持停車"
                        elif not valid_distance:
                            run.recovery_hold_reason = "前方距離無效，保持停車"
                        elif rear_range_m is None or rear_range_m <= config.RECOVERY_REAR_MIN_RANGE_M:
                            run.recovery_hold_reason = "後方回波不足或有障礙，保持停車"
                    if run.recovery_hold_reason is None:
                        final_vx = -OCCLUSION_REVERSE_VX
                        lidar_status = "遮擋後直線後退，等待前方達 100 cm"
                    else:
                        lidar_status = (run.recovery_hold_reason
                                        + "（切手動模式可由駕駛接手，砲塔照常射擊）")
            elif (
                run.vision_wait_until is not None
                and run.locked_vy is None
                and now >= run.vision_wait_until
            ):
                complete = True
                completion_reason = "等待視覺橫移方向 10 秒逾時"
            elif not fresh_scan:
                lidar_status = "等待有效前方掃描"
            elif front_distance_m is None:
                lidar_status = "偵測範圍內無前方回波"
            elif blocked:
                lidar_status = "起步前有障礙，等待前方清空"
            else:
                arrived = (
                    run.vision_wait_until is not None
                    or front_distance_m < AUTO_TARGET_DISTANCE_M
                )
                if arrived and run.locked_vy is None:
                    if run.vision_wait_until is None:
                        run.vision_wait_until = now + VISION_WAIT_SECONDS
                        next_status_at = 0.0
                        vision_valid_after = time.monotonic()
                        latest_vy = None
                        send_aim_command(aim_process, "VISION_ON")
                    if latest_vy is not None:
                        run.locked_vy = latest_vy[1]
                        send_aim_command(aim_process, "VISION_OFF")
                lidar_status = "抵達目標距離" if arrived else "尚未抵達目標距離"
                final_vx = 0 if arrived else AUTO_FORWARD_VX
                if run.locked_vy is not None and run.strafe_until is None:
                    duration = 0.0
                    if run.locked_vy:
                        duration = NORMAL_STRAFE_DURATION_SCALE * min(
                            MAX_STRAFE_SECONDS, (abs(run.locked_vy) + 3) / 12.0
                        )
                    run.strafe_until = now + duration
                    next_status_at = 0.0
                    print(
                        f"[Auto Drive] Vision vy={run.locked_vy}，正常橫移指令 "
                        f"vy={run.locked_vy}，持續 {duration:.2f} 秒",
                        flush=True,
                    )
                if run.strafe_until is not None:
                    if now < run.strafe_until:
                        final_vy = max(-100, min(100, run.locked_vy))
                        if not arrived:
                            final_vx = abs(final_vy)
                    else:
                        complete = arrived
                        positioning_succeeded = front_distance_m < AUTO_TARGET_DISTANCE_M
                        if complete and not positioning_succeeded:
                            completion_reason = "橫移後未能確認仍在目標距離內"
                run.moved = run.moved or bool(final_vx or final_vy)

            if complete:
                final_vx, final_vy, final_wz = 0, 0, 0

            # 自動模式的底盤前後方向與 Xavier 的正負號相反。
            tx_vx = -final_vx
            tx_vy = final_vy
            if run.vy_recorded_before_release and run.brake_until is None:
                tx_vy = max(-100, min(100, round(final_vy * 1.5)))
            uart.send_packet(mode="A", vx=tx_vx, vy=tx_vy, wz=final_wz,
                             yaw=50, pitch=50, fire=0, speed=0)
            if run.kind == "initial":
                # The match departure (not a re-departure): replayed backwards
                # to return for the reload.
                drive_log.append((now, tx_vx, tx_vy, final_wz))

            if complete:
                end_run()
                print("[LiDAR] 定位結束，關閉點雲、IMU 與心跳", flush=True)
                print(f"[Auto Drive] {distance_text}，{completion_reason}，底盤已送出停止",
                      flush=True)
                if not positioning_succeeded:
                    # Never give up the round here: a cancelled round scores
                    # zero, while the turret (+/-28..34.5 deg) still reaches
                    # most holes from an imperfect stop, and holes it cannot
                    # reach are skipped after 3 s by the aim process.
                    print(f"[Aim] 定位不完整（{completion_reason}），仍照常開始投球；"
                          "打不到的孔會自動跳過", flush=True)
                finish_positioning()
            elif now >= next_status_at:
                next_status_at = now + AUTO_STATUS_INTERVAL_SECONDS
                if run.locked_vy is None:
                    vision_status = ("等待首次偵測靶面" if run.vision_wait_until is not None
                                     else "尚未偵測到靶面")
                else:
                    vision_status = f"已鎖定 vy={run.locked_vy}"
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
                lidar.join(timeout=2.0)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import socket
import struct
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

# Force stdout/stderr to UTF-8 so emoji/output never breaks on odd terminal codepages.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add parent directory to path so 'config' resolves correctly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 網路與埠號設定 (對應 Livox Mid-360)
# ==========================================
HOST_IP = "0.0.0.0"      # 監聽本地所有網卡
POINT_PORT = 56301       # 點雲接收埠號 (mid360_config.json point_data_port)
IMU_PORT = 56401         # IMU 數據接收埠號 (mid360_config.json imu_data_port)
CMD_PORT = 56100         # 命令傳送埠號
LIDAR_IP = "192.168.1.163"

HEARTBEAT_INTERVAL_S = 1.0   # Livox 要求持續收到心跳才會維持串流，單次啟動指令不夠
HEARTBEAT_MSG = b"\x01\x00\x00\x00\x00\x00\x00\x00"

# Livox 點雲封包常見的 data_type 標記 (header 偏移量 9, 1 byte)
# 0: Cartesian 座標 (int32 x,y,z + tag/reflectivity, 14 bytes/point)
# 其餘型別 (spherical 等) 不在本腳本解析範圍內，需跳過避免解析出垃圾座標
SUPPORTED_POINT_DATA_TYPE = 0
POINT_HEADER_LEN = 36
POINT_STRIDE = 14

IMU_HEADER_LEN = 24
IMU_PAYLOAD_LEN = 24  # 6 * float32

# 場地邊界 (僅用於繪圖顯示，可依實際場地大小在 config 覆寫)
FIELD_LENGTH_M = getattr(config, 'FIELD_LENGTH_M', 4.5)
FIELD_WIDTH_M = getattr(config, 'FIELD_WIDTH_M', 3.0)


# ==========================================
# 2. LiDAR DWA 避障與點雲/IMU 解析類別
# ==========================================
class LidarAvoidance(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.buffer_lock = threading.Lock()

        # 資料快取與統計
        self.point_buffer = []          # 點雲數據快取 [x, y, z] (公尺)
        self.latest_sample_point = None # 最新一個點的採樣 (x, y, z)
        self.latest_imu_data = None     # 最新 IMU telemetry 數據 (gyro, accel)
        self.point_packet_count = 0
        self.imu_packet_count = 0
        self.dropped_point_packets = 0  # 因 data_type 不支援或封包過短而丟棄的封包數
        self.obstacle_rel_pos = None    # 偵測到的最近動態障礙物相對位置 [x, y]

        self.running = True
        self.is_active = True           # 支援手/自動模式切換時的休眠控制
        self.last_grid = None

        # 動態障礙物位置的短時間窗口，用來做簡單平滑，降低單幀雜訊造成的跳動
        self._obs_history = deque(maxlen=5)
        # 對手軌跡 (供地圖繪製拖尾效果用，只存實際偵測到的位置，不含平滑後的 None 幀)
        self.enemy_trail = deque(maxlen=30)

        # ---- Connected-component clustering thresholds for enemy-blob detection ----
        self._min_dynamic_cells = getattr(config, 'MIN_DYNAMIC_CELLS', 5)     # skip clustering below this many changed cells
        self._min_cluster_cells = getattr(config, 'MIN_CLUSTER_CELLS', 3)     # reject blobs smaller than this (noise)
        self._robot_min_diameter_m = getattr(config, 'ROBOT_MIN_DIAMETER_M', 0.15)  # reject blobs smaller than a plausible robot
        self._robot_max_diameter_m = getattr(config, 'ROBOT_MAX_DIAMETER_M', 1.0)   # reject blobs larger than a plausible robot (walls/scene shifts)

        # ---- Cluster-selection scoring weights ----
        # Which blob among several valid candidates is "the enemy" is decided by a
        # weighted score combining: (1) proximity to the robot/sensor, (2) cluster
        # size (a real robot has a nontrivial footprint, so bigger blobs among the
        # size-filtered candidates are more likely to be genuine), and (3) direction
        # consistency with the recent tracked velocity (a wall segment that flickers
        # in/out of the dynamic-cell diff won't move coherently frame-to-frame the way
        # a real moving robot does, so this specifically helps reject wall clutter).
        self._cluster_size_cap = getattr(config, 'CLUSTER_SIZE_SCORE_CAP', 60)  # point count that saturates the size score
        self._min_velocity_for_dir_check_mps = getattr(config, 'MIN_VELOCITY_FOR_DIR_CHECK_MPS', 0.1)
        # Weights when a track is already active (association: favor staying with the
        # same object and moving consistently with its established heading)
        self._w_track_dist = getattr(config, 'W_TRACK_DIST', 0.35)
        self._w_track_dir = getattr(config, 'W_TRACK_DIR', 0.35)
        self._w_track_size = getattr(config, 'W_TRACK_SIZE', 0.20)
        self._w_track_sensor_dist = getattr(config, 'W_TRACK_SENSOR_DIST', 0.10)
        # Weights for fresh acquisition (no active track yet): nearest + largest wins,
        # since there's no established motion history to check consistency against
        self._w_acq_sensor_dist = getattr(config, 'W_ACQ_SENSOR_DIST', 0.5)
        self._w_acq_size = getattr(config, 'W_ACQ_SIZE', 0.5)

        # ---- Position continuity / movement-confirmation filter ----
        # Raw candidate history (before continuity/movement validation), used to judge
        # whether a candidate is genuinely moving.
        self._raw_candidate_history = deque(maxlen=15)
        self._last_accepted_pos = None   # last position confirmed as the enemy robot
        self._last_accepted_time = 0.0
        # Estimated enemy velocity (m/s, EMA-smoothed), used for direction-consistency scoring
        self._track_velocity = None
        self._velocity_ema_alpha = getattr(config, 'TRACK_VELOCITY_EMA_ALPHA', 0.5)
        # Track lost after this many seconds with no accepted detection -> requires
        # fresh movement evidence before re-acquiring
        self._track_lost_timeout_s = getattr(config, 'TRACK_LOST_TIMEOUT_S', 1.0)
        # Reject jumps implying speed above this (m/s) as noise/unrelated detections
        self._max_track_speed_mps = getattr(config, 'MAX_TRACK_SPEED_MPS', 4.0)
        # Extra slack (m) on the jump-gate to absorb grid quantization / sensor noise
        self._jump_margin_m = getattr(config, 'JUMP_MARGIN_M', 0.3)
        # Window/threshold used to confirm a candidate is genuinely moving (not static clutter)
        self._movement_window_s = getattr(config, 'MOVEMENT_WINDOW_S', 0.6)
        self._min_move_m = getattr(config, 'MIN_ENEMY_MOVE_M', 0.15)
        # 最新一批點雲的歷史紀錄 (含時間戳)，用短時間窗口累積後再繪圖，
        # 因為 Mid-360 屬非重複掃描，單一封包只是一小段掃描弧線，
        # 只畫最新一包看起來會像雜亂的線條；累積 0.3~0.5 秒才會呈現完整點雲外形。
        # Rolling buffer of recent point-cloud batches (with timestamps), so the display
        # can show a short accumulated window instead of one scan-line-thin UDP packet.
        # Kept small (maxlen + window_s below) since rebuilding this every frame at 30fps
        # is the most expensive part of the render loop — bigger values look "fuller"
        # but cost more per frame.
        self.point_history = deque(maxlen=150)

        # DWA 避障參數設定
        self.search_radius_m = getattr(config, 'DETECTION_RADIUS_M', 2.5)
        self.safe_dist_m = getattr(config, 'SAFE_DISTANCE_M', 1.2)
        self.robot_safe_radius_m = getattr(config, 'ROBOT_SAFE_RADIUS_M', 0.5)
        self.max_vx = getattr(config, 'MAX_VX', 100)
        self.max_vy = getattr(config, 'MAX_VY', 100)
        self.min_vx = getattr(config, 'MIN_VX', -self.max_vx)  # 允許減速/倒退閃避

        self._heartbeat_sock = None

    # ------------------------------------------------------------
    # 心跳 / 啟動指令
    # ------------------------------------------------------------
    def _heartbeat_loop(self):
        """持續傳送心跳指令，維持光達串流不中斷。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._heartbeat_sock = sock
        while self.running:
            try:
                sock.sendto(HEARTBEAT_MSG, (LIDAR_IP, CMD_PORT))
            except OSError:
                pass
            time.sleep(HEARTBEAT_INTERVAL_S)
        sock.close()

    def run(self):
        """背景線程：啟動心跳、點雲與 IMU 監聽工作"""
        t_heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        t_imu = threading.Thread(target=self._listen_imu, daemon=True)
        t_point = threading.Thread(target=self._listen_point_cloud, daemon=True)
        t_heartbeat.start()
        t_imu.start()
        t_point.start()

        while self.running:
            time.sleep(0.1)

    # ------------------------------------------------------------
    # IMU 監聽
    # ------------------------------------------------------------
    def _listen_imu(self):
        """監聽 UDP 埠號 56401 的 IMU Telemetry 數據流"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST_IP, IMU_PORT))
        sock.settimeout(1.0)

        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
                self.imu_packet_count += 1

                # Livox IMU 封包：Data Header 約 24 bytes，後接 6 個 float32
                if len(data) >= IMU_HEADER_LEN + IMU_PAYLOAD_LEN:
                    payload = data[IMU_HEADER_LEN:IMU_HEADER_LEN + IMU_PAYLOAD_LEN]
                    gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = struct.unpack('<ffffff', payload)
                    with self.buffer_lock:
                        self.latest_imu_data = {
                            "gyro": (gyro_x, gyro_y, gyro_z),
                            "accel": (acc_x, acc_y, acc_z)
                        }
            except socket.timeout:
                continue
            except Exception:
                pass
        sock.close()

    # ------------------------------------------------------------
    # 點雲監聽
    # ------------------------------------------------------------
    def _listen_point_cloud(self):
        """監聽 UDP 埠號 56301 的 Point Cloud 點雲數據流"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST_IP, POINT_PORT))
        sock.settimeout(1.0)

        while self.running:
            if not self.is_active:
                time.sleep(0.05)
                continue

            try:
                data, _ = sock.recvfrom(65535)
                self.point_packet_count += 1

                if len(data) <= POINT_HEADER_LEN:
                    self.dropped_point_packets += 1
                    continue

                # 檢查 data_type (header offset 9, 1 byte)，非支援型別直接跳過，
                # 避免用錯誤的 stride/格式解析出無意義座標
                data_type = data[9]
                if data_type != SUPPORTED_POINT_DATA_TYPE:
                    self.dropped_point_packets += 1
                    continue

                payload = data[POINT_HEADER_LEN:]
                points_count = len(payload) // POINT_STRIDE

                batch = []
                sample_pt = None
                for i in range(points_count):
                    offset = i * POINT_STRIDE
                    if offset + 12 <= len(payload):
                        x, y, z = struct.unpack('<iii', payload[offset:offset + 12])
                        if x != 0 or y != 0 or z != 0:
                            pt_m = [x / 1000.0, y / 1000.0, z / 1000.0]
                            batch.append(pt_m)
                            if sample_pt is None:
                                sample_pt = pt_m

                if batch:
                    batch_np = np.array(batch)
                    obs_pos = self._extract_dynamic_obstacle(batch_np)

                    with self.buffer_lock:
                        if sample_pt:
                            self.latest_sample_point = sample_pt
                        self.obstacle_rel_pos = obs_pos
                        self.point_history.append((time.time(), batch_np))
                        self.point_buffer.extend(batch)
                        if len(self.point_buffer) > 30000:
                            self.point_buffer = self.point_buffer[-30000:]

            except socket.timeout:
                continue
            except Exception:
                pass
        sock.close()

    # ------------------------------------------------------------
    # Dynamic obstacle detection
    # ------------------------------------------------------------
    def _extract_dynamic_obstacle(self, pts):
        """
        Height/range crop, then find the enemy robot via 2D occupancy-grid diffing
        followed by connected-component clustering of the newly-occupied cells.

        This replaces the earlier approach of just taking the single nearest new
        grid cell (which could be one stray noisy point). Instead:
          1. Group all "new" cells into spatially-connected clusters (blobs).
          2. Reject blobs whose size doesn't match a plausible robot footprint
             (too small = noise, too large = a wall/whole-room shift).
          3. Among the remaining candidate blobs, prefer the one closest to the
             last confirmed track (association across frames) when a track is
             active, otherwise prefer the blob with the most supporting points
             (strongest evidence of being a real object).
          4. The final position is the centroid of the *actual points* in that
             blob's cells (not just the grid-cell center), for better precision.
        """
        if len(pts) < 10:
            return self._smoothed_obstacle(None)

        # Filter out floor / ceiling returns
        z_mask = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.8)
        dist_xy = np.hypot(pts[:, 0], pts[:, 1])
        valid_pts = pts[z_mask & (dist_xy <= self.search_radius_m)]

        if len(valid_pts) < 5:
            return self._smoothed_obstacle(None)

        grid_size = 0.1  # 10cm grid cells
        # floor() rather than astype(int) to avoid inconsistent truncation near zero
        grid_coords = np.floor(valid_pts[:, :2] / grid_size).astype(int)

        cell_to_points = {}
        for pt, cell in zip(valid_pts, map(tuple, grid_coords)):
            cell_to_points.setdefault(cell, []).append(pt)
        current_grid = set(cell_to_points.keys())

        dynamic_pos = None
        if self.last_grid is not None:
            dynamic_cells = current_grid - self.last_grid
            if len(dynamic_cells) >= self._min_dynamic_cells:
                clusters = self._cluster_grid_cells(dynamic_cells)
                dynamic_pos = self._select_best_cluster(clusters, cell_to_points, grid_size)

        self.last_grid = current_grid

        # Candidates then pass continuity + movement validation before display/smoothing.
        validated_pos = self._validate_candidate(dynamic_pos)
        return self._smoothed_obstacle(validated_pos)

    @staticmethod
    def _cluster_grid_cells(cells):
        """8-connectivity BFS clustering of a set of grid-cell coordinates into blobs."""
        cells = set(cells)
        visited = set()
        clusters = []
        neighbor_offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                             (0, 1), (1, -1), (1, 0), (1, 1))

        for start_cell in cells:
            if start_cell in visited:
                continue
            queue = deque([start_cell])
            visited.add(start_cell)
            component = [start_cell]
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in neighbor_offsets:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in cells and neighbor not in visited:
                        visited.add(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            clusters.append(component)
        return clusters

    def _select_best_cluster(self, clusters, cell_to_points, grid_size):
        """Filter clusters by plausible robot footprint size, then pick the best
        candidate with a weighted score combining: proximity, cluster size, and
        (once a track is established) direction consistency with the enemy's
        recent movement — this last factor is what specifically filters out a
        wall segment that flickers into the dynamic-cell diff but doesn't move
        coherently the way a real robot does."""
        candidates = []
        for comp in clusters:
            if len(comp) < self._min_cluster_cells:
                continue

            xs = [c[0] for c in comp]
            ys = [c[1] for c in comp]
            extent_x = (max(xs) - min(xs) + 1) * grid_size
            extent_y = (max(ys) - min(ys) + 1) * grid_size
            diameter = np.hypot(extent_x, extent_y)
            if not (self._robot_min_diameter_m <= diameter <= self._robot_max_diameter_m):
                continue  # too small (noise) or too large (wall / scene shift)

            cluster_pts = [p for cell in comp for p in cell_to_points.get(cell, [])]
            if not cluster_pts:
                continue
            centroid = np.mean(np.array(cluster_pts)[:, :2], axis=0)
            candidates.append({'centroid': centroid, 'size': len(cluster_pts)})

        if not candidates:
            return None

        now = time.time()
        track_active = (self._last_accepted_pos is not None and
                         now - self._last_accepted_time <= self._track_lost_timeout_s)
        have_velocity = (track_active and self._track_velocity is not None and
                          np.hypot(*self._track_velocity) > self._min_velocity_for_dir_check_mps)

        best_score, best_centroid = -1.0, None
        for c in candidates:
            centroid = c['centroid']
            size_score = min(c['size'] / self._cluster_size_cap, 1.0)
            sensor_dist = np.hypot(centroid[0], centroid[1])
            sensor_dist_score = 1.0 / (1.0 + sensor_dist)

            if track_active:
                track_dist = np.hypot(centroid[0] - self._last_accepted_pos[0],
                                       centroid[1] - self._last_accepted_pos[1])
                track_dist_score = 1.0 / (1.0 + track_dist)

                if have_velocity:
                    disp = centroid - np.array(self._last_accepted_pos)
                    disp_mag = np.linalg.norm(disp)
                    if disp_mag > 1e-6:
                        vel = np.array(self._track_velocity)
                        cos_sim = float(np.dot(disp / disp_mag, vel / np.linalg.norm(vel)))
                        dir_score = (cos_sim + 1.0) / 2.0  # map [-1,1] -> [0,1]
                    else:
                        dir_score = 0.5  # candidate didn't move relative to last fix; neutral
                else:
                    dir_score = 0.5  # not enough history yet to judge direction consistency

                score = (self._w_track_dist * track_dist_score +
                         self._w_track_dir * dir_score +
                         self._w_track_size * size_score +
                         self._w_track_sensor_dist * sensor_dist_score)
            else:
                # Fresh acquisition: nearest to the robot + largest plausible blob wins,
                # since there's no motion history yet to check consistency against.
                score = (self._w_acq_sensor_dist * sensor_dist_score +
                         self._w_acq_size * size_score)

            if score > best_score:
                best_score, best_centroid = score, centroid

        return best_centroid

    def _validate_candidate(self, raw_candidate):
        """
        Two-stage gate applied after cluster selection:
        1) Continuity -- if a track is already active, the new candidate must fall
           within "last position + a plausible max-speed-implied radius" or it's
           rejected as an unrelated jump. Track lost after _track_lost_timeout_s
           with no accepted detection resets this so a fresh target can be acquired.
        2) Movement confirmation -- a real enemy should be moving; if displacement
           over the recent _movement_window_s history is under _min_move_m, treat
           it as static clutter (sensor jitter, a fixed fixture) and reject it.
        On acceptance, also updates the EMA-smoothed velocity estimate used by
        _select_best_cluster's direction-consistency scoring.
        """
        now = time.time()

        # Track timed out -> forget it, require fresh movement evidence to re-acquire.
        if (self._last_accepted_pos is not None and
                now - self._last_accepted_time > self._track_lost_timeout_s):
            self._last_accepted_pos = None
            self._track_velocity = None
            self._raw_candidate_history.clear()

        if raw_candidate is None:
            return None

        # Keep only the recent window of raw candidates for the movement check.
        self._raw_candidate_history.append((now, raw_candidate))
        while (self._raw_candidate_history and
               now - self._raw_candidate_history[0][0] > self._movement_window_s):
            self._raw_candidate_history.popleft()

        # 1) Continuity check
        prev_pos, prev_time = self._last_accepted_pos, self._last_accepted_time
        if prev_pos is not None:
            dt = max(now - prev_time, 1e-3)
            max_jump = self._max_track_speed_mps * dt + self._jump_margin_m
            jump_dist = np.hypot(raw_candidate[0] - prev_pos[0], raw_candidate[1] - prev_pos[1])
            if jump_dist > max_jump:
                # Implausible jump -- unrelated to the current track, reject this frame
                # (history keeps accumulating; if it keeps recurring nearby, the
                # timeout above will let it start a fresh track instead).
                return None

        # 2) Movement check: need at least two history samples spanning the window,
        # with total displacement above the noise threshold, to call it "moving".
        if len(self._raw_candidate_history) < 2:
            return None
        oldest_pos = self._raw_candidate_history[0][1]
        newest_pos = self._raw_candidate_history[-1][1]
        moved_dist = np.hypot(newest_pos[0] - oldest_pos[0], newest_pos[1] - oldest_pos[1])
        if moved_dist < self._min_move_m:
            return None  # insufficient displacement -> static object or noise, not the enemy

        # Passed all checks -- accept as the enemy's current position, and update the
        # smoothed velocity estimate (used for direction-consistency scoring next frame).
        if prev_pos is not None:
            dt = max(now - prev_time, 1e-3)
            inst_vel = ((raw_candidate[0] - prev_pos[0]) / dt, (raw_candidate[1] - prev_pos[1]) / dt)
            if self._track_velocity is None:
                self._track_velocity = inst_vel
            else:
                a = self._velocity_ema_alpha
                self._track_velocity = (a * inst_vel[0] + (1 - a) * self._track_velocity[0],
                                         a * inst_vel[1] + (1 - a) * self._track_velocity[1])

        self._last_accepted_pos = raw_candidate
        self._last_accepted_time = now
        return raw_candidate

    def _smoothed_obstacle(self, new_pos):
        """將本幀偵測結果加入歷史窗口並回傳平均位置，None 代表本幀未偵測到。"""
        self._obs_history.append(new_pos)
        valid = [p for p in self._obs_history if p is not None]
        if not valid:
            return None
        arr = np.array(valid)
        smoothed = arr.mean(axis=0)
        self.enemy_trail.append(tuple(smoothed))
        return smoothed

    # ------------------------------------------------------------
    # DWA 決策
    # ------------------------------------------------------------
    def apply_priority_override(self, raw_vx, raw_vy, raw_wz):
        """最高優先級 DWA 覆寫計算"""
        with self.buffer_lock:
            obs = self.obstacle_rel_pos

        if obs is None or np.hypot(obs[0], obs[1]) > self.safe_dist_m:
            return raw_vx, raw_vy, raw_wz, False

        best_score = -999999
        best_cmd = (0, 0, 0)

        # 允許減速甚至倒退，而不是永遠維持正向高速前進
        vx_samples = np.linspace(self.min_vx, self.max_vx, 7)
        vy_samples = np.linspace(-self.max_vy, self.max_vy, 9)
        wz_samples = np.linspace(-30, 30, 3)

        for vx in vx_samples:
            for vy in vy_samples:
                for wz in wz_samples:
                    pred_x = (vx * 0.6) / 100.0
                    pred_y = (vy * 0.6) / 100.0
                    dist_to_obs = np.hypot(pred_x - obs[0], pred_y - obs[1])

                    if dist_to_obs < self.robot_safe_radius_m:
                        continue

                    score = (2.0 * np.hypot(vx, vy)) + \
                            (1.0 * -np.hypot(vx - raw_vx, vy - raw_vy)) + \
                            (1.0 * dist_to_obs * 20.0)

                    if score > best_score:
                        best_score = score
                        best_cmd = (int(vx), int(vy), int(wz))

        if best_score == -999999:
            return 0, 0, 0, True

        return best_cmd[0], best_cmd[1], best_cmd[2], True

    # ------------------------------------------------------------
    # 對外查詢介面
    # ------------------------------------------------------------
    def get_telemetry_snapshot(self):
        """抓取目前最即時的數據快照"""
        with self.buffer_lock:
            return (
                self.point_packet_count,
                self.imu_packet_count,
                self.dropped_point_packets,
                self.latest_sample_point,
                self.latest_imu_data,
                self.obstacle_rel_pos
            )

    def get_recent_points(self, window_s=0.25):
        """Return points accumulated over the last window_s seconds (for display).
        Smaller window = cheaper per-frame rebuild, at the cost of a slightly
        thinner-looking point cloud."""
        now = time.time()
        with self.buffer_lock:
            recent = [pts for t, pts in self.point_history if now - t <= window_s]
        if not recent:
            return None
        return np.vstack(recent)

    def get_map_snapshot(self, window_s=0.25):
        """Data needed to render the map: accumulated point cloud + enemy trail."""
        pts = self.get_recent_points(window_s)
        with self.buffer_lock:
            trail = list(self.enemy_trail)
            obs = self.obstacle_rel_pos
        return pts, trail, obs

    def set_active(self, state: bool):
        self.is_active = state

    def stop(self):
        self.running = False


# ==========================================
# 3. 即時動態 2D 雷達地圖 (matplotlib)
# ==========================================
class LiveRadarMap:
    """
    以 matplotlib 動畫持續更新的 2D 俯視雷達地圖：
    - 灰點: 即時點雲 (依高度 Z 上色)
    - 藍色三角: 本車機器人 (固定於圖中心偏下，朝上為前方)
    - 紅點 + 拖尾: 動態對手位置與其近期軌跡
    - 虛線圓: 偵測半徑 (search_radius_m)
    - 橘色圓: 安全距離 (safe_dist_m)，對手進入此圈即觸發 DWA 避障
    - 綠/紅 外框: 依是否正在避障變色的場地邊界
    """

    def __init__(self, lidar: LidarAvoidance, field_length=FIELD_LENGTH_M, field_width=FIELD_WIDTH_M):
        self.lidar = lidar
        self.field_length = field_length
        self.field_width = field_width

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.manager.set_window_title("LiDAR DWA Live Avoidance Map")

        half_l = field_length / 2
        half_w = field_width / 2
        self.ax.set_xlim(-half_w - 0.5, half_w + 0.5)
        self.ax.set_ylim(-0.5, field_length + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("Y (m) — left/right")
        self.ax.set_ylabel("X (m) — forward")

        # 場地邊界 (機器人視角座標系：本車在下方，前方朝上，繪圖時 x/y 互換以符合直覺)
        self.field_patch = Rectangle(
            (-half_w, 0), field_width, field_length,
            fill=False, edgecolor='green', linewidth=2
        )
        self.ax.add_patch(self.field_patch)

        # 偵測半徑 / 安全距離圓 (以機器人為圓心)
        self.robot_x, self.robot_y = 0.0, 0.3  # 機器人固定顯示位置 (前方座標略高於 0)
        self.search_circle = Circle(
            (self.robot_x, self.robot_y), lidar.search_radius_m,
            fill=False, linestyle='--', edgecolor='gray', linewidth=1
        )
        self.safe_circle = Circle(
            (self.robot_x, self.robot_y), lidar.safe_dist_m,
            fill=False, linestyle='-', edgecolor='orange', linewidth=1.5
        )
        self.ax.add_patch(self.search_circle)
        self.ax.add_patch(self.safe_circle)

        # 點雲散點圖 (依高度上色)
        self.cloud_scatter = self.ax.scatter([], [], s=3, c=[], cmap='viridis', alpha=0.6, vmin=-0.2, vmax=0.8)

        # Robot marker (triangle)
        self.robot_marker, = self.ax.plot([self.robot_x], [self.robot_y], marker='^', markersize=16,
                                           color='royalblue', markeredgecolor='black', zorder=5)

        # Heading arrow: shows the direction of the robot's current (post-DWA) commanded
        # velocity. Initialized pointing forward with a tiny nonzero length; updated each
        # frame via set_UVC so no artist is recreated per frame.
        self._arrow_len_m = 0.6  # fixed visual arrow length in meters, independent of speed
        self.heading_quiver = self.ax.quiver(
            [self.robot_x], [self.robot_y], [0.0], [self._arrow_len_m],
            color='royalblue', angles='xy', scale_units='xy', scale=1,
            width=0.012, zorder=6
        )

        # Enemy marker and trail
        self.enemy_marker, = self.ax.plot([], [], marker='o', markersize=12,
                                           color='red', markeredgecolor='black', zorder=5)
        self.enemy_trail_line, = self.ax.plot([], [], color='red', alpha=0.4, linewidth=2)

        self.status_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.ax.set_title("LiDAR DWA Live Avoidance Map", fontsize=13)

        # 固定的模擬前進指令，與 DWA 覆寫做比較 (可依實際控制程式替換)
        self.fake_raw_vx = 80
        self.fake_raw_vy = 0
        self.fake_raw_wz = 0

    @staticmethod
    def _to_plot_xy(rel_x, rel_y):
        """將雷達相對座標 (x: 前, y: 左) 轉成畫布座標 (plot_x: 左右, plot_y: 前方)。"""
        return -rel_y, rel_x + 0.3

    def _update(self, _frame):
        pts, trail, obs_pos = self.lidar.get_map_snapshot()
        p_cnt, imu_cnt, dropped_cnt, sample_pt, imu_data, _ = self.lidar.get_telemetry_snapshot()
        final_vx, final_vy, final_wz, is_evading = self.lidar.apply_priority_override(
            self.fake_raw_vx, self.fake_raw_vy, self.fake_raw_wz
        )

        # 1. 更新點雲
        if pts is not None and len(pts) > 0:
            plot_x = -pts[:, 1]
            plot_y = pts[:, 0] + 0.3
            self.cloud_scatter.set_offsets(np.column_stack([plot_x, plot_y]))
            self.cloud_scatter.set_array(pts[:, 2])
        else:
            self.cloud_scatter.set_offsets(np.empty((0, 2)))

        # 2. 更新對手位置與拖尾 (拖尾越舊透明度越低，用線段模擬移動軌跡)
        if obs_pos is not None:
            ex, ey = self._to_plot_xy(obs_pos[0], obs_pos[1])
            self.enemy_marker.set_data([ex], [ey])
        else:
            self.enemy_marker.set_data([], [])

        if trail:
            trail_xy = [self._to_plot_xy(p[0], p[1]) for p in trail]
            txs, tys = zip(*trail_xy)
            self.enemy_trail_line.set_data(txs, tys)
        else:
            self.enemy_trail_line.set_data([], [])

        # 3. Heading arrow: direction of the final (post-DWA) commanded velocity.
        # vx/vy are raw command units (not m/s), so only the direction matters here;
        # arrow length stays fixed so it's readable regardless of command magnitude.
        cmd_speed = np.hypot(final_vx, final_vy)
        if cmd_speed > 1e-6:
            dir_x, dir_y = final_vx / cmd_speed, final_vy / cmd_speed
            arrow_u = -dir_y * self._arrow_len_m
            arrow_v = dir_x * self._arrow_len_m
        else:
            arrow_u, arrow_v = 0.0, 0.0  # robot commanded to stop: no direction to show
        self.heading_quiver.set_UVC([arrow_u], [arrow_v])
        self.heading_quiver.set_color('red' if is_evading else 'royalblue')

        # 4. Field boundary color reflects current avoidance state at a glance
        self.field_patch.set_edgecolor('red' if is_evading else 'green')

        # 5. Status text
        dist_str = "--"
        if obs_pos is not None:
            dist_str = f"{np.hypot(obs_pos[0], obs_pos[1]):.2f} m"

        imu_str = "waiting..."
        if imu_data:
            g = imu_data["gyro"]
            imu_str = f"gyroZ={g[2]:+.2f}"

        evade_str = "DWA evading" if is_evading else "path clear"
        dist_label = "Opponent dist"
        cmd_label = "Cmd"
        lines = [
            f"Packets: pts={p_cnt}(drop={dropped_cnt}) imu={imu_cnt}",
            f"IMU: {imu_str}",
            f"{dist_label}: {dist_str}",
            f"{evade_str}",
            f"{cmd_label}: raw=({self.fake_raw_vx},{self.fake_raw_vy},{self.fake_raw_wz}) "
            f"-> final=({final_vx},{final_vy},{final_wz})",
        ]
        self.status_text.set_text("\n".join(lines))

        return (self.cloud_scatter, self.enemy_marker, self.enemy_trail_line,
                self.heading_quiver, self.field_patch, self.status_text)

    def run(self, target_fps=30):
        """Run the live animation. blit=True is required to sustain ~30fps —
        without it, matplotlib redraws the entire figure (including the static
        detection/safety circles and axes) every frame, which is far slower."""
        interval_ms = 1000.0 / target_fps
        self._anim = animation.FuncAnimation(
            self.fig, self._update, interval=interval_ms, blit=True, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()


# ==========================================
# 4. Main entry point
# ==========================================
def main():
    print("==================================================")
    print(" Livox Mid-360 Point Cloud / IMU / DWA Avoidance Test")
    print("==================================================")

    lidar = LidarAvoidance()
    lidar.start()
    lidar.set_active(True)

    print("LiDAR receiver threads started, listening on port 56301 (Point) & 56401 (IMU), heartbeat active...")
    print("Opening the live radar map window — move an object near the LiDAR to see updates...\n")

    radar_map = LiveRadarMap(lidar)
    try:
        radar_map.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nInterrupted by user, test ended.")
        lidar.stop()


if __name__ == '__main__':
    main()

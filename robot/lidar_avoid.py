# lidar_avoid.py
import socket
import struct
import threading
import time
from collections import deque
import numpy as np

try:
    import config
except ImportError:
    # 若在非標準路徑下測試，提供 fallback 預設值
    class config:
        MIN_DYNAMIC_CELLS = 5
        MIN_CLUSTER_CELLS = 3
        ROBOT_MIN_DIAMETER_M = 0.15
        ROBOT_MAX_DIAMETER_M = 1.0
        CLUSTER_SIZE_SCORE_CAP = 60
        MIN_VELOCITY_FOR_DIR_CHECK_MPS = 0.1
        W_TRACK_DIST = 0.35
        W_TRACK_DIR = 0.35
        W_TRACK_SIZE = 0.20
        W_TRACK_SENSOR_DIST = 0.10
        W_ACQ_SENSOR_DIST = 0.5
        W_ACQ_SIZE = 0.5
        TRACK_VELOCITY_EMA_ALPHA = 0.5
        TRACK_LOST_TIMEOUT_S = 1.0
        MAX_TRACK_SPEED_MPS = 4.0
        JUMP_MARGIN_M = 0.3
        MOVEMENT_WINDOW_S = 0.6
        MIN_ENEMY_MOVE_M = 0.15
        DETECTION_RADIUS_M = 2.5
        SAFE_DISTANCE_M = 1.2
        ROBOT_SAFE_RADIUS_M = 0.5
        MAX_VX = 100
        MAX_VY = 100
        MIN_VX = -100

# ==========================================
# 網路與埠號設定 (對應 Livox Mid-360)
# ==========================================
HOST_IP = "0.0.0.0"
POINT_PORT = 56301
IMU_PORT = 56401
CMD_PORT = 56100
LIDAR_IP = "192.168.1.163"

HEARTBEAT_INTERVAL_S = 1.0
HEARTBEAT_MSG = b"\x01\x00\x00\x00\x00\x00\x00\x00"

SUPPORTED_POINT_DATA_TYPE = 0
POINT_HEADER_LEN = 36
POINT_STRIDE = 14
IMU_HEADER_LEN = 24
IMU_PAYLOAD_LEN = 24


class LidarAvoidance(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.buffer_lock = threading.Lock()

        # 資料快取與統計
        self.point_buffer = []
        self.latest_sample_point = None
        self.latest_imu_data = None
        self.point_packet_count = 0
        self.imu_packet_count = 0
        self.dropped_point_packets = 0
        self.obstacle_rel_pos = None

        self.running = True
        self.is_active = True
        self.last_grid = None

        self._obs_history = deque(maxlen=5)
        self.enemy_trail = deque(maxlen=30)

        # 演算法參數
        self._min_dynamic_cells = getattr(config, 'MIN_DYNAMIC_CELLS', 5)
        self._min_cluster_cells = getattr(config, 'MIN_CLUSTER_CELLS', 3)
        self._robot_min_diameter_m = getattr(config, 'ROBOT_MIN_DIAMETER_M', 0.15)
        self._robot_max_diameter_m = getattr(config, 'ROBOT_MAX_DIAMETER_M', 1.0)
        self._cluster_size_cap = getattr(config, 'CLUSTER_SIZE_SCORE_CAP', 60)
        self._min_velocity_for_dir_check_mps = getattr(config, 'MIN_VELOCITY_FOR_DIR_CHECK_MPS', 0.1)
        self._w_track_dist = getattr(config, 'W_TRACK_DIST', 0.35)
        self._w_track_dir = getattr(config, 'W_TRACK_DIR', 0.35)
        self._w_track_size = getattr(config, 'W_TRACK_SIZE', 0.20)
        self._w_track_sensor_dist = getattr(config, 'W_TRACK_SENSOR_DIST', 0.10)
        self._w_acq_sensor_dist = getattr(config, 'W_ACQ_SENSOR_DIST', 0.5)
        self._w_acq_size = getattr(config, 'W_ACQ_SIZE', 0.5)

        self._raw_candidate_history = deque(maxlen=15)
        self._last_accepted_pos = None
        self._last_accepted_time = 0.0
        self._track_velocity = None
        self._velocity_ema_alpha = getattr(config, 'TRACK_VELOCITY_EMA_ALPHA', 0.5)
        self._track_lost_timeout_s = getattr(config, 'TRACK_LOST_TIMEOUT_S', 1.0)
        self._max_track_speed_mps = getattr(config, 'MAX_TRACK_SPEED_MPS', 4.0)
        self._jump_margin_m = getattr(config, 'JUMP_MARGIN_M', 0.3)
        self._movement_window_s = getattr(config, 'MOVEMENT_WINDOW_S', 0.6)
        self._min_move_m = getattr(config, 'MIN_ENEMY_MOVE_M', 0.15)
        self.point_history = deque(maxlen=150)

        self.search_radius_m = getattr(config, 'DETECTION_RADIUS_M', 2.5)
        self.safe_dist_m = getattr(config, 'SAFE_DISTANCE_M', 1.2)
        self.robot_safe_radius_m = getattr(config, 'ROBOT_SAFE_RADIUS_M', 0.5)
        self.max_vx = getattr(config, 'MAX_VX', 100)
        self.max_vy = getattr(config, 'MAX_VY', 100)
        self.min_vx = getattr(config, 'MIN_VX', -self.max_vx)

        self._heartbeat_sock = None

    def _heartbeat_loop(self):
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
        t_heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        t_imu = threading.Thread(target=self._listen_imu, daemon=True)
        t_point = threading.Thread(target=self._listen_point_cloud, daemon=True)
        t_heartbeat.start()
        t_imu.start()
        t_point.start()

        while self.running:
            time.sleep(0.1)

    def _listen_imu(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST_IP, IMU_PORT))
        sock.settimeout(1.0)

        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
                self.imu_packet_count += 1
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

    def _listen_point_cloud(self):
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

                if data[9] != SUPPORTED_POINT_DATA_TYPE:
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

    def _extract_dynamic_obstacle(self, pts):
        if len(pts) < 10:
            return self._smoothed_obstacle(None)

        z_mask = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.8)
        dist_xy = np.hypot(pts[:, 0], pts[:, 1])
        valid_pts = pts[z_mask & (dist_xy <= self.search_radius_m)]

        if len(valid_pts) < 5:
            return self._smoothed_obstacle(None)

        grid_size = 0.1
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
        validated_pos = self._validate_candidate(dynamic_pos)
        return self._smoothed_obstacle(validated_pos)

    @staticmethod
    def _cluster_grid_cells(cells):
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
                continue

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
                        dir_score = (cos_sim + 1.0) / 2.0
                    else:
                        dir_score = 0.5
                else:
                    dir_score = 0.5

                score = (self._w_track_dist * track_dist_score +
                         self._w_track_dir * dir_score +
                         self._w_track_size * size_score +
                         self._w_track_sensor_dist * sensor_dist_score)
            else:
                score = (self._w_acq_sensor_dist * sensor_dist_score +
                         self._w_acq_size * size_score)

            if score > best_score:
                best_score, best_centroid = score, centroid

        return best_centroid

    def _validate_candidate(self, raw_candidate):
        now = time.time()
        if (self._last_accepted_pos is not None and
                now - self._last_accepted_time > self._track_lost_timeout_s):
            self._last_accepted_pos = None
            self._track_velocity = None
            self._raw_candidate_history.clear()

        if raw_candidate is None:
            return None

        self._raw_candidate_history.append((now, raw_candidate))
        while (self._raw_candidate_history and
               now - self._raw_candidate_history[0][0] > self._movement_window_s):
            self._raw_candidate_history.popleft()

        prev_pos, prev_time = self._last_accepted_pos, self._last_accepted_time
        if prev_pos is not None:
            dt = max(now - prev_time, 1e-3)
            max_jump = self._max_track_speed_mps * dt + self._jump_margin_m
            jump_dist = np.hypot(raw_candidate[0] - prev_pos[0], raw_candidate[1] - prev_pos[1])
            if jump_dist > max_jump:
                return None

        if len(self._raw_candidate_history) < 2:
            return None
        oldest_pos = self._raw_candidate_history[0][1]
        newest_pos = self._raw_candidate_history[-1][1]
        moved_dist = np.hypot(newest_pos[0] - oldest_pos[0], newest_pos[1] - oldest_pos[1])
        if moved_dist < self._min_move_m:
            return None

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
        self._obs_history.append(new_pos)
        valid = [p for p in self._obs_history if p is not None]
        if not valid:
            return None
        arr = np.array(valid)
        smoothed = arr.mean(axis=0)
        self.enemy_trail.append(tuple(smoothed))
        return smoothed

    def apply_priority_override(self, raw_vx, raw_vy, raw_wz):
        with self.buffer_lock:
            obs = self.obstacle_rel_pos

        if obs is None or np.hypot(obs[0], obs[1]) > self.safe_dist_m:
            return raw_vx, raw_vy, raw_wz, False

        best_score = -999999
        best_cmd = (0, 0, 0)

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

    def get_telemetry_snapshot(self):
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
        now = time.time()
        with self.buffer_lock:
            recent = [pts for t, pts in self.point_history if now - t <= window_s]
        if not recent:
            return None
        return np.vstack(recent)

    def get_map_snapshot(self, window_s=0.25):
        pts = self.get_recent_points(window_s)
        with self.buffer_lock:
            trail = list(self.enemy_trail)
            obs = self.obstacle_rel_pos
        return pts, trail, obs

    def set_active(self, state: bool):
        self.is_active = state

    def stop(self):
        self.running = False

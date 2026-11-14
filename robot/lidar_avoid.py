#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket
import math
import struct
import threading
import time
from collections import deque
import numpy as np

import config

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

        self.enemy_trail = deque(maxlen=30)
        self.point_history = deque(maxlen=150)
        self._front_hits = deque()
        self._last_scan_at = None
        self.search_radius_m = config.DETECTION_RADIUS_M
        self.safe_dist_m = config.FRONT_STOP_DISTANCE_M

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
                    now = time.monotonic()
                    front = batch_np[self._front_fan_mask(batch_np)]
                    obs_pos = tuple(front[front[:, 0].argmin(), :2]) if len(front) else None

                    with self.buffer_lock:
                        if sample_pt:
                            self.latest_sample_point = sample_pt
                        if not self.is_active:
                            continue
                        if obs_pos is not None:
                            self._last_scan_at = now
                            if obs_pos[0] <= self.search_radius_m:
                                self._front_hits.append((now, obs_pos))
                        self.obstacle_rel_pos = self._nearest_front_hit(now)
                        if self.obstacle_rel_pos is not None:
                            self.enemy_trail.append(self.obstacle_rel_pos)
                        self.point_history.append((time.time(), front))
                        self.point_buffer.extend(front.tolist())
                        if len(self.point_buffer) > 30000:
                            self.point_buffer = self.point_buffer[-30000:]

            except socket.timeout:
                continue
            except Exception:
                pass
        sock.close()

    def _front_fan_mask(self, points):
        """Filter in LiDAR coordinates. The mounted sensor's +X faces forward."""
        x, y, z = points.T
        fan_slope = math.tan(math.radians(config.FRONT_LIDAR_HALF_ANGLE_DEG))
        return ((x > 0.0)
                & (np.abs(y) <= x * fan_slope)
                & (y >= -config.FRONT_LIDAR_HALF_WIDTH_M)
                & (y <= 0.25)
                & (z > config.FRONT_LIDAR_MIN_Z_M)
                & (z < config.FRONT_LIDAR_MAX_Z_M))

    def _nearest_front_hit(self, now):
        # UDP packets cover only part of a scan. Preserve near hits across packets.
        while self._front_hits and now - self._front_hits[0][0] > config.FRONT_LIDAR_MAX_AGE_S:
            self._front_hits.popleft()
        return min((point for _, point in self._front_hits),
                   key=lambda point: point[0], default=None)

    def get_front_obstacle(self):
        """Return (fresh_scan, nearest forward range from bumper in metres or None)."""
        with self.buffer_lock:
            now = time.monotonic()
            fresh = (self.is_active and self._last_scan_at is not None
                     and now - self._last_scan_at <= config.FRONT_LIDAR_MAX_AGE_S)
            hit = self._nearest_front_hit(now) if fresh else None
        distance = None if hit is None else max(0.0, hit[0] - config.LIDAR_TO_FRONT_M)
        return fresh, distance

    def apply_priority_override(self, raw_vx, raw_vy, raw_wz):
        fresh, distance = self.get_front_obstacle()
        if not fresh or (distance is not None and distance <= self.safe_dist_m):
            return 0, 0, 0, True
        return raw_vx, raw_vy, raw_wz, False

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
        with self.buffer_lock:
            self.is_active = state
            self._last_scan_at = None
            self._front_hits.clear()
            self.enemy_trail.clear()
            self.obstacle_rel_pos = None

    def stop(self):
        self.running = False

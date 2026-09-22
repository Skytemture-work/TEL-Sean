#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from lidar_avoid import LidarAvoidance

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 即時動態 2D 雷達地圖 (matplotlib)
# ==========================================
class LiveRadarMap:
    def __init__(self, lidar: LidarAvoidance, field_length=4.5, field_width=3.0):
        self.lidar = lidar
        self.field_length = getattr(config, 'FIELD_LENGTH_M', field_length)
        self.field_width = getattr(config, 'FIELD_WIDTH_M', field_width)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.manager.set_window_title("LiDAR DWA Live Avoidance Map")

        half_l = self.field_length / 2
        half_w = self.field_width / 2
        self.ax.set_xlim(-half_w - 0.5, half_w + 0.5)
        self.ax.set_ylim(-0.5, self.field_length + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("Y (m) — left/right")
        self.ax.set_ylabel("X (m) — forward")

        self.field_patch = Rectangle(
            (-half_w, 0), self.field_width, self.field_length,
            fill=False, edgecolor='green', linewidth=2
        )
        self.ax.add_patch(self.field_patch)

        self.robot_x, self.robot_y = 0.0, 0.3
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

        self.cloud_scatter = self.ax.scatter([], [], s=3, c=[], cmap='viridis', alpha=0.6, vmin=-0.2, vmax=0.8)
        self.robot_marker, = self.ax.plot([self.robot_x], [self.robot_y], marker='^', markersize=16,
                                          color='royalblue', markeredgecolor='black', zorder=5)

        self._arrow_len_m = 0.6
        self.heading_quiver = self.ax.quiver(
            [self.robot_x], [self.robot_y], [0.0], [self._arrow_len_m],
            color='royalblue', angles='xy', scale_units='xy', scale=1,
            width=0.012, zorder=6
        )

        self.enemy_marker, = self.ax.plot([], [], marker='o', markersize=12,
                                          color='red', markeredgecolor='black', zorder=5)
        self.enemy_trail_line, = self.ax.plot([], [], color='red', alpha=0.4, linewidth=2)

        self.status_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.ax.set_title("LiDAR DWA Live Avoidance Map", fontsize=13)

        self.fake_raw_vx = 80
        self.fake_raw_vy = 0
        self.fake_raw_wz = 0

    @staticmethod
    def _to_plot_xy(rel_x, rel_y):
        return -rel_y, rel_x + 0.3

    def _update(self, _frame):
        pts, trail, obs_pos = self.lidar.get_map_snapshot()
        p_cnt, imu_cnt, dropped_cnt, sample_pt, imu_data, _ = self.lidar.get_telemetry_snapshot()
        final_vx, final_vy, final_wz, is_evading = self.lidar.apply_priority_override(
            self.fake_raw_vx, self.fake_raw_vy, self.fake_raw_wz
        )

        if pts is not None and len(pts) > 0:
            plot_x = -pts[:, 1]
            plot_y = pts[:, 0] + 0.3
            self.cloud_scatter.set_offsets(np.column_stack([plot_x, plot_y]))
            self.cloud_scatter.set_array(pts[:, 2])
        else:
            self.cloud_scatter.set_offsets(np.empty((0, 2)))

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

        cmd_speed = np.hypot(final_vx, final_vy)
        if cmd_speed > 1e-6:
            dir_x, dir_y = final_vx / cmd_speed, final_vy / cmd_speed
            arrow_u = -dir_y * self._arrow_len_m
            arrow_v = dir_x * self._arrow_len_m
        else:
            arrow_u, arrow_v = 0.0, 0.0
        self.heading_quiver.set_UVC([arrow_u], [arrow_v])
        self.heading_quiver.set_color('red' if is_evading else 'royalblue')

        self.field_patch.set_edgecolor('red' if is_evading else 'green')

        dist_str = "--"
        if obs_pos is not None:
            dist_str = f"{np.hypot(obs_pos[0], obs_pos[1]):.2f} m"

        imu_str = "waiting..."
        if imu_data:
            g = imu_data["gyro"]
            imu_str = f"gyroZ={g[2]:+.2f}"

        evade_str = "DWA evading" if is_evading else "path clear"
        lines = [
            f"Packets: pts={p_cnt}(drop={dropped_cnt}) imu={imu_cnt}",
            f"IMU: {imu_str}",
            f"Opponent dist: {dist_str}",
            f"{evade_str}",
            f"Cmd: raw=({self.fake_raw_vx},{self.fake_raw_vy},{self.fake_raw_wz}) "
            f"-> final=({final_vx},{final_vy},{final_wz})",
        ]
        self.status_text.set_text("\n".join(lines))

        return (self.cloud_scatter, self.enemy_marker, self.enemy_trail_line,
                self.heading_quiver, self.field_patch, self.status_text)

    def run(self, target_fps=30):
        interval_ms = 1000.0 / target_fps
        self._anim = animation.FuncAnimation(
            self.fig, self._update, interval=interval_ms, blit=True, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()

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

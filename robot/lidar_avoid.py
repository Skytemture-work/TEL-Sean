# lidar_avoid.py
import numpy as np
import threading
import time
import config

class LidarAvoidance(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.obstacle_rel_pos = None  
        self.last_grid = None
        self.is_active = True

    def run(self):
        while True:
            if not self.is_active:
                time.sleep(0.1)
                continue

            pts = self._fetch_livox_point_cloud()
            
            if pts is not None and len(pts) > 0:
                # 高度與半徑過濾
                z_mask = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.8)
                dist_xy = np.hypot(pts[:, 0], pts[:, 1])
                range_mask = z_mask & (dist_xy <= config.DETECTION_RADIUS_M)
                valid_pts = pts[range_mask]

                obs_pos = self._extract_dynamic_obstacle(valid_pts)
                
                with self.lock:
                    self.obstacle_rel_pos = obs_pos

            time.sleep(0.04)

    def _fetch_livox_point_cloud(self):
        # TODO: 接入 Livox SDK2 取點雲
        return None 

    def _extract_dynamic_obstacle(self, pts):
        if len(pts) < 10: return None
        grid_size = 0.1
        grid_coords = (pts[:, :2] / grid_size).astype(int)
        current_grid = set(map(tuple, grid_coords))

        dynamic_pos = None
        if self.last_grid is not None:
            dynamic_cells = current_grid - self.last_grid
            if len(dynamic_cells) > 5:
                dynamic_pts = np.array(list(dynamic_cells)) * grid_size
                dists = np.hypot(dynamic_pts[:, 0], dynamic_pts[:, 1])
                min_idx = np.argmin(dists)
                dynamic_pos = dynamic_pts[min_idx]

        self.last_grid = current_grid
        return dynamic_pos

    def apply_priority_override(self, raw_vx, raw_vy, raw_wz):
        """
        評估避障優先級，若有風險則覆蓋底盤運動指令 (DWA 切線超車)
        回傳: (final_vx, final_vy, final_wz, is_evading)
        """
        with self.lock:
            obs = self.obstacle_rel_pos

        if obs is None:
            return raw_vx, raw_vy, raw_wz, False

        obs_dist = np.hypot(obs[0], obs[1])
        if obs_dist > config.SAFE_DISTANCE_M:
            return raw_vx, raw_vy, raw_wz, False

        # --- 觸發 DWA 切線閃避 ---
        best_score = -999999
        best_cmd = (0, 0, 0)

        vx_samples = np.linspace(30, config.MAX_VX, 4)
        vy_samples = np.linspace(-config.MAX_VY, config.MAX_VY, 9)
        wz_samples = np.linspace(-30, 30, 3)

        for vx in vx_samples:
            for vy in vy_samples:
                for wz in wz_samples:
                    pred_x = (vx * 0.6) / 100.0
                    pred_y = (vy * 0.6) / 100.0
                    dist_to_obs = np.hypot(pred_x - obs[0], pred_y - obs[1])

                    if dist_to_obs < config.ROBOT_SAFE_RADIUS_M:
                        continue

                    speed_score = np.hypot(vx, vy)
                    target_score = -np.hypot(vx - raw_vx, vy - raw_vy)
                    safety_score = dist_to_obs * 20.0

                    score = (2.0 * speed_score) + (1.0 * target_score) + (1.0 * safety_score)

                    if score > best_score:
                        best_score = score
                        best_cmd = (int(vx), int(vy), int(wz))

        if best_score == -999999:
            return 0, 0, 0, True # 完全死胡同，緊急煞車

        return best_cmd[0], best_cmd[1], best_cmd[2], True

    def set_active(self, state):
        self.is_active = state
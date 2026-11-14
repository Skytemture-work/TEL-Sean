# config.py

# ==========================================
# 1. UART 通訊設定
# ==========================================
SERIAL_PORT = "/dev/ttyTHS0"  # Xavier UART 腳位
BAUDRATE = 115200

# ==========================================
# 2. 視覺與標靶設定 (ZED 2)
# ==========================================
TARGET_WIDTH_CM = 40.0         # 標靶實際寬度 (cm)
TARGET_HEIGHT_CM = 40.0        # 標靶實際高度 (cm)
ZED_TO_FRONT_M = 0.0             # 實測 ZED 到車體前緣的前向距離
AUTO_CENTER_TOLERANCE_PX = 18
AUTO_VISION_MAX_AGE_S = 0.5
AUTO_MAX_FORWARD = 40           # UART 指令，需依實機煞停距離調整
AUTO_STRAFE_SIGN = 1            # +1: 正 vy 向右，實車方向相反時設為 -1

# MID-360 正前方障礙偵測，依遮擋範圍和安裝位置校正
DISTANCE_PLATE_CM = 260.0       # 車頭到正前方定位面的目標距離
DETECTION_RADIUS_M = 10.0      # LiDAR 前向距離上限（公尺）
FRONT_LIDAR_HALF_ANGLE_DEG = 20.0
FRONT_LIDAR_HALF_WIDTH_M = 0.55
FRONT_LIDAR_MIN_Z_M = -0.2
FRONT_LIDAR_MAX_Z_M = 0.8
FRONT_LIDAR_MAX_AGE_S = 0.25
LIDAR_TO_FRONT_M = 0.0           # 實測 LiDAR 到車體前緣的前向距離
FRONT_STOP_DISTANCE_M = 1.0
FRONT_OBSTACLE_RESUME_WINDOW_S = 3.0
FRONT_BRAKE_TIME_S = 0.15
FRONT_HARD_STOP_M = 0.6
# ==========================================
# 6. 雷達地圖顯示參數 (測試工具用)
# ==========================================
FIELD_LENGTH_M = 4.5           # 場地長度顯示範圍 (公尺)
FIELD_WIDTH_M = 3.0            # 場地寬度顯示範圍 (公尺)

# ==========================================
# 7. DM 手動目標角度邊界（實際角度，單位：度）
# ==========================================
# manual.py 仍會與校正後的機構限位、工作空間取交集。
DM_YAW_MIN_DEG = -34.5
DM_YAW_MAX_DEG = 34.5
# 仰角活動範圍統一由 limits.py 定義；30°以上的水平角限制由 manual.py 套用。
from turret_control.limits import (
    PITCH_MIN_DEG as DM_PITCH_MIN_DEG,
    PITCH_MAX_DEG as DM_PITCH_MAX_DEG,
)

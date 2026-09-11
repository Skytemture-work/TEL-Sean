import time
import sys
import os

# 將上一層目錄加入系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from lidar_avoid import LidarAvoidance

def main():
    print("========================================")
    print(" LiDAR DWA 極速避障單元測試")
    print(" 測試當發生碰撞危險時，優先級覆寫是否生效")
    print("========================================")

    lidar = LidarAvoidance()
    lidar.start()
    lidar.set_active(True)

    # 模擬視覺系統傳來的理想指令 (假設目標在正前方，機器人想全速前進)
    fake_raw_vx = 80
    fake_raw_vy = 0
    fake_raw_wz = 0

    print(f"✅ 啟動測試！視覺原始需求速度 -> VX:{fake_raw_vx}, VY:{fake_raw_vy}\n")

    try:
        while True:
            # 呼叫優先級覆寫函數
            final_vx, final_vy, final_wz, is_evading = lidar.apply_priority_override(
                fake_raw_vx, fake_raw_vy, fake_raw_wz
            )

            if is_evading:
                print(f"🚨 [警告！觸發緊急避障] 對手距離過近！")
                print(f"   -> 原始速度: VX={fake_raw_vx}, VY={fake_raw_vy}")
                # 你會看到 VY 突然有數值，這代表 DWA 正在叫底盤橫移閃避
                print(f"   -> DWA 覆寫: VX={final_vx}, VY={final_vy} (切線超車)")
                print("-" * 40)
            else:
                print(f"🟢 [路線安全] 附近無威脅，維持原速度 (VX={final_vx}, VY={final_vy})")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n測試結束")

if __name__ == '__main__':
    main()
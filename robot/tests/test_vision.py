import time
import sys
import os
import cv2

# 將上一層目錄加入系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_target import VisionTracker

def main():
    print("========================================")
    print(" 視覺標靶 PnP 單元測試 (ZED 2)")
    print(" 請將標靶放置於鏡頭前，觀察解算速度")
    print("========================================")

    # 啟動視覺追蹤背景執行緒
    vision = VisionTracker()
    vision.start()
    
    # 確保視覺模組處於啟動狀態
    vision.set_active(True)

    print("✅ 視覺線程已啟動，等待相機影像...\n")

    try:
        while True:
            # 獲取視覺解算結果
            (vx, vy, wz), (yaw, pitch), detected = vision.get_vision_data()

            if detected:
                print(f"🎯 [已鎖定標靶]")
                print(f"   -> 底盤切線控制: 前後 VX = {vx:4d} | 側向 VY = {vy:4d} | 旋轉 WZ = {wz:4d}")
                print(f"   -> 砲台獨立瞄準: YAW = {yaw:3d} | PITCH = {pitch:3d}")
                print("-" * 40)
            else:
                print("🔍 [尋找標靶中...] 尚未偵測到九宮格角點")

            time.sleep(0.1) # 10Hz 印出頻率

    except KeyboardInterrupt:
        print("\n測試結束")

if __name__ == '__main__':
    main()
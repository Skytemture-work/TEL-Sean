import time
import sys
import os
import cv2

# 將上一層目錄加入系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from vision_target import VisionTracker

def main():
    print("========================================")
    print(" 視覺標靶最密集點測距與姿態單元測試 (ZED 2)")
    print(f" 設定停靠目標距離: {config.DISTANCE_PLATE_CM} cm")
    print(" 按下 'q' 鍵可關閉視窗與測試程式")
    print("========================================")

    # 初始化並啟動視覺背景執行緒
    vision = VisionTracker()
    vision.start()
    vision.set_active(True)

    print("✅ 視覺線程已啟動，開啟即時顯示視窗中...\n")

    try:
        while True:
            (vx, vy, wz), (yaw, pitch), detected = vision.get_vision_data()

            if detected:
                print(f"🎯 [已鎖定標靶] VX={vx:5.1f} | VY={vy:5.1f} | WZ={wz:5.1f} | YAW={yaw:3d} | PITCH={pitch:3d}")
            else:
                print("🔍 [尋找標靶中...] 未偵測到目標物體")

            # 在主執行緒中讀取畫面並渲染 (綠色十字星號代表最密集採樣點)
            frame = vision.get_annotated_frame()
            if frame is not None:
                cv2.imshow("ZED 2 Concentrated Depth Sensing", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User Interrupt] 按下 'q' 鍵，關閉程式...")
                break

            time.sleep(0.02)  # ~50Hz 刷新頻率

    except KeyboardInterrupt:
        print("\n[User Interrupt] 使用者按下 Ctrl+C 結束測試")
    finally:
        vision.set_active(False)
        cv2.destroyAllWindows()
        print("系統資源與視窗已安全關閉。")

if __name__ == '__main__':
    main()

# Turret control

水平／垂直雙軸瞄準控制，給東京威力科創機器人大賽的棒球發射機構用。

## 硬體對應（2026-09-24 實測）

| 軸 | 馬達 | ESC_ID | MST_ID |
|---|---|---|---|
| 水平 yaw | 馬達 2 | `0x03` | `0x13` |
| 垂直 pitch | 馬達 1 | `0x01` | `0x00` |

兩顆的 `MST_ID` **不同**，backend 會開機自動讀取，不要改成寫死。

相機**固定在底座**（`camera_on_turret=False`）—— 砲塔轉動不會改變畫面，所以
「是否瞄準完成」必須看馬達回饋，不能看像素誤差。

## 靶板與接戰順序

3 欄 × 4 列共 12 個孔。**上排 Ø20 cm（小）、下三排 Ø40 cm（大）。**

順序：**大的先** → **逐欄由左到右** → **欄內由下往上**。
水平馬達每欄只移動一次，共 3 次。

```
       欄1        欄2        欄3
上  ⑩ Ø20     ⑪ Ø20     ⑫ Ø20      ← 最後打
    ③ Ø40     ⑥ Ø40     ⑨ Ø40
    ② Ø40     ⑤ Ø40     ⑧ Ø40
下  ① Ø40     ④ Ø40     ⑦ Ø40      ← 從這裡開始
```

### 🟢 孔徑已知 = 單目測距

`distance = focal_px × diameter_m / 像素寬度`

這是距離的來源，**不需要 ZED 深度**。有了距離，視差修正和重力補正才能真的用。
YOLO 若只有一個類別，`classify_by_size()` 會用「大孔是小孔的 2 倍」把兩種尺寸分開；
比例不對時（例如畫面只有一種尺寸）會標成未知，讓開火閘門擋下，而不是亂猜距離。

### 🟢 相機固定在車身 = 三維關節幾何

實機的 `use_axis_geometry=True` 會把每個孔的像素與測距結果轉成相機座標，
再扣除相機到**水平關節**的固定偏移，並換算成水平關節與垂直關節的絕對目標角度。
因此車子停在場上不同位置時，不需要假設靶板在固定像素或固定距離；每一幀會依靶板
目前的距離重新計算角度。`geometry.py` 也會補償水平關節與垂直關節不重合造成的近距離視差。

`run_board_real` 會用靶板 300 × 340 cm 的實際尺寸與至少 4 個非共線孔心做 PnP；完整靶面
四邊形可作為備援，但仍要至少 3 個孔位支持，不能只靠黑色外觀，以免把旁邊鐵架當成靶。
每幀重新投影 H01～H12 並更新距離，但不重建射擊計畫，因此已射球數、
目前索引與計分模式的重複射擊都會保留。車體仍在移動、重投影誤差過大或姿態逾時時，
砲塔可以繼續追瞄，但發射與逾時跳孔都會暫停，直到姿態連續穩定。

螢幕測試的 `--virtual-board` 會自動關閉這段幾何，因為螢幕上的孔徑不代表真實 20/40 cm。

## 模組

| 檔案 | 說明 |
|---|---|
| `config.py` | 所有參數。**大部分仍是暫時值**，見下方 |
| `aiming.py` | 像素 → 角度，含穩定判定與開火閘門 |
| `geometry.py` | 相機座標 → 水平／垂直關節座標與絕對馬達角度 |
| `ballistics.py` | 視差修正 + 重力彈道求解（純數學） |
| `backends.py` | `MockMotorBackend`（安全模擬）與 `DamiaoSerialBackend`（實機） |
| `limits.py` | 馬達座標安全範圍讀寫與最後一道軟體限位 |
| `run_limit_calibration.py` | 失能後手動示教 home 與水平／垂直安全活動範圍 |
| `controller.py` | 接合瞄準與馬達輸出 |
| `control_loop.py` | 200 Hz 固定頻率迴圈 + 偵測執行緒 |
| `board.py` | 靶板模型：測距、尺寸分類、接戰排序 |
| `board_pose.py` | 每幀 PnP 靶面姿態、平滑、移動判定與 12 孔重投影 |
| `dashboard.py` / `dashboard.html` | 區網即時影像、孔位、射擊佇列與砲塔角度儀表板 |
| `engagement.py` | 逐孔接戰驅動，含射擊冷卻與逾時跳過 |
| `detection.py` | YOLO 結果轉靶板目標 |

## 環境限制

⚠ **AGX 的 JetPack R35.4.1 是 Python 3.8.10**，不是 3.9/3.10。

所以**不能直接寫** `tuple[float, float]`、`X | None` 這類語法 —— 型別註解在 def 時就會求值，
會直接 `TypeError: 'type' object is not subscriptable`，連 import 都失敗。

每個檔案開頭都要有：

```python
from __future__ import annotations
```

這在筆電上測不出來（筆電多半是 3.10+），**只有在 AGX 上才會炸**。

`backends.py` 需要 `DM_CAN.py` 和 `dm_port.py`。找不到時會自動去
`../dm_motor_test/` 撈，也可以用 `DamiaoSerialBackend(dm_path=...)` 指定。

## 執行

```bash
# 連續平滑移動（MIT：位置 + 速度前饋，實機前確認急停可用）
python3 -m turret_control.run_manual_move --yaw 10 --pitch 15 --speed 0.30
# 若垂直軸受機構重量影響而有靜摩擦，可逐步小量調整力矩前饋
python3 -m turret_control.run_manual_move --pitch 15 --pitch-tau-ff -3.0

# 砲塔雙軸手動控制面板：預設失能，直接輸入實體角度（度）
python3 -m turret_control.run_vertical_panel
# SSH／無螢幕環境可強制使用終端介面
python3 -m turret_control.run_vertical_panel --cli
# 例如在 CLI 輸入 `set 15`，砲口會平滑移到 +15°
# `yaw 30` 向右 30°，`yaw -20` 向左 20°
# `aim 20 15` 讓水平與垂直在同一控制週期開始移向 +20° / +15°
# `enable` 後會直接詢問垂直、水平目標；之後輸入 `move` 可再次開啟相同問答
# CLI 的 `enable` 與 `disable` 會直接執行，不需要再輸入 `yes`
# 垂直軸預設使用實機已驗證可移動的位置－速度模式；必要時才覆寫為 MIT
python3 -m turret_control.run_vertical_panel --cli --control-mode mit

# 手動示教：失能兩顆馬達；完成 home 後，用手沿安全外框走一圈
# 程式先收集 30 秒；若尚未回到起點會繼續錄，直到閉合或再過 30 秒
# 原始樣本會先自動存到 mechanism_limits_trace_latest.json，再進行輪廓驗證
# 預設從整個輪廓向內保留 0.5 個馬達角度，不會把工作區錯當成長方形
python3 -m turret_control.run_limit_calibration
# 若只重測活動範圍、不更新 home 原點：
python3 -m turret_control.run_limit_calibration --skip-origin

# 角度模擬（不碰硬體）
python3 -m turret_control.run_demo --u 900 --v 200 --distance 5

# 完整靶板接戰（12 個孔，200 Hz，不碰硬體）
python3 -m turret_control.run_board_demo --cooldown 0.2 --log logs/board_engagement.jsonl

# 主迴圈 demo：200 Hz 馬達迴圈 + 30 Hz 模擬偵測
python3 -m turret_control.run_loop_demo --seconds 3
python3 -m turret_control.run_loop_demo --drop-after 1.0   # 驗證目標消失時仍持續送命令

# YOLO + 模擬馬達
python3 -m pip install ultralytics
python3 -m turret_control.run_yolo_demo --device cpu --zed-uvc

# 真實影像靶板預覽：偵測全部 12 孔、尺寸分類、單目測距、接戰排序
python3 -m turret_control.run_yolo_demo --board --device cpu --zed-uvc
# 預覽視窗按 p 可在終端列印目前完整計畫，按 q/Esc 離開

# 實機自動瞄準 + 區網即時儀表板（會連接並使能真實馬達，先確認急停）
python3 -m turret_control.run_board_real --zed-uvc --web-port 8080
# 同一區網的電腦／平板開啟：http://<robot-ip>:8080

# 測試
python3 -m unittest discover -s tests
```

垂直面板預設使用實機已驗證可移動的 `posvel` 位置－速度模式，外層仍使用加速度限制的
連續軌跡（預設速度 `0.06 rad/s`、加速度 `0.05 rad/s²`），並將命令最大領先實際位置限在 10°，
避免機構落後時累積大量位置誤差後突然跳動。預設 `kp=50`、`kd=1.5`、
重力補償前饋 `0 Nm`（非零值會在使能後 1.5 秒漸增，必須先實機確認方向）；若 2 秒內移動不到 0.25°，
會顯示停滯警告但保留原目標，不再偷偷改成目前位置。終端與面板會同時顯示實際扭矩及
命令－回饋誤差，方便區分控制命令、負載與馬達回饋。每次使能只會先保持當前位置，
必須再輸入一次目標才會移動。

面板輸入的是**物理砲口仰角**，換算式為
`馬達角 = 水平砲口馬達角 + 方向 × 機構比例 × 砲口仰角`。
校正值放在 `startup_pose.json`：目前水平砲口約為 `+0.249997 rad`、方向為 `-1`、
直驅比例為 `1.0`。若日後加入連桿或齒輪，請先更新
`motor_rad_per_muzzle_rad`；也可用 `--pitch-sign` 和
`--motor-rad-per-muzzle-rad` 暫時覆寫。砲口仰角安全範圍會由
`mechanism_limits.json` 的馬達端點換算；`--min-angle`／`--max-angle` 只能再縮小，不能放寬。
畫面中的馬達角是編碼器**絕對座標**，正負號不是移動方向；例如砲口從
`+15°` 回到 `0°` 時，垂直馬達目標會增加約 `15°`，砲口則持續向下回到水平。
若在前一個目標尚未到達時改目標，控制器會先按加速度上限煞車，再連續反轉。

同一程序會同時保持水平與垂直兩顆馬達。水平輸入是以目前正前方校正點為 `0°` 的
絕對角度，正值向右、負值向左，限制會由 `mechanism_limits.json` 的水平馬達端點換算（目前暫值為
`-35°～+35°`）。水平安全聯鎖使用垂直馬達
實際回饋：砲口仰角 `≤35°` 才允許水平移動。高於 `35°` 時若收到水平目標，控制器會
先降低垂直；回饋確認到達 `35.0°` 或更低後，水平立即開始、垂直繼續同時移動。
若最終垂直目標仍高於 `35°`，會先以 `34.5°` 作為安全中繼點；確認水平與垂直都到達
中繼姿態後，再回到最終垂直角度。`home` 使用相同流程回到水平 `0°`、垂直原點 `45°`。
實機方向校正為 `yaw_sign=-1`：
馬達編碼器角度減少時，砲塔實際向右。

CLI 的 `aim <水平角度> <垂直角度>` 在目前與目標垂直角度都不高於 `35°` 時，會讓兩軸在
下一個 200 Hz 控制週期一起開始平滑作動；需要跨越高角度區時則自動改用上述三階段避讓流程。
`enable` 與 `disable` 會在輸入後直接執行，不會再要求輸入 `yes`；其中 `disable`
可能讓垂直軸下墜，輸入前仍需先支撐機構。一般的 `set`、`yaw`、`aim` 也都不需要確認。

若機械原點有偏差，可在 CLI 輸入 `calibrate`，或使用上面的
`run_limit_calibration` 一次記錄 home 與安全範圍。程式會先要求確認並失能兩軸；確認
兩顆 ERR 都是 0 後，手動把機構放到「水平 0°、垂直初始 45°」，再輸入
`capture-origin` 並二次確認。程式只更新軟體校正檔，不改馬達內部零點；舊的
`startup_pose.json` 會備份成 `startup_pose.json.bak`。擷取後維持失能，必須等人員
離開機構才可重新 `enable`。

`--board` 模式只開相機與 YOLO，不開 CAN、不建立馬達控制器；必須偵測到 12
個孔且尺寸分類成功，畫面才會顯示 `BOARD READY`。這是接上真機構前的影像流程驗證。

推送到隊友專案後，YOLO 程式預設使用 `robot/best_v2.pt`；舊版
`robot/best.pt` 會保留，不會被覆蓋。也可以用 `--model` 指定其他權重。

靶板孔洞有固定空間 ID：`H01`～`H09` 是三欄由下往上的大孔，`H10`～`H12`
是上排小孔。這個 ID 和接戰順序分開；目前第一個接戰目標是底部的 `H01`，
最後才是上排的 `H10`～`H12`。
`BoardEngagement` 會把 `fired`／`skipped` 事件留在記憶體，若指定 `--log` 則另外
以一行一筆 JSONL 持久保存，包含孔洞 ID、欄列、像素位置、孔徑和距離。

`fired` 代表射擊 callback 回報「球已送出」，不是相機確認球已穿過孔洞；命中確認仍需
另外接射後影像或感測器。

## 不可以拿掉的安全性質

**1. 每一圈都要送馬達命令。** 馬達有通訊丟失防護，停止送命令會自動失能；
垂直軸沒有煞車，失能就直接掉下來。沒有目標時要 `hold()`，不是什麼都不做。

**2. 每個實機位置命令都會在 DAMIAO backend 最後檢查 `mechanism_limits.json`。**
超出各軸範圍或描繪出的 yaw/pitch 多邊形工作區時，目標會停在第一個安全邊界內；這是軟體保護，不能取代急停、機械止擋或人員支撐。

**3. 射擊要有冷卻。** 迴圈 200 Hz，`fire_allowed` 會連續成立 —— 不擋的話一秒射 200 發。
`BoardEngagement` 的 `shot_cooldown_s` 負責這件事，別拿掉。

**4. 開火閘門 fail closed。** 下列任一條成立就不開火：

- 沒有目標或信心度不足
- 馬達還沒到位（看回饋，不是看像素誤差）
- 沒有馬達回饋（無法確認就不開火）
- 目標超出機構角度限位（`clamp()` 夾到限位 ≠ 瞄準完成）
- 目標超出彈道射程
- 需要距離但沒有距離
- 靶面 PnP 無效、已逾時，或車體／相機仍在移動

## 還是暫時值的參數

這些**必須在真機構上量過才能用**：

| 參數 | 現值 | 要怎麼得到 |
|---|---|---|
| `focal_length_x/y_px` | 700.0 | 相機標定 |
| `camera_principal_x/y_px`、`camera_distortion_coeffs` | 畫面中心／零畸變暫值 | 同一解析度的相機內參標定；ZED rectified 左圖畸變為 0 |
| `yaw_zero_rad` / `pitch_zero_rad` | 由 `startup_pose.json` 載入 | 手動校正後於下次啟動自動套用 |
| `yaw_sign` / `pitch_sign` | 1 / −1 | 垂直方向已確認為馬達角增加時砲口下降 |
| `mechanism_limits.json` 的 yaw/pitch 工作區 | 目前是由舊校正推導的保守暫值 | 請用 `run_limit_calibration` 沿安全外框手動描邊 30 秒重測 |
| `camera_to_yaw_yaw/pitch_rad` | 0.0 | 炮塔零點時，相機光軸相對水平軸的偏角 |
| `yaw_axis_offset_x/y/z_m` | 0.0 | 量相機原點到水平關節原點，使用相機座標 |
| `pitch_axis_offset_x/y/z_m` | 0.0 | 量水平關節到垂直關節，使用水平零點座標 |
| `muzzle_offset_x/y/z_m` | 0.0 | 舊版彈道補償用：量相機到發射口的距離 |
| `muzzle_speed_m_s` | 20.0 | 實際測球速 |

**距離已經有解**：用已知孔徑反推（見上面）。但這需要 YOLO 的框寬夠準 ——
框抖動會直接變成距離抖動，進而抖動仰角。實機上要量一下框寬的穩定度。

參考：5 公尺外、20 m/s 的球若不補重力會低約 **30 公分**；10 公分的視差偏移在 2 公尺處是 **2.9°**。

## 驗證狀況（2026-09-24）

| 項目 | 狀態 |
|---|---|
| 65 個單元測試 | ✅ 在 AGX 的 Python 3.8 上全過 |
| `DamiaoSerialBackend` 實機 | ✅ 兩顆馬達**分別**測過：自動讀 MST_ID、200 Hz hold 漂移 0.0000/0.0004 rad |
| 全靶板接戰模擬 | ✅ 12/12，200.0 Hz，0 圈來不及 |
| **兩顆馬達同時上線** | ❌ **沒測過** —— 只有一條 CAN 線，需要菊花鏈或 Y 型分接線 |
| 真實 YOLO 影像靶板流程 | ✅ `run_yolo_demo --board` 已接上；待現場影像實測 |
| 孔洞 ID／接戰事件紀錄 | ✅ `H01`～`H12`，可用 `--log` 輸出 JSONL |
| 真機構 | ❌ 製作中 |

兩顆併線時記得：CAN 匯流排**只能頭尾兩端各一顆 120Ω**（量起來 60Ω）。
三顆並聯變 40Ω 會讓整條線失效。

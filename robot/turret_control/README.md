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

## 模組

| 檔案 | 說明 |
|---|---|
| `config.py` | 所有參數。**大部分仍是暫時值**，見下方 |
| `aiming.py` | 像素 → 角度，含穩定判定與開火閘門 |
| `ballistics.py` | 視差修正 + 重力彈道求解（純數學） |
| `backends.py` | `MockMotorBackend`（安全模擬）與 `DamiaoSerialBackend`（實機） |
| `controller.py` | 接合瞄準與馬達輸出 |
| `control_loop.py` | 200 Hz 固定頻率迴圈 + 偵測執行緒 |
| `board.py` | 靶板模型：測距、尺寸分類、接戰排序 |
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

# 測試
python3 -m unittest discover -s tests -t .
```

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

## 三個不可以拿掉的安全性質

**1. 每一圈都要送馬達命令。** 馬達有通訊丟失防護，停止送命令會自動失能；
垂直軸沒有煞車，失能就直接掉下來。沒有目標時要 `hold()`，不是什麼都不做。

**2. 射擊要有冷卻。** 迴圈 200 Hz，`fire_allowed` 會連續成立 —— 不擋的話一秒射 200 發。
`BoardEngagement` 的 `shot_cooldown_s` 負責這件事，別拿掉。

**3. 開火閘門 fail closed。** 下列任一條成立就不開火：

- 沒有目標或信心度不足
- 馬達還沒到位（看回饋，不是看像素誤差）
- 沒有馬達回饋（無法確認就不開火）
- 目標超出機構角度限位（`clamp()` 夾到限位 ≠ 瞄準完成）
- 目標超出彈道射程
- 需要距離但沒有距離

## 還是暫時值的參數

這些**必須在真機構上量過才能用**：

| 參數 | 現值 | 要怎麼得到 |
|---|---|---|
| `focal_length_x/y_px` | 700.0 | 相機標定 |
| `yaw_zero_rad` / `pitch_zero_rad` | 0.0 | 機械零點對位 |
| `yaw_sign` / `pitch_sign` | 1 | 馬達裝上去後看轉向 |
| `yaw_min/max`、`pitch_min/max` | ±80° / −20°~35° | 機構實際行程 |
| `muzzle_offset_x/y/z_m` | 0.0 | 量相機到發射口的距離 |
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

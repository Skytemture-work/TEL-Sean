# 視覺、砲塔與發射器整合介面

這份文件是給主控與發射機構組直接對接用的。正式靶板流程以
`turret_control/run_board_real.py` 為唯一入口；`vision_target.py`、`main.py`
與 `comm_uart.py` 是早期相容流程，角度單位與控制架構不同，不要混接。

## 1. 整體資料流

```text
ZED SDK 校正後左眼影像 + 左右眼立體深度
  → YOLO segmentation mask
  → 黑色靶面與孔位支持檢查（排除場外亮點、黑色鐵架）
  → 官方 3×4、±2 cm 幾何檢查／缺孔補點
  → 每幀 PnP 靶面姿態與 H01～H12 重投影
  → 優先 ZED 靶面深度平面；無有效平面時退回單孔深度或 PnP 幾何
  → 相機／兩軸／轉動發射孔的幾何修正（目前按直線出球）
  → DAMIAO 水平／垂直絕對目標角（rad）
  → 共用手動馬達執行緒：檢查路徑、降仰角、水平移動、回復仰角
  → 馬達回饋到位 + 安全閘門
  → fire_callback(target)
```

主流程自己擁有相機、YOLO、PnP、射擊計畫和兩顆 DAMIAO 馬達。外部整合程式只需：

1. 實作一次發射的 callback。
2. 需要狀態時讀 HTTP API，或傳入 `telemetry_callback`。
3. 不要再根據紫色線或像素位置自行算第二套角度。

自動與手動使用同一套雙軸馬達工作執行緒（不能同時啟動兩個行程搶同一序列埠）。自動模式採 `posvel`、速度 `0.06 rad/s`、加速度 `0.05 rad/s²`；避讓完成、回饋新鮮且馬達停止後才能發射。移動中更新的目標會等目前避讓完成再重新規劃。路徑以馬達角度不超過 `0.1°` 的間隔檢查，仍需實機驗證。

依操作人員確認，活動範圍為仰角 `0°～30°`／水平 `±34.5°`，`30°～33°` 依三角形收窄，`33°～home` 水平 `±10°`；home 保留已記錄的馬達位置，以水平零點和已確認的 `1:1` 比例換算約 `44.63°`。活動多邊形沒有再額外內縮 `0.5°`；水平 `±34.5°` 本身相對機構設計 `±35°` 保留 `0.5°`。

實機入口採用 CAD 設計幾何：相機固定、仰角 `20°`，左眼在旋轉軸左側 `60 mm`；相機中心在旋轉軸前方 `101.662 mm`，並比基準面高 `1.998 mm`；垂直軸位於水平軸上方 `72.1 mm`。設計 home `45°` 時發射孔相對垂直軸前方 `150 mm`、上方 `96.5 mm`，發射孔與仰角一起轉動。此模型假設左眼光學中心與 CAD 相機中心同高、同前後位置，且相機無橫滾；仍須確認實機裝配及光學中心。

依操作人員決定，目前採**直線出球、不做重力或空氣阻力補償**，距離仍用於幾何視差；球速 `20 m/s` 的舊預設不會用於實機瞄準。輕球不代表軌跡一定是直線，需在實際距離與球速下驗證落點。若將來開啟重力補償，須先實作轉動發射孔與彈道的聯合求解；目前組態會拒絕這種組合。

孔位 ID 固定如下；計分策略可能讓同一個孔在射擊序列中出現多次。

```text
H10  H11  H12    上排 Ø20 cm
H03  H06  H09
H02  H05  H08    下三排 Ø40 cm
H01  H04  H07
```

## 2. 直接執行完整流程

從 `robot/` 目錄執行：

```bash
python3 -m turret_control.run_board_real \
  --model ./best.pt \
  --device 0 \
  --port /dev/ttyACM0 \
  --match-phase upper \
  --cooldown 1.0 \
  --target-timeout 6.0
```

- `--port` 是 DAMIAO 轉接器，不是發射器的 VESC port；兩套硬體不要搶同一個裝置。
- 預設使用 ZED SDK 的 HD720 校正後左眼影像、配對深度和 SDK 回傳的左眼相機內參；需要安裝可用的 ZED SDK 與 `pyzed`。SDK 無法開啟時會報錯，不會默默切換相機。
- `--zed-uvc` 明確切換到不含雙眼深度的 UVC 原始影像；可搭配 `--zed-calibration /usr/local/zed/settings/SN實際序號.conf` 載入該相機原廠左眼原始影像內參。`--opencv-camera` 則使用一般 OpenCV 相機，`--camera` 選擇兩種 OpenCV 相機的編號。
- `--match-phase` 依當場半場改成 `upper` 或 `lower`。
- `--target-timeout` 是一個孔等不到成功發射時的略過秒數；整合測試可設 `0` 停用自動略過。
- `--headless` 可關閉本機 OpenCV 視窗，HTTP 儀表板仍可用。
- 網頁儀表板預設使用 `8080` 埠；`--web-port 其他埠號` 可更改，`--web-port 0` 可停用。
- 競賽計分模式預設啟用，按 `G` 開始辨識紅燈；`--no-score-mode` 可切換一般模式。既有的 `--score-mode` 參數仍可使用。
- 競賽模式啟用少孔遮蔽備援：最近一次穩定 PnP 後最多 1 秒內，若仍有至少一個符合位置、尺寸及信心度的孔，會以原距離和朝向做平移追蹤；單次降級期間最多成功發射一球。每兩次有效姿態間隔不得超過 0.25 秒，之後需重新取得穩定 PnP。
- 模型預設依序尋找 `best_v2.pt`、`best.pt`、訓練目錄下的 `best.pt`，正式整合仍建議明確傳 `--model`。
- 這個命令會連接並使能真實 DAMIAO 馬達；先確認急停、機械限位與支撐。

命令列預設仍是人工確認模式：

- `G`：比賽哨聲後開始辨識三個小孔的紅燈，建立當回合最高分計畫。
- `F`：只確認目前一發。現在的預設 callback 只記錄成功，尚未控制實際發射器。
- `S`：略過目前孔位。
- `Q` / `Esc`：離開並關閉裝置。

少孔備援不能量測靶板或相機的旋轉、距離變化，也無法保證單孔不會誤匹配；應在比賽場地驗證。完全看不到孔或更新失敗時不會盲射。儀表板 `pose.source` 為 `pnp` 或 `partial`；後者的位置殘差是可見孔的平移一致性誤差，不代表完整姿態誤差。

## 3. Python 內嵌呼叫（建議整合方式）

`run_board_real.main()` 現在接受兩個明確的整合 callback：

```python
from queue import SimpleQueue

from turret_control.run_board_real import main as run_board


class Launcher:
    def ready(self):
        # 飛輪已到速、推球器已復位、實體安全開關允許
        ...

    def fire_once(self, target):
        """成功送出一顆球並收到硬體確認才回傳 True。"""
        if not self.ready():
            return False

        return self.trigger_and_wait_ack(
            hole_id=target.hole_id,
            distance_m=target.distance_m,
        )

    def close(self):
        # 停推球、停止飛輪並關閉通訊
        ...


launcher = Launcher()
telemetry_queue = SimpleQueue()


def on_telemetry(state):
    # callback 在控制執行緒內，這裡只做快速複製或放入 queue。
    telemetry_queue.put(state)


try:
    run_board(
        [
            "--model", "./best.pt",
            "--device", "0",
            "--port", "/dev/ttyACM0",
            "--web-port", "8080",
            "--match-phase", "upper",
            "--cooldown", "1.0",
            "--target-timeout", "6.0",
        ],
        fire_callback=launcher.fire_once,
        telemetry_callback=on_telemetry,
    )
finally:
    launcher.close()
```

傳入 `fire_callback` 就代表啟用外部自動發射：當所有安全條件成立時，主流程會直接呼叫它，
不再等待鍵盤 `F`。若不傳，行為維持人工 `F` 確認。

### `fire_callback` 合約

型別：

```python
Callable[[BoardTarget], bool]
```

輸入 `BoardTarget` 的常用欄位：

| 欄位 | 單位 | 說明 |
|---|---:|---|
| `hole_id` | — | `H01`～`H12` |
| `distance_m` | m | 孔位光軸深度；優先 ZED 靶面平面，再依序使用單孔深度與 PnP |
| `distance_source` | — | `zed_plane`：ZED 深度平面；`zed_depth`：孔位邊緣深度；`board_pose`：PnP 估算 |
| `u_px`, `v_px` | px | 孔心影像座標 |
| `diameter_m` | m | 小孔 `0.20`、大孔 `0.40` |
| `confidence` | 0～1 | YOLO 信心度 |
| `column` | 0～2 | 左到右 |
| `row` | 0～3 | 下到上 |

回傳值的語意必須一致：

- `True`：硬體已確認這一顆球真的送出。射擊計畫會記成 `fired` 並前進一格。
- `False`：沒有送出、尚未 ready、正在等待 ACK，或發射失敗。計畫停在同一格。
- callback 丟出例外：主流程離開並進入清理；不要吞掉未知的硬體錯誤後回 `True`。

`BoardEngagement` 會以 `--cooldown` 限制所有 callback 嘗試；即使回傳 `False`，也不會被
影像／馬達迴圈每幀重複呼叫。callback 仍應快速返回，建議不要在控制執行緒長時間等待。
若硬體 ACK 可能超過幾十毫秒，請在發射器物件內使用 worker/state machine：第一次呼叫送出
帶流水號的請求並回 `False`，之後只在收到同一流水號的 `FIRED` ACK 時回一次 `True`。
同一加分孔可能連續投多球，不能只用 `hole_id` 去重。

`True` 只代表發射器確認球已送出，不代表影像已確認穿孔或得分。

### callback 被呼叫前已通過的條件

- 已有目前射擊目標，且信心度足夠。
- 相機解析度符合標定內參。
- 靶面 PnP 有效、未逾時，且連續穩定。
- 距離可用，視差／彈道計算成功。
- 目標未超出機械限位且位於發射孔前方。
- 水平、垂直馬達都有回饋，且連續到位。
- 上一個發射嘗試已超過 cooldown。

實體發射安全開關、飛輪到速、推球器復位與彈藥狀態仍由發射器 callback 再檢查。

## 4. 狀態與影像的讀取方式

儀表板預設啟用於 `8080` 埠，提供：

| 路徑 | 內容 |
|---|---|
| `GET /` | 完整儀表板 |
| `GET /stream` | 標註後的 MJPEG 即時影像 |
| `GET /api/status` | JSON 狀態，與 `telemetry_callback` 收到的資料相同 |
| `GET /health` | `ok` |

例如：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/status
```

```python
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8080/api/status", timeout=0.2) as response:
    state = json.load(response)

next_hole = state["target"]["hole_id"]
can_fire = state["firing"]["allowed"]
yaw_change_deg = state["target"]["yaw"]["delta_deg"]
pitch_change_deg = state["target"]["pitch"]["delta_deg"]
```

主要 JSON 欄位：

| JSON 路徑 | 說明 |
|---|---|
| `vision.raw_detections` | YOLO 原始候選數 |
| `vision.board_detections` | 通過靶面篩選的候選數 |
| `vision.board_surface_found` | 是否找到有孔位支持的黑色靶面 |
| `pose.state` / `pose.ready` | `invalid`、`settling`、`ready`；只有 ready 可發射 |
| `pose.reprojection_error_px` | PnP 重投影誤差 |
| `target.hole_id` / `distance_m` / `distance_source` | 即將發射的孔、光軸深度與距離來源 |
| `target.yaw.current_deg` / `target_deg` / `delta_deg` | 砲口水平目前、目標、尚需變化量 |
| `target.pitch.current_deg` / `target_deg` / `delta_deg` | 砲口垂直目前、目標、尚需變化量 |
| `target.yaw.target_motor_deg` | DAMIAO 編碼器絕對目標，只供診斷 |
| `firing.allowed` | 軟體瞄準閘門是否已開 |
| `firing.blocked_reason` | 尚不可發射的原因 |
| `firing.fired_this_frame` | 只在 callback 成功的那一幀為 true |
| `plan.fired_holes` | 至少已發射一次的孔位 |
| `plan.unfired_holes` | 尚未發射的孔位 |
| `plan.sequence` | 完整逐球序列；包含重複加分孔與每球狀態 |
| `score.active_bonus_hole` | 目前紅燈指定的小孔 |

`delta_deg` 是給人看的物理砲口變化量；主流程內部送給 DAMIAO 的是絕對馬達角 `rad`。
外部程式不要把 dashboard 的角度再送回馬達，否則會形成兩個控制器互相搶命令。

HTTP server 綁定 `0.0.0.0` 且目前沒有驗證，僅限比賽區域網路使用。瀏覽器若從另一個
網域自行開發前端，還要處理同源／CORS；Python、ROS 或同一個 dashboard 不受影響。

## 5. 若只想使用底層物件

每次 `driver.update()` 都會同時：讀馬達回饋、送新的水平／垂直命令、檢查開火條件，
並在符合條件時呼叫發射器。回傳 `EngagementStatus`：

```python
status = driver.update()

status.target                 # BoardTarget 或 None
status.command.fire_allowed   # bool
status.command.target_yaw_rad
status.command.target_pitch_rad
status.command.yaw_servo_error_rad
status.command.pitch_servo_error_rad
status.command.blocked_reason
status.fired                  # 本 tick 是否確認發射
status.skipped                # 本 tick 是否略過
status.remaining
status.finished
```

所有 `*_rad` 都是弧度。`target_yaw_rad`、`target_pitch_rad` 是 DAMIAO **絕對編碼器目標**，
不是「還要轉多少」。實體程式必須持續呼叫 `update()`；沒有目標時控制器仍會 `hold()`，
不能停止送 CAN frame，因為垂直軸沒有煞車。

一般整合不應另外呼叫這層；讓 `run_board_real.main()` 保持相機、計畫與馬達的單一擁有者較安全。

## 6. 舊版 UART 介面不能直接接新砲塔

舊版 `VisionTracker.get_vision_data()` 回傳：

```python
((vx, vy, wz), (yaw, pitch), detected)
```

其中 `yaw`、`pitch` 是 `0～100` 的舊控制量、更新率約 1 Hz，不是角度也不是 DAMIAO rad。
新靶板流程則使用 PnP、固定孔位、彈道補償與絕對馬達 rad；兩者不可互換。

`UARTController.send_packet()` 的舊封包是：

```text
X,A,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n
```

目前 `controller.ino` 只用 `VX/VY/WZ` 驅動底盤。`YAW/PITCH/FIRE/SPEED` 只被保存並在
`E,...` 狀態封包中回傳；`FIRE` 會再受遙控器安全開關限制，但程式內沒有飛輪、推球馬達或
「球已送出」ACK。因此現有 UART 的 `FIRE=1` 不能當成新流程的成功回覆。

`tests/control_vesc_rpm.py` 與 `tests/mks_2reload.py` 也是單機、阻塞式測試程式，不能直接
import 到影像控制迴圈。發射機構組需要把其中的硬體操作包成非互動式 `Launcher`，提供：

- 飛輪預備／到速狀態。
- 單次推球命令與唯一流水號。
- 實際送球 ACK。
- reload 完成狀態。
- `close()`／急停時停止推球與飛輪。

目前實機瞄準不使用發射速度做彈道修正。若實測落點顯示需要補償重力或阻力，須量測出球速度、重新實作並驗證模型；UART 的舊 `speed=0～100` 不是 m/s。

## 7. 建議整合驗收順序

1. `python3 -m turret_control.run_yolo_demo --board --model ./best.pt --zed-uvc`：只驗證影像，不開 CAN。
2. `python3 -m turret_control.run_board_demo --cooldown 0.2`：只用模擬馬達跑完整計畫。
3. 真馬達、不接外部 callback：確認靶面、PnP、角度、限位與儀表板。
4. 接 dry-run callback，固定回 `False`：不得前進孔位，也不得高頻重送。
5. 每次只對一個 request 回一次 `True`：計畫必須只前進一球。
6. 測試同一紅燈孔連續多球，每球都要有不同發射流水號。
7. 移動車體、遮住靶面、拔掉馬達回饋：`firing.allowed` 必須變成 false，callback 不得發射。
8. 最後才接球，以實體安全開關、急停與機械防護做低速單發測試。

正式自動發射前仍須確認 ZED SDK 使用該相機的標定，並以實機驗證 CAD 安裝位置、方向、
活動範圍與不同距離的落點；程式會計算角度不代表已完成實機校準。

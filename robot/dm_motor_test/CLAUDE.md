# DM-J4340P-2EC 馬達測試

達妙科技 DM-J4340P-2EC 關節馬達的測試與控制。馬達通訊與三種控制模式**已在 AGX Xavier
上驗證完成**（見下面「已驗證」兩節）。

## 這是什麼專案

**東京威力科創機器人大賽**的棒球發射機構。用 YOLO 認靶板上的孔，兩顆 DM-J4340P-2EC
分別控制水平轉向與垂直俯仰，把棒球打進去。機器人上的電腦是 NVIDIA AGX Xavier。

這個資料夾（`dm_motor_test`）只放**馬達底層驗證工具**。
上層的瞄準／彈道／接戰邏輯在隔壁：

```
~/Documents/
├── dm_motor_test/      ← 這裡：馬達底層、DM_CAN.py、01~03 測試腳本
├── turret_control/     ← 瞄準、彈道、靶板接戰、200Hz 控制迴圈（見其 README.md）
└── tests/              ← turret_control 的單元測試（65 個）
```

`turret_control/backends.py` 會 import 這裡的 `DM_CAN.py` 和 `dm_port.py`，
找不到時自動往 `../dm_motor_test/` 撈。**改動這兩個檔案會影響上層。**

### ⚠ Python 3.8

JetPack R35.4.1 是 **Python 3.8.10**。`tuple[float, float]`、`X | None` 這類 3.9/3.10
語法在型別註解位置會直接 `TypeError`，連 import 都失敗。每個檔案要加
`from __future__ import annotations`。**這在筆電（3.10+）上測不出來，只有 AGX 會炸。**

## 硬體

- **馬達**：DAMIAO DM-J4340P-2EC，馬達驅動一體、雙編碼器（輸出軸單圈絕對位置）
- **通訊板**：達妙 USB-to-CAN 調試板，HDSC CDC Device，**VID:PID = `2e88:4603`**
- **電源**：24V，建議 ≥ 5A（額定 2.5A / 峰值 8A）

### 規格（官方手冊 V1.0，2025.01.10）

| 項目 | 值 |
|---|---|
| 額定電壓 | 24V（另有 48V 版） |
| 額定 / 峰值扭矩 | 9 N·m / 27 N·m |
| 減速比 | 40:1 |
| 額定 / 空載最高轉速 | 36 rpm / 52 rpm（**≈ 5.4 rad/s 輸出軸**） |
| 控制介面 | CAN @ 1 Mbps 標準幀 |
| 保護 | 過壓建議 ≤ 32V、欠壓 ≥ 15V、驅動過溫 120°C |

**位置與速度的回饋、命令都是「輸出軸」的量**，不是轉子端。所以 `1 rad/s` 就已經很快了。

### 兩顆馬達的 ID（2026-09-24 確認）

| | ESC_ID | MST_ID | 用途 |
|---|---|---|---|
| 馬達 1 | `0x01` | `0x00`（出廠預設） | **垂直**俯仰 pitch |
| 馬達 2 | `0x03` | `0x13` | **水平**旋轉 yaw |

⚠ **馬達 2 不是出廠預設值**，已經被預先設定過。兩顆 ID 不衝突，**可以直接併線，不需要跑
`03_set_id.py`**（省一次 flash 擦寫）。

⚠ **兩顆的 `MST_ID` 不同**，這會咬人：`addMotor()` 只在 `MasterID != 0` 時才註冊
`motors_map[MasterID]`（[DM_CAN.py:322-324](DM_CAN.py#L322-L324)）。如果對馬達 2 用
`MasterID=0`，它的回饋幀（`CANID=0x13`）會被丟掉，位置永遠讀到 0，**而且不會報錯**。
`turret_control/backends.py` 已改成開機自動讀 `MST_ID`，不要改回寫死。

⚠ **兩顆同時上線還沒測過** —— 目前只有一條 CAN 線，只能一次接一顆。
併線時 CAN 匯流排**只能頭尾兩端各一顆 120Ω**（量起來 60Ω）；三顆並聯變 40Ω 會整條失效。
需要菊花鏈或 Y 型分接線。

兩顆分別測過的結果（`turret_control` 的 `DamiaoSerialBackend`）：
自動讀出 MST_ID 正確、200 Hz `hold()` 漂移分別是 0.0000 / 0.0004 rad。

### 這顆的實測值（2026-08-11 讀出，馬達 1）

```
CAN ID (ESC_ID)     0x01
MST_ID              0x00        ← 出廠預設，函式庫有 fallback 能正確收到回饋
CTRL_MODE           1 (MIT)     ← 出廠預設
Gr                  40.0
PMAX / VMAX / TMAX  12.5 / 10.0 / 28.0
ACC / DEC           2.0 / -2.0
MAX_SPD             600.0       ← posvel 追不上的真正原因
sw_ver              925970741 = 0x37333535 = ASCII "5537"（新版韌體）
```

⚠ **`VMAX = 10.0` 不等於函式庫 `DM4310`→`DM4340` 內建的 `8.0`。**
`02_move.py` 已改成開機自動讀暫存器並呼叫 `change_limit_param()` 對齊，**不要移除這段**。

未確認：`VMAX=10` 剛好等於函式庫裡 `DM4340_48V` 的值。馬達標籤是 24V 還是 48V 版尚未核對。
（VMAX 只是映射範圍暫存器，不足以判定版本。24V 版餵 48V 會過壓，反之只是無力。）

## 檔案

| 檔案 | 說明 |
|---|---|
| `DM_CAN.py` | 官方函式庫，**原版未修改**，來自 github.com/cmjang/DM_Control_Python |
| `dm_port.py` | 用 VID:PID 自動找調試板，找不到會列出現有序列埠 |
| `01_check.py` | 只讀參數 / 掃 CAN ID `0x01`~`0x08`，馬達不會動 |
| `02_move.py` | vel / posvel / mit 三種模式的運轉測試 |
| `03_set_id.py` | 改 ESC_ID / MST_ID。**匯流排上只能有一顆**，預設不寫 flash |
| `99-damiao.rules` | udev 規則，把裝置固定成 `/dev/damiao` |

## 常用指令

```bash
python3 01_check.py                                        # 先跑這個，確認通訊
python3 02_move.py --mode vel    --vel 0.5                 # 速度模式
python3 02_move.py --mode posvel --amp 1.57 --vel 2.5 --set-zero
python3 02_move.py --mode mit    --amp 0.5 --kp 30 --kd 1.0
```

`--port` 不給就自動偵測。Ctrl+C 會先把命令歸零、失能，再關埠。

⚠ 沒重新登入過的 shell 要包 `sg dialout -c "..."`，見下面「環境」。

上層（`turret_control`，全部不碰硬體）：

```bash
cd ~/Documents
python3 -m unittest discover -s tests -t .            # 65 個測試
python3 -m turret_control.run_board_demo --cooldown 0.2   # 12 個孔完整接戰
python3 -m turret_control.run_loop_demo --drop-after 1.0  # 驗證目標消失仍持續送命令
```

## 環境

AGX 實際狀況：`Python 3.8.10`、`numpy 1.24.4`（JetPack 內建）、`pyserial 3.5`（`pip --user`）、
`ultralytics 8.0.200`。**沒有裝 `python-can`，也沒有任何 `can0` 介面** —— 見陷阱 7。

- `pyserial`、`numpy`
- 使用者必須在 **`dialout`** 群組（`sudo usermod -aG dialout $USER`，需重新登入），
  否則 `/dev/ttyACM*` 開啟時 `PermissionError: [Errno 13]`
- ⚠ `usermod` **不會影響已經開著的 shell** —— 程序憑證是登入時決定的，改 `/etc/group`
  對現有 session 無效。不想登出就用 `sg dialout -c "python3 01_check.py"` 包起來跑。
  重新登入後就不必了。

## 已驗證（2026-08-11，桌機 Ubuntu）

三種模式全部通過：

- **vel**：15 秒正弦速度命令的位置積分理論值 0.9549 rad，實測 +0.954 rad（誤差 < 0.1%）
- **mit**：`kp=30 kd=1.0`，零交越時速度理論值 -0.524 rad/s，實測 -0.525 rad/s；
  位置落後 0.039 rad（無速度前饋時的正常穩態誤差）
- **posvel**：功能正常。首次測試用 `--vel 1.0` 出現明顯落後，**原因是速度飽和不是故障** ——
  `--amp 1.57 --period 6` 的峰值速度需求是 `1.57 × 2π/6 = 1.644 rad/s`，超過限速值

空載摩擦扭矩約 0.4~0.6 N·m。

## 已驗證（2026-08-11，AGX Xavier）

三種模式在 AGX 上全部重跑通過，數據與桌機一致：

- **vel**：`--vel 0.5`，行程實測 0.954 rad vs 理論 0.9549（誤差 0.1%），峰值速度 ±0.501
- **mit**：`kp=30 kd=1.0`，零交越速度理論 0.5236 rad/s、實測 ~0.515；位置落後 **0.0375 rad**
  （桌機是 0.039 rad，吻合）。振幅衰減 2.8%（命令 0.5、實測 0.486~0.488）
- **posvel**：`--amp 1.57 --vel 2.5`，仍有落後 —— 見下面「posvel 的速度上限」

迴圈頻率實測 **200.0 Hz，來不及的圈數 0/3000**（修正排程後，見陷阱 1）。

### ⚠ posvel 的速度上限不只是 `--vel`

即使 `--vel 2.5` 遠大於需求的 1.644 rad/s，實測峰值速度仍只到 **1.30 rad/s**，
位置只到 ±1.24（命令 ±1.57），且峰值落後命令約 0.57 秒。

讀出來的暫存器：`ACC = 2.0`、`DEC = -2.0`、`MAX_SPD = 600.0`。

`MAX_SPD=600` 若是轉子端 rpm，換算輸出軸是 `600/40 = 15 rpm ≈ 1.571 rad/s`，
剛好低於需求的 1.644 —— 加上 ACC/DEC 梯形斜坡的爬升時間，就得到實測的 1.30。
**所以 `--vel` 只是上層要求，真正的天花板是馬達內部的 `MAX_SPD` 與 ACC/DEC。**

未確認：`MAX_SPD` 的單位（轉子 rpm 是推測，實測 1.30 與推算的 1.571 並不完全吻合）。
要讓 posvel 跟得上就得調高 `MAX_SPD`／`ACC`／`DEC`，但**寫暫存器前先看陷阱 5**。

## 陷阱

1. **必須持續送命令**。有「通訊丟失防護」，設定週期內沒收到 CAN 指令會自動失能。
   控制迴圈不能停（`02_move.py` 跑 200 Hz）。
   ⚠ 迴圈**不可以寫成 `time.sleep(DT)`** —— 那是固定延遲疊加在工作時間之上，
   USB CDC 來回(~1ms)加 `print` 會讓實際週期變成 5.87 ms（**170 Hz**，不是 200 Hz）。
   已改成補償式排程（睡到固定的下一個時刻），實測回到 200.0 Hz。
   之後若要上 1 kHz，這個寫法差異會直接決定成敗。
2. **MIT 模式做位置控制時 `kd` 不可為 0**，否則震盪甚至失控（手冊明列）。
3. **設零點必須在失能狀態**。
4. **切控制模式不寫 flash**，掉電後回到上次存檔的模式（這顆是 MIT）。
5. **`save_motor_param()` 只在失能時生效，且 flash 僅約 1 萬次擦寫**，絕不可放進迴圈。
6. `control_Vel()` / `control_Pos_Vel()` 送的是**原始 float**，PMAX/VMAX/TMAX 只影響
   **回饋解碼**與 **MIT 模式的命令編碼**。
7. `DM_CAN.py` 走的是達妙 USB2CAN 板的**私有序列協定**（幀頭 `0x55 0xAA`，921600 bps），
   **不是 socketcan**，換一般 CAN 卡不能用。
   ⚠ 這點踩過一次：`turret_control/backends.py` 原本用 `python-can` + `socketcan` + `can0` 寫，
   完全不能動 —— 這台 `ip link show type can` 是空的，`import can` 也 ModuleNotFoundError。
   已改成走 `DM_CAN.py`。舊版留在 `backends.py.socketcan.bak` 當反例。
8. **`hold()` 不能是「什麼都不送」**。沒有目標時要重送當前位置，不是停止送命令。
   理由見「安全」那節的垂直軸。
9. **射擊要有冷卻**。控制迴圈 200 Hz，開火條件會連續成立；不擋的話一秒射 200 發。

## 安全

執行 `02_move.py` 前：馬達鎖在夾具上、輸出軸空載、手離開轉子。峰值 27 N·m 會傷人。
馬達 24V 電源絕不可從 AGX 取電。

### ⚠ 垂直軸沒有煞車，失能就會掉下來

這顆**沒有煞車**，40:1 減速機也不是自鎖式的（空載摩擦只有 0.4~0.6 N·m，擋不住任何負載）。
下列任一情況垂直軸都會**自由落下**：

- 程式結束或 Ctrl+C（`02_move.py` 收尾就是 `disable`）
- 24V 斷電
- 觸發保護（過流、過溫、**通訊丟失**）

「通訊丟失」最危險 —— USB 線鬆、AGX 當機、程式卡住超過保護週期，馬達就自己失能。
所以**控制迴圈任何時候都不能停止送命令**（`turret_control` 沒有目標時送 `hold()`）。

機構端要解：配重／氣彈簧抵銷重力矩（最省事，也讓馬達不用持續出力發熱）、
蝸桿自鎖、或至少加機械限位擋塊。**已告知機構團隊。**

扭矩檢核：輸出軸額定 **9 N·m**、峰值 27 N·m。靜態重力矩 = `質量 × 9.81 × 重心到軸距離`。
例：3 kg 重心在 0.2 m → 5.9 N·m（額定內但會持續發熱）；同樣 3 kg 拉到 0.35 m → 10.3 N·m
（超過額定，只能靠峰值，久了過溫保護在 120°C）。

指示燈：**紅燈常亮 = 失能待命**、**綠燈常亮 = 使能**、**紅燈閃爍 = 故障**
（8 超壓 / 9 欠壓 / A 過流 / B MOS過溫 / C 線圈過溫 / D 通訊丟失 / E 過載）

## AGX Xavier 移植（✅ 通訊已完成，2026-08-11）

**路線：把同一塊 USB-to-CAN 板搬到 AGX**，程式碼一行沒改就通了（`dm_port.py` 靠 VID:PID
認裝置，不受 `ttyACM` 編號變動影響）。

實際環境：`JetPack R35.4.1` / `Linux 5.10.120-tegra` / `aarch64` / `BOARD: t186ref`，
專案路徑 `~/Documents/dm_motor_test`，使用者 `lluser`（不是 `nvidia`）。

四個步驟都已執行完畢：

```bash
# 1. 複製（已完成）
# 2. 相依套件：JetPack 內建 numpy 1.24.4；pyserial 3.5 用 pip --user 裝
# 3. 權限 + udev（已完成）
sudo usermod -aG dialout $USER
sudo cp ~/Documents/dm_motor_test/99-damiao.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 4. 驗證（已通過）
python3 01_check.py
```

`01_check.py` 在 AGX 上讀到的值**與桌機完全一致**：
`ID=0x01`、`Gr=40.0`、`MST_ID=0x00`、`CTRL_MODE=1`、`PMAX/VMAX/TMAX=12.5/10.0/28.0`、
`sw_ver=925970741`。

實測差異（與原本預期不同的兩點）：

- 這台 Xavier 上**只有 `/dev/ttyACM0` 一個**，內建除錯口並沒有佔號碼。不過還是靠
  VID:PID 認，不受影響。`/dev/damiao` symlink 已生效。
- 上面 `## 環境` 那條 `sg dialout` 陷阱就是在這台踩到的。

`02_move.py` 三種模式也已在 AGX 上跑過並通過，見上面「已驗證（2026-08-11，AGX Xavier）」。

注意事項：

- **接地**：USB 會讓 AGX GND 與 24V 電源 GND 共地，這是 CAN 需要的，但要確保馬達大電流
  的回流路徑不經過 AGX。
- **上電順序**：先給馬達 24V，等 2~3 秒再使能。
- **時序**：USB CDC 每次來回約 1 ms，200 Hz 綽綽有餘。若之後要 1 kHz 閉環，這條 USB
  路徑會是瓶頸。

### 未來（尚未開始）

若要降延遲或空出 USB 埠，改走 Xavier 原生 CAN（`mttcan`，需外接 3.3V CAN 收發器如
SN65HVD230 / TCAN332，40-pin header）。屆時要自己用 `python-can` 按協定打包，
把 `dm_port.py` 換成 CAN bus 抽象，上層邏輯不動。ROS 2 節點（接 `~/ros2_ws`）在那之後再做。

CAN 匯流排兩端需 120Ω 終端電阻（達妙調試板與馬達端通常內建）。

## CAN 協定速查

| 功能 | CAN ID | 資料 |
|---|---|---|
| 使能 | `ID` | `FF FF FF FF FF FF FF FC` |
| 失能 | `ID` | `FF FF FF FF FF FF FF FD` |
| 設零點 | `ID` | `FF FF FF FF FF FF FF FE` |
| MIT 控制 | `ID` | 位元打包 p(16) v(12) kp(12) kd(12) tau(12) |
| 位置速度 | `0x100+ID` | float p_des, float v_des（小端） |
| 速度 | `0x200+ID` | float v_des |
| 力位混控 | `0x300+ID` | float p_des, uint16 v×100, uint16 i×10000 |
| 讀暫存器 | `0x7FF` | `CANID_L, CANID_H, 0x33, RID` |
| 寫暫存器 | `0x7FF` | `CANID_L, CANID_H, 0x55, RID, data[4]` |
| 存 flash | `0x7FF` | `CANID_L, CANID_H, 0xAA, 0x01` |

回饋幀（ID = MST_ID）：`D0=ID|ERR<<4, D1~2=POS(16), D3~4=VEL(12), D4~5=T(12), D6=T_MOS, D7=T_Rotor`

`kp` 範圍 `[0,500]`、`kd` 範圍 `[0,5]`。

常用暫存器：`ACC=0x04`、`DEC=0x05`（位置模式梯形加減速，換向抖動時往上調）、
`MAX_SPD=0x06`（**posvel 的真正速度天花板**，見上面 posvel 那節）、
`MST_ID=0x07`、`ESC_ID=0x08`、`CTRL_MODE=0x0A`、`Gr=0x14`、
`PMAX=0x15`、`VMAX=0x16`、`TMAX=0x17`、`can_br=0x23`、`p_m=0x50`、`xout=0x51`

## 參考

- 官方手冊：https://doc.switch-science.com/media/files/5e033168-99f1-4d2d-b758-e03192a1f071.pdf
- 資料源頭：https://gitee.com/kit-miao/DM-J4340P-2EC
- Python 函式庫：https://github.com/cmjang/DM_Control_Python

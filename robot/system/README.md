# 開機啟動 CAN 與主控程式

適用於使用者 `robot`、專案位於 `/home/robot/robot/TEL-Sean/robot` 的 Xavier。
服務由 systemd 啟動，不需要登入桌面，也不會在開機時要求 sudo 密碼。

## 啟動時的實際動作

保留目前 `main.py` 的流程：DM 砲口定位到約 20°、MKS 馬達使能，
VESC 初始保持停轉，收到手動模式的非零轉速封包後才運轉。
這不是「等待解鎖才使能」模式；若要通電後完全不動，需要另外實作待命流程。

## 安裝

在主控程式關閉時執行：

```sh
cd /home/robot/robot/TEL-Sean/robot
sudo ./system/install-robot-service.sh
```

安裝程式會：

- 安裝 `can0-up.service`，設定 CAN0 為 500000 bit/s。
- 將 `can0.service` 設為同一個服務的別名，避免兩個服務重複設定 CAN。
- 安裝 `robot.service`，設定這兩項服務下次開機自動執行。
- 安裝 USB ACM 序列埠權限規則，將 `robot` 加入 `dialout`、`video`、`render` 群組。

安裝當下不啟動或重新啟動主控，也不重新設定正在運作的 CAN。
如果原本已存在另一份獨立的 `/etc/systemd/system/can0.service`，安裝會停止；
先停妥主控並整合舊設定，避免兩個 CAN 初始化服務並存。

僅需安裝 CAN 與序列權限時，可執行：

```sh
sudo ./system/install-hardware-config.sh
```

這個指令會啟用開機設定，並在 CAN 裝置存在時啟動 CAN 服務；若服務已啟動，
不會重新啟動它。USB ACM 是序列裝置，只有 CAN 網路介面有 UP 狀態。

## 手動啟動與查看日誌

確認沒有其他 `main.py`、瞄準或校準程式佔用硬體後，再啟動服務。
啟動主控會執行上述砲口定位動作。

```sh
sudo systemctl start can0.service
ip -details link show dev can0
sudo systemctl start robot.service
systemctl status robot.service --no-pager
sudo journalctl -u robot.service -b -f
```

主控會先等候 CAN、ESP32 UART、兩顆 VESC 與 DM 轉接板的裝置節點和讀寫權限，
最多 30 秒。這個檢查不開啟序列埠，也不發送馬達命令；裝置節點就緒不代表馬達
已回覆遙測，馬達回覆仍由主程式檢查。

## 停止與除錯

```sh
sudo systemctl stop robot.service
sudo systemctl restart robot.service
sudo journalctl -u can0-up.service -b
sudo systemctl disable robot.service
```

服務停止時送出 SIGINT，讓既有程式執行收尾：砲口回到 yaw=0°、pitch=30°，
再失能。允許最多 60 秒收尾；`systemctl stop` 不是立即切斷馬達電源。

異常退出後間隔 5 秒重啟，180 秒內最多啟動 3 次；每次重啟都可能重新定位砲口。
若觸發啟動次數限制，先處理日誌中的原因，再執行：

```sh
sudo systemctl reset-failed robot.service
sudo systemctl start robot.service
```

手動執行 `python3 main.py`、`run_board_real` 或馬達校準前，先停用主控服務。
重新設定或停止 CAN 前，也先停止主控服務。

## 下次開機確認

重新開機或重新上電會自動執行主控和砲口定位。準備好機構後，由操作者安排測試。
重新連線後查看：

```sh
systemctl is-enabled can0-up.service robot.service
systemctl is-active can0.service robot.service
sudo journalctl -u robot.service -b --no-pager
```

`active` 只表示服務程序仍在執行，仍須查看日誌確認硬體初始化。
本服務不會修復 ZED 的 USB 連線問題；先前 ZED 僅連上 USB 2.0，
需要將相機接到 USB 3.0 並確認連線速度後，再確認影像與自動瞄準功能。

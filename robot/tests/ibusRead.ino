// 不使用 IBusBM 函式庫，直接在程式內解析 FlySky iBUS
// 適用 ESP32 Arduino Core 3.x

// =====================
// FlySky iBUS (自行解析)
// =====================
#define IBUS_RX 13
#define IBUS_TX 14
#define IBUS_CH_NUM 14
#define IBUS_TIMEOUT 500

HardwareSerial ibusSerial(1);

uint16_t ibusCh[IBUS_CH_NUM];
unsigned long lastIbusTime = 0;

// =====================
// Xavier UART (Serial2)
// =====================
#define XAVIER_RX 1
#define XAVIER_TX 2

HardwareSerial xavierSerial(2);

#define DEADZONE 50

// =====================
// GB42 四輪腳位
// =====================
// Front Left
#define FL_PWM 4
#define FL_DIR 5

// Front Right
#define FR_PWM 6
#define FR_DIR 7

// Rear Left
#define RL_PWM 15
#define RL_DIR 16

// Rear Right
#define RR_PWM 17
#define RR_DIR 18

// =====================
// GB42 四輪煞車腳位
// =====================
#define FL_BRAKE 9
#define FR_BRAKE 10
#define RL_BRAKE 11
#define RR_BRAKE 12

// 暫定 HIGH 啟動煞車、LOW 解除煞車
// 請依馬達驅動器規格確認有效電位
#define BRAKE_ON HIGH
#define BRAKE_OFF LOW

#define PWM_FREQ 20000
#define PWM_RESOLUTION 8

// =====================
// 系統狀態變數
// =====================
char currentMode = 'A';  // 'M' = Manual, 'A' = Auto

int manual_yaw = 50;
int manual_pitch = 50;

// 自動模式接收來自 Xavier 的控制量
int auto_vx = 0;
int auto_vy = 0;
int auto_wz = 0;
int auto_yaw = 50;
int auto_pitch = 50;
int auto_fire = 0;
int auto_speed = 0;

unsigned long lastXavierTime = 0;
String xavierRxBuffer = "";

// =====================
// iBUS 安全讀取
// =====================
int safeRead(int ch) {
  if (millis() - lastIbusTime > IBUS_TIMEOUT)
    return 1500;

  int v = ibusCh[ch];

  if (v < 900 || v > 2100)
    return 1500;

  return v;
}

// =====================
// Deadzone
// =====================
int applyDeadzone(int v) {
  if (abs(v - 1500) < DEADZONE)
    return 1500;

  return v;
}

// =====================
// 開關 -> 0 / 1
// =====================
int toSwitch(int v) {
  return (v > 1500) ? 1 : 0;
}

// =====================
// 油門 0~100
// =====================
int throttleMap(int v) {
  v = constrain(v, 1000, 2000);

  if (v < 1100)
    return 0;

  return map(v, 1100, 2000, 0, 100);
}

// =====================
// 搖桿 正向 -100~100
// =====================
int joystickNormal(int v) {
  v = constrain(v, 1000, 2000);
  return map(v, 1000, 2000, -100, 100);
}

// =====================
// 搖桿 反向 -100~100
// =====================
int joystickReverse(int v) {
  v = constrain(v, 1000, 2000);
  return map(v, 1000, 2000, 100, -100);
}

// =====================
// 四輪煞車控制
// =====================
void setBrake(bool enable) {
  int brakeState = enable ? BRAKE_ON : BRAKE_OFF;

  digitalWrite(FL_BRAKE, brakeState);
  digitalWrite(FR_BRAKE, brakeState);
  digitalWrite(RL_BRAKE, brakeState);
  digitalWrite(RR_BRAKE, brakeState);
}

// =====================
// 馬達控制
// ESP32 Arduino Core 3.x
// ledcWrite 第一個參數是 PWM 腳位
// =====================
void motorControl(int dirPin, int pwmPin, int speed) {
  speed = constrain(speed, -255, 255);

  if (speed > 0) {
    digitalWrite(dirPin, HIGH);
    ledcWrite(pwmPin, speed);
  }
  else if (speed < 0) {
    digitalWrite(dirPin, LOW);
    ledcWrite(pwmPin, abs(speed));
  }
  else {
    ledcWrite(pwmPin, 0);
  }
}

// =====================
// 麥克納姆輪解算與輸出
// =====================
void driveMecanum(int forward, int strafe, int rotate, int throttle) {
  int FL = rotate + strafe - forward;
  int FR = rotate + strafe + forward;
  int RL = rotate - strafe - forward;
  int RR = rotate - strafe + forward;

  // 等比例縮放，保留移動方向
  int maxV = max(
    max(abs(FL), abs(FR)),
    max(abs(RL), abs(RR))
  );

  if (maxV > 100) {
    FL = FL * 100 / maxV;
    FR = FR * 100 / maxV;
    RL = RL * 100 / maxV;
    RR = RR * 100 / maxV;
  }

  // 油門倍率
  FL = FL * throttle / 100;
  FR = FR * throttle / 100;
  RL = RL * throttle / 100;
  RR = RR * throttle / 100;

  // PWM 映射
  FL = map(FL, -100, 100, -255, 255);
  FR = map(FR, -100, 100, -255, 255);
  RL = map(RL, -100, 100, -255, 255);
  RR = map(RR, -100, 100, -255, 255);

  // 四輪輸出與煞車
  bool allStopped =
    (FL == 0 && FR == 0 && RL == 0 && RR == 0);

  if (allStopped) {
    // 先將四輪 PWM 歸零
    ledcWrite(FL_PWM, 0);
    ledcWrite(FR_PWM, 0);
    ledcWrite(RL_PWM, 0);
    ledcWrite(RR_PWM, 0);

    // 再啟動煞車
    setBrake(true);
  }
  else {
    // 馬達需要移動時先解除煞車
    setBrake(false);

    motorControl(FL_DIR, FL_PWM, FL);
    motorControl(FR_DIR, FR_PWM, FR);
    motorControl(RL_DIR, RL_PWM, RL);
    motorControl(RR_DIR, RR_PWM, RR);
  }
}

// =====================
// 解析來自 Xavier 的指令
// 格式：X,A,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n
// =====================
void parseXavierPacket(String packet) {
  packet.trim();

  if (packet.length() == 0 || packet.charAt(0) != 'X')
    return;

  int index = 0;
  String values[9];
  int lastComma = -1;

  for (int i = 0; i < packet.length(); i++) {
    if (packet.charAt(i) == ',') {
      if (index < 9) {
        values[index] = packet.substring(lastComma + 1, i);
        index++;
      }

      lastComma = i;
    }
  }

  if (index < 9 && lastComma < packet.length()) {
    values[index] = packet.substring(lastComma + 1);
    index++;
  }

  // 確保收到完整的 9 個欄位
  if (index == 9) {
    auto_vx    = values[2].toInt();
    auto_vy    = values[3].toInt();
    auto_wz    = values[4].toInt();
    auto_yaw   = values[5].toInt();
    auto_pitch = values[6].toInt();
    auto_fire  = values[7].toInt();
    auto_speed = values[8].toInt();

    lastXavierTime = millis();
  }
}

// =====================
// Setup
// =====================
void setup() {
  Serial.begin(115200);

  // Xavier UART
  xavierSerial.begin(
    115200, SERIAL_8N1, XAVIER_RX, XAVIER_TX
  );

  // iBUS UART
  ibusSerial.begin(
    115200, SERIAL_8N1, IBUS_RX, IBUS_TX
  );

  for (int i = 0; i < IBUS_CH_NUM; i++)
    ibusCh[i] = 1500;

  // DIR 初始化
  pinMode(FL_DIR, OUTPUT);
  pinMode(FR_DIR, OUTPUT);
  pinMode(RL_DIR, OUTPUT);
  pinMode(RR_DIR, OUTPUT);

  // BRAKE 初始化
  pinMode(FL_BRAKE, OUTPUT);
  pinMode(FR_BRAKE, OUTPUT);
  pinMode(RL_BRAKE, OUTPUT);
  pinMode(RR_BRAKE, OUTPUT);

  // PWM 初始化：ESP32 Arduino Core 3.x
  ledcAttach(FL_PWM, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(FR_PWM, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(RL_PWM, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(RR_PWM, PWM_FREQ, PWM_RESOLUTION);

  // 啟動時先停止 PWM，再啟動煞車
  ledcWrite(FL_PWM, 0);
  ledcWrite(FR_PWM, 0);
  ledcWrite(RL_PWM, 0);
  ledcWrite(RR_PWM, 0);

  setBrake(true);

  Serial.println("MECANUM & XAVIER UART READY");
}

// =====================
// Loop
// =====================
void loop() {

  // ==========================================
  // 0. 更新 iBUS 資料
  // ==========================================
  ibusUpdate();

  bool ibusConnected =
    (millis() - lastIbusTime <= IBUS_TIMEOUT);

  // ==========================================
  // 1. 讀取遙控器 iBUS 通道資料
  // ==========================================
  int ch1 = applyDeadzone(safeRead(0));
  int ch2 = applyDeadzone(safeRead(1));
  int ch3 = applyDeadzone(safeRead(2));
  int ch4 = applyDeadzone(safeRead(3));

  // Switch
  int vrB = safeRead(5);
  int swA = toSwitch(safeRead(6));
  int swB = safeRead(7);
  int swC = safeRead(8);
  int swD = safeRead(9);

  // 模式切換
  currentMode = (swA == 1) ? 'M' : 'A';

  // SW2：底盤 / 砲台控制
  bool isTurretControl = (swB > 1500);

  // SW1：射擊安全開關
  int fire_allow = (swD > 1500) ? 1 : 0;

  // VRB：射擊速度
  int speed_val = map(
    constrain(vrB, 1000, 2000),
    1000, 2000, 0, 100
  );

  // 油門倍率
  int throttle = throttleMap(ch3);

  // 手動控制變數
  int forward = 0;
  int strafe = 0;
  int rotate = joystickNormal(ch4);

  if (!isTurretControl) {
    // 底盤控制模式
    forward = joystickReverse(ch2);
    strafe = joystickNormal(ch1);
  }
  else {
    // 砲台控制模式
    manual_yaw = map(
      constrain(ch1, 1000, 2000),
      1000, 2000, 0, 100
    );

    manual_pitch = map(
      constrain(ch2, 1000, 2000),
      1000, 2000, 0, 100
    );
  }

  // ==========================================
  // 2. 接收 Xavier UART 指令
  // ==========================================
  while (xavierSerial.available() > 0) {
    char c = (char)xavierSerial.read();

    if (c == '\n') {
      parseXavierPacket(xavierRxBuffer);
      xavierRxBuffer = "";
    }
    else if (c != '\r') {
      xavierRxBuffer += c;

      // 避免異常封包造成 String 無限增長
      if (xavierRxBuffer.length() > 128)
        xavierRxBuffer = "";
    }
  }

  // ==========================================
  // 3. 底盤馬達控制與安全保護
  // ==========================================
  if (!ibusConnected) {
    // 遙控器失聯：停止底盤並啟動煞車
    driveMecanum(0, 0, 0, 0);
  }
  else if (currentMode == 'M') {
    // 手動模式
    driveMecanum(forward, strafe, rotate, throttle);
  }
  else if (currentMode == 'A') {
    // 自動模式：Xavier 超時保護
    if (millis() - lastXavierTime > 500) {
      driveMecanum(0, 0, 0, 0);
    }
    else {
      driveMecanum(auto_vx, auto_vy, auto_wz, 100);
    }
  }

  // ==========================================
  // 4. 發送狀態封包給 Xavier
  // E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n
  // ==========================================
  int send_vx =
    (currentMode == 'M') ? forward : auto_vx;

  int send_vy =
    (currentMode == 'M') ? strafe : auto_vy;

  int send_wz =
    (currentMode == 'M') ? rotate : auto_wz;

  int send_yaw =
    (currentMode == 'M') ? manual_yaw : auto_yaw;

  int send_pitch =
    (currentMode == 'M') ? manual_pitch : auto_pitch;

  int send_fire =
    (fire_allow == 1)
      ? ((currentMode == 'M') ? 0 : auto_fire)
      : 0;

  int send_speed =
    (currentMode == 'M') ? speed_val : auto_speed;

  String txPacket =
    "E," + String(currentMode) + "," +
    String(send_vx) + "," +
    String(send_vy) + "," +
    String(send_wz) + "," +
    String(send_yaw) + "," +
    String(send_pitch) + "," +
    String(send_fire) + "," +
    String(send_speed) + "\n";

  xavierSerial.print(txPacket);

  // ==========================================
  // Debug：原本的輸出順序
  // ==========================================
  Serial.print("TX->Xavier: ");
  Serial.print(txPacket);

  Serial.print("Mode:");
  Serial.print(currentMode);

  Serial.print("\tThrottle:");
  Serial.print(throttle);

  Serial.print("\tForward:");
  Serial.print(forward);

  Serial.print("\tStrafe:");
  Serial.print(strafe);

  Serial.print("\tRotate:");
  Serial.println(rotate);

  delay(20);

  Serial.print("\tSWA:");
  Serial.print(swA);

  Serial.print("\tswC:");
  Serial.print(swC);

  Serial.print("\tswB:");
  Serial.print(swB);

  Serial.print("\tswD:");
  Serial.print(swD);

  Serial.print("\tvrB:");
  Serial.println(vrB);

  delay(20);
}

// =====================
// FlySky iBUS 封包解析
// 32 bytes：0x20 0x40 + 14ch*2 + checksum
// =====================
void ibusUpdate() {
  static uint8_t buf[32];
  static uint8_t idx = 0;

  while (ibusSerial.available() > 0) {
    uint8_t b = (uint8_t)ibusSerial.read();

    if (idx == 0) {
      if (b != 0x20)
        continue;

      buf[idx++] = b;
      continue;
    }

    if (idx == 1) {
      if (b != 0x40) {
        idx = (b == 0x20) ? 1 : 0;
        if (idx == 1)
          buf[0] = b;
        continue;
      }

      buf[idx++] = b;
      continue;
    }

    buf[idx++] = b;

    if (idx == 32) {
      idx = 0;

      // checksum = 0xFFFF - 前 30 bytes 總和
      uint16_t sum = 0xFFFF;

      for (uint8_t i = 0; i < 30; i++)
        sum -= buf[i];

      uint16_t rxSum =
        (uint16_t)buf[30] |
        ((uint16_t)buf[31] << 8);

      if (sum == rxSum) {
        for (uint8_t i = 0; i < IBUS_CH_NUM; i++) {
          ibusCh[i] =
            (uint16_t)buf[2 + i * 2] |
            ((uint16_t)buf[3 + i * 2] << 8);
        }

        lastIbusTime = millis();
      }
    }
  }
}

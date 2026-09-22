// CH2的前後相反!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include <IBusBM.h>

IBusBM ibus;
HardwareSerial ibusSerial(1);

// =====================
// Xavier UART (Serial2)
// =====================
#define XAVIER_RX 16
#define XAVIER_TX 17
HardwareSerial xavierSerial(2);

// =====================
// FlySky iBUS
// =====================
#define IBUS_RX 13
#define IBUS_TX 14

#define DEADZONE 50

// =====================
// GB42 四輪腳位 (未更改)
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

#define FL_CH 0
#define FR_CH 1
#define RL_CH 2
#define RR_CH 3

#define PWM_FREQ 20000
#define PWM_RESOLUTION 8

// =====================
// 系統狀態變數
// =====================
char currentMode = 'A';          // 'M' = Manual, 'A' = Auto
int manual_yaw = 50;             // 砲台 YAW (0~100)
int manual_pitch = 50;           // 砲台 PITCH (0~100)

// 自動模式接收來自 Xavier 的控制量
int auto_vx = 0;
int auto_vy = 0;
int auto_wz = 0;
int auto_yaw = 50;
int auto_pitch = 50;
int auto_fire = 0;
int auto_speed = 0;

unsigned long lastXavierTime = 0; // 超時保護計時
String xavierRxBuffer = "";      // 串口接收緩衝區

// =====================
// iBUS安全讀取 (未更改)
// =====================
int safeRead(int ch){
  int v = ibus.readChannel(ch);
  if(v < 900 || v > 2100)
    return 1500;
  return v;
}

// =====================
// Deadzone (未更改)
// =====================
int applyDeadzone(int v){
  if(abs(v - 1500) < DEADZONE)
    return 1500;
  return v;
}

// =====================
// 油門 0~100 (未更改)
// =====================
int throttleMap(int v){
  v = constrain(v, 1000, 2000);
  if(v < 1100)
    return 0;
  return map(v, 1100, 2000, 0, 100);
}

// =====================
// 搖桿 正向 -100~100 (未更改)
// =====================
int joystickNormal(int v){
  v = constrain(v, 1000, 2000);
  return map(v, 1000, 2000, -100, 100);
}

// =====================
// 搖桿 反向 -100~100 (未更改)
// =====================
int joystickReverse(int v){
  v = constrain(v, 1000, 2000);
  return map(v, 1000, 2000, 100, -100);
}

// =====================
// 馬達控制 (未更改)
// =====================
void motorControl(int dirPin, int channel, int speed){
  speed = constrain(speed, -255, 255);

  if(speed > 0){
    digitalWrite(dirPin, HIGH);
    ledcWrite(channel, speed);
  }
  else if(speed < 0){
    digitalWrite(dirPin, LOW);
    ledcWrite(channel, abs(speed));
  }
  else{
    ledcWrite(channel, 0);
  }
}

// =====================
// 麥克納姆輪底盤解算與輸出
// =====================
void driveMecanum(int forward, int strafe, int rotate, int throttle) {
  int FL = rotate + strafe - forward;
  int FR = rotate + strafe + forward;
  int RL = rotate - strafe - forward;
  int RR = rotate - strafe + forward;

  // 限制
  FL = constrain(FL, -100, 100);
  FR = constrain(FR, -100, 100);
  RL = constrain(RL, -100, 100);
  RR = constrain(RR, -100, 100);

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

  // 四輪輸出
  motorControl(FL_DIR, FL_CH, FL);
  motorControl(FR_DIR, FR_CH, FR);
  motorControl(RL_DIR, RL_CH, RL);
  motorControl(RR_DIR, RR_CH, RR);
}

// =====================
// 解析來自 Xavier 的指令 (X,A,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n)
// =====================
void parseXavierPacket(String packet) {
  packet.trim();
  if (packet.length() == 0 || packet.charAt(0) != 'X') return;

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
    lastXavierTime = millis(); // 更新收到資料的時間戳
  }
}

// =====================
// Setup
// =====================
void setup(){
  Serial.begin(115200);

  // Xavier UART (Serial2)
  xavierSerial.begin(115200, SERIAL_8N1, XAVIER_RX, XAVIER_TX);

  // iBUS
  ibusSerial.begin(115200, SERIAL_8N1, IBUS_RX, IBUS_TX);
  ibus.begin(ibusSerial);

  // DIR
  pinMode(FL_DIR, OUTPUT);
  pinMode(FR_DIR, OUTPUT);
  pinMode(RL_DIR, OUTPUT);
  pinMode(RR_DIR, OUTPUT);

  // PWM
  ledcSetup(FL_CH, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(FR_CH, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(RL_CH, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(RR_CH, PWM_FREQ, PWM_RESOLUTION);

  ledcAttachPin(FL_PWM, FL_CH);
  ledcAttachPin(FR_PWM, FR_CH);
  ledcAttachPin(RL_PWM, RL_CH);
  ledcAttachPin(RR_PWM, RR_CH);

  Serial.println("MECANUM & XAVIER UART READY");
}

// =====================
// Loop
// =====================
void loop(){

  // ==========================================
  // 1. 讀取遙控器 iBUS 通道資料 (CH1 ~ CH10)
  // ==========================================
  int ch1 = applyDeadzone(safeRead(0)); // CH1: 右搖桿左右
  int ch2 = applyDeadzone(safeRead(1)); // CH2: 右搖桿上下
  int ch3 = applyDeadzone(safeRead(2)); // CH3: 左搖桿上下 (油門)
  int ch4 = applyDeadzone(safeRead(3)); // CH4: 左搖桿左右
  int ch5 = safeRead(4);                // SW1: 射擊安全開關
  int ch6 = safeRead(5);                // SW2: 模式切換 (Manual / Auto)
  int ch7 = safeRead(6);                // SW3: 切換 (底盤 / 砲台)
  int ch8 = safeRead(7);                // SW4: 射擊模式 (OFF / SINGLE / BURST)
  int ch10 = safeRead(9);               // VRB: 射擊速度旋鈕

  // 判斷 SW2: 控制模式切換 ('M' 或 'A')
  currentMode = (ch6 > 1500) ? 'A' : 'M';

  // 判斷 SW3: 手動時 CH1/CH2 用途切換
  bool isTurretControl = (ch7 > 1500);

  // 判斷 SW1: 射擊安全允許
  int fire_allow = (ch5 > 1500) ? 1 : 0;

  // 讀取 VRB 轉速 (0 ~ 100)
  int speed_val = map(constrain(ch10, 1000, 2000), 1000, 2000, 0, 100);

  // 計算油門倍率 (0 ~ 100)
  int throttle = throttleMap(ch3);

  // 手動控制變數宣告
  int forward = 0;
  int strafe = 0;
  int rotate = joystickNormal(ch4);

  if (!isTurretControl) {
    // SW3 = 底盤控制模式
    forward = joystickReverse(ch2);
    strafe = joystickNormal(ch1);
  } else {
    // SW3 = 砲台控制模式 (搖桿輸入轉換為 0~100 控制量，可根據需求微調)
    manual_yaw = map(constrain(ch1, 1000, 2000), 1000, 2000, 0, 100);
    manual_pitch = map(constrain(ch2, 1000, 2000), 1000, 2000, 0, 100);
  }

  // ==========================================
  // 2. 接收並解析來自 Xavier 的 UART 指令
  // ==========================================
  while (xavierSerial.available() > 0) {
    char c = (char)xavierSerial.read();
    if (c == '\n') {
      parseXavierPacket(xavierRxBuffer);
      xavierRxBuffer = "";
    } else if (c != '\r') {
      xavierRxBuffer += c;
    }
  }

  // ==========================================
  // 3. 依據模式執行底盤馬達驅動
  // ==========================================
  if (currentMode == 'M') {
    // 【手動模式】: 由 ESP32 解析遙控器直接驅動底盤
    driveMecanum(forward, strafe, rotate, throttle);
  } 
  else if (currentMode == 'A') {
    // 【自動模式】: 聽從 Xavier 發送的 VX, VY, WZ 控制指令
    // 超時安全保護：若超過 500ms 沒收到 Xavier 封包，底盤停等
    if (millis() - lastXavierTime > 500) {
      driveMecanum(0, 0, 0, 0);
    } else {
      // auto_vx = 前後, auto_vy = 左右平移, auto_wz = 旋轉
      driveMecanum(auto_vx, auto_vy, auto_wz, 100);
    }
  }

  // ==========================================
  // 4. 定期發送狀態封包給 Xavier (E,MODE,VX,VY,WZ,YAW,PITCH,FIRE,SPEED\n)
  // ==========================================
  // 決定要送給 Xavier 的 YAW, PITCH, FIRE, SPEED
  int send_vx = (currentMode == 'M') ? forward : auto_vx;
  int send_vy = (currentMode == 'M') ? strafe : auto_vy;
  int send_wz = (currentMode == 'M') ? rotate : auto_wz;
  int send_yaw = (currentMode == 'M') ? manual_yaw : auto_yaw;
  int send_pitch = (currentMode == 'M') ? manual_pitch : auto_pitch;
  int send_fire = (fire_allow == 1) ? ((currentMode == 'M') ? 0 : auto_fire) : 0;
  int send_speed = (currentMode == 'M') ? speed_val : auto_speed;

  String txPacket = "E," + String(currentMode) + "," +
                    String(send_vx) + "," +
                    String(send_vy) + "," +
                    String(send_wz) + "," +
                    String(send_yaw) + "," +
                    String(send_pitch) + "," +
                    String(send_fire) + "," +
                    String(send_speed) + "\n";

  xavierSerial.print(txPacket);

  // Debug 訊息
  Serial.print("Mode:"); Serial.print(currentMode);
  Serial.print("\tThrottle:"); Serial.print(throttle);
  Serial.print("\tForward:"); Serial.print(forward);
  Serial.print("\tStrafe:"); Serial.print(strafe);
  Serial.print("\tRotate:"); Serial.println(rotate);

  delay(20); // 50Hz 控制週期
}

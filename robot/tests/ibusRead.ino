#include <IBusBM.h>

IBusBM ibus;
HardwareSerial ibusSerial(1);

#define IBUS_RX 13
#define IBUS_TX 14

#define DEADZONE 50

// ================== 工具函式 ==================

int safeRead(int ch) {
  int v = ibus.readChannel(ch);
  if (v < 900 || v > 2100) return 1500;
  return v;
}

int applyDeadzone(int v) {
  if (abs(v - 1500) < DEADZONE) return 1500;
  return v;
}

// VR → 0~100
int toPercent(int v) {
  v = constrain(v, 1000, 2000);
  return map(v, 1000, 2000, 0, 100);
}

// Switch → 0/1
int toSwitch(int v) {
  return (v > 1500) ? 1 : 0;
}

//新增：SWC 三段模式
int to3State(int v) {
  if (v < 1300) return 0;   // 下
  if (v > 1700) return 2;   // 上
  return 1;                 // 中
}

// ================== setup ==================

void setup() {
  Serial.begin(115200);
  delay(500);

  ibusSerial.begin(115200, SERIAL_8N1, IBUS_RX, IBUS_TX);
  ibus.begin(ibusSerial);

  Serial.println("IBUS READY");
}

// ================== loop ==================

void loop() {

  // ===== 搖桿（CH1~CH4）=====
  int ch1 = applyDeadzone(safeRead(0));
  int ch2 = applyDeadzone(safeRead(1));
  int ch3 = applyDeadzone(safeRead(2));
  int ch4 = applyDeadzone(safeRead(3));

  // ===== Switch=====
  int swA = toSwitch(safeRead(6));
  int swB = toSwitch(safeRead(7));

  //SWC 三段模式
  int swC = to3State(safeRead(8));
  int swD = toSwitch(safeRead(9));

  // ===== VR=====
  int vra = toPercent(safeRead(4));
  int vrb = toPercent(safeRead(5));

  // ===== fail-safe（保留你原本邏輯）=====
  if (ibus.readChannel(0) < 900) {
    ch1 = ch2 = ch3 = ch4 = 1500;
    swA = swB = swC = swD = 0;
    vra = vrb = 0;
  }

  // ===== 輸出（完全不改你的順序）=====

  Serial.print(swA); Serial.print("\t");
  Serial.print(swB); Serial.print("\t");
  Serial.print(vra); Serial.print("\t");
  Serial.print(vrb); Serial.print("\t");

  Serial.print(swC); Serial.print("\t");
  Serial.println(swD);

  Serial.print(ch3); Serial.print("\t");
  Serial.print(ch4); Serial.print("\t");
  Serial.print(ch1); Serial.print("\t");
  Serial.println(ch2);

  delay(20);
}
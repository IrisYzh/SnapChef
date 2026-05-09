// FridgeLens - Button + OLED prototype
//
// Wiring:
//   HC-SR04:  VCC -> 5V,   GND -> GND, TRIG -> GPIO2, ECHO -> GPIO1 (分压: 10kΩ + 20kΩ)
//   OLED:     VCC -> 3.3V, GND -> GND, SDA -> GPIO5,  SCL -> GPIO6
//   Mode btn: GPIO9 -> GND   （任何界面都跳到模式选择）
//   Encoder:  CLK -> GPIO3, DT -> GPIO4, SW -> GPIO7, VCC -> 3.3V, GND -> GND
//             旋钮转动 = 滚动/切换，旋钮按下 = 确认/删除
//
// 状态说明:
//   STATE_IDLE            待机，传感器触发或按钮进入模式选择
//   STATE_MODE_SELECT     三选一菜单，10s 无操作返回待机
//   STATE_SCANNING        扫描动画，扫完自动跳转
//   STATE_RESULT          显示菜谱，一直停留直到用户按返回
//   STATE_INGREDIENT_MENU 食材列表，30s 无操作返回待机
//
// Libraries: Adafruit SSD1306, Adafruit GFX

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ─── 引脚定义 ────────────────────────────────────────────────
#define TRIG_PIN      2
#define ECHO_PIN      1    // 已分压（10kΩ + 20kΩ）
#define MODE_BTN_PIN  9    // 独立按钮，任何界面跳到模式选择
#define ENC_CLK_PIN   3
#define ENC_DT_PIN    4
#define ENC_SW_PIN    7    // 旋钮按下 = 确认 / 删除

// ─── OLED 设置 ───────────────────────────────────────────────
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDR     0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── 状态机 ──────────────────────────────────────────────────
enum State {
  STATE_IDLE,
  STATE_MODE_SELECT,
  STATE_SCANNING,
  STATE_RESULT,
  STATE_INGREDIENT_MENU
};

State currentState = STATE_IDLE;

// ─── 模式选择 ────────────────────────────────────────────────
// 0 = Ingredient Scan, 1 = Receipt Scan, 2 = Edit Ingredients
const char* modeLabels[] = {
  "Ingredient Scan",
  "Receipt Scan",
  "Edit Ingredients"
};
const int modeCount = 3;
int selectedMode = 0;

// ─── Inventory（可动态删除）──────────────────────────────────
#define MAX_ITEMS    20
#define MAX_NAME_LEN 16

char inventory[MAX_ITEMS][MAX_NAME_LEN] = {
  "Tomato", "Egg", "Cucumber", "Tofu", "Carrot"
};
int inventoryCount = 5;
int menuIndex      = 0;   // 食材菜单当前选中项

// ─── 模拟菜名（后续替换为真实 API 输出）─────────────────────
const char* recipes[][3] = {
  {"Tomato Egg Stir-fry", "Tomato Soup",    "Bruschetta"      },
  {"Steamed Egg",         "Egg Fried Rice", "Omelette"        },
  {"Cucumber Salad",      "Cucumber Stir",  "Pickled Cucumber"},
  {"Mapo Tofu",           "Tofu Soup",      "Fried Tofu"      },
  {"Glazed Carrots",      "Carrot Soup",    "Carrot Stir-fry" }
};
const int recipeCount = 5;
int currentFood = 0;

// ─── 超时设置 ────────────────────────────────────────────────
const unsigned long MODE_TIMEOUT_MS = 10000; // 模式选择 10s
const unsigned long MENU_TIMEOUT_MS = 30000; // 食材菜单 30s
unsigned long       lastActionTime  = 0;

// ─── HC-SR04 设置 ────────────────────────────────────────────
const float         DETECT_CM     = 10.0;
const unsigned long COOLDOWN_MS   = 3000;
unsigned long       lastDetectTime = 0;

// ─── 传感器数据流 ─────────────────────────────────────────────
const unsigned long STREAM_INTERVAL_MS = 500;  // 每 500ms 输出一次
unsigned long       lastStreamTime     = 0;

// ─── 按钮和编码器状态 ────────────────────────────────────────
bool lastModeBtn = HIGH;
bool lastEncBtn  = HIGH;
int  lastEncCLK  = HIGH;

unsigned long lastDebounceModeBtn = 0;
unsigned long lastDebounceEncBtn  = 0;
const unsigned long DEBOUNCE_MS   = 50;

// ─── 长按检测 ────────────────────────────────────────────────
const unsigned long LONG_PRESS_MS = 3000; // 长按阈值 3s
unsigned long       encBtnDownTime = 0;   // 记录按下的时刻
bool                encBtnHeld     = false; // 当前是否处于按住状态

// ─── 前向声明（解决函数顺序依赖）────────────────────────────
void showIdle();
void showModeSelect();
void showScanning();
void showResult(int recipeIndex);
void showIngredientMenu();

// ─── Inventory 操作 ──────────────────────────────────────────

void deleteItem(int idx) {
  if (inventoryCount == 0 || idx < 0 || idx >= inventoryCount) return;
  for (int i = idx; i < inventoryCount - 1; i++) {
    strncpy(inventory[i], inventory[i + 1], MAX_NAME_LEN);
  }
  inventoryCount--;
  if (menuIndex >= inventoryCount && menuIndex > 0) menuIndex--;
}

// ─── 工具函数 ────────────────────────────────────────────────

float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return 999.0;
  return duration * 0.0343 / 2.0;
}

void centerText(const char* text, int y, int size) {
  display.setTextSize(size);
  int16_t x1, y1;
  uint16_t w, h;
  display.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, y);
  display.print(text);
}

// ─── 状态切换函数 ────────────────────────────────────────────

void goToIdle() {
  currentState = STATE_IDLE;
  showIdle();
}

void goToModeSelect() {
  currentState   = STATE_MODE_SELECT;
  lastActionTime = millis();
  showModeSelect();
}

void goToIngredientMenu() {
  currentState   = STATE_INGREDIENT_MENU;
  menuIndex      = 0;
  lastActionTime = millis();
  showIngredientMenu();
}

// ─── 屏幕显示函数 ────────────────────────────────────────────

void showIdle() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  centerText("SnapChef", 6, 2);
  display.drawLine(0, 24, SCREEN_WIDTH, 24, SSD1306_WHITE);
  centerText("Hold item near", 30, 1);
  centerText("sensor to scan", 42, 1);
  centerText("or press button", 54, 1);
  display.display();
}

void showModeSelect() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  centerText("Select Mode", 0, 1);
  display.drawLine(0, 10, SCREEN_WIDTH, 10, SSD1306_WHITE);

  for (int i = 0; i < modeCount; i++) {
    int y = 14 + i * 16;
    if (i == selectedMode) {
      display.fillRect(0, y, SCREEN_WIDTH, 14, SSD1306_WHITE);
      display.setTextColor(SSD1306_BLACK);
      display.setCursor(4, y + 3);
      display.print("> ");
      display.print(modeLabels[i]);
      display.setTextColor(SSD1306_WHITE);
    } else {
      display.setCursor(12, y + 3);
      display.print(modeLabels[i]);
    }
  }

  centerText("Knob:sel  SW:ok  Btn:back", 57, 1);
  display.display();
}

void showScanning() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (selectedMode == 1) {
    centerText("Reading receipt", 20, 1);
  } else {
    centerText("Scanning...", 20, 2);
  }
  display.display();
}

void showResult(int recipeIndex) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Detected: ");
  if (inventoryCount > 0) {
    display.println(inventory[recipeIndex % inventoryCount]);
  }
  display.drawLine(0, 10, SCREEN_WIDTH, 10, SSD1306_WHITE);

  display.setCursor(0, 14);
  display.println("Try cooking:");
  for (int i = 0; i < 3; i++) {
    display.print(" ");
    display.println(recipes[recipeIndex][i]);
  }

  display.drawLine(0, 54, SCREEN_WIDTH, 54, SSD1306_WHITE);
  display.setCursor(0, 56);
  display.print("SW:scan again  Btn:back");
  display.display();
}

void showIngredientMenu() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  char title[24];
  snprintf(title, sizeof(title), "Ingredients (%d)", inventoryCount);
  centerText(title, 0, 1);
  display.drawLine(0, 10, SCREEN_WIDTH, 10, SSD1306_WHITE);

  if (inventoryCount == 0) {
    centerText("No items", 28, 1);
  } else {
    for (int i = -1; i <= 1; i++) {
      int idx = (menuIndex + i + inventoryCount) % inventoryCount;
      int y   = 28 + i * 14;
      if (i == 0) {
        display.fillRect(0, y - 2, SCREEN_WIDTH, 13, SSD1306_WHITE);
        display.setTextColor(SSD1306_BLACK);
        display.setCursor(4, y);
        display.print("> ");
        display.print(inventory[idx]);
        display.setTextColor(SSD1306_WHITE);
      } else {
        display.setCursor(12, y);
        display.print(inventory[idx]);
      }
    }
  }

  display.drawLine(0, 54, SCREEN_WIDTH, 54, SSD1306_WHITE);
  display.setCursor(0, 56);
  display.print("Knob:scroll  Hold:del  Btn:back");
  display.display();
}

// ─── 输入读取 ────────────────────────────────────────────────

bool modeBtnPressed() {
  bool reading = digitalRead(MODE_BTN_PIN);
  if (reading == LOW && lastModeBtn == HIGH) {
    if (millis() - lastDebounceModeBtn > DEBOUNCE_MS) {
      lastDebounceModeBtn = millis();
      lastModeBtn = reading;
      return true;
    }
  }
  lastModeBtn = reading;
  return false;
}

// 旋钮刚按下（用于菜谱界面的单击确认）
bool encBtnPressed() {
  bool reading = digitalRead(ENC_SW_PIN);
  if (reading == LOW && lastEncBtn == HIGH) {
    if (millis() - lastDebounceEncBtn > DEBOUNCE_MS) {
      lastDebounceEncBtn = millis();
      lastEncBtn = reading;
      return true;
    }
  }
  lastEncBtn = reading;
  return false;
}

// 旋钮当前是否处于按住状态（用于长按进度计算）
bool encBtnIsHeld() {
  return digitalRead(ENC_SW_PIN) == LOW;
}

int readEncoder() {
  int clk = digitalRead(ENC_CLK_PIN);
  if (clk != lastEncCLK && clk == LOW) {
    lastEncCLK = clk;
    return (digitalRead(ENC_DT_PIN) == HIGH) ? 1 : -1;
  }
  lastEncCLK = clk;
  return 0;
}

// ─── Setup ───────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(TRIG_PIN,     OUTPUT);
  pinMode(ECHO_PIN,     INPUT);
  pinMode(MODE_BTN_PIN, INPUT_PULLUP);
  pinMode(ENC_CLK_PIN,  INPUT_PULLUP);
  pinMode(ENC_DT_PIN,   INPUT_PULLUP);
  pinMode(ENC_SW_PIN,   INPUT_PULLUP);

  // 等待引脚稳定，读取初始状态防止误触发
  delay(100);
  lastModeBtn = digitalRead(MODE_BTN_PIN);
  lastEncBtn  = digitalRead(ENC_SW_PIN);
  lastEncCLK  = digitalRead(ENC_CLK_PIN);

  Wire.begin(5, 6);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed");
    while (true);
  }

  Serial.println("FridgeLens ready");
  showIdle();
}

// ─── Main Loop ───────────────────────────────────────────────

void loop() {
  unsigned long now         = millis();
  bool          modePressed = modeBtnPressed();
  bool          encPressed  = encBtnPressed();
  int           encDelta    = readEncoder();

  // 持续流式输出传感器数据（CSV 格式，Serial Plotter 可直接使用）
  if (now - lastStreamTime >= STREAM_INTERVAL_MS) {
    lastStreamTime = now;
    float d = readDistance();
    Serial.print("timestamp:");
    Serial.print(now);
    Serial.print(",distance_cm:");
    Serial.print(d, 1);
    Serial.print(",state:");
    Serial.println(currentState);
  }

  // 独立按钮：任何状态都跳到模式选择
  if (modePressed) {
    Serial.println("Mode button -> MODE_SELECT");
    selectedMode = 0;
    goToModeSelect();
    return;
  }

  switch (currentState) {

    // ── 待机 ─────────────────────────────────────────────────
    case STATE_IDLE: {
      float dist = readDistance();
      if (dist < DETECT_CM && now - lastDetectTime > COOLDOWN_MS) {
        lastDetectTime = now;
        Serial.print("Detected at ");
        Serial.print(dist);
        Serial.println(" cm -> MODE_SELECT");
        selectedMode = 0;
        goToModeSelect();
      }
      break;
    }

    // ── 模式选择 ─────────────────────────────────────────────
    case STATE_MODE_SELECT: {
      if (now - lastActionTime > MODE_TIMEOUT_MS) {
        Serial.println("Timeout -> IDLE");
        goToIdle();
        break;
      }
      if (encDelta != 0) {
        lastActionTime = now;
        selectedMode   = (selectedMode + encDelta + modeCount) % modeCount;
        showModeSelect();
      }
      if (encPressed) {
        lastActionTime = now;
        Serial.print("Mode confirmed: ");
        Serial.println(modeLabels[selectedMode]);
        if (selectedMode == 2) {
          goToIngredientMenu();
        } else {
          showScanning();
          delay(800);
          currentState = STATE_SCANNING;
        }
      }
      break;
    }

    // ── 扫描中 ───────────────────────────────────────────────
    case STATE_SCANNING: {
      if (selectedMode == 0) {
        Serial.print("Simulating ingredient: ");
        Serial.println(inventory[currentFood % inventoryCount]);
        showResult(currentFood % recipeCount);
        currentFood  = (currentFood + 1) % recipeCount;
        currentState = STATE_RESULT;
      } else {
        Serial.println("Simulating receipt scan -> INGREDIENT_MENU");
        goToIngredientMenu();
      }
      break;
    }

    // ── 显示菜谱（一直停留）─────────────────────────────────
    case STATE_RESULT: {
      if (encPressed) {
        showScanning();
        delay(800);
        currentState = STATE_SCANNING;
      }
      break;
    }

    // ── 食材菜单 ─────────────────────────────────────────────
    case STATE_INGREDIENT_MENU: {
      if (now - lastActionTime > MENU_TIMEOUT_MS) {
        Serial.println("Timeout -> IDLE");
        encBtnHeld = false;
        goToIdle();
        break;
      }
      if (encDelta != 0 && inventoryCount > 0) {
        lastActionTime = now;
        encBtnHeld     = false;  // 滚动时取消长按
        menuIndex      = (menuIndex + encDelta + inventoryCount) % inventoryCount;
        showIngredientMenu();
      }

      bool held = encBtnIsHeld();

      if (held && inventoryCount > 0) {
        if (!encBtnHeld) {
          // 刚开始按下，记录时刻
          encBtnHeld    = true;
          encBtnDownTime = now;
          lastActionTime = now;
        } else {
          // 持续按住，计算进度
          unsigned long elapsed = now - encBtnDownTime;
          lastActionTime = now;

          if (elapsed >= LONG_PRESS_MS) {
            // 达到 3s，执行删除
            Serial.print("Long press delete: ");
            Serial.println(inventory[menuIndex]);
            encBtnHeld = false;
            deleteItem(menuIndex);
            showIngredientMenu();
          } else {
            // 显示长按进度条
            int progress = map(elapsed, 0, LONG_PRESS_MS, 0, SCREEN_WIDTH);
            showIngredientMenu();
            // 在底部叠加进度条
            display.fillRect(0, 54, progress, 10, SSD1306_WHITE);
            centerText("Hold to delete...", 56, 1);
            display.display();
          }
        }
      } else if (!held && encBtnHeld) {
        // 松开但没达到 3s，取消
        encBtnHeld = false;
        showIngredientMenu();
      }

      break;
    }
  }

  delay(10);
}

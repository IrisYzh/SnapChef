// FridgeLens - IR Proximity Sensor + OLED Display Prototype
// Hardware: XIAO ESP32S3 + IR sensor (3-pin) + SSD1306 OLED (128x64 I2C)
//
// Wiring:
//   IR Sensor:  VCC -> 3.3V, GND -> GND, OUT -> GPIO2 (D1)
//   OLED:       VCC -> 3.3V, GND -> GND, SDA -> GPIO5, SCL -> GPIO6
//
// Arduino IDE setup:
//   1. Board: "XIAO_ESP32S3"
//   2. Install libraries: Adafruit SSD1306, Adafruit GFX

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ----- Pin definitions -----
#define IR_PIN        2       // IR sensor OUT pin (LOW = object detected)
#define SDA_PIN       5       // I2C SDA
#define SCL_PIN       6       // I2C SCL

// ----- OLED settings -----
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDR     0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ----- Simulated food items and recipes -----
// In the final version, food labels come from TFLite model output.
// For now we cycle through a test list to verify the display pipeline.

const char* foodItems[] = {"Tomato", "Egg", "Cucumber", "Tofu", "Carrot"};
const int foodCount = 5;
int currentFood = 0;

// Each food item maps to a few dish suggestions
const char* recipes[][3] = {
  {"Tomato Egg Stir-fry", "Tomato Soup", "Bruschetta"},           // Tomato
  {"Steamed Egg", "Egg Fried Rice", "Omelette"},                  // Egg
  {"Cucumber Salad", "Cucumber Egg Stir-fry", "Pickled Cucumber"},// Cucumber
  {"Mapo Tofu", "Tofu Soup", "Fried Tofu"},                      // Tofu
  {"Glazed Carrots", "Carrot Soup", "Carrot Stir-fry"}            // Carrot
};

// ----- State management -----
bool lastDetected = false;
unsigned long detectTime = 0;
const unsigned long COOLDOWN_MS = 3000;  // wait 3s between detections

// ----- Helper: show centered text -----
void centerText(const char* text, int y, int size) {
  display.setTextSize(size);
  int16_t x1, y1;
  uint16_t w, h;
  display.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, y);
  display.print(text);
}

// ----- Display: idle screen -----
void showIdle() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  centerText("SnapChef", 8, 2);
  centerText("Hold item near", 36, 1);
  centerText("sensor to scan", 48, 1);
  display.display();
}

// ----- Display: detecting screen -----
void showDetecting() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  centerText("Scanning...", 24, 2);
  display.display();
}

// ----- Display: result screen -----
void showResult(int foodIndex) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Show detected item name
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Detected: ");
  display.println(foodItems[foodIndex]);

  // Draw a separator line
  display.drawLine(0, 12, SCREEN_WIDTH, 12, SSD1306_WHITE);

  // Show dish suggestions
  display.setCursor(0, 18);
  display.println("Dish suggestions:");
  display.println();
  for (int i = 0; i < 3; i++) {
    display.print(" ");
    display.println(recipes[foodIndex][i]);
  }

  display.display();
}

// ----- Setup -----
void setup() {
  Serial.begin(115200);
  delay(500);

  // IR sensor pin
  pinMode(IR_PIN, INPUT);

  // Initialize I2C with custom SDA/SCL pins
  Wire.begin(SDA_PIN, SCL_PIN);

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed");
    while (true);
  }

  Serial.println("FridgeLens prototype ready");
  showIdle();
}

// ----- Main loop -----
void loop() {
  bool detected = (digitalRead(IR_PIN) == LOW);  // most IR modules: LOW = object near
  unsigned long now = millis();

  // Rising edge: object just appeared, and cooldown has passed
  if (detected && !lastDetected && (now - detectTime > COOLDOWN_MS)) {
    detectTime = now;

    Serial.print("Object detected! Simulating: ");
    Serial.println(foodItems[currentFood]);

    // Step 1: show scanning animation
    showDetecting();
    delay(800);

    // Step 2: show result with dish suggestions
    showResult(currentFood);

    // Cycle to next food item for the next detection
    currentFood = (currentFood + 1) % foodCount;

    // Keep result on screen for a while
    delay(5000);

    // Return to idle
    showIdle();
  }

  lastDetected = detected;
  delay(50);  // debounce
}

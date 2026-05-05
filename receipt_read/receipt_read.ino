/*
 * SnapChef - Receipt Reader (XIAO ESP32S3 Sense)
 *
 * Press an external button -> capture JPEG from OV2640/OV3660 ->
 * POST multipart/form-data to /receipts/analyze -> print JSON to Serial.
 *
 * Hardware:
 *   - Seeed Studio XIAO ESP32S3 Sense (with camera FFC)
 *   - Push button: one leg -> GPIO2 (D1), other leg -> GND
 *     (uses internal pull-up, active LOW)
 *
 * WiFi:
 *   - UW MPSK is regular WPA2-PSK from the device's POV (each MAC has its own
 *     pre-shared key). Plain WiFi.begin(ssid, mpsk_password) works.
 *   - The MPSK password is the per-device key you generated at
 *     https://uwconnect.uw.edu/ (Wi-Fi -> Manage Devices). It is NOT your UW
 *     NetID password.
 *
 * Arduino IDE setup:
 *   - Board: "XIAO_ESP32S3"
 *   - PSRAM: "OPI PSRAM"
 *   - Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <esp_camera.h>

// ==================== USER CONFIG ====================

// UW MPSK credentials — fill in before flashing.
// Register the board's MAC address at https://uwconnect.uw.edu/ to get an MPSK.
static const char* WIFI_SSID = "UW MPSK";
static const char* WIFI_PASS = "45yVuUeU7At7gnxt";

// Backend — get the API key from the project owner. Do NOT commit a real key.
static const char* API_URL = "https://snapchef-production.up.railway.app/receipts/analyze";
static const char* HEALTHZ_URL = "https://snapchef-production.up.railway.app/healthz";
static const char* API_KEY = "snapchefasdfghjkl123456789";

// Multipart boundary — any unique ASCII token.
static const char* BOUNDARY = "----snapchef32boundary";

// Button pin (active LOW, internal pull-up).
static const int BUTTON_PIN = 2;

// HTTP timeout (ms). Backend says typical 4-5s, recommends 15-20s.
static const int HTTP_TIMEOUT_MS = 20000;

// ==================== Last-capture preview (HTTP) ====================
static WebServer previewServer(80);
static uint8_t* lastJpeg     = nullptr;
static size_t   lastJpegLen  = 0;
static uint16_t lastJpegW    = 0;
static uint16_t lastJpegH    = 0;

static void cacheLastJpeg(const uint8_t* buf, size_t len, uint16_t w, uint16_t h) {
  if (lastJpeg) { free(lastJpeg); lastJpeg = nullptr; lastJpegLen = 0; }
  lastJpeg = (uint8_t*)heap_caps_malloc(len, MALLOC_CAP_SPIRAM);
  if (!lastJpeg) { Serial.println("cache: malloc failed"); return; }
  memcpy(lastJpeg, buf, len);
  lastJpegLen = len;
  lastJpegW   = w;
  lastJpegH   = h;
}

static void handleLastJpeg() {
  if (!lastJpeg || lastJpegLen == 0) {
    previewServer.send(404, "text/plain", "No capture yet. Press the button.");
    return;
  }
  previewServer.sendHeader("Cache-Control", "no-cache");
  previewServer.send_P(200, "image/jpeg", (const char*)lastJpeg, lastJpegLen);
}

static void handlePreviewIndex() {
  String html = "<!doctype html><meta charset=utf-8>"
                "<title>SnapChef last capture</title>"
                "<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:12px}"
                "img{max-width:100%;border:2px solid #4ecca3;border-radius:6px}</style>"
                "<h2>Last capture</h2>";
  if (lastJpegLen) {
    html += "<p>" + String(lastJpegW) + "x" + String(lastJpegH) +
            " &middot; " + String((unsigned)lastJpegLen) + " bytes</p>";
    html += "<img src=\"/last?t=" + String(millis()) + "\">";
  } else {
    html += "<p>No capture yet. Press the button.</p>";
  }
  html += "<p><a style=color:#4ecca3 href=\"/\">refresh</a></p>";
  previewServer.send(200, "text/html", html);
}

// ==================== Camera pins (XIAO ESP32S3 Sense) ====================
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40
#define SIOC_GPIO_NUM   39
#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15
#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13

// ==================== Camera init ====================
static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  // Receipts are text-heavy — use a high resolution so Textract can OCR clearly.
  // UXGA = 1600x1200; payload usually ~150-400 KB at quality 12, well under 8 MB.
  config.frame_size   = FRAMESIZE_UXGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    if (s->id.PID == OV3660_PID) {
      s->set_brightness(s, 1);
      s->set_saturation(s, -1);
      s->set_vflip(s, 1);
    }
    // Good defaults for printed text on white paper.
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);
  }
  return true;
}

// ==================== WiFi ====================
static bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  delay(100);
  Serial.printf("MAC address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("Connecting to \"%s\"", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  // Wait up to 30s.
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 30000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connect FAILED");
    return false;
  }
  Serial.printf("WiFi connected. IP: %s  RSSI: %d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
  return true;
}

// ==================== Healthz self-check ====================
static void healthzSelfCheck() {
  WiFiClientSecure client;
  client.setInsecure();  // skip CA verification for the demo
  HTTPClient http;
  http.setTimeout(5000);
  if (!http.begin(client, HEALTHZ_URL)) {
    Serial.println("healthz: http.begin failed");
    return;
  }
  int code = http.GET();
  String body = http.getString();
  Serial.printf("healthz: HTTP %d  body: %s\n", code, body.c_str());
  http.end();
}

// ==================== Upload one captured JPEG ====================
static void uploadReceipt(const uint8_t* jpeg, size_t jpeg_len) {
  Serial.printf("Uploading %u bytes...\n", (unsigned)jpeg_len);

  // Build multipart body in PSRAM (UXGA frames can be a few hundred KB).
  String prefix = String("--") + BOUNDARY + "\r\n"
                  "Content-Disposition: form-data; name=\"image\"; filename=\"receipt.jpg\"\r\n"
                  "Content-Type: image/jpeg\r\n\r\n";
  String suffix = String("\r\n--") + BOUNDARY + "--\r\n";

  size_t total = prefix.length() + jpeg_len + suffix.length();
  uint8_t* body = (uint8_t*)heap_caps_malloc(total, MALLOC_CAP_SPIRAM);
  if (!body) {
    Serial.println("malloc failed (need PSRAM enabled)");
    return;
  }
  memcpy(body, prefix.c_str(), prefix.length());
  memcpy(body + prefix.length(), jpeg, jpeg_len);
  memcpy(body + prefix.length() + jpeg_len, suffix.c_str(), suffix.length());

  WiFiClientSecure client;
  client.setInsecure();  // demo: skip CA verification
  HTTPClient http;
  http.setTimeout(HTTP_TIMEOUT_MS);

  if (!http.begin(client, API_URL)) {
    Serial.println("http.begin failed");
    free(body);
    return;
  }

  String contentType = String("multipart/form-data; boundary=") + BOUNDARY;
  http.addHeader("Content-Type", contentType);
  http.addHeader("X-API-Key", API_KEY);

  unsigned long t0 = millis();
  int code = http.POST(body, total);
  unsigned long dt = millis() - t0;

  Serial.printf("HTTP %d in %lu ms\n", code, dt);
  String resp = http.getString();
  Serial.println("--- response body ---");
  Serial.println(resp);
  Serial.println("--- end ---");

  http.end();
  free(body);
}

static void captureAndSend() {
  // Drop a couple of stale frames so AE/AWB can settle on the receipt.
  for (int i = 0; i < 2; i++) {
    camera_fb_t* drop = esp_camera_fb_get();
    if (drop) esp_camera_fb_return(drop);
    delay(80);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    return;
  }
  if (fb->format != PIXFORMAT_JPEG) {
    Serial.println("Frame is not JPEG (camera misconfigured)");
    esp_camera_fb_return(fb);
    return;
  }
  Serial.printf("Captured JPEG: %ux%u, %u bytes\n",
                (unsigned)fb->width, (unsigned)fb->height, (unsigned)fb->len);

  cacheLastJpeg(fb->buf, fb->len, fb->width, fb->height);

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi dropped, reconnecting...");
    connectWiFi();
  }
  if (WiFi.status() == WL_CONNECTED) {
    uploadReceipt(fb->buf, fb->len);
  }
  esp_camera_fb_return(fb);
}

// ==================== Button (active LOW, debounced edge) ====================
static bool lastBtn = HIGH;
static unsigned long lastDebounceMs = 0;
static const unsigned long DEBOUNCE_MS = 50;

static bool buttonPressed() {
  bool reading = digitalRead(BUTTON_PIN);
  if (reading == LOW && lastBtn == HIGH &&
      millis() - lastDebounceMs > DEBOUNCE_MS) {
    lastDebounceMs = millis();
    lastBtn = reading;
    return true;
  }
  lastBtn = reading;
  return false;
}

// ==================== Setup / Loop ====================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== SnapChef Receipt Reader ===");

  if (!psramFound()) {
    Serial.println("FATAL: PSRAM not detected. Enable 'OPI PSRAM' in Tools menu.");
    while (true) delay(1000);
  }

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  delay(50);
  lastBtn = digitalRead(BUTTON_PIN);

  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed.");
    while (true) delay(1000);
  }
  Serial.println("Camera OK");

  if (!connectWiFi()) {
    Serial.println("Continuing without WiFi; will retry on button press.");
  } else {
    healthzSelfCheck();
    previewServer.on("/", handlePreviewIndex);
    previewServer.on("/last", handleLastJpeg);
    previewServer.begin();
    Serial.printf("Preview: http://%s/  (or /last for raw JPEG)\n",
                  WiFi.localIP().toString().c_str());
  }

  Serial.println("Ready. Press the button to capture and upload a receipt.");
}

void loop() {
  previewServer.handleClient();
  if (buttonPressed()) {
    Serial.println("\n[BUTTON] capture + upload");
    captureAndSend();
    Serial.println("Ready for next press.");
  }
  delay(10);
}

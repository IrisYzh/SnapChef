/*
 * Vegetable Classification on XIAO ESP32S3 Sense
 * Edge Impulse (Veggie_inferencing, 96x96 INT8) + WiFi preview
 *
 * Hardware:
 *   - Seeed Studio XIAO ESP32S3 Sense (with camera FFC expansion)
 *   - OV2640 / OV3660 / OV5640 camera module
 *
 * Arduino IDE setup:
 *   1. Board package: "esp32 by Espressif", v3.x
 *   2. Board:              "XIAO_ESP32S3"
 *   3. PSRAM:              "OPI PSRAM"                       <-- REQUIRED
 *   4. USB CDC On Boot:    "Enabled"                         <-- REQUIRED for Serial
 *   5. Partition Scheme:   "Huge APP (3MB No OTA/1MB SPIFFS)"
 *   6. Library: import  train/ei-veggie-arduino-1.0.5-impulse-#1.zip
 *              via  Sketch -> Include Library -> Add .ZIP Library
 *   7. Update WIFI_SSID / WIFI_PASS below
 */

#include <Veggie_inferencing.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_heap_caps.h"
#include <WiFi.h>
#include <WebServer.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ==================== XIAO ESP32S3 Sense camera pins ====================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ==================== WiFi Config ====================
const char* WIFI_SSID = "iPhone";
const char* WIFI_PASS = "mlh020702";

WebServer server(80);

// ==================== Constants ====================
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS   96
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS   96
#define EI_CAMERA_FRAME_BYTE_SIZE         3

// Camera capture frame size (JPEG), decoded to RGB888 then cropped/resized to 96x96.
#define CAMERA_FRAMESIZE      FRAMESIZE_QVGA    // 320x240
constexpr int kCaptureWidth  = 320;
constexpr int kCaptureHeight = 240;

constexpr float CONFIDENCE_THRESHOLD = 0.85f;
constexpr unsigned long kInferenceIntervalMs = 500;

// ==================== Buffers (PSRAM) ====================
static uint8_t *snapshot_buf = nullptr;   // 96*96*3  = 27648 B
static uint8_t *rgb888_buf   = nullptr;   // 320*240*3 = 230400 B

// ==================== Shared inference state (loop writes, handlers read) ====================
String        last_label        = "N/A";
float         last_confidence   = 0.0f;   // 0..1
unsigned long last_inference_ms = 0;

// ==================== Camera init ====================
bool ei_camera_init() {
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
    config.frame_size   = CAMERA_FRAMESIZE;
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return false;
    }
    sensor_t* s = esp_camera_sensor_get();
    if (s && s->id.PID == OV3660_PID) {
        // OV3660 on XIAO Sense needs vflip + a brightness/saturation nudge.
        s->set_brightness(s, 1);
        s->set_saturation(s, -1);
        s->set_vflip(s, 1);
        s->set_hmirror(s, 0);
    }
    return true;
}

// Grab a JPEG frame, decode to RGB888, crop+resize to img_width x img_height,
// write into out_buf as tightly-packed RGB bytes.
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }
    if (fb->format != PIXFORMAT_JPEG) {
        Serial.printf("Unexpected format %d (expected JPEG)\n", (int)fb->format);
        esp_camera_fb_return(fb);
        return false;
    }
    if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf)) {
        Serial.println("JPEG decode failed");
        esp_camera_fb_return(fb);
        return false;
    }
    int src_w = fb->width;
    int src_h = fb->height;
    esp_camera_fb_return(fb);

    int r = ei::image::processing::crop_and_interpolate_rgb888(
        rgb888_buf, src_w, src_h, out_buf, img_width, img_height);
    if (r != 0) {
        Serial.printf("crop_and_interpolate_rgb888 failed: %d\n", r);
        return false;
    }
    return true;
}

// Edge Impulse feature extractor: pack each RGB byte triple into a float
// holding 0xRRGGBB (what run_classifier() expects for camera input).
static int ei_get_feature_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 3;
    size_t out_ix = 0;

    while (length--) {
        out_ptr[out_ix] =
            (snapshot_buf[pixel_ix] << 16) +
            (snapshot_buf[pixel_ix + 1] << 8) +
            snapshot_buf[pixel_ix + 2];
        pixel_ix += 3;
        out_ix++;
    }
    return 0;
}

// ==================== Web Server Handlers ====================
void handleRoot() {
    String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Veggie Classifier (Edge Impulse)</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
    h1 { color: #4ecca3; }
    img { border: 3px solid #4ecca3; border-radius: 8px; max-width: 100%; }
    .result { font-size: 28px; margin: 20px; padding: 15px; border-radius: 8px; }
    .known { background: #16213e; border: 2px solid #4ecca3; }
    .unknown { background: #16213e; border: 2px solid #e94560; }
    .info { font-size: 14px; color: #999; }
  </style>
</head>
<body>
  <h1>Vegetable Classifier</h1>
  <div><img id="cam" src="/capture" /></div>
  <div id="result" class="result known">Loading...</div>
  <p class="info">XIAO ESP32S3 Sense | Edge Impulse 96x96 INT8 | auto-refresh 2s</p>
  <script>
    function refresh() {
      document.getElementById('cam').src = '/capture?t=' + Date.now();
      fetch('/result').then(r => r.json()).then(data => {
        var el = document.getElementById('result');
        if (data.known) {
          el.className = 'result known';
          el.innerHTML = data.label + ' <b>' + data.confidence + '%</b><br><span class="info">' + data.time + 'ms inference</span>';
        } else {
          el.className = 'result unknown';
          el.innerHTML = 'Object not known<br><span class="info">best guess: ' + data.label + ' ' + data.confidence + '%  |  ' + data.time + 'ms</span>';
        }
      });
    }
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>
)rawliteral";
    server.send(200, "text/html", html);
}

void handleCapture() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb || fb->format != PIXFORMAT_JPEG) {
        if (fb) esp_camera_fb_return(fb);
        server.send(500, "text/plain", "Capture failed");
        return;
    }
    server.sendHeader("Cache-Control", "no-cache");
    server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
}

void handleResult() {
    bool known = last_confidence >= CONFIDENCE_THRESHOLD;
    String json = "{\"label\":\""   + last_label + "\","
                  "\"confidence\":" + String(last_confidence * 100.0f, 1) + ","
                  "\"time\":"       + String(last_inference_ms) + ","
                  "\"known\":"      + String(known ? "true" : "false") + "}";
    server.send(200, "application/json", json);
}

// ==================== Setup ====================
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=== Veggie Classifier (Edge Impulse on XIAO ESP32S3 Sense) ===\n");

    if (!psramFound()) {
        Serial.println("FATAL: PSRAM not detected. Set PSRAM=\"OPI PSRAM\" in "
                       "Tools menu and re-flash. Halting.");
        while (true) delay(1000);
    }
    Serial.printf("PSRAM free: %u bytes\n", (unsigned)ESP.getFreePsram());

    snapshot_buf = (uint8_t*)heap_caps_malloc(
        EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE,
        MALLOC_CAP_SPIRAM);
    if (!snapshot_buf) {
        Serial.println("FATAL: snapshot_buf allocation failed. Halting.");
        while (true) delay(1000);
    }

    rgb888_buf = (uint8_t*)heap_caps_malloc(
        kCaptureWidth * kCaptureHeight * 3, MALLOC_CAP_SPIRAM);
    if (!rgb888_buf) {
        Serial.println("FATAL: rgb888_buf allocation failed. Halting.");
        while (true) delay(1000);
    }

    if (!ei_camera_init()) {
        Serial.println("FATAL: Failed to initialize camera. Halting.");
        while (true) delay(1000);
    }
    Serial.println("Camera initialized");

    WiFi.mode(WIFI_STA);
    delay(100);
    Serial.printf("Connecting to %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nWiFi connected!\nOpen http://%s in browser\n\n",
                  WiFi.localIP().toString().c_str());

    server.on("/",        handleRoot);
    server.on("/capture", handleCapture);
    server.on("/result",  handleResult);
    server.begin();

    Serial.printf("Threshold: %.0f%%\n", CONFIDENCE_THRESHOLD * 100.0f);
    Serial.println("Ready.\n");
}

// ==================== Loop ====================
void loop() {
    server.handleClient();

    static unsigned long last_run = 0;
    if (millis() - last_run < kInferenceIntervalMs) return;
    last_run = millis();

    if (!ei_camera_capture(EI_CLASSIFIER_INPUT_WIDTH,
                           EI_CLASSIFIER_INPUT_HEIGHT,
                           snapshot_buf)) {
        Serial.println("Capture failed");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_get_feature_data;

    ei_impulse_result_t result = {0};
    unsigned long t0 = millis();
    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
    unsigned long dt = millis() - t0;
    if (res != EI_IMPULSE_OK) {
        Serial.printf("run_classifier failed (%d)\n", res);
        return;
    }

    float best_conf = -1.0f;
    int   best_idx  = -1;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > best_conf) {
            best_conf = result.classification[ix].value;
            best_idx  = (int)ix;
        }
    }

    last_label        = (best_idx >= 0) ? String(result.classification[best_idx].label) : "N/A";
    last_confidence   = (best_conf > 0) ? best_conf : 0.0f;
    last_inference_ms = dt;

    Serial.printf("[%lums] ", dt);
    if (best_conf >= CONFIDENCE_THRESHOLD) {
        Serial.printf("%s (%.1f%%)\n", last_label.c_str(), best_conf * 100.0f);
    } else {
        Serial.printf("Object not known (best: %s %.1f%%)\n",
                      last_label.c_str(), best_conf * 100.0f);
    }
}

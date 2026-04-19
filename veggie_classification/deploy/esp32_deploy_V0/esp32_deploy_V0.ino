/*
 * Vegetable Classification on XIAO ESP32S3 Sense
 *
 * Uses OV2640 camera + TFLite Micro to classify 10 vegetables.
 * Outputs "Object not known" when confidence is below threshold.
 * Includes WiFi web server to preview camera images in browser.
 *
 * Hardware: Seeed Studio XIAO ESP32S3 Sense (with OV2640)
 * Model: Custom tiny CNN, INT8 quantized (~24KB)
 *
 * Setup:
 *   1. Arduino IDE: install esp32 board package v3.x
 *      URL: https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *   2. Board: XIAO_ESP32S3, PSRAM: OPI PSRAM
 *   3. Library: install "TensorFlowLite_ESP32" from Library Manager
 *   4. Update WIFI_SSID and WIFI_PASS below
 */

#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "labels.h"

// ==================== WiFi Config ====================
const char* WIFI_SSID = "iPhone";
const char* WIFI_PASS = "mlh020702";

WebServer server(80);

// ==================== Camera Pins (XIAO ESP32S3 Sense) ====================
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

// ==================== Model Constants ====================
constexpr int kImageWidth    = 96;
constexpr int kImageHeight   = 96;
constexpr int kImageChannels = 3;

constexpr int kTensorArenaSize = 100 * 1024;  // 100KB arena
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr unsigned long kInferenceIntervalMs = 2000;  // classify every 2s

// ==================== TFLite Globals ====================
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

float input_scale = 0.0f;
int32_t input_zero_point = 0;
float output_scale = 0.0f;
int32_t output_zero_point = 0;

// Last inference result (shared with web server)
String last_label = "N/A";
float last_confidence = 0.0f;
unsigned long last_inference_ms = 0;

// ==================== Camera Init ====================
bool initCamera() {
  camera_config_t config;
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
  config.pixel_format = PIXFORMAT_JPEG;    // JPEG for web streaming
  config.frame_size   = FRAMESIZE_QVGA;    // 320x240 for viewing
  config.jpeg_quality = 10;
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ==================== Preprocessing ====================
// Convert JPEG frame to 96x96 INT8 RGB for model input
// We capture JPEG for web display, then re-capture RGB565 for inference
void captureAndPreprocess(int8_t* input_data) {
  // Temporarily switch to RGB565 at 96x96 for inference
  sensor_t* s = esp_camera_sensor_get();
  s->set_pixformat(s, PIXFORMAT_RGB565);
  s->set_framesize(s, FRAMESIZE_96X96);
  delay(100);  // let sensor settle

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb || fb->format != PIXFORMAT_RGB565) {
    if (fb) esp_camera_fb_return(fb);
    // Fallback: try with QQVGA
    s->set_framesize(s, FRAMESIZE_QQVGA);
    delay(100);
    fb = esp_camera_fb_get();
    if (!fb) {
      // Restore JPEG mode
      s->set_pixformat(s, PIXFORMAT_JPEG);
      s->set_framesize(s, FRAMESIZE_QVGA);
      return;
    }
  }

  int src_w = fb->width;
  int src_h = fb->height;
  uint16_t* src = (uint16_t*)fb->buf;

  // Center-crop to square, then nearest-neighbor resize
  int crop = min(src_w, src_h);
  int x_off = (src_w - crop) / 2;
  int y_off = (src_h - crop) / 2;

  for (int y = 0; y < kImageHeight; y++) {
    for (int x = 0; x < kImageWidth; x++) {
      int sx = x_off + (x * crop) / kImageWidth;
      int sy = y_off + (y * crop) / kImageHeight;
      uint16_t pixel = src[sy * src_w + sx];

      uint8_t r = ((pixel >> 11) & 0x1F) * 255 / 31;
      uint8_t g = ((pixel >> 5)  & 0x3F) * 255 / 63;
      uint8_t b = (pixel         & 0x1F) * 255 / 31;

      int idx = (y * kImageWidth + x) * kImageChannels;
      input_data[idx + 0] = (int8_t)((int)r - 128);
      input_data[idx + 1] = (int8_t)((int)g - 128);
      input_data[idx + 2] = (int8_t)((int)b - 128);
    }
  }

  esp_camera_fb_return(fb);

  // Restore JPEG mode for web streaming
  s->set_pixformat(s, PIXFORMAT_JPEG);
  s->set_framesize(s, FRAMESIZE_QVGA);
}

// ==================== Web Server Handlers ====================

// Serve the main page with live camera view + classification result
void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Veggie Classifier</title>
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
  <div>
    <img id="cam" src="/capture" />
  </div>
  <div id="result" class="result known">Loading...</div>
  <p class="info">XIAO ESP32S3 Sense | 96x96 INT8 TFLite | Auto-refresh 2s</p>
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

// Serve a JPEG capture
void handleCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Capture failed");
    return;
  }
  server.sendHeader("Cache-Control", "no-cache");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// Serve latest classification result as JSON
void handleResult() {
  String json = "{\"label\":\"" + last_label + "\","
                "\"confidence\":" + String(last_confidence, 1) + ","
                "\"time\":" + String(last_inference_ms) + ","
                "\"known\":" + String(last_confidence >= CONFIDENCE_THRESHOLD * 100 ? "true" : "false") + "}";
  server.send(200, "application/json", json);
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Vegetable Classifier ===\n");

  // Init camera
  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed. Halting.");
    while (true) delay(1000);
  }
  Serial.println("Camera OK");

  // Print MAC address before connecting
  WiFi.mode(WIFI_STA);
  delay(100);  // Wait for WiFi hardware init
  Serial.printf("MAC address: %s\n", WiFi.macAddress().c_str());

  // Connect WiFi
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nWiFi connected!\n");
  Serial.printf("MAC address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("Open http://%s in browser\n\n", WiFi.localIP().toString().c_str());

  // Start web server
  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.on("/result", handleResult);
  server.begin();

  // Load model
  tfl_model = tflite::GetModel(model_data);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model version mismatch: %d vs %d\n",
                  tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  // Register ops
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddRelu();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FATAL: AllocateTensors() failed.");
    while (true) delay(1000);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  input_scale      = input_tensor->params.scale;
  input_zero_point = input_tensor->params.zero_point;
  output_scale      = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;

  Serial.printf("Arena used: %d bytes\n", interpreter->arena_used_bytes());
  Serial.printf("Input:  [%d,%d,%d,%d] scale=%.6f zp=%d\n",
      input_tensor->dims->data[0], input_tensor->dims->data[1],
      input_tensor->dims->data[2], input_tensor->dims->data[3],
      input_scale, input_zero_point);
  Serial.printf("Output: %d classes, scale=%.6f zp=%d\n",
      output_tensor->dims->data[1], output_scale, output_zero_point);
  Serial.printf("Threshold: %.0f%%\n\n", CONFIDENCE_THRESHOLD * 100);
}

// ==================== Loop ====================
void loop() {
  server.handleClient();  // Handle web requests

  static unsigned long last_run = 0;
  if (millis() - last_run < kInferenceIntervalMs) return;
  last_run = millis();

  // Switch to RGB565 for inference, preprocess, then switch back
  captureAndPreprocess(input_tensor->data.int8);

  // Inference
  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }
  unsigned long dt = millis() - t0;

  // Read & dequantize output
  int8_t* out = output_tensor->data.int8;
  float max_conf = -1.0f;
  int max_idx = -1;

  for (int i = 0; i < NUM_CLASSES; i++) {
    float conf = (out[i] - output_zero_point) * output_scale;
    if (conf > max_conf) {
      max_conf = conf;
      max_idx = i;
    }
  }

  // Update shared result for web server
  last_label = LABELS[max_idx];
  last_confidence = max_conf * 100.0f;
  last_inference_ms = dt;

  // Serial output
  Serial.printf("[%lums] ", dt);
  if (max_conf >= CONFIDENCE_THRESHOLD) {
    Serial.printf("%s (%.1f%%)\n", LABELS[max_idx], max_conf * 100.0f);
  } else {
    Serial.printf("%s (best: %s %.1f%%)\n",
                  UNKNOWN_LABEL, LABELS[max_idx], max_conf * 100.0f);
  }
}

/*
 * Vegetable Classification on XIAO ESP32S3 Sense + OV3660 + SSD1306 OLED
 * (V5, MobileNetV2 V3 weights — serial-only, no WiFi)
 *
 * OV3660 camera + TFLite Micro + MobileNetV2 (alpha=0.35, INT8) classifies
 * 9 vegetables and shows the result on a 128x64 SSD1306 OLED. Results are
 * smoothed (EMA + hysteresis) so brief confidence dips don't flicker the
 * display to "Uncertain".
 *
 * Hardware:
 *   - Seeed Studio XIAO ESP32S3 Sense (camera FFC)
 *   - OV3660 camera module (3MP)
 *   - SSD1306 128x64 I2C OLED  —  SDA=GPIO5, SCL=GPIO6, addr 0x3C
 *
 * Libraries (install via Library Manager):
 *   - TensorFlowLite_ESP32
 *   - Adafruit GFX Library
 *   - Adafruit SSD1306
 *
 * Arduino IDE setup:
 *   1. Board package: esp32 by Espressif, v3.x
 *   2. Board: "XIAO_ESP32S3"
 *   3. PSRAM: "OPI PSRAM"  (REQUIRED — arena lives in PSRAM)
 *   4. Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"  (model is ~625 KB)
 */

#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "labels.h"

// ==================== OLED (SSD1306 128x64) ====================
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT    64
#define OLED_RESET       -1
#define SCREEN_ADDRESS 0x3C
#define I2C_SDA           5
#define I2C_SCL           6
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ==================== Camera Pins (XIAO ESP32S3 Sense) ====================
// These pins are identical for OV2640 / OV3660 / OV5640 on the XIAO Sense —
// the FFC connector mapping is fixed by the board, not the sensor.
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

// MobileNetV2-0.35 @ 96x96 INT8 needs ~200-300 KB of arena. 512 KB gives
// headroom for the baked preprocess ops + persistent input/output tensors.
constexpr int kTensorArenaSize = 512 * 1024;
uint8_t* tensor_arena = nullptr;

constexpr unsigned long kInferenceIntervalMs = 500;

// JPEG at QVGA used for capture; decoded to RGB888 via fmt2rgb888().
#define CAMERA_FRAMESIZE      FRAMESIZE_QVGA     // 320x240
constexpr int kCaptureWidth  = 320;
constexpr int kCaptureHeight = 240;

// Persistent RGB888 decode buffer (allocated once in PSRAM).
uint8_t* rgb888_buf = nullptr;

// ==================== TFLite Globals ====================
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

float input_scale = 0.0f;
int32_t input_zero_point = 0;
float output_scale = 0.0f;
int32_t output_zero_point = 0;

// ==================== Prediction smoothing ====================
// UX target: user pulls a veggie from the fridge, holds it up, and expects a
// fast AND stable answer. Two failure modes to avoid:
//   (a) long wait — a clearly-visible veggie (single frame >=80%) sits at
//       "Uncertain" while EMA slowly catches up.
//   (b) flicker — model glitches for one frame but user hasn't moved the
//       veggie, yet the display flips to Uncertain and back.
//
// Asymmetric filter, three tiers:
//   1. Snap-lock: if RAW confidence >= HIGH_CONF_LOCK (0.80), display that
//      label immediately. This covers both "showed a clear veggie" and "user
//      swapped to a different veggie". Bypasses EMA lag.
//   2. EMA band: when raw is borderline, argmax the low-pass-filtered
//      probability vector. alpha=0.5 ~= 2-frame memory.
//   3. Sticky release: once a label is shown, only flip to Uncertain after
//      UNKNOWN_HOLD (4) consecutive below-threshold frames ~= 8 s of
//      sustained uncertainty. One glitch frame is ignored.
constexpr float HIGH_CONF_LOCK = 0.80f;
constexpr float EMA_ALPHA      = 0.5f;
constexpr int   UNKNOWN_HOLD   = 4;

float ema_probs[NUM_CLASSES] = {0};
bool  ema_init = false;
int   below_threshold_streak = 0;

// What the OLED is currently showing. -1 means "Uncertain".
int           displayed_idx = -1;
float         displayed_conf = 0.0f;
unsigned long displayed_dt   = 0;

// ==================== Camera Init ====================
bool initCamera() {
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
  // OV3660 tolerates 20 MHz XCLK on ESP32S3. If you see stripes/glitches,
  // drop to 10 MHz.
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = CAMERA_FRAMESIZE;
  config.jpeg_quality = 10;
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (!s) {
    Serial.println("esp_camera_sensor_get() returned null");
    return false;
  }
  Serial.printf("Sensor PID=0x%04X (OV2640=0x%04X, OV3660=0x%04X, OV5640=0x%04X)\n",
                s->id.PID, OV2640_PID, OV3660_PID, OV5640_PID);
  if (s->id.PID != OV3660_PID) {
    Serial.println("WARNING: sensor is not OV3660. Code will still work but "
                   "color/brightness tweaks may need adjusting.");
  } else {
    s->set_brightness(s, 1);
    s->set_saturation(s, -1);
    s->set_contrast(s, 0);
    s->set_vflip(s, 1);
    s->set_hmirror(s, 0);
  }
  return true;
}

// ==================== Preprocessing ====================
// Grab the latest JPEG frame, decode it to RGB888 in a persistent PSRAM
// buffer, center-crop to square, nearest-neighbor downsample to 96x96, and
// quantize into the model's INT8 input tensor.
//
// MobileNetV2 here has `preprocess_input` ((x/127.5)-1) baked into the graph,
// so the model's float input range is raw [0, 255]. Quant formula:
// int8 = round(pixel_u8 / input_scale + zp).  (Expect input_scale ≈ 1.0, zp ≈ -128.)
bool captureAndPreprocess(int8_t* input_data) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
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
  const int src_w = fb->width;
  const int src_h = fb->height;
  esp_camera_fb_return(fb);

  const int crop   = src_w < src_h ? src_w : src_h;
  const int x_off  = (src_w - crop) / 2;
  const int y_off  = (src_h - crop) / 2;
  const float inv_input_scale = 1.0f / input_scale;

  for (int y = 0; y < kImageHeight; y++) {
    const int sy = y_off + (y * crop) / kImageHeight;
    for (int x = 0; x < kImageWidth; x++) {
      const int sx = x_off + (x * crop) / kImageWidth;
      const int pi = (sy * src_w + sx) * 3;
      const uint8_t r = rgb888_buf[pi + 0];
      const uint8_t g = rgb888_buf[pi + 1];
      const uint8_t b = rgb888_buf[pi + 2];

      const int idx = (y * kImageWidth + x) * kImageChannels;
      input_data[idx + 0] = (int8_t)lroundf((float)r * inv_input_scale + input_zero_point);
      input_data[idx + 1] = (int8_t)lroundf((float)g * inv_input_scale + input_zero_point);
      input_data[idx + 2] = (int8_t)lroundf((float)b * inv_input_scale + input_zero_point);
    }
  }
  return true;
}

// ==================== Display ====================
// Full redraw each inference (128x64 is tiny — cheap). Layout:
//   row 0:     "Veggie Classifier"  size=1
//   row 2-5:    class label         size=2 (or "Uncertain")
//   row 6-7:    confidence + timing size=1 (or "best: X  NN%")
void renderDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(F("Veggie Classifier"));
  display.drawFastHLine(0, 10, SCREEN_WIDTH, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 20);
  if (displayed_idx >= 0) {
    display.println(LABELS[displayed_idx]);
  } else {
    display.println(F("Uncertain"));
  }

  display.setTextSize(1);
  display.setCursor(0, 52);
  if (displayed_idx >= 0) {
    display.printf("%.1f%%  %lums", displayed_conf * 100.0f, displayed_dt);
  } else if (ema_init) {
    // Even when Uncertain, surface the smoothed best guess so the user can see
    // what the model is leaning toward.
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
      if (ema_probs[i] > ema_probs[best]) best = i;
    }
    display.printf("best: %s %.0f%%",
                   LABELS[best], ema_probs[best] * 100.0f);
  }
  display.display();
}

void showBootMessage(const char* msg) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("Veggie Classifier"));
  display.setCursor(0, 20);
  display.println(msg);
  display.display();
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Veggie Classifier V5 (MobileNetV2 V3 weights, OLED + serial) ===\n");

  Wire.begin(I2C_SDA, I2C_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    // OLED missing isn't fatal — keep running with serial output only.
    Serial.println("WARNING: SSD1306 init failed (check I2C wiring on SDA=5/SCL=6).");
  } else {
    showBootMessage("Booting...");
  }

  if (!psramFound()) {
    Serial.println("FATAL: PSRAM not detected. Set PSRAM=\"OPI PSRAM\" in "
                   "Tools menu and re-flash. Halting.");
    while (true) delay(1000);
  }
  Serial.printf("PSRAM free: %u bytes\n", (unsigned)ESP.getFreePsram());

  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed. Halting.");
    while (true) delay(1000);
  }
  Serial.println("Camera OK");

  const size_t rgb888_size = kCaptureWidth * kCaptureHeight * 3;
  rgb888_buf = (uint8_t*)heap_caps_malloc(rgb888_size, MALLOC_CAP_SPIRAM);
  if (!rgb888_buf) {
    Serial.printf("FATAL: Failed to allocate %u bytes RGB888 decode buffer in PSRAM.\n",
                  (unsigned)rgb888_size);
    while (true) delay(1000);
  }
  Serial.printf("RGB888 decode buffer: %u bytes @ %p\n",
                (unsigned)rgb888_size, rgb888_buf);

  tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM);
  if (!tensor_arena) {
    Serial.printf("FATAL: Failed to allocate %d bytes of tensor arena in PSRAM.\n",
                  kTensorArenaSize);
    while (true) delay(1000);
  }
  Serial.printf("Tensor arena allocated in PSRAM: %d bytes @ %p\n",
                kTensorArenaSize, tensor_arena);

  tfl_model = tflite::GetModel(model_data);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("FATAL: Model schema %lu != expected %d\n",
                  (unsigned long)tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  // Ops used by MobileNetV2-0.35 INT8 + baked Lambda(preprocess_input).
  // If AllocateTensors() later complains "Didn't find op for builtin opcode
  // ...", add that op here and re-flash.
  static tflite::MicroMutableOpResolver<14> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddAdd();                  // inverted-residual skip connections
  resolver.AddPad();                  // stride-2 block padding
  resolver.AddMean();                 // GlobalAveragePooling2D
  resolver.AddReshape();              // after GAP, before Dense
  resolver.AddSoftmax();
  resolver.AddMul();                  // baked preprocess: x * (1/127.5)
  resolver.AddSub();                  // baked preprocess: (x/127.5) - 1
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddLogistic();             // reserved — in case of hard_sigmoid
  resolver.AddRelu6();                // MobileNetV2 activation

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FATAL: AllocateTensors() failed. Check missing ops or "
                   "increase kTensorArenaSize.");
    while (true) delay(1000);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  input_scale       = input_tensor->params.scale;
  input_zero_point  = input_tensor->params.zero_point;
  output_scale      = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;

  Serial.printf("Arena used: %u / %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("Input:  [%d,%d,%d,%d] dtype=%d scale=%.6f zp=%ld\n",
      input_tensor->dims->data[0], input_tensor->dims->data[1],
      input_tensor->dims->data[2], input_tensor->dims->data[3],
      (int)input_tensor->type, input_scale, (long)input_zero_point);
  Serial.printf("Output: %d classes, dtype=%d scale=%.6f zp=%ld\n",
      output_tensor->dims->data[1], (int)output_tensor->type,
      output_scale, (long)output_zero_point);
  Serial.printf("Threshold: %.0f%%\n\n", CONFIDENCE_THRESHOLD * 100);

  showBootMessage("Ready");
}

// ==================== Loop ====================
void loop() {
  static unsigned long last_run = 0;
  if (millis() - last_run < kInferenceIntervalMs) return;
  last_run = millis();

  if (!captureAndPreprocess(input_tensor->data.int8)) return;

  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }
  unsigned long dt = millis() - t0;

  // Dequantize all 9 class probabilities (softmax is baked into the graph,
  // so these already sum to ~1 and are valid probabilities).
  int8_t* out = output_tensor->data.int8;
  float probs[NUM_CLASSES];
  float raw_max = -1e30f;
  int   raw_idx = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] = (out[i] - output_zero_point) * output_scale;
    if (probs[i] > raw_max) { raw_max = probs[i]; raw_idx = i; }
  }

  // --- Filter 1: EMA on the probability vector ---
  if (!ema_init) {
    for (int i = 0; i < NUM_CLASSES; i++) ema_probs[i] = probs[i];
    ema_init = true;
  } else {
    for (int i = 0; i < NUM_CLASSES; i++) {
      ema_probs[i] = EMA_ALPHA * probs[i] + (1.0f - EMA_ALPHA) * ema_probs[i];
    }
  }

  int   ema_idx  = 0;
  float ema_conf = ema_probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (ema_probs[i] > ema_conf) { ema_conf = ema_probs[i]; ema_idx = i; }
  }

  // Three-tier decision (see smoothing comment block above).
  const char* decision;
  if (raw_max >= HIGH_CONF_LOCK) {
    // Tier 1 — snap-lock on raw. User showed the veggie clearly; don't wait.
    displayed_idx  = raw_idx;
    displayed_conf = raw_max;
    displayed_dt   = dt;
    below_threshold_streak = 0;
    decision = "snap";
  } else if (ema_conf >= CONFIDENCE_THRESHOLD) {
    // Tier 2 — smoothed above threshold. Trust the EMA.
    displayed_idx  = ema_idx;
    displayed_conf = ema_conf;
    displayed_dt   = dt;
    below_threshold_streak = 0;
    decision = "ema";
  } else {
    // Tier 3 — below threshold. Hold the last confident label until
    // UNKNOWN_HOLD frames in a row are uncertain.
    below_threshold_streak++;
    if (below_threshold_streak >= UNKNOWN_HOLD) {
      displayed_idx  = -1;
      displayed_conf = ema_conf;
      displayed_dt   = dt;
      decision = "uncertain";
    } else {
      decision = "hold";   // keep previous displayed_idx / conf / dt
    }
  }

  renderDisplay();

  // Serial shows which tier fired, so filter behaviour is easy to eyeball.
  Serial.printf("[%lums] raw:%s %.1f%%  ema:%s %.1f%%  [%s]  -> ",
                dt,
                LABELS[raw_idx], raw_max * 100.0f,
                LABELS[ema_idx], ema_conf * 100.0f,
                decision);
  if (displayed_idx >= 0) {
    Serial.printf("%s (%.1f%%)\n", LABELS[displayed_idx], displayed_conf * 100.0f);
  } else {
    Serial.printf("%s\n", UNKNOWN_LABEL);
  }
}

/*
 * Vegetable Classification on XIAO ESP32S3 Sense + OV3660
 * Serial-only build (V5, MobileNetV2 V3 weights)
 *
 * Uses OV3660 camera + TFLite Micro + MobileNetV2 (alpha=0.35, INT8) to
 * classify 10 vegetables. Outputs "Object not known" when confidence is below
 * threshold. Results are printed over the USB serial port only — no WiFi,
 * no web server.
 *
 * Hardware:
 *   - Seeed Studio XIAO ESP32S3 Sense (the Sense expansion with camera FFC)
 *   - OV3660 camera module (3MP) plugged into the FFC connector
 *
 * Model: MobileNetV2 (alpha=0.35) @ 96x96, INT8 quantized, tensor arena in
 * PSRAM. `preprocess_input` ((x/127.5)-1) is baked into the graph, so the
 * firmware feeds raw [0,255] pixels quantized by the model's own input scale.
 * Weights are from classification_V3 (video-frame augmented training set).
 *
 * Arduino IDE setup:
 *   1. Board package: esp32 by Espressif, v3.x
 *   2. Board: "XIAO_ESP32S3"
 *   3. PSRAM: "OPI PSRAM"  (REQUIRED — arena lives in PSRAM)
 *   4. Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"  (model is ~625 KB)
 *   5. Library: "TensorFlowLite_ESP32" from Library Manager
 */

#include <esp_camera.h>
#include <esp_heap_caps.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "labels.h"

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

constexpr unsigned long kInferenceIntervalMs = 2000;

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

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Veggie Classifier V5 (MobileNetV2 V3 weights, serial-only) ===\n");

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

  int8_t* out = output_tensor->data.int8;
  float max_conf = -1e30f;
  int   max_idx  = -1;
  for (int i = 0; i < NUM_CLASSES; i++) {
    float conf = (out[i] - output_zero_point) * output_scale;
    if (conf > max_conf) {
      max_conf = conf;
      max_idx  = i;
    }
  }

  Serial.printf("[%lums] ", dt);
  if (max_conf >= CONFIDENCE_THRESHOLD) {
    Serial.printf("%s (%.1f%%)\n", LABELS[max_idx], max_conf * 100.0f);
  } else {
    Serial.printf("%s (best: %s %.1f%%)\n",
                  UNKNOWN_LABEL, LABELS[max_idx], max_conf * 100.0f);
  }
}

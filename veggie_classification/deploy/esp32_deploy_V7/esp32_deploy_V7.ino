/*
 * Vegetable Classification on XIAO ESP32S3 Sense + OV3660 — display-agnostic
 * (V7, MobileNetV2 V4 weights — 128x128 input, Rescaling preprocess)
 *
 * Same pipeline as V6 (camera -> INT8 MobileNetV2 -> 3-tier smoothing) with
 * three changes to match the V4 training notebook:
 *   - kImageWidth/Height bumped 96 -> 128.
 *   - Tensor arena bumped 512KB -> 700KB (~(128/96)^2 grew the activations).
 *   - Inference interval kept at 500 ms. Measured invoke is ~320 ms at 128x128
 *     vs ~210 ms at 96x96 — only 1.52x, not the (128/96)^2 = 1.78x the activation
 *     volume would suggest. On this board MobileNetV2 is PSRAM-bandwidth bound
 *     (lots of 1x1 + depthwise convs) plus a fixed per-invoke overhead, so the
 *     wall-clock scales sub-linearly with input area. 500 ms cadence still leaves
 *     ~180 ms slack for the display hook.
 *
 * The classification result is exposed through a single `ClassificationResult`
 * struct and a single hook function, `displayResult()`. Reuse this sketch on
 * any display by replacing only that one function.
 *
 * Hardware:
 *   - Seeed Studio XIAO ESP32S3 Sense (camera FFC)
 *   - OV3660 camera module (3MP)
 *
 * Libraries:
 *   - TensorFlowLite_ESP32  (Library Manager)
 *
 * Arduino IDE setup:
 *   1. Board: "XIAO_ESP32S3"
 *   2. PSRAM: "OPI PSRAM"  (required — tensor arena lives in PSRAM)
 *   3. Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 */

#include <esp_camera.h>
#include <esp_heap_caps.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "labels.h"

// ============================================================================
//                                  CONFIG
// ============================================================================

// Inference cadence — loop() runs at this rate (ms). Inference at 128x128 is
// ~480 ms on PSRAM arena, so 800 ms gives ~320 ms of slack for the display hook.
constexpr unsigned long kInferenceIntervalMs = 800;

// Model input — must match the trained model.
// V7: 128x128 (was 96x96 in V6).
constexpr int kImageWidth    = 128;
constexpr int kImageHeight   = 128;
constexpr int kImageChannels = 3;

// Tensor arena (PSRAM). 700 KB covers MobileNetV2-0.35 @ 128x128 with headroom.
// V6 used 512 KB at 96x96; activations grow ~(128/96)^2 ~= 1.78x.
constexpr int kTensorArenaSize = 700 * 1024;

// Smoothing tunables — symmetric streak filter. Same as V6.
//   - A frame counts as a "detection" iff its top-1 confidence >= CONF_LOCK
//     AND its top-1 class is NOT the trained "Unknown" class.
//   - LOCK_STREAK_NEEDED detection frames in a row of the SAME class are
//     required before that class is shown.
//   - RELEASE_STREAK_NEEDED non-confirming frames in a row are required
//     before a shown label is cleared.
constexpr float CONF_LOCK             = 0.80f;
constexpr int   LOCK_STREAK_NEEDED    = 3;
constexpr int   RELEASE_STREAK_NEEDED = 3;

// Camera pins — XIAO ESP32S3 Sense FFC. Same on OV2640 / OV3660 / OV5640.
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

// JPEG capture format (decoded to RGB888 in memory).
#define CAMERA_FRAMESIZE      FRAMESIZE_QVGA     // 320x240
constexpr int kCaptureWidth   = 320;
constexpr int kCaptureHeight  = 240;

// ============================================================================
//                          PUBLIC RESULT TYPE
// ============================================================================

// Everything a display hook might want to know about a single classification.
struct ClassificationResult {
  enum Tier { TIER_LOCKED, TIER_HOLD, TIER_BUILDING, TIER_IDLE };

  bool          known;          // false => "Object not known"
  int           labelIdx;       // -1 if !known, else index into LABELS[]
  const char*   label;          // LABELS[labelIdx] or UNKNOWN_LABEL
  float         confidence;     // confidence shown to the user
  unsigned long inferenceMs;    // last invoke() wall time

  // Diagnostic — which smoothing tier produced this frame's decision.
  //   locked   : displayed label, this frame confirmed it
  //   hold     : displayed label, this frame did NOT confirm it (sticky)
  //   building : nothing displayed, accumulating consecutive >=CONF_LOCK frames
  //   idle     : nothing displayed, no candidate streak in progress
  Tier          tier;
  const char*   tierName;
  float         rawConfidence;  // top-1 confidence before any smoothing
  int           rawLabelIdx;    // top-1 class index before any smoothing
};

// ============================================================================
//                              PRIVATE GLOBALS
// ============================================================================

namespace {

uint8_t* tensor_arena = nullptr;
uint8_t* rgb888_buf   = nullptr;

const tflite::Model*       tfl_model       = nullptr;
tflite::MicroInterpreter*  interpreter     = nullptr;
TfLiteTensor*              input_tensor    = nullptr;
TfLiteTensor*              output_tensor   = nullptr;

float   input_scale       = 0.0f;
int32_t input_zero_point  = 0;
float   output_scale      = 0.0f;
int32_t output_zero_point = 0;

// Smoothing state — symmetric streak filter.
int   current_display_idx     = -1;
float current_display_conf    = 0.0f;
int   lock_candidate_idx      = -1;
int   lock_streak             = 0;
int   release_streak          = 0;

// Index of the trained "Unknown" class within LABELS[]; -1 if not present.
int   unknown_class_idx       = -1;

}  // namespace

// ============================================================================
//                              CAMERA INIT
// ============================================================================

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
  config.frame_size   = CAMERA_FRAMESIZE;
  config.jpeg_quality = 10;
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) return false;

  sensor_t* s = esp_camera_sensor_get();
  if (s && s->id.PID == OV3660_PID) {
    s->set_brightness(s, 1);
    s->set_saturation(s, -1);
    s->set_vflip(s, 1);
  }

  rgb888_buf = (uint8_t*)heap_caps_malloc(kCaptureWidth * kCaptureHeight * 3,
                                          MALLOC_CAP_SPIRAM);
  return rgb888_buf != nullptr;
}

// ============================================================================
//                              MODEL INIT
// ============================================================================

static bool initModel() {
  tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM);
  if (!tensor_arena) return false;

  tfl_model = tflite::GetModel(model_data);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) return false;

  // Ops used by MobileNetV2-0.35 INT8 + baked Rescaling preprocess.
  // Same set as V6 (Rescaling lowers to Mul+Add, both already in the resolver).
  static tflite::MicroMutableOpResolver<14> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddAdd();
  resolver.AddPad();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddLogistic();
  resolver.AddRelu6();

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) return false;

  input_tensor      = interpreter->input(0);
  output_tensor     = interpreter->output(0);
  input_scale       = input_tensor->params.scale;
  input_zero_point  = input_tensor->params.zero_point;
  output_scale      = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;
  return true;
}

// ============================================================================
//                       CAPTURE + PREPROCESS + INVOKE
// ============================================================================

static bool captureAndPreprocess(int8_t* dst) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  if (fb->format != PIXFORMAT_JPEG ||
      !fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf)) {
    esp_camera_fb_return(fb);
    return false;
  }
  const int src_w = fb->width;
  const int src_h = fb->height;
  esp_camera_fb_return(fb);

  // Center-crop to square, then nearest-neighbour resize to kImageWidth.
  // Quantise pixels into INT8 using the model's input scale + zero point.
  // (Rescaling((x/127.5)-1) is baked into the graph — feed raw [0,255] pixels.)
  const int crop  = src_w < src_h ? src_w : src_h;
  const int x_off = (src_w - crop) / 2;
  const int y_off = (src_h - crop) / 2;
  const float inv_scale = 1.0f / input_scale;

  for (int y = 0; y < kImageHeight; y++) {
    const int sy = y_off + (y * crop) / kImageHeight;
    for (int x = 0; x < kImageWidth; x++) {
      const int sx = x_off + (x * crop) / kImageWidth;
      const int pi = (sy * src_w + sx) * 3;
      const int idx = (y * kImageWidth + x) * kImageChannels;
      dst[idx + 0] = (int8_t)lroundf((float)rgb888_buf[pi + 0] * inv_scale + input_zero_point);
      dst[idx + 1] = (int8_t)lroundf((float)rgb888_buf[pi + 1] * inv_scale + input_zero_point);
      dst[idx + 2] = (int8_t)lroundf((float)rgb888_buf[pi + 2] * inv_scale + input_zero_point);
    }
  }
  return true;
}

// ============================================================================
//                              SMOOTHING
// ============================================================================

static void applySmoothing(int   raw_idx, float raw_max,
                           unsigned long dt,
                           ClassificationResult& out) {
  out.rawLabelIdx   = raw_idx;
  out.rawConfidence = raw_max;
  out.inferenceMs   = dt;

  const bool is_unknown_class = (raw_idx == unknown_class_idx);
  const bool is_detection     = (raw_max >= CONF_LOCK) && !is_unknown_class;

  if (is_detection) {
    if (raw_idx == lock_candidate_idx) {
      lock_streak++;
    } else {
      lock_candidate_idx = raw_idx;
      lock_streak = 1;
    }

    if (raw_idx == current_display_idx) {
      release_streak = 0;
      current_display_conf = raw_max;
    } else {
      release_streak++;
    }

    if (lock_streak >= LOCK_STREAK_NEEDED && raw_idx != current_display_idx) {
      current_display_idx  = raw_idx;
      current_display_conf = raw_max;
      release_streak = 0;
    }
  } else {
    lock_candidate_idx = -1;
    lock_streak        = 0;
    if (current_display_idx != -1) {
      release_streak++;
      if (release_streak >= RELEASE_STREAK_NEEDED) {
        current_display_idx = -1;
        current_display_conf = 0.0f;
        release_streak = 0;
      }
    }
  }

  if (current_display_idx >= 0) {
    out.known      = true;
    out.labelIdx   = current_display_idx;
    out.label      = LABELS[current_display_idx];
    out.confidence = current_display_conf;
    if (is_detection && raw_idx == current_display_idx) {
      out.tier     = ClassificationResult::TIER_LOCKED;
      out.tierName = "locked";
    } else {
      out.tier     = ClassificationResult::TIER_HOLD;
      out.tierName = "hold";
    }
  } else {
    out.known      = false;
    out.labelIdx   = -1;
    out.label      = UNKNOWN_LABEL;
    out.confidence = 0.0f;
    if (is_detection) {
      out.tier     = ClassificationResult::TIER_BUILDING;
      out.tierName = "building";
    } else {
      out.tier     = ClassificationResult::TIER_IDLE;
      out.tierName = "idle";
    }
  }
}

// ============================================================================
//                              CLASSIFY ONE FRAME
// ============================================================================

static bool classifyOnce(ClassificationResult& out) {
  if (!captureAndPreprocess(input_tensor->data.int8)) return false;

  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) return false;
  unsigned long dt = millis() - t0;

  // Argmax over INT8 outputs is monotone — no need to dequantise everything.
  int8_t* q = output_tensor->data.int8;
  int   raw_idx = 0;
  int8_t raw_q  = q[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (q[i] > raw_q) { raw_q = q[i]; raw_idx = i; }
  }
  const float raw_max = (raw_q - output_zero_point) * output_scale;

  applySmoothing(raw_idx, raw_max, dt, out);
  return true;
}

// ============================================================================
//             USER-REPLACEABLE DISPLAY HOOK  <<<<<  EDIT ONLY THIS
// ============================================================================
//
// `r` is fully populated; render it however you want. Default below prints
// one line per inference to USB serial — useful for benchmarking the V4 model.
//
// Output format (per frame):
//   [<invoke_ms>ms] raw:<top1_label> <conf>%  [<tier>]  -> <displayed_label> (<conf>%)
//
// Watch the `raw:` column to compare against val-set predictions, and the
// `[tier]` column to see whether the smoothing layer is suppressing useful
// signal (lots of `building` with no `locked`) or holding stale labels (lots
// of `hold`). Tweak CONF_LOCK / *_STREAK_NEEDED above accordingly.

void printResult(const ClassificationResult& r) {
  Serial.printf("[%lums] raw:%s %.1f%%  [%s]  -> ",
                r.inferenceMs,
                LABELS[r.rawLabelIdx], r.rawConfidence * 100.0f,
                r.tierName);
  if (r.known) {
    Serial.printf("%s (%.1f%%)\n", r.label, r.confidence * 100.0f);
  } else {
    Serial.printf("%s\n", r.label);
  }
}

// ============================================================================
//                              setup() / loop()
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Veggie Classifier V7 (MobileNetV2 V4 weights, 128x128) ===");

  if (!psramFound())  { Serial.println("FATAL: PSRAM not detected."); while (1) delay(1000); }
  if (!initCamera())  { Serial.println("FATAL: Camera init failed."); while (1) delay(1000); }
  if (!initModel())   { Serial.println("FATAL: Model init failed.");  while (1) delay(1000); }

  // Locate the trained "Unknown" class so the smoothing layer can treat it
  // as "no detection". -1 means the model has no Unknown class (older models).
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (strcmp(LABELS[i], "Unknown") == 0) { unknown_class_idx = i; break; }
  }
  Serial.printf("Unknown class index: %d\n", unknown_class_idx);

  Serial.printf("Model: %u bytes, input %dx%d, arena %d KB\n",
                model_data_len, kImageWidth, kImageHeight,
                kTensorArenaSize / 1024);
  Serial.printf("Input quant: scale=%.6f zp=%d   Output quant: scale=%.6f zp=%d\n",
                input_scale, (int)input_zero_point,
                output_scale, (int)output_zero_point);
  Serial.printf("Ready. %d classes, lock threshold %.0f%%, interval %lu ms.\n",
                NUM_CLASSES, CONF_LOCK * 100, kInferenceIntervalMs);
}

void loop() {
  static unsigned long last_run = 0;
  if (millis() - last_run < kInferenceIntervalMs) return;
  last_run = millis();

  ClassificationResult r;
  if (!classifyOnce(r)) return;
  printResult(r);
}

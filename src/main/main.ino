/*
 * SnapChef — Main controller (XIAO ESP32S3 Sense)
 *
 * Hosts the camera, HC-SR04 ultrasonic, WiFi, and a NimBLE GATT server.
 * The display device (Waveshare 4.3" Touch LCD) connects over BLE and
 * drives the UI; this firmware is "headless".
 *
 * Capabilities exposed over BLE:
 *   1. Idle proximity wake — emits {"evt":"proximity_wake"} when an object
 *      is detected within ~10 cm of the HC-SR04.
 *   2. Veggie classification — on {"cmd":"start_veggie_scan"} runs the
 *      MobileNetV2 INT8 classifier (128x128, V4 weights) lifted from esp32_deploy_V7.
 *      Locks on >=80% top-1 confidence over 3 consecutive frames; gives up
 *      after 20 s.
 *   3. Receipt scan — on {"cmd":"capture_receipt"} switches the camera to
 *      UXGA, captures a single JPEG, POSTs it to /receipts/analyze, and
 *      streams the raw JSON response back on the data characteristic.
 *   4. Recipe lookup — on {"cmd":"get_recipe","ingredients":[...]} returns a
 *      hard-coded mock recipe (Carrot/Eggplant/generic) — backend not built.
 *
 * Wiring (XIAO ESP32S3 Sense):
 *   HC-SR04: VCC -> 5V, GND -> GND, TRIG -> GPIO2, ECHO -> GPIO1
 *   Camera : FFC connector (OV2640 / OV3660)
 *
 * Arduino IDE setup:
 *   - Board: "XIAO_ESP32S3"
 *   - PSRAM: "OPI PSRAM"
 *   - Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 *
 * Required libraries:
 *   - TensorFlowLite_ESP32
 *   - NimBLE-Arduino (h2zero) — small heap footprint vs built-in BLE stack
 *   - WiFi / WiFiClientSecure / HTTPClient (built in)
 */

#include <Arduino.h>
#include <vector>

#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <mbedtls/base64.h>

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

#include <NimBLEDevice.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "labels.h"
#include "snapchef_ble.h"

// ============================================================================
//                                  CONFIG
// ============================================================================

// --- WiFi (UW MPSK; same credentials as receipt_read.ino) ---
static const char* WIFI_SSID = "UW MPSK";
static const char* WIFI_PASS = "45yVuUeU7At7gnxt";

// --- Backend ---
static const char* API_URL     = "https://snapchef-production.up.railway.app/receipts/analyze";
static const char* HEALTHZ_URL = "https://snapchef-production.up.railway.app/healthz";
// Recipe endpoints (Claude-backed on the backend; change BASE_URL if you move
// these to a different host). Same X-API-Key as receipts.
static const char* RECIPES_LIST_URL  = "https://snapchef-production.up.railway.app/recipes/list";
static const char* RECIPES_STEPS_URL = "https://snapchef-production.up.railway.app/recipes/steps";
static const char* API_KEY     = "snapchefasdfghjkl123456789";
static const char* BOUNDARY    = "----snapchef32boundary";
static const int   HTTP_TIMEOUT_MS = 20000;
static const int   RECIPE_HTTP_TIMEOUT_MS = 30000;   // LLM generation can be slow

// --- HC-SR04 ---
static const int   TRIG_PIN     = 2;
static const int   ECHO_PIN     = 1;
static const float DETECT_CM    = 10.0f;
static const unsigned long PROXIMITY_INTERVAL_MS = 200;
static const unsigned long PROXIMITY_COOLDOWN_MS = 3000;

// --- Veggie classifier (mirrors esp32_deploy_V6.ino) ---
static const unsigned long INFERENCE_INTERVAL_MS = 500;
static const unsigned long VEGGIE_TIMEOUT_MS     = 20000;
static const int  IMG_W = 128, IMG_H = 128, IMG_C = 3;     // V4 model: 128x128
static const int  TENSOR_ARENA_SIZE = 700 * 1024;          // ~(128/96)^2 of V3's 512KB
static const float CONF_LOCK = 0.80f;
static const int   LOCK_STREAK_NEEDED    = 3;
static const int   RELEASE_STREAK_NEEDED = 3;

// --- Camera pins (XIAO ESP32S3 Sense FFC) ---
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

// QVGA RGB888 buffer used by the veggie pipeline.
static const int CAPTURE_W = 320;
static const int CAPTURE_H = 240;

// ============================================================================
//                              GLOBAL STATE
// ============================================================================

enum MainState {
    STATE_IDLE,
    STATE_VEGGIE_SCAN,
    STATE_RECEIPT_CAPTURE,
    STATE_RECEIPT_UPLOAD,
    STATE_RECIPE,
};

static volatile MainState gState = STATE_IDLE;
static volatile bool gCancelRequested = false;
static String  gPendingPurpose = "in";           // for veggie scan
static String  gPendingIngredientsJson = "[]";   // for recipe
static String  gPendingTrigger = "";             // for recipe list (just-removed item)
static String  gPendingDish    = "";             // for recipe steps (chosen dish)

// --- TFLM ---
static uint8_t* tensor_arena = nullptr;
static uint8_t* rgb888_buf   = nullptr;
static const tflite::Model*       tfl_model     = nullptr;
static tflite::MicroInterpreter*  interpreter   = nullptr;
static TfLiteTensor*              input_tensor  = nullptr;
static TfLiteTensor*              output_tensor = nullptr;
static float   input_scale = 0, output_scale = 0;
static int32_t input_zero_point = 0, output_zero_point = 0;
static int     unknown_class_idx = -1;

// --- Veggie smoothing ---
static int   current_display_idx  = -1;
static float current_display_conf = 0.0f;
static int   lock_candidate_idx   = -1;
static int   lock_streak          = 0;
static int   release_streak       = 0;

// --- BLE ---
static NimBLEServer*         bleServer    = nullptr;
static NimBLECharacteristic* chrCmd       = nullptr;
static NimBLECharacteristic* chrEvt       = nullptr;
static NimBLECharacteristic* chrData      = nullptr;
static volatile bool         bleConnected = false;

// --- Pending commands queue (filled in BLE callback, drained in loop()) ---
static volatile bool gCmdPending = false;
static String gPendingCmd;

// ============================================================================
//                              FORWARD DECLS
// ============================================================================

static bool initCamera();
static bool initModel();
static bool captureAndPreprocess(int8_t* dst);
static void resetSmoothing();
static void runVeggieScan();
static void runReceiptCapture();
static void runRecipe();
static void runRecipeList();
static void runRecipeSteps();
static void sendEvent(const String& json);
static void sendData(const String& payload);

// ============================================================================
//                               CAMERA
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
    // Init at UXGA so the camera DMA buffer is sized for the largest capture
    // we'll ever do (receipt path). Sensor is immediately downshifted to QVGA
    // below for the veggie streaming path; runtime set_framesize() can then
    // safely upshift back to UXGA without FB-OVF.
    config.frame_size   = FRAMESIZE_UXGA;
    config.jpeg_quality = 12;
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
    if (s) {
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 1);
        s->set_gain_ctrl(s, 1);
        // DMA buffer is sized for UXGA above; drop the sensor down to QVGA for
        // veggie streaming. Receipt path will upshift back to UXGA on demand.
        s->set_framesize(s, FRAMESIZE_QVGA);
        s->set_quality(s, 10);
    }

    rgb888_buf = (uint8_t*)heap_caps_malloc(CAPTURE_W * CAPTURE_H * 3, MALLOC_CAP_SPIRAM);
    return rgb888_buf != nullptr;
}

// Drains stale frames from the queue — call after framesize/quality changes
// so the next esp_camera_fb_get() returns one at the new settings.
static void drainCameraFrames(int n) {
    for (int i = 0; i < n; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(60);
    }
}

// ============================================================================
//                                TFLM MODEL
// ============================================================================

static bool initModel() {
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) return false;

    tfl_model = tflite::GetModel(model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) return false;

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
        tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return false;

    input_tensor      = interpreter->input(0);
    output_tensor     = interpreter->output(0);
    input_scale       = input_tensor->params.scale;
    input_zero_point  = input_tensor->params.zero_point;
    output_scale      = output_tensor->params.scale;
    output_zero_point = output_tensor->params.zero_point;

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (strcmp(LABELS[i], "Unknown") == 0) { unknown_class_idx = i; break; }
    }
    return true;
}

// ============================================================================
//                       VEGGIE: capture + preprocess + invoke
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

    // Center-crop to square, nearest-neighbour resize to IMG_W, INT8 quantise.
    const int crop  = src_w < src_h ? src_w : src_h;
    const int x_off = (src_w - crop) / 2;
    const int y_off = (src_h - crop) / 2;
    const float inv_scale = 1.0f / input_scale;

    for (int y = 0; y < IMG_H; y++) {
        const int sy = y_off + (y * crop) / IMG_H;
        for (int x = 0; x < IMG_W; x++) {
            const int sx = x_off + (x * crop) / IMG_W;
            const int pi = (sy * src_w + sx) * 3;
            const int idx = (y * IMG_W + x) * IMG_C;
            dst[idx + 0] = (int8_t)lroundf((float)rgb888_buf[pi + 0] * inv_scale + input_zero_point);
            dst[idx + 1] = (int8_t)lroundf((float)rgb888_buf[pi + 1] * inv_scale + input_zero_point);
            dst[idx + 2] = (int8_t)lroundf((float)rgb888_buf[pi + 2] * inv_scale + input_zero_point);
        }
    }
    return true;
}

static void resetSmoothing() {
    current_display_idx  = -1;
    current_display_conf = 0.0f;
    lock_candidate_idx   = -1;
    lock_streak          = 0;
    release_streak       = 0;
}

// Returns true if a class is currently locked (display-ready).
static bool classifyOnce(int& outIdx, float& outConf, bool& outLocked) {
    outLocked = false;
    if (!captureAndPreprocess(input_tensor->data.int8)) return false;
    if (interpreter->Invoke() != kTfLiteOk) return false;

    int8_t* q = output_tensor->data.int8;
    int   raw_idx = 0;
    int8_t raw_q  = q[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (q[i] > raw_q) { raw_q = q[i]; raw_idx = i; }
    }
    const float raw_max = (raw_q - output_zero_point) * output_scale;

    const bool is_unknown_class = (raw_idx == unknown_class_idx);
    const bool is_detection     = (raw_max >= CONF_LOCK) && !is_unknown_class;

    if (is_detection) {
        if (raw_idx == lock_candidate_idx) lock_streak++;
        else { lock_candidate_idx = raw_idx; lock_streak = 1; }

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
        outIdx    = current_display_idx;
        outConf   = current_display_conf;
        outLocked = true;
    } else {
        outIdx  = raw_idx;
        outConf = raw_max;
    }
    return true;
}

// ============================================================================
//                                   WiFi
// ============================================================================

static bool connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return true;
    WiFi.mode(WIFI_STA);
    delay(50);
    Serial.printf("[wifi] mac=%s connecting to %s\n",
                  WiFi.macAddress().c_str(), WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 30000) {
        delay(400);
        Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[wifi] connect FAILED");
        return false;
    }
    Serial.printf("[wifi] connected ip=%s rssi=%d\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    return true;
}

static void healthzSelfCheck() {
    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;
    http.setTimeout(5000);
    if (!http.begin(client, HEALTHZ_URL)) return;
    int code = http.GET();
    Serial.printf("[wifi] healthz %d %s\n", code, http.getString().c_str());
    http.end();
}

// ============================================================================
//                              RECEIPT UPLOAD
// ============================================================================

// Returns the full HTTP response body (raw JSON) on 200, empty string on err.
// On error also fills `errCode` / `errMsg` for an event payload.
static String uploadReceipt(const uint8_t* jpeg, size_t jpeg_len,
                            String& errCode, String& errMsg) {
    errCode = ""; errMsg = "";
    if (!connectWiFi()) {
        errCode = "wifi"; errMsg = "WiFi unavailable";
        return "";
    }

    String prefix = String("--") + BOUNDARY + "\r\n"
                    "Content-Disposition: form-data; name=\"image\"; filename=\"receipt.jpg\"\r\n"
                    "Content-Type: image/jpeg\r\n\r\n";
    String suffix = String("\r\n--") + BOUNDARY + "--\r\n";
    size_t total = prefix.length() + jpeg_len + suffix.length();
    uint8_t* body = (uint8_t*)heap_caps_malloc(total, MALLOC_CAP_SPIRAM);
    if (!body) {
        errCode = "oom"; errMsg = "PSRAM alloc failed";
        return "";
    }
    memcpy(body, prefix.c_str(), prefix.length());
    memcpy(body + prefix.length(), jpeg, jpeg_len);
    memcpy(body + prefix.length() + jpeg_len, suffix.c_str(), suffix.length());

    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;
    http.setTimeout(HTTP_TIMEOUT_MS);
    if (!http.begin(client, API_URL)) {
        free(body);
        errCode = "http_begin"; errMsg = "begin() failed";
        return "";
    }
    http.addHeader("Content-Type", String("multipart/form-data; boundary=") + BOUNDARY);
    http.addHeader("X-API-Key", API_KEY);

    unsigned long t0 = millis();
    int code = http.POST(body, total);
    unsigned long dt = millis() - t0;
    String resp = http.getString();
    http.end();
    free(body);

    Serial.printf("[receipt] HTTP %d in %lu ms, %u bytes\n",
                  code, dt, (unsigned)resp.length());
    Serial.print("[receipt] body: ");
    Serial.println(resp);
    if (code != 200) {
        errCode = "http_" + String(code);
        errMsg  = resp.length() ? resp : "HTTP error";
        return "";
    }
    return resp;
}

// ============================================================================
//                                   RECIPE
// ============================================================================

// Hard-coded mocks until the backend lands. Returns full JSON.
// `ingredientsJson` is the raw JSON array string from the command, e.g.
//   ["Carrot","Tomato","Onion"]
static String buildRecipeMock(const String& ingredientsJson) {
    bool hasCarrot   = ingredientsJson.indexOf("\"Carrot\"")   >= 0;
    bool hasEggplant = ingredientsJson.indexOf("\"Eggplant\"") >= 0;

    String title;
    String steps;   // JSON array body, no surrounding brackets
    int    timeMin = 20;

    if (hasCarrot) {
        title = "Honey Glazed Carrots";
        timeMin = 20;
        steps = "\"Peel and slice carrots into 1\\/2-inch coins.\","
                "\"Saute in butter over medium heat for 6 min until tender.\","
                "\"Stir in 2 tbsp honey, a pinch of salt, finish with parsley.\"";
    } else if (hasEggplant) {
        title = "Eggplant Parmesan";
        timeMin = 35;
        steps = "\"Slice eggplant 1\\/4-inch thick; salt and rest 10 min.\","
                "\"Pan-fry slices in olive oil until golden on both sides.\","
                "\"Layer with marinara and mozzarella; bake 18 min at 200C.\"";
    } else {
        title = "Stir-Fried Vegetable Medley";
        timeMin = 15;
        steps = "\"Chop everything you took out into thumb-sized pieces.\","
                "\"Hot wok, oil, garlic 10s, harder veg first 2 min, soft veg 1 min.\","
                "\"Splash soy sauce + sesame oil, toss, plate.\"";
    }

    String json = "{";
    json += "\"evt\":\"recipe_result\",";
    json += "\"title\":\""  + title + "\",";
    json += "\"time_min\":" + String(timeMin) + ",";
    json += "\"steps\":[" + steps + "],";
    json += "\"ingredients\":" + ingredientsJson;
    json += "}";
    return json;
}

// Generic HTTPS JSON POST. Returns the response body on 200, empty on err.
// On error, fills errCode/errMsg for a payload back to the display.
static String httpPostJson(const char* url, const String& body,
                            int timeout_ms, String& errCode, String& errMsg) {
    errCode = ""; errMsg = "";
    if (!connectWiFi()) { errCode = "wifi"; errMsg = "WiFi unavailable"; return ""; }

    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;
    http.setTimeout(timeout_ms);
    if (!http.begin(client, url)) { errCode = "http_begin"; errMsg = "begin() failed"; return ""; }
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-API-Key", API_KEY);

    unsigned long t0 = millis();
    int code = http.POST((uint8_t*)body.c_str(), body.length());
    unsigned long dt = millis() - t0;
    String resp = http.getString();
    http.end();

    Serial.printf("[http] POST %s -> %d in %lu ms (%u bytes)\n",
                  url, code, dt, (unsigned)resp.length());
    if (code != 200) {
        errCode = "http_" + String(code);
        errMsg  = resp.length() ? resp : "HTTP error";
        return "";
    }
    return resp;
}

// Trim wrapping whitespace/quotes from a dish name.
static String trimQuoted(const String& s) {
    int a = 0, b = (int)s.length();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t')) b--;
    return s.substring(a, b);
}

// Pull a string array out of {"dishes":["A","B","C"]} into a pipe-joined
// "A|B|C". Tolerant — uses simple scanning so we don't need a JSON lib.
static String dishesFromJson(const String& json) {
    String pat = "\"dishes\":[";
    int i = json.indexOf(pat); if (i < 0) return "";
    i += pat.length();
    int end = json.indexOf(']', i); if (end < 0) end = json.length();
    String out;
    while (i < end) {
        int q1 = json.indexOf('"', i); if (q1 < 0 || q1 >= end) break;
        int q2 = json.indexOf('"', q1 + 1); if (q2 < 0 || q2 > end) break;
        String name = trimQuoted(json.substring(q1 + 1, q2));
        if (name.length()) { if (out.length()) out += '|'; out += name; }
        i = q2 + 1;
    }
    return out;
}

// POST to /recipes/list — backend should return {"dishes":["A","B","C"]}.
// We forward dishes as a pipe-joined list inside a small JSON envelope so
// the display can splice them straight into populateTakeOutRecipes().
static void runRecipeList() {
    gState = STATE_RECIPE;
    Serial.printf("[recipe_list] trigger=%s fridge=%s\n",
                  gPendingTrigger.c_str(), gPendingIngredientsJson.c_str());

    String body = "{\"trigger\":\"";
    body += gPendingTrigger;
    body += "\",\"fridge\":";
    body += gPendingIngredientsJson;
    body += "}";

    String errCode, errMsg;
    String resp = httpPostJson(RECIPES_LIST_URL, body, RECIPE_HTTP_TIMEOUT_MS,
                                errCode, errMsg);

    String dishes;
    if (resp.length()) dishes = dishesFromJson(resp);

    if (!dishes.length()) {
        // Fall back to a tiny mock so the UI still has something to show
        // when the backend isn't ready / call failed.
        Serial.printf("[recipe_list] fallback (err=%s)\n", errCode.c_str());
        if (gPendingTrigger.indexOf("Carrot") >= 0)
            dishes = "Honey Glazed Carrots|Carrot Soup|Roasted Carrots";
        else if (gPendingTrigger.indexOf("Tomato") >= 0)
            dishes = "Caprese Salad|Tomato Omelette|Pasta Pomodoro";
        else if (gPendingTrigger.indexOf("Eggplant") >= 0)
            dishes = "Eggplant Parmesan|Baba Ganoush|Ratatouille";
        else
            dishes = "Stir-Fried Medley|Quick Soup|Veggie Wrap";
    }

    String payload = "{\"evt\":\"recipe_list\",\"dishes\":\"";
    payload += dishes;
    payload += "\"}";
    sendEvent("{\"evt\":\"recipe_list\"}");
    sendData(payload);
    gState = STATE_IDLE;
}

// POST to /recipes/steps — backend should return
//   {"title":"...","time_min":15,"steps":["...","..."]}.
// We add the evt envelope and forward to the display.
static void runRecipeSteps() {
    gState = STATE_RECIPE;
    Serial.printf("[recipe_steps] dish=%s fridge=%s\n",
                  gPendingDish.c_str(), gPendingIngredientsJson.c_str());

    String body = "{\"dish\":\"";
    body += gPendingDish;
    body += "\",\"fridge\":";
    body += gPendingIngredientsJson;
    body += "}";

    String errCode, errMsg;
    String resp = httpPostJson(RECIPES_STEPS_URL, body, RECIPE_HTTP_TIMEOUT_MS,
                                errCode, errMsg);

    String payload;
    if (resp.length()) {
        // Backend returns body without evt envelope; splice one in.
        // The response starts with '{'; insert "evt":"recipe_result", right after.
        int brace = resp.indexOf('{');
        if (brace >= 0) {
            payload  = resp.substring(0, brace + 1);
            payload += "\"evt\":\"recipe_result\",";
            payload += resp.substring(brace + 1);
        }
    }
    if (payload.length() == 0) {
        // Fall back to the existing mock-builder when the API fails so the
        // UI still gets a recipe.
        Serial.printf("[recipe_steps] fallback (err=%s)\n", errCode.c_str());
        payload = buildRecipeMock(gPendingIngredientsJson);
    }

    sendEvent("{\"evt\":\"recipe_result\"}");
    sendData(payload);
    gState = STATE_IDLE;
}

// ============================================================================
//                                  HC-SR04
// ============================================================================

static float readDistanceCm() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long dur = pulseIn(ECHO_PIN, HIGH, 30000);  // 30 ms timeout (~5 m)
    if (dur == 0) return 999.0f;
    return dur * 0.0343f / 2.0f;
}

// ============================================================================
//                                BLE: helpers
// ============================================================================

static void sendEvent(const String& json) {
    if (!chrEvt || !bleConnected) return;
    chrEvt->setValue((uint8_t*)json.c_str(), json.length());
    chrEvt->notify();
    Serial.printf("[ble:evt] %s\n", json.c_str());
}

// Splits `payload` (raw text, typically JSON) into framed chunks of the form
//   "<seq>/<total>|<frag>"
// and sends each as a single notify on the data characteristic. Display
// reassembles by parsing the header.
static void sendData(const String& payload) {
    if (!chrData || !bleConnected) return;
    const int frag = SNAPCHEF_DATA_FRAG_MAX;
    int total = (payload.length() + frag - 1) / frag;
    if (total == 0) total = 1;
    Serial.printf("[ble:data] sending %d bytes in %d frames\n",
                  (int)payload.length(), total);
    for (int i = 0; i < total; i++) {
        String chunk;
        chunk.reserve(frag + 16);
        chunk += String(i + 1);
        chunk += '/';
        chunk += String(total);
        chunk += '|';
        chunk += payload.substring(i * frag,
                                    min((int)payload.length(), (i + 1) * frag));
        chrData->setValue((uint8_t*)chunk.c_str(), chunk.length());
        chrData->notify();
        delay(20);   // give stack room to push between fragments
    }
}

// ============================================================================
//                          BLE: server callbacks
// ============================================================================

// NimBLE-Arduino v2.x callback signatures (v1 had no NimBLEConnInfo arg).
class ServerCb : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* /*srv*/, NimBLEConnInfo& /*info*/) override {
        bleConnected = true;
        Serial.println("[ble] client connected");
        // Suspend advertising while connected (single-peer design).
    }
    void onDisconnect(NimBLEServer* /*srv*/, NimBLEConnInfo& /*info*/, int /*reason*/) override {
        bleConnected = false;
        Serial.println("[ble] client disconnected, restarting adv");
        gCancelRequested = true;       // abort any in-flight scan
        NimBLEDevice::startAdvertising();
    }
};

class CmdCb : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& /*info*/) override {
        std::string v = c->getValue();
        String s(v.c_str());
        Serial.printf("[ble:cmd] %s\n", s.c_str());
        // Cancel is handled inline; everything else queued for the main loop.
        if (s.indexOf("\"cancel\"") >= 0) {
            gCancelRequested = true;
            return;
        }
        gPendingCmd = s;
        gCmdPending = true;
    }
};

static void initBle() {
    NimBLEDevice::init(SNAPCHEF_BLE_NAME);
    NimBLEDevice::setMTU(247);
    NimBLEDevice::setPower(9);   // +9 dBm; v2 takes raw int8_t

    bleServer = NimBLEDevice::createServer();
    bleServer->setCallbacks(new ServerCb());

    NimBLEService* svc = bleServer->createService(SNAPCHEF_SVC_UUID);

    chrCmd = svc->createCharacteristic(SNAPCHEF_CHR_CMD_UUID,
                                       NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
    chrCmd->setCallbacks(new CmdCb());

    chrEvt = svc->createCharacteristic(SNAPCHEF_CHR_EVT_UUID,
                                       NIMBLE_PROPERTY::NOTIFY | NIMBLE_PROPERTY::READ);
    chrData = svc->createCharacteristic(SNAPCHEF_CHR_DATA_UUID,
                                        NIMBLE_PROPERTY::NOTIFY | NIMBLE_PROPERTY::READ);

    svc->start();

    NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(SNAPCHEF_SVC_UUID);
    adv->setName(SNAPCHEF_BLE_NAME);
    adv->enableScanResponse(true);   // v2 rename of setScanResponse
    adv->start();
    Serial.println("[ble] advertising");
}

// ============================================================================
//                            COMMAND HANDLERS
// ============================================================================

// Tiny purpose-built JSON helpers to avoid pulling in ArduinoJson.
// They assume the message comes from our own display firmware (well-formed).
static String extractStringField(const String& json, const char* key) {
    String pat = String("\"") + key + "\":\"";
    int i = json.indexOf(pat);
    if (i < 0) return "";
    i += pat.length();
    int j = json.indexOf('"', i);
    if (j < 0) return "";
    return json.substring(i, j);
}

// Returns "true" or "false" as a string (ready to splice into an outgoing
// JSON payload). Defaults to "false" when the key is missing.
static String extractBoolField(const String& json, const char* key) {
    String pat = String("\"") + key + "\":";
    int i = json.indexOf(pat);
    if (i < 0) return "false";
    i += pat.length();
    while (i < (int)json.length() && (json[i] == ' ' || json[i] == '\t')) i++;
    if (json.substring(i, i + 4) == "true") return "true";
    return "false";
}

// Walks "items":[...] in a receipt response and streams the first `maxItems`
// out as one single-frame BLE event each:
//   {"evt":"receipt_item","idx":k,"total":N,"name":"…","needs_refrigeration":bool}
// Each event is well under the 244-byte single-notify ceiling, so this path
// avoids the chunked-data reassembler entirely. Returns the number of items
// emitted.
static int sendReceiptTestStream(const String& resp, int maxItems) {
    int items_pos = resp.indexOf("\"items\":[");
    if (items_pos < 0) return 0;

    // Pass 1: extract item ranges so we know `total` before emitting any event.
    std::vector<int> starts, ends;
    int p = items_pos + 9;
    int depth = 0;
    int item_start = -1;
    while (p < (int)resp.length() && (int)starts.size() < maxItems) {
        char c = resp[p];
        if (c == '{') {
            if (depth == 0) item_start = p;
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && item_start >= 0) {
                starts.push_back(item_start);
                ends.push_back(p + 1);
                item_start = -1;
            }
        } else if (c == ']' && depth == 0) {
            break;
        }
        p++;
    }

    int total = (int)starts.size();

    // Always emit a "begin" frame first so the display can switch to the
    // receipt-result screen and clear stale rows even when total == 0
    // (empty items array — Textract OK but no products parsed).
    {
        String j = "{\"evt\":\"receipt_test_begin\",\"total\":";
        j += String(total);
        j += "}";
        sendEvent(j);
        delay(80);
    }

    for (int i = 0; i < total; i++) {
        String item   = resp.substring(starts[i], ends[i]);
        String name   = extractStringField(item, "name");
        String refrig = extractBoolField(item, "needs_refrigeration");
        String safeName;
        safeName.reserve(name.length() + 4);
        for (size_t k = 0; k < name.length(); k++) {
            char ch = name[k];
            if (ch == '"' || ch == '\\') safeName += '\\';
            safeName += ch;
        }
        String j = "{\"evt\":\"receipt_item\",\"idx\":";
        j += String(i);
        j += ",\"total\":";
        j += String(total);
        j += ",\"name\":\"";
        j += safeName;
        j += "\",\"needs_refrigeration\":";
        j += refrig;
        j += "}";
        sendEvent(j);
        delay(80);   // spacing so consecutive notifies don't overflow TX queue
    }
    return total;
}

static String extractArrayField(const String& json, const char* key) {
    String pat = String("\"") + key + "\":[";
    int i = json.indexOf(pat);
    if (i < 0) return "[]";
    i += pat.length() - 1;            // point at '['
    int depth = 0;
    for (int j = i; j < (int)json.length(); j++) {
        char c = json[j];
        if (c == '[') depth++;
        else if (c == ']') {
            depth--;
            if (depth == 0) return json.substring(i, j + 1);
        }
    }
    return "[]";
}

static void runVeggieScan() {
    gState = STATE_VEGGIE_SCAN;
    gCancelRequested = false;
    resetSmoothing();
    sendEvent(String("{\"evt\":\"veggie_scanning\",\"purpose\":\"") +
              gPendingPurpose + "\"}");

    unsigned long t0 = millis();
    unsigned long lastInfer = 0;
    int   foundIdx  = -1;
    float foundConf = 0.0f;

    while (millis() - t0 < VEGGIE_TIMEOUT_MS) {
        if (gCancelRequested) {
            Serial.println("[veggie] cancelled");
            sendEvent("{\"evt\":\"veggie_cancelled\"}");
            gState = STATE_IDLE;
            return;
        }
        if (millis() - lastInfer < INFERENCE_INTERVAL_MS) { delay(10); continue; }
        lastInfer = millis();

        int idx; float conf; bool locked;
        if (!classifyOnce(idx, conf, locked)) continue;
        Serial.printf("[veggie] %s%s %.2f\n",
                      locked ? "LOCK " : "...  ", LABELS[idx], conf);
        if (locked) { foundIdx = idx; foundConf = conf; break; }
    }

    if (foundIdx >= 0) {
        String j = "{";
        j += "\"evt\":\"veggie_result\",";
        j += "\"label\":\""; j += LABELS[foundIdx]; j += "\",";
        j += "\"confidence\":"; j += String(foundConf, 3); j += ",";
        j += "\"purpose\":\""; j += gPendingPurpose; j += "\"";
        j += "}";
        sendEvent(j);
    } else {
        sendEvent(String("{\"evt\":\"veggie_unknown\",\"purpose\":\"") +
                  gPendingPurpose + "\"}");
    }
    gState = STATE_IDLE;
}

// One-shot receipt flow: capture UXGA → upload to /receipts/analyze →
// stream the OCR JSON result back on the data characteristic.
static void runReceiptCapture() {
    gState = STATE_RECEIPT_CAPTURE;
    sendEvent("{\"evt\":\"receipt_capturing\"}");

    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        sendEvent("{\"evt\":\"receipt_error\",\"code\":\"sensor\",\"msg\":\"no sensor\"}");
        gState = STATE_IDLE; return;
    }

    // Upshift to UXGA for max OCR fidelity. DMA buffer was sized for UXGA at
    // init, so this shift is safe (no FB-OVF).
    s->set_framesize(s, FRAMESIZE_UXGA);
    s->set_quality(s, 12);
    drainCameraFrames(3);

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        sendEvent("{\"evt\":\"receipt_error\",\"code\":\"capture\",\"msg\":\"capture failed\"}");
        s->set_framesize(s, FRAMESIZE_QVGA);
        s->set_quality(s, 10);
        drainCameraFrames(2);
        gState = STATE_IDLE; return;
    }
    Serial.printf("[receipt] captured %ux%u %u bytes (UXGA)\n",
                  (unsigned)fb->width, (unsigned)fb->height, (unsigned)fb->len);

    size_t   jlen = fb->len;
    uint8_t* jbuf = (uint8_t*)heap_caps_malloc(jlen, MALLOC_CAP_SPIRAM);
    bool copied = false;
    if (jbuf) { memcpy(jbuf, fb->buf, jlen); copied = true; }
    esp_camera_fb_return(fb);

    // Restore sensor for veggie streaming before doing the slow upload.
    s->set_framesize(s, FRAMESIZE_QVGA);
    s->set_quality(s, 10);
    drainCameraFrames(2);

    if (!copied) {
        if (jbuf) heap_caps_free(jbuf);
        sendEvent("{\"evt\":\"receipt_error\",\"code\":\"oom\",\"msg\":\"jpeg copy failed\"}");
        gState = STATE_IDLE; return;
    }

    gState = STATE_RECEIPT_UPLOAD;
    sendEvent("{\"evt\":\"receipt_uploading\"}");

    String errCode, errMsg;
    String resp = uploadReceipt(jbuf, jlen, errCode, errMsg);
    heap_caps_free(jbuf);

    if (resp.length() == 0) {
        String j = "{\"evt\":\"receipt_error\",\"code\":\"";
        j += errCode; j += "\",\"msg\":\"";
        for (size_t i = 0; i < errMsg.length() && i < 200; i++) {
            char c = errMsg[i];
            if (c == '"' || c == '\\' || c == '\n' || c == '\r') c = ' ';
            j += c;
        }
        j += "\"}";
        sendEvent(j);
        gState = STATE_IDLE; return;
    }

    // Heavy WiFi/TLS just finished; give BLE coex a generous moment to drain
    // its queue before pushing notifications. Without this, the first
    // notifications after upload are silently dropped on the radio.
    delay(500);

    // DEBUG: stream the first 10 items as individual single-frame events.
    // Small payloads, no chunked reassembler — easiest path to verify items
    // are arriving on the display.
    int n = sendReceiptTestStream(resp, 10);
    Serial.printf("[receipt_test] streamed %d items\n", n);
    gState = STATE_IDLE;
}

static void runRecipe() {
    gState = STATE_RECIPE;
    String recipe = buildRecipeMock(gPendingIngredientsJson);
    sendEvent("{\"evt\":\"recipe_result\"}");
    sendData(recipe);
    gState = STATE_IDLE;
}

static void handleCmd(const String& cmd) {
    String name = extractStringField(cmd, "cmd");
    if (gState != STATE_IDLE && name != "cancel") {
        Serial.printf("[cmd] busy, dropping %s\n", name.c_str());
        sendEvent("{\"evt\":\"error\",\"code\":\"busy\",\"msg\":\"already running\"}");
        return;
    }

    if (name == "start_veggie_scan") {
        gPendingPurpose = extractStringField(cmd, "purpose");
        if (gPendingPurpose.length() == 0) gPendingPurpose = "in";
        runVeggieScan();
    } else if (name == "capture_receipt") {
        runReceiptCapture();
    } else if (name == "get_recipe") {
        // Legacy mock path; not used by the new UI flow.
        gPendingIngredientsJson = extractArrayField(cmd, "ingredients");
        runRecipe();
    } else if (name == "get_recipe_list") {
        gPendingTrigger         = extractStringField(cmd, "trigger");
        gPendingIngredientsJson = extractArrayField(cmd, "ingredients");
        runRecipeList();
    } else if (name == "get_recipe_steps") {
        gPendingDish            = extractStringField(cmd, "dish");
        gPendingIngredientsJson = extractArrayField(cmd, "ingredients");
        runRecipeSteps();
    } else {
        Serial.printf("[cmd] unknown: %s\n", name.c_str());
    }
}

// ============================================================================
//                              setup() / loop()
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(400);
    Serial.println("\n=== SnapChef Main ===");

    if (!psramFound()) {
        Serial.println("FATAL: PSRAM required (enable 'OPI PSRAM').");
        while (true) delay(1000);
    }

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIG_PIN, LOW);

    if (!initCamera()) { Serial.println("FATAL: camera init failed"); while (true) delay(1000); }
    Serial.println("camera OK");

    if (!initModel())  { Serial.println("FATAL: model init failed");  while (true) delay(1000); }
    Serial.printf("model OK, unknown_class_idx=%d\n", unknown_class_idx);

    connectWiFi();
    if (WiFi.status() == WL_CONNECTED) healthzSelfCheck();

    initBle();
    Serial.println("ready");
}

void loop() {
    // 1. Pull pending command (filled by BLE write callback).
    if (gCmdPending) {
        String c = gPendingCmd;
        gCmdPending = false;
        handleCmd(c);
    }

    // 2. HC-SR04 proximity wake (only when idle and a peer is connected).
    static unsigned long lastProx = 0;
    static unsigned long lastWake = 0;
    if (millis() - lastProx >= PROXIMITY_INTERVAL_MS) {
        lastProx = millis();
        if (gState == STATE_IDLE && bleConnected) {
            float d = readDistanceCm();
            if (d > 0 && d < DETECT_CM &&
                millis() - lastWake > PROXIMITY_COOLDOWN_MS) {
                lastWake = millis();
                Serial.printf("[proximity] %.1f cm — wake\n", d);
                sendEvent("{\"evt\":\"proximity_wake\"}");
            }
        }
    }

    delay(10);
}

/*
 * Read classification results from Grove Vision AI V2 and print them to Serial.
 *
 * The Grove module runs the Vela-compiled MobileNetV2 on its Ethos-U55 NPU;
 * the ESP32S3 is just an I2C host that polls inference results and formats
 * them for the serial monitor.
 *
 * Hardware (XIAO ESP32S3 Sense):
 *   - Grove Vision AI V2  <->  XIAO Grove I2C port
 *     SDA = GPIO5, SCL = GPIO6  (default Wire pins on XIAO)
 *     Grove module powered from 3V3, shares GND with XIAO
 *   - Grove module default I2C address: 0x62
 *
 * Before flashing:
 *   1. Upload the Vela model (veggie_mnv2_sensecraft_int8_vela.tflite) to the
 *      Grove Vision AI V2 via SenseCraft AI Web Toolkit, and confirm inference
 *      previews OK there. This sketch does NOT push the model — it only reads
 *      results from whatever model is already running on the module.
 *   2. Install library: "Seeed Arduino SSCMA" (Library Manager).
 *
 * Arduino IDE:
 *   - Board: "XIAO_ESP32S3"
 *   - USB CDC On Boot: Enabled  (so Serial prints over USB)
 */

#include <Wire.h>
#include <Seeed_Arduino_SSCMA.h>

#include "labels.h"

// XIAO ESP32S3 Grove I2C pins
#define I2C_SDA 5
#define I2C_SCL 6

// Print cadence: Grove Vision AI V2 runs ~10-30 fps internally, we poll slower
// so the serial log stays readable. 300ms leaves the module enough slack to
// finish the previous frame before we ask for the next one.
static const uint32_t POLL_INTERVAL_MS = 3000;

// Transient invoke failures (module busy, frame dropped, I2C NACK) are normal.
// We only complain after this many consecutive failures so the log stays clean.
static const uint8_t FAIL_WARN_THRESHOLD = 1;

SSCMA AI;
static uint32_t last_poll_ms = 0;
static uint8_t consecutive_fails = 0;


static const char* labelFor(int class_id) {
  if (class_id >= 0 && class_id < NUM_CLASSES) return LABELS[class_id];
  return "?";
}

static void printHeader() {
  Serial.println();
  Serial.println(F("=== Grove Vision AI V2 -> ESP32S3 Serial Reader ==="));
  Serial.print(F("Model ID: "));   Serial.println(AI.ID());
  Serial.print(F("Name: "));       Serial.println(AI.name());
  Serial.println(F("---------------------------------------------------"));
  Serial.println(F("Polling inference results. Ctrl+C to stop.\n"));
}


void setup() {
  Serial.begin(115200);
  // Give USB CDC a moment to enumerate so the header isn't lost.
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 2000) { delay(10); }

  Wire.begin(I2C_SDA, I2C_SCL);

  // SSCMA::begin() negotiates transport with the module. It retries internally
  // but we loop here so the sketch waits for the Grove board to come up.
  while (!AI.begin(&Wire)) {
    Serial.println(F("[!] Grove Vision AI V2 not responding on I2C, retrying..."));
    delay(1000);
  }

  printHeader();
}


void loop() {
  if (millis() - last_poll_ms < POLL_INTERVAL_MS) return;
  last_poll_ms = millis();

  // invoke(times, filter, show)
  //   times  = 1   : run one inference
  //   filter = false: return all classes, not just top-k
  //   show   = false: do NOT stream the jpeg preview (saves I2C bandwidth)
  if (!AI.invoke(1, false, false)) {
    consecutive_fails = 0;
    // Classification head -> AI.classes() populated. Object-detection models
    // populate AI.boxes() instead; we support both so you can reuse this
    // sketch if you later flash a detection model to the Grove module.
    if (AI.classes().size() > 0) {
      // classes() returns entries sorted by score descending. Print top-1 plus
      // the runners-up so you can see what the model is confusing it with.
      const auto& top = AI.classes()[0];
      const char* name = labelFor(top.target);
      float conf = top.score / 100.0f;  // SSCMA reports 0-100

      Serial.print(F("["));
      Serial.print(millis() / 1000.0f, 2);
      Serial.print(F("s]  "));

      if (conf >= CONFIDENCE_THRESHOLD) {
        Serial.print(name);
      } else {
        Serial.print(UNKNOWN_LABEL);
        Serial.print(F(" (top="));
        Serial.print(name);
        Serial.print(F(")"));
      }
      Serial.print(F("  conf="));
      Serial.print(conf, 2);

      // Show up to 2 runners-up for context
      size_t n = AI.classes().size();
      if (n > 1) {
        Serial.print(F("   | "));
        for (size_t i = 1; i < n && i < 3; ++i) {
          const auto& c = AI.classes()[i];
          Serial.print(labelFor(c.target));
          Serial.print(F(":"));
          Serial.print(c.score / 100.0f, 2);
          if (i + 1 < n && i + 1 < 3) Serial.print(F(", "));
        }
      }
      Serial.println();
      return;
    }

    if (AI.boxes().size() > 0) {
      Serial.print(F("[boxes] "));
      for (size_t i = 0; i < AI.boxes().size(); ++i) {
        const auto& b = AI.boxes()[i];
        Serial.print(labelFor(b.target));
        Serial.print(F("("));
        Serial.print(b.score / 100.0f, 2);
        Serial.print(F(") @["));
        Serial.print(b.x);  Serial.print(F(","));
        Serial.print(b.y);  Serial.print(F(" "));
        Serial.print(b.w);  Serial.print(F("x"));
        Serial.print(b.h);  Serial.print(F("]"));
        if (i + 1 < AI.boxes().size()) Serial.print(F("  "));
      }
      Serial.println();
      return;
    }

    Serial.println(F("[.] no objects"));
  } else {
    // Transient failure: module busy, frame dropped, or I2C NACK. Only warn
    // after several in a row so we don't spam the log on every missed frame.
    if (++consecutive_fails == FAIL_WARN_THRESHOLD) {
      Serial.print(F("[!] "));
      Serial.print(FAIL_WARN_THRESHOLD);
      Serial.println(F(" invokes failed in a row — close any SenseCraft browser"));
      Serial.println(F("    tab, check I2C wiring, or re-flash the model."));
      consecutive_fails = 0;  // reset so we warn again if it stays broken
    }
  }
}

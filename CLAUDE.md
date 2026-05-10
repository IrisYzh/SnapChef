# SnapChef — guide for Claude

A smart-fridge sticker built from **two ESP32-S3 boards** that talk over BLE.

## What ships

| Folder | Purpose | Status |
|---|---|---|
| `src/main/` | Main controller firmware — XIAO ESP32S3 Sense (camera + HC-SR04 + WiFi). NimBLE GATT server. | **Production target** |
| `src/display/` | Display firmware — Waveshare ESP32-S3-Touch-LCD-4.3 (LVGL UI). NimBLE GATT client. | **Production target** |
| `veggie_classification/` | Reference: TFLM vegetable classifier (V0–V6) + training notebooks. `deploy/esp32_deploy_V6/` is the version actually used by `src/main`. | Reference only |
| `receipt_read/` | Reference: standalone receipt-OCR sketch. Upload logic ported into `src/main`. | Reference only |
| `UI/SnapChef_UI/` | Reference: standalone LVGL mock UI. Forked into `src/display`. | Reference only |
| `UI/ESP32-S3-Touch-LCD-4.3-Demo/` | Vendor demos for the Waveshare board. | Reference only |
| `fridgelens_button_oled/` | Older single-board prototype on a tiny OLED (HC-SR04 + encoder). | Historical |
| `API.md` | Backend contract for `/receipts/analyze`. **Read this before touching receipt code.** | Spec |

When making changes, edit `src/main/` and `src/display/`. The other folders
exist so you can re-derive a subsystem if needed; they are not built.

## How the two devices interact

```
display (UI, IN_FRIDGE in NVS) ──BLE──▶ main (sensors, camera, WiFi, recipes)
        ◀──BLE notifies (evt, data)──
```

- Main = GATT server, name `SnapChef-Main`. Service + characteristic UUIDs in [src/main/snapchef_ble.h](src/main/snapchef_ble.h) (mirror in [src/display/snapchef_ble.h](src/display/snapchef_ble.h) — keep them in sync).
- Three characteristics: `cmd` (display→main, write), `evt` (main→display, notify, small JSON), `data` (main→display, notify, chunked `<seq>/<total>|…` framing for receipt + recipe payloads).
- MTU is negotiated to 247. `data` fragments cap at `SNAPCHEF_DATA_FRAG_MAX = 180` bytes.
- `IN_FRIDGE` lives **only on the display**, persisted via `Preferences` under `snapchef/inv` as a `|`-delimited string. Main is stateless about inventory.

### Commands display → main

```
{"cmd":"start_veggie_scan", "purpose":"in"|"out"}
{"cmd":"start_receipt_scan"}
{"cmd":"cancel"}
{"cmd":"get_recipe", "ingredients":["Carrot","Tomato",...]}
```

### Events main → display

`ready` · `proximity_wake` · `veggie_scanning` · `veggie_result` · `veggie_unknown` · `veggie_cancelled` · `receipt_capturing` · `receipt_uploading` · `receipt_result` (body on `data`) · `receipt_error` · `recipe_result` (body on `data`) · `error`.

## User flows

1. **Put in** — display: *Put In* → *Scan Veggie* (added on confirm) **or** *Scan Receipt* (checklist; checked rows merged into `IN_FRIDGE`; rows pre-checked when `needs_refrigeration` is true).
2. **Take out** — display: *Take Out* → veggie scan; on success the matched item is removed and the user is offered a recipe (`get_recipe` is sent with `[removed_veggie, ...IN_FRIDGE]`).
3. **My Fridge** — browse / hold-3s-to-remove items.
4. **Idle wake** — main's HC-SR04 emits `proximity_wake` at <10 cm, 3 s cooldown; display jumps from idle → action-select.

## Build matrix

### `src/main/` — XIAO ESP32S3 Sense

- Board: `XIAO_ESP32S3` · PSRAM: **OPI PSRAM** · Partition: **Huge APP (3 MB No OTA / 1 MB SPIFFS)** · USB CDC On Boot: **Enabled**
  - The default 4 MB partition (1.25 MB app) is too small — the binary lands around 2 MB once TFLM model + WiFi/SSL + NimBLE all link in. Huge App fits with ~30 % headroom. Re-confirm this in *Tools → Partition Scheme* every time the board is changed; Arduino IDE 2.x sometimes resets it.
- Libraries: `TensorFlowLite_ESP32`, `NimBLE-Arduino` (~v1.4.x signatures used), built-in `WiFi` / `WiFiClientSecure` / `HTTPClient`
- Hardware: HC-SR04 → `TRIG = GPIO2`, `ECHO = GPIO1` (10 kΩ + 20 kΩ divider on echo). Camera on the FFC.
- Bundles `model_data.h` + `labels.h` copied verbatim from `veggie_classification/deploy/esp32_deploy_V6/` — the 10 trained classes are **Bellpepper, Broccoli, Cabbage, Carrot, Eggplant, Garlic, Onion, Potato, Tomato, Unknown**.
- WiFi credentials and the receipt API key are hardcoded near the top of [src/main/main.ino](src/main/main.ino) (UW MPSK; rotate before publishing).

### `src/display/` — Waveshare ESP32-S3-Touch-LCD-4.3

- Board: Waveshare ESP32-S3 4.3" board · PSRAM: **OPI PSRAM** · Partition: **Huge App + 4 MB**
- Libraries: `ESP32_Display_Panel`, `lvgl 8.4.0`, `NimBLE-Arduino`, `ArduinoJson`
- The `esp_panel_*.h`, `lv_conf.h`, and `lvgl_v8_port.*` files are verbatim copies from `UI/SnapChef_UI/` and only need re-syncing if the upstream demo updates.

## Where things live in `src/main/main.ino`

Single sketch, ordered top-to-bottom:

1. Config (WiFi, API, HC-SR04 thresholds, classifier tunables, camera pins).
2. Camera init + `drainCameraFrames` helper for QVGA↔UXGA framesize swaps (do **not** double-init the camera).
3. TFLM bring-up + `captureAndPreprocess` + `classifyOnce` + smoothing (lifted from V6).
4. WiFi + `uploadReceipt` (multipart POST, returns raw response body).
5. `buildRecipeMock` — Carrot / Eggplant / generic three-step recipes. Replace once the backend lands.
6. HC-SR04 read.
7. NimBLE server (`ServerCb`, `CmdCb`) + `sendEvent` / `sendData` (chunked).
8. Command handlers: `runVeggieScan`, `runReceiptScan`, `runRecipe`. State machine in `gState`; `gCancelRequested` aborts the veggie loop.
9. `setup()` / `loop()` — loop drains queued commands, then ticks the proximity sensor.

## Where things live in `src/display/SnapChef_UI.ino`

Top-to-bottom: colour palette → inventory NVS helpers → BLE client (scan, reconnect, chunked-data reassembly) → UI globals → screen builders (`buildIdle` … `buildMenu`) → BLE event/data handlers (which acquire the LVGL lock before touching widgets) → `setup` / `loop`.

Anything called from a NimBLE callback must `lvgl_port_lock(-1)` before touching widgets and `lvgl_port_unlock()` after — this is enforced inside `onEvent` / `onData`.

## Pitfalls worth remembering

- **Init the camera at the largest framesize you'll ever use, then `set_framesize` down at runtime.** `esp_camera_init` sizes the DMA / JPEG buffer once based on `config.frame_size`; `set_framesize` only reprograms the sensor, not the buffer. Upshifting from a smaller init size (e.g. QVGA → UXGA) overflows the DMA buffer and you get `cam_hal: FB-OVF` followed by `esp_camera_fb_get()` returning NULL. Current pattern: init at UXGA + quality 12, immediately `set_framesize(QVGA)` + `set_quality(10)` for veggie streaming, upshift back to UXGA inside `runReceiptScan`. Always `drainCameraFrames(2-3)` after a sensor change so the next grab is at the new settings; never `esp_camera_deinit` + reinit.
- **Don't add a UXGA RGB888 buffer.** The TFLM path decodes to a 320×240 buffer; receipt path forwards JPEG bytes only. Keeping that invariant lets the static `rgb888_buf` stay small.
- **Recipe ingredient list ordering matters** for the mock: `buildRecipeMock` matches `Carrot` first, then `Eggplant`, else generic. If you add another mock, add it in `src/main/main.ino` and update this section.
- **API key + WiFi password** are hardcoded — never commit a real key. Scrub before pushing.
- **NimBLE-Arduino major versions changed callback signatures** (v1.x → v2.x adds `NimBLEConnInfo` args). The current code targets ~v1.4.x; if a fresh install pulls v2.x, the `onConnect` / `onDisconnect` / `onWrite` / `onResult` overrides need extra parameters.

## Backend

- `/healthz` and `/receipts/analyze` are documented in [API.md](API.md). Typical end-to-end latency 4–5 s; client timeout 20 s.
- Recipe endpoint **does not exist yet** — main returns mocks. When it lands, replace `buildRecipeMock` and add an HTTP path mirroring `uploadReceipt`.

## Planning artefact

The original cross-device design lives in `~/.claude/plans/esp32s3-esp32s3-sense-hc-sr04-1-hc-sr04-partitioned-twilight.md` (BLE protocol decisions, screen list, verification steps). Read it if you're rearchitecting; otherwise the inline comments in the two sketches are usually enough.

# SnapChef (FridgeLens)

A smart fridge assistant that tracks ingredients using proximity sensing, on-device ML classification (Grove Vision AI V2), and cloud OCR for receipt scanning.

---

## Project Structure

```
SnapChef/
├── fridgelens_button_oled/
│   └── fridgelens_button_oled.ino   # Main prototype firmware
├── veggie_classification/
│   ├── deploy/
│   │   ├── esp32_deploy_V0/         # Custom Tiny CNN (TFLite Micro, 23 KB)
│   │   ├── esp32_deploy_V1/         # YOLOv8n-cls (TFLite Micro, 1.5 MB)
│   │   ├── esp32_deploy_V2/         # Grove Vision AI V2 via SSCMA  ← current
│   │   └── esp32_deploy_V3/         # Edge Impulse library
│   └── train/
│       ├── download_dataset.ipynb   # Kaggle dataset download
│       ├── classification_V0/       # Custom CNN training + TFLite export
│       └── classification_V1/       # YOLOv8 training + TFLite export
└── README.md
```

---

## Part 1 — Main Prototype Firmware

### Hardware

| Component | Part |
|-----------|------|
| Microcontroller | Seeed XIAO ESP32S3 |
| Display | 0.96" OLED SSD1306 (I2C, 128×64) |
| Proximity sensor | HC-SR04 Ultrasonic |
| Input | Rotary encoder (with push button) + momentary push button |

### Wiring

```
HC-SR04:   VCC → 5V,   GND → GND
           TRIG → GPIO2
           ECHO → GPIO1  (voltage divider: 10 kΩ + 20 kΩ to GND)

OLED:      VCC → 3.3V, GND → GND
           SDA → GPIO5,  SCL → GPIO6

Mode btn:  one leg → GPIO9, other leg → GND

Encoder:   CLK → GPIO3,  DT → GPIO4,  SW → GPIO7
           VCC → 3.3V,   GND → GND
```

### Arduino IDE Setup

1. Install Arduino IDE ≥ 2.0. Add the ESP32 board package:
   - Preferences → Additional boards manager URLs:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Boards Manager → install **esp32 by Espressif** (v3.x)
2. Install libraries via Library Manager:
   - **Adafruit SSD1306** (≥ 2.5.0)
   - **Adafruit GFX Library** (≥ 1.11.0)
3. Board settings:
   - Board: **XIAO_ESP32S3**
   - USB CDC On Boot: **Enabled**
   - PSRAM: **OPI PSRAM**

### Flash & Run

1. Open `fridgelens_button_oled/fridgelens_button_oled.ino`
2. Connect XIAO via USB-C → click **Upload**
3. Open **Serial Monitor** at **115200 baud**

### Sensor Data Stream

The firmware continuously outputs HC-SR04 readings every 500 ms:

```
timestamp:1234,distance_cm:23.5,state:0
timestamp:1734,distance_cm:8.2,state:0
```

| Field | Description |
|-------|-------------|
| `timestamp` | Milliseconds since boot |
| `distance_cm` | Distance in cm; 999.0 = no echo |
| `state` | 0=IDLE, 1=MODE_SELECT, 2=SCANNING, 3=RESULT, 4=INGREDIENT_MENU |

Open **Tools → Serial Plotter** to visualise `distance_cm` live.

### Usage

| Interaction | Action |
|-------------|--------|
| Hold item within 10 cm of sensor | Wake device, enter mode selection |
| Press Mode Button (any screen) | Jump to mode selection |
| Rotate encoder | Scroll / change selection |
| Press encoder knob | Confirm |
| Hold encoder knob 3 s | Delete selected ingredient |
| 10 s idle on mode select | Auto-return to standby |

Modes: **Ingredient Scan** · **Receipt Scan** · **Edit Ingredients**

---

## Part 2 — Vegetable Classification (ML)

All four deploy versions target **Seeed XIAO ESP32S3**. Pre-trained model files (`model_data.h`, `labels.h`) are already included — no training required to run.

**10 classes:** Bell pepper, Broccoli, Cabbage, Carrot, Cucumber, Eggplant, Garlic, Onion, Potato, Tomato

---

### Deploy V2 — Grove Vision AI V2 (Current)

**Hardware:** XIAO ESP32S3 + Grove Vision AI V2 module

**Wiring:** Plug Grove Vision AI V2 directly onto the XIAO ESP32S3 rear header pins; connect USB-C to the XIAO port.

**Flash the ML model onto the Vision AI V2 (one-time setup):**
1. Open [SenseCraft AI](https://sensecraft.seeed.cc/ai) in a browser (Chrome recommended)
2. Connect the Grove Vision AI V2 via its own USB-C port
3. Upload `veggie_classification/train/sensecraft/Grove_20260418.tflite`
4. Disconnect from SenseCraft; reconnect XIAO USB-C for Arduino

**Arduino IDE setup:**
1. Board package: esp32 by Espressif v3.x (same URL as Part 1)
2. Board: **XIAO_ESP32S3**, USB CDC On Boot: **Enabled**, PSRAM: **OPI PSRAM**
3. Library: install **Seeed_Arduino_SSCMA** from Seeed GitHub or Library Manager

**Flash & Run:**
1. Open `veggie_classification/deploy/esp32_deploy_V2/esp32_deploy_V2.ino`
2. Upload; open Serial Monitor at **115200 baud**
3. Point the Vision AI V2 camera at a vegetable; results appear at 500 ms intervals:
   ```
   [Perf] pre=2ms, infer=45ms, post=1ms
   [Result] Tomato (95%)
   ```

---

### Deploy V0 — Custom Tiny CNN (TFLite Micro, 23 KB)

**Hardware:** XIAO ESP32S3 Sense (with OV2640 camera FFC expansion)

**Arduino IDE setup:**
1. Board: **XIAO_ESP32S3**, PSRAM: **OPI PSRAM**
2. Library: install **TensorFlowLite_ESP32** from Library Manager
3. Open `esp32_deploy_V0/esp32_deploy_V0.ino`; update `WIFI_SSID` / `WIFI_PASS` near the top
4. Upload; navigate to the device IP in a browser for the live camera preview

**Expected Serial output:**
```
Detected: Carrot (conf: 0.82)
```

---

### Deploy V1 — YOLOv8n-cls (TFLite Micro, 1.5 MB)

**Hardware:** XIAO ESP32S3 Sense (with OV3660 camera FFC expansion)

**Arduino IDE setup:**
1. Board: **XIAO_ESP32S3**
2. PSRAM: **OPI PSRAM** ← required (1 MB tensor arena lives in PSRAM)
3. Partition Scheme: **Huge APP (3MB No OTA / 1MB SPIFFS)** ← required (model ~1.5 MB)
4. Library: **TensorFlowLite_ESP32** from Library Manager
5. Open `esp32_deploy_V1/esp32_deploy_V1.ino`; update `WIFI_SSID` / `WIFI_PASS`
6. Upload; navigate to device IP for live preview

---

### Deploy V3 — Edge Impulse

**Hardware:** XIAO ESP32S3 Sense (OV2640 / OV3660 / OV5640)

**Arduino IDE setup:**
1. Board: **XIAO_ESP32S3**, PSRAM: **OPI PSRAM**, USB CDC On Boot: **Enabled**
2. Partition Scheme: **Huge APP (3MB No OTA / 1MB SPIFFS)**
3. Install the Edge Impulse library:
   - Sketch → Include Library → **Add .ZIP Library…**
   - Select `veggie_classification/train/ei-veggie-arduino-1.0.5-impulse-#1.zip`
4. Open `esp32_deploy_V3/esp32_deploy_V3.ino`; update `WIFI_SSID` / `WIFI_PASS`
5. Upload; open Serial Monitor at **115200 baud** and navigate to device IP for preview

---

## Part 3 — Model Training (optional)

Pre-trained models are already included. Follow these steps only if you want to retrain from scratch.

### Python Environment

```bash
pip install tensorflow==2.18.0 numpy matplotlib seaborn scikit-learn kaggle
pip install ultralytics   # for V1 (YOLOv8)
```

Python 3.10–3.12 recommended.

### Step 1 — Download Dataset

1. Open `veggie_classification/train/download_dataset.ipynb` in Jupyter
2. Set your Kaggle credentials in Cell 2 (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
   - Get your key from kaggle.com → Account → Create API Token
3. Run all cells — downloads ~1.8 GB to `veggie_classification/train/data/`

### Step 2 — Train Custom CNN (V0)

1. Open `veggie_classification/train/classification_V0/food_classification.ipynb`
2. Verify `BASE_DIR` points to `../data/huggingface` (default is correct)
3. Run all cells in order:
   - Cells 1–4: load data, build model
   - Cell 5: train (up to 100 epochs, early stopping at patience=15); ~78% validation accuracy
   - Cell 8: export `veggie_model.tflite` (23 KB INT8)
   - Cell 10: export `model_data.h` + `labels.h` to the same folder
4. Copy `model_data.h` and `labels.h` into `deploy/esp32_deploy_V0/` to deploy

### Step 3 — Train YOLOv8n-cls (V1)

1. Open `veggie_classification/train/classification_V1/yolo_classification.ipynb`
2. Run all cells:
   - Trains YOLOv8n-cls for 50 epochs; ~93% validation accuracy
   - Exports `veggie_yolo.tflite` (1.5 MB INT8), `model_data.h`, `labels.h`
3. Copy headers into `deploy/esp32_deploy_V1/` to deploy

---

## Accuracy Summary

| Version | Architecture | Model size | Validation accuracy |
|---------|-------------|-----------|-------------------|
| V0 | Custom Tiny CNN | 23 KB | 74.3% (INT8) |
| V1 | YOLOv8n-cls | 1.5 MB | 93.3% (INT8) |
| V2 | Grove Vision AI V2 | 285 KB | — (proprietary) |
| V3 | Edge Impulse | — | 85% threshold |

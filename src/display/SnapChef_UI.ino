/*
 * SnapChef — Display device (Waveshare ESP32-S3-Touch-LCD-4.3)
 *
 * LVGL UI driving the user flows; all sensing happens on the main board
 * (XIAO ESP32S3 Sense), reachable over BLE.
 *
 * Flows:
 *   PUT IN  → Scan Veggie  → veggie added to IN_FRIDGE
 *           → Scan Receipt → confirmed items merged into IN_FRIDGE
 *   TAKE OUT → Scan Veggie → veggie removed from IN_FRIDGE
 *                          → optional recipe based on IN_FRIDGE
 *   EDIT  → browse / hold-to-remove items in IN_FRIDGE
 *
 * Build:
 *   - Board: Waveshare ESP32-S3-Touch-LCD-4.3
 *   - Libraries: ESP32_Display_Panel, lvgl 8.4.0, NimBLE-Arduino, ArduinoJson
 *   - Partition: Huge App + 4MB flash
 *   - PSRAM: OPI PSRAM
 */

#define BOARD_WAVESHARE_ESP32_S3_TOUCH_LCD_4_3 1
#define LV_USE_PRIVATE_API 1

#include <Arduino.h>
#include <vector>
#include <esp_display_panel.hpp>
#include <lvgl.h>
#include "lvgl_v8_port.h"

#include <NimBLEDevice.h>
#include <ArduinoJson.h>
#include <Preferences.h>

#include "snapchef_ble.h"

using namespace esp_panel::drivers;
using namespace esp_panel::board;

// ─── Colour palette ──────────────────────────────────────────────────────────
#define COLOR_BG        lv_color_hex(0x0D0D0D)
#define COLOR_CARD      lv_color_hex(0x1A1A1A)
#define COLOR_ACCENT    lv_color_hex(0x00C896)
#define COLOR_ACCENT2   lv_color_hex(0xFF6B35)
#define COLOR_TEXT      lv_color_hex(0xF0F0F0)
#define COLOR_SUBTEXT   lv_color_hex(0x888888)
#define COLOR_DELETE    lv_color_hex(0xFF3B3B)
#define COLOR_OK        lv_color_hex(0x4ECCA3)

// ============================================================================
//                              INVENTORY (NVS)
// ============================================================================

static Preferences gPrefs;
static std::vector<String> gFridge;

static void inventoryLoad() {
    gFridge.clear();
    gPrefs.begin("snapchef", true);
    String s = gPrefs.getString("inv", "");
    gPrefs.end();
    int start = 0;
    while (start < (int)s.length()) {
        int sep = s.indexOf('|', start);
        if (sep < 0) sep = s.length();
        String item = s.substring(start, sep);
        item.trim();
        if (item.length()) gFridge.push_back(item);
        start = sep + 1;
    }
}

static void inventorySave() {
    String s;
    for (size_t i = 0; i < gFridge.size(); i++) {
        if (i) s += '|';
        s += gFridge[i];
    }
    gPrefs.begin("snapchef", false);
    gPrefs.putString("inv", s);
    gPrefs.end();
}

static void inventoryAdd(const String& name) {
    if (name.length() == 0) return;
    gFridge.push_back(name);
    inventorySave();
}

// Removes the first matching entry (case-insensitive). Returns true if found.
static bool inventoryRemove(const String& name) {
    for (size_t i = 0; i < gFridge.size(); i++) {
        if (gFridge[i].equalsIgnoreCase(name)) {
            gFridge.erase(gFridge.begin() + i);
            inventorySave();
            return true;
        }
    }
    return false;
}

static void inventoryRemoveAt(int idx) {
    if (idx < 0 || idx >= (int)gFridge.size()) return;
    gFridge.erase(gFridge.begin() + idx);
    inventorySave();
}

// ============================================================================
//                              BLE CLIENT
// ============================================================================

static NimBLEClient*         bleCli      = nullptr;
static NimBLERemoteCharacteristic* bleCmd  = nullptr;
static NimBLERemoteCharacteristic* bleEvt  = nullptr;
static NimBLERemoteCharacteristic* bleData = nullptr;
static volatile bool         bleConnected = false;
static volatile bool         bleShouldScan = true;
static NimBLEAdvertisedDevice* bleTarget   = nullptr;

// Reassembly of "<seq>/<total>|<frag>" frames on the data characteristic.
static String dataAccum;
static int    dataExpectedSeq = 1;
static int    dataExpectedTotal = 0;

// Forward decl from UI section.
static void onEvent(const String& json);
static void onData(const String& payload);

static void resetDataAccum() {
    dataAccum = "";
    dataExpectedSeq = 1;
    dataExpectedTotal = 0;
}

static void evtNotifyCb(NimBLERemoteCharacteristic* /*c*/,
                         uint8_t* data, size_t len, bool /*isNotify*/) {
    String s; s.reserve(len);
    for (size_t i = 0; i < len; i++) s += (char)data[i];
    Serial.printf("[ble:evt<-] %s\n", s.c_str());
    onEvent(s);
}

static void dataNotifyCb(NimBLERemoteCharacteristic* /*c*/,
                          uint8_t* data, size_t len, bool /*isNotify*/) {
    // Frame: "<seq>/<total>|<payload>"
    int sep1 = -1, sep2 = -1;
    for (size_t i = 0; i < len; i++) {
        if (data[i] == '/' && sep1 < 0) sep1 = i;
        else if (data[i] == '|' && sep1 >= 0) { sep2 = i; break; }
    }
    if (sep1 < 0 || sep2 < 0) return;
    int seq   = atoi((const char*)data);
    int total = atoi((const char*)data + sep1 + 1);
    if (seq == 1) { resetDataAccum(); dataExpectedTotal = total; }
    if (seq != dataExpectedSeq || total != dataExpectedTotal) {
        Serial.printf("[ble:data] frame oos seq=%d expected=%d total=%d\n",
                      seq, dataExpectedSeq, total);
        resetDataAccum();
        return;
    }
    int payloadStart = sep2 + 1;
    dataAccum.concat((const char*)(data + payloadStart), len - payloadStart);
    dataExpectedSeq++;
    if (seq == total) {
        Serial.printf("[ble:data<-] %d bytes\n", (int)dataAccum.length());
        onData(dataAccum);
        resetDataAccum();
    }
}

class CliCb : public NimBLEClientCallbacks {
    void onConnect(NimBLEClient* /*c*/) override {
        Serial.println("[ble] connected to peer");
    }
    void onDisconnect(NimBLEClient* /*c*/) override {
        Serial.println("[ble] disconnected");
        bleConnected = false;
        bleShouldScan = true;
    }
};

class ScanCb : public NimBLEAdvertisedDeviceCallbacks {
    void onResult(NimBLEAdvertisedDevice* dev) override {
        if (dev->isAdvertisingService(NimBLEUUID(SNAPCHEF_SVC_UUID))) {
            Serial.printf("[ble] found %s rssi=%d\n",
                          dev->getName().c_str(), dev->getRSSI());
            NimBLEDevice::getScan()->stop();
            if (bleTarget) delete bleTarget;
            bleTarget = new NimBLEAdvertisedDevice(*dev);
        }
    }
};

static void bleSendCmd(const String& json) {
    if (!bleConnected || !bleCmd) {
        Serial.printf("[ble:cmd->] DROPPED (no conn): %s\n", json.c_str());
        return;
    }
    Serial.printf("[ble:cmd->] %s\n", json.c_str());
    bleCmd->writeValue((uint8_t*)json.c_str(), json.length(), false);
}

static bool bleConnectTo(NimBLEAdvertisedDevice* dev) {
    if (!bleCli) {
        bleCli = NimBLEDevice::createClient();
        bleCli->setClientCallbacks(new CliCb(), false);
    }
    if (!bleCli->connect(dev)) return false;

    NimBLERemoteService* svc = bleCli->getService(SNAPCHEF_SVC_UUID);
    if (!svc) { bleCli->disconnect(); return false; }
    bleCmd  = svc->getCharacteristic(SNAPCHEF_CHR_CMD_UUID);
    bleEvt  = svc->getCharacteristic(SNAPCHEF_CHR_EVT_UUID);
    bleData = svc->getCharacteristic(SNAPCHEF_CHR_DATA_UUID);
    if (!bleCmd || !bleEvt || !bleData) { bleCli->disconnect(); return false; }

    if (bleEvt->canNotify())  bleEvt->subscribe(true,  evtNotifyCb);
    if (bleData->canNotify()) bleData->subscribe(true, dataNotifyCb);
    bleCli->setMTU(247);

    bleConnected = true;
    return true;
}

static void bleInit() {
    NimBLEDevice::init("SnapChef-Display");
    NimBLEDevice::setMTU(247);
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);
    NimBLEScan* scan = NimBLEDevice::getScan();
    scan->setAdvertisedDeviceCallbacks(new ScanCb(), false);
    scan->setActiveScan(true);
    scan->setInterval(80);
    scan->setWindow(40);
}

static void bleTick() {
    if (bleConnected) return;
    if (bleTarget) {
        Serial.println("[ble] connecting…");
        bool ok = bleConnectTo(bleTarget);
        delete bleTarget;
        bleTarget = nullptr;
        if (!ok) {
            Serial.println("[ble] connect failed");
            bleShouldScan = true;
        }
        return;
    }
    if (bleShouldScan) {
        bleShouldScan = false;
        Serial.println("[ble] scanning…");
        NimBLEDevice::getScan()->start(3, [](NimBLEScanResults){
            // Restart scan on next tick if nothing matched.
            bleShouldScan = (bleTarget == nullptr) && !bleConnected;
        }, false);
    }
}

// ============================================================================
//                                 UI: globals
// ============================================================================

enum DisplayState {
    UI_IDLE,
    UI_ACTION_SELECT,         // Put In / Take Out / Edit
    UI_SUBMODE_SELECT,        // Veggie / Receipt (Put In only)
    UI_SCANNING,
    UI_VEGGIE_RESULT,
    UI_RECEIPT_RESULT,
    UI_RECIPE_PROMPT,
    UI_RECIPE_RESULT,
    UI_MENU,
};

static DisplayState gUiState = UI_IDLE;
static String       gPurpose = "in";          // "in" or "out"
static String       gScanKind = "veggie";     // "veggie" or "receipt"

static lv_obj_t* scr_idle           = nullptr;
static lv_obj_t* scr_action         = nullptr;
static lv_obj_t* scr_submode        = nullptr;
static lv_obj_t* scr_scan           = nullptr;
static lv_obj_t* scr_veggie_result  = nullptr;
static lv_obj_t* scr_receipt_result = nullptr;
static lv_obj_t* scr_recipe_prompt  = nullptr;
static lv_obj_t* scr_recipe_result  = nullptr;
static lv_obj_t* scr_menu           = nullptr;

static lv_obj_t* scan_status_label  = nullptr;
static lv_obj_t* scan_label_dots    = nullptr;
static lv_timer_t* scan_anim_timer  = nullptr;
static int       scan_dots_phase    = 0;

static lv_obj_t* connection_dot     = nullptr;   // small green/grey indicator on idle

// Veggie result widgets (filled when result shown)
static lv_obj_t* vr_name_label      = nullptr;
static lv_obj_t* vr_conf_label      = nullptr;
static lv_obj_t* vr_status_label    = nullptr;
static lv_obj_t* vr_action_btn      = nullptr;
static lv_obj_t* vr_action_btn_lbl  = nullptr;
static String    vr_pending_label;
static String    vr_pending_purpose;
static bool      vr_known           = false;

// Recipe prompt context (Take-Out → "Generate recipe?")
static String    rp_pending_veggie;

// Receipt result state
struct ReceiptItem {
    String name;
    bool   needs_refrig;
    bool   checked;
};
static std::vector<ReceiptItem> rcp_items;
static lv_obj_t* rcp_list_obj = nullptr;

// Recipe result widgets
static lv_obj_t* rec_title_lbl = nullptr;
static lv_obj_t* rec_time_lbl  = nullptr;
static lv_obj_t* rec_steps_obj = nullptr;

// Menu state
static lv_obj_t* menu_list_obj = nullptr;
struct MenuRowData { int idx; lv_obj_t* row; lv_timer_t* hold_timer; };
static MenuRowData menu_rows[40];
static lv_obj_t* menu_confirm_box = nullptr;
static int       menu_pending_delete = -1;

// ============================================================================
//                              UI helpers
// ============================================================================

static lv_obj_t* makeScreen() {
    lv_obj_t* scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, COLOR_BG, 0);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);
    return scr;
}

static lv_obj_t* makeCard(lv_obj_t* parent, int x, int y, int w, int h) {
    lv_obj_t* c = lv_obj_create(parent);
    lv_obj_set_size(c, w, h);
    lv_obj_set_pos(c, x, y);
    lv_obj_set_style_bg_color(c, COLOR_CARD, 0);
    lv_obj_set_style_bg_opa(c, LV_OPA_COVER, 0);
    lv_obj_set_style_border_color(c, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(c, 1, 0);
    lv_obj_set_style_radius(c, 12, 0);
    lv_obj_set_style_pad_all(c, 16, 0);
    lv_obj_clear_flag(c, LV_OBJ_FLAG_SCROLLABLE);
    return c;
}

static lv_obj_t* makeLabel(lv_obj_t* parent, const char* text,
                           const lv_font_t* font, lv_color_t color) {
    lv_obj_t* l = lv_label_create(parent);
    lv_label_set_text(l, text);
    lv_obj_set_style_text_font(l, font, 0);
    lv_obj_set_style_text_color(l, color, 0);
    return l;
}

static lv_obj_t* makeButton(lv_obj_t* parent, const char* text, lv_color_t bg,
                            lv_event_cb_t cb) {
    lv_obj_t* b = lv_btn_create(parent);
    lv_obj_set_style_bg_color(b, bg, 0);
    lv_obj_set_style_bg_color(b, lv_color_darken(bg, 40), LV_STATE_PRESSED);
    lv_obj_set_style_radius(b, 10, 0);
    lv_obj_set_style_border_width(b, 0, 0);
    lv_obj_set_style_shadow_width(b, 0, 0);
    if (cb) lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, NULL);
    lv_obj_t* l = lv_label_create(b);
    lv_label_set_text(l, text);
    lv_obj_set_style_text_font(l, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(l, COLOR_TEXT, 0);
    lv_obj_center(l);
    return b;
}

static void switchScreen(lv_obj_t* target) {
    lv_scr_load_anim(target, LV_SCR_LOAD_ANIM_FADE_ON, 200, 0, false);
}

// Forward declarations of screen builders so callbacks can navigate.
static void buildIdle();
static void buildAction();
static void buildSubmode();
static void buildScan();
static void buildVeggieResult();
static void buildReceiptResult();
static void buildRecipePrompt();
static void buildRecipeResult();
static void buildMenu();

static void rebuildMenu();   // tear down + rebuild list when fridge changes
static void startScanAnim();
static void stopScanAnim();

// ============================================================================
//                              SCREEN: idle
// ============================================================================

static void onIdleStartCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildIdle() {
    scr_idle = makeScreen();

    lv_obj_t* bar = lv_obj_create(scr_idle);
    lv_obj_set_size(bar, 800, 4); lv_obj_set_pos(bar, 0, 0);
    lv_obj_set_style_bg_color(bar, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(bar, 0, 0);
    lv_obj_set_style_radius(bar, 0, 0);

    lv_obj_t* title = makeLabel(scr_idle, "SnapChef", &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(title, LV_ALIGN_CENTER, 0, -90);

    lv_obj_t* tag = makeLabel(scr_idle, "Smart Fridge Assistant",
                              &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align_to(tag, title, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

    lv_obj_t* div = lv_obj_create(scr_idle);
    lv_obj_set_size(div, 120, 2);
    lv_obj_set_style_bg_color(div, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(div, 0, 0);
    lv_obj_set_style_radius(div, 0, 0);
    lv_obj_align_to(div, tag, LV_ALIGN_OUT_BOTTOM_MID, 0, 20);

    lv_obj_t* prompt = makeLabel(scr_idle,
        "Hold an ingredient near the sensor\nor tap to get started",
        &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(prompt, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(prompt, div, LV_ALIGN_OUT_BOTTOM_MID, 0, 20);

    lv_obj_t* btn = makeButton(scr_idle, "Get Started", COLOR_ACCENT, onIdleStartCb);
    lv_obj_set_size(btn, 200, 52);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -48);

    // Tiny connection dot in the top-right corner.
    connection_dot = lv_obj_create(scr_idle);
    lv_obj_set_size(connection_dot, 12, 12);
    lv_obj_align(connection_dot, LV_ALIGN_TOP_RIGHT, -16, 12);
    lv_obj_set_style_radius(connection_dot, 6, 0);
    lv_obj_set_style_border_width(connection_dot, 0, 0);
    lv_obj_set_style_bg_color(connection_dot, COLOR_SUBTEXT, 0);
}

static void updateConnectionDot() {
    if (!connection_dot) return;
    lv_obj_set_style_bg_color(connection_dot,
                              bleConnected ? COLOR_OK : COLOR_SUBTEXT, 0);
}

// ============================================================================
//                              SCREEN: action select
// ============================================================================

static void onActionPutInCb(lv_event_t* /*e*/) {
    gPurpose = "in";
    gUiState = UI_SUBMODE_SELECT;
    switchScreen(scr_submode);
}

static void onActionTakeOutCb(lv_event_t* /*e*/) {
    gPurpose = "out";
    gScanKind = "veggie";
    gUiState = UI_SCANNING;
    bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"out\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Show the veggie to remove");
    startScanAnim();
    switchScreen(scr_scan);
}

static void onActionMenuCb(lv_event_t* /*e*/) {
    gUiState = UI_MENU;
    rebuildMenu();
    switchScreen(scr_menu);
}

static void onActionBackCb(lv_event_t* /*e*/) {
    gUiState = UI_IDLE;
    switchScreen(scr_idle);
}

static void buildAction() {
    scr_action = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_action);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* htitle = makeLabel(header, "SnapChef", &lv_font_montserrat_30, COLOR_ACCENT);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);
    lv_obj_t* sub = makeLabel(header, "Choose an action", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_RIGHT_MID, -24, 0);

    int cw = 220, ch = 300, gap = 30, cy = 110;
    int total = cw * 3 + gap * 2;
    int sx = (800 - total) / 2;

    // Put In
    lv_obj_t* c1 = makeCard(scr_action, sx, cy, cw, ch);
    lv_obj_set_style_border_color(c1, COLOR_ACCENT, 0);
    lv_obj_t* i1 = makeLabel(c1, LV_SYMBOL_DOWNLOAD, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(i1, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t* t1 = makeLabel(c1, "Put In\nFridge", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t1, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t1, i1, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t* d1 = makeLabel(c1, "Add veggies or\na receipt's items",
                             &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d1, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d1, t1, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t* b1 = makeButton(c1, "Start", COLOR_ACCENT, onActionPutInCb);
    lv_obj_set_size(b1, cw - 32, 44);
    lv_obj_align(b1, LV_ALIGN_BOTTOM_MID, 0, 0);

    // Take Out
    lv_obj_t* c2 = makeCard(scr_action, sx + cw + gap, cy, cw, ch);
    lv_obj_set_style_border_color(c2, COLOR_ACCENT2, 0);
    lv_obj_t* i2 = makeLabel(c2, LV_SYMBOL_UPLOAD, &lv_font_montserrat_48, COLOR_ACCENT2);
    lv_obj_align(i2, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t* t2 = makeLabel(c2, "Take Out\nof Fridge", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t2, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t2, i2, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t* d2 = makeLabel(c2, "Scan a veggie to\nremove and cook it",
                             &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d2, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d2, t2, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t* b2 = makeButton(c2, "Scan", COLOR_ACCENT2, onActionTakeOutCb);
    lv_obj_set_size(b2, cw - 32, 44);
    lv_obj_align(b2, LV_ALIGN_BOTTOM_MID, 0, 0);

    // My Fridge
    lv_obj_t* c3 = makeCard(scr_action, sx + (cw + gap) * 2, cy, cw, ch);
    lv_obj_set_style_border_color(c3, COLOR_SUBTEXT, 0);
    lv_obj_t* i3 = makeLabel(c3, LV_SYMBOL_LIST, &lv_font_montserrat_48, COLOR_SUBTEXT);
    lv_obj_align(i3, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t* t3 = makeLabel(c3, "My\nFridge", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t3, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t3, i3, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t* d3 = makeLabel(c3, "View / remove\nstored ingredients",
                             &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d3, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d3, t3, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t* b3 = makeButton(c3, "Open", lv_color_hex(0x444444), onActionMenuCb);
    lv_obj_set_size(b3, cw - 32, 44);
    lv_obj_align(b3, LV_ALIGN_BOTTOM_MID, 0, 0);

    lv_obj_t* back = makeButton(scr_action, LV_SYMBOL_LEFT " Back",
                                lv_color_hex(0x333333), onActionBackCb);
    lv_obj_set_size(back, 120, 40);
    lv_obj_align(back, LV_ALIGN_BOTTOM_LEFT, 24, -20);
}

// ============================================================================
//                            SCREEN: submode select
// ============================================================================

static void onSubmodeVeggieCb(lv_event_t* /*e*/) {
    gScanKind = "veggie";
    gUiState = UI_SCANNING;
    bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"in\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Show the veggie to add");
    startScanAnim();
    switchScreen(scr_scan);
}

static void onSubmodeReceiptCb(lv_event_t* /*e*/) {
    gScanKind = "receipt";
    gUiState = UI_SCANNING;
    bleSendCmd("{\"cmd\":\"start_receipt_scan\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Hold receipt steady");
    startScanAnim();
    switchScreen(scr_scan);
}

static void onSubmodeBackCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildSubmode() {
    scr_submode = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_submode);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* htitle = makeLabel(header, "Put In Fridge", &lv_font_montserrat_30, COLOR_ACCENT);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);
    lv_obj_t* sub = makeLabel(header, "What are you adding?", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_RIGHT_MID, -24, 0);

    int cw = 280, ch = 300, gap = 40, cy = 110;
    int total = cw * 2 + gap;
    int sx = (800 - total) / 2;

    lv_obj_t* c1 = makeCard(scr_submode, sx, cy, cw, ch);
    lv_obj_set_style_border_color(c1, COLOR_ACCENT, 0);
    lv_obj_t* i1 = makeLabel(c1, LV_SYMBOL_EYE_OPEN, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(i1, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t* t1 = makeLabel(c1, "Scan a Veggie", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align_to(t1, i1, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t* d1 = makeLabel(c1, "Hold one item up\nto the camera",
                             &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d1, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d1, t1, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t* b1 = makeButton(c1, "Scan", COLOR_ACCENT, onSubmodeVeggieCb);
    lv_obj_set_size(b1, cw - 32, 48);
    lv_obj_align(b1, LV_ALIGN_BOTTOM_MID, 0, 0);

    lv_obj_t* c2 = makeCard(scr_submode, sx + cw + gap, cy, cw, ch);
    lv_obj_set_style_border_color(c2, COLOR_ACCENT2, 0);
    lv_obj_t* i2 = makeLabel(c2, LV_SYMBOL_FILE, &lv_font_montserrat_48, COLOR_ACCENT2);
    lv_obj_align(i2, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t* t2 = makeLabel(c2, "Scan a Receipt", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align_to(t2, i2, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t* d2 = makeLabel(c2, "We'll snap one frame\nand parse the items",
                             &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d2, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d2, t2, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t* b2 = makeButton(c2, "Capture", COLOR_ACCENT2, onSubmodeReceiptCb);
    lv_obj_set_size(b2, cw - 32, 48);
    lv_obj_align(b2, LV_ALIGN_BOTTOM_MID, 0, 0);

    lv_obj_t* back = makeButton(scr_submode, LV_SYMBOL_LEFT " Back",
                                lv_color_hex(0x333333), onSubmodeBackCb);
    lv_obj_set_size(back, 120, 40);
    lv_obj_align(back, LV_ALIGN_BOTTOM_LEFT, 24, -20);
}

// ============================================================================
//                              SCREEN: scanning
// ============================================================================

static void scanAnimCb(lv_timer_t* /*t*/) {
    if (!scan_label_dots) return;
    scan_dots_phase = (scan_dots_phase + 1) % 4;
    char d[5] = "    ";
    for (int i = 0; i < scan_dots_phase; i++) d[i] = '.';
    d[scan_dots_phase] = '\0';
    lv_label_set_text(scan_label_dots, d);
}

static void startScanAnim() {
    scan_dots_phase = 0;
    if (scan_anim_timer) lv_timer_del(scan_anim_timer);
    scan_anim_timer = lv_timer_create(scanAnimCb, 400, NULL);
}

static void stopScanAnim() {
    if (scan_anim_timer) { lv_timer_del(scan_anim_timer); scan_anim_timer = nullptr; }
}

static void onScanCancelCb(lv_event_t* /*e*/) {
    bleSendCmd("{\"cmd\":\"cancel\"}");
    stopScanAnim();
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildScan() {
    scr_scan = makeScreen();

    lv_obj_t* icon = makeLabel(scr_scan, LV_SYMBOL_REFRESH, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(icon, LV_ALIGN_CENTER, 0, -100);

    lv_obj_t* lbl = makeLabel(scr_scan, "Scanning", &lv_font_montserrat_30, COLOR_TEXT);
    lv_obj_align_to(lbl, icon, LV_ALIGN_OUT_BOTTOM_MID, -10, 16);

    scan_label_dots = makeLabel(scr_scan, "   ", &lv_font_montserrat_30, COLOR_ACCENT);
    lv_obj_align_to(scan_label_dots, lbl, LV_ALIGN_OUT_RIGHT_MID, 4, 0);

    scan_status_label = makeLabel(scr_scan, "Please hold still",
                                  &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(scan_status_label, LV_ALIGN_CENTER, 0, 0);

    lv_obj_t* tip = makeLabel(scr_scan, "Up to 20 seconds",
                              &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(tip, LV_ALIGN_CENTER, 0, 36);

    lv_obj_t* cancel = makeButton(scr_scan, "Cancel", lv_color_hex(0x444444), onScanCancelCb);
    lv_obj_set_size(cancel, 160, 48);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, 0, -32);
}

// ============================================================================
//                          SCREEN: veggie result
// ============================================================================

static void vrConfirmInCb(lv_event_t* /*e*/) {
    inventoryAdd(vr_pending_label);
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void vrConfirmOutCb(lv_event_t* /*e*/) {
    bool removed = inventoryRemove(vr_pending_label);
    if (!removed) {
        // Veggie wasn't in fridge — go straight back to action.
        gUiState = UI_ACTION_SELECT;
        switchScreen(scr_action);
        return;
    }
    rp_pending_veggie = vr_pending_label;
    gUiState = UI_RECIPE_PROMPT;
    switchScreen(scr_recipe_prompt);
}

static void vrRetryCb(lv_event_t* /*e*/) {
    if (gPurpose == "out") {
        gUiState = UI_SCANNING;
        bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"out\"}");
        if (scan_status_label) lv_label_set_text(scan_status_label, "Show the veggie to remove");
    } else {
        gUiState = UI_SCANNING;
        bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"in\"}");
        if (scan_status_label) lv_label_set_text(scan_status_label, "Show the veggie to add");
    }
    startScanAnim();
    switchScreen(scr_scan);
}

static void vrCancelCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildVeggieResult() {
    scr_veggie_result = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_veggie_result);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    vr_status_label = makeLabel(header, LV_SYMBOL_OK " Detected",
                                &lv_font_montserrat_22, COLOR_ACCENT);
    lv_obj_align(vr_status_label, LV_ALIGN_LEFT_MID, 24, 0);

    lv_obj_t* card = makeCard(scr_veggie_result, 230, 110, 340, 280);
    lv_obj_set_style_border_color(card, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(card, 2, 0);

    lv_obj_t* icon = makeLabel(card, LV_SYMBOL_IMAGE, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(icon, LV_ALIGN_TOP_MID, 0, 8);

    vr_name_label = makeLabel(card, "—", &lv_font_montserrat_36, COLOR_TEXT);
    lv_obj_align_to(vr_name_label, icon, LV_ALIGN_OUT_BOTTOM_MID, 0, 12);

    vr_conf_label = makeLabel(card, "", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align_to(vr_conf_label, vr_name_label, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

    vr_action_btn = makeButton(scr_veggie_result, "Add to Fridge", COLOR_ACCENT, vrConfirmInCb);
    lv_obj_set_size(vr_action_btn, 220, 52);
    lv_obj_align(vr_action_btn, LV_ALIGN_BOTTOM_MID, -130, -32);
    vr_action_btn_lbl = lv_obj_get_child(vr_action_btn, 0);

    lv_obj_t* retry = makeButton(scr_veggie_result, LV_SYMBOL_REFRESH " Retry",
                                  lv_color_hex(0x444444), vrRetryCb);
    lv_obj_set_size(retry, 160, 52);
    lv_obj_align(retry, LV_ALIGN_BOTTOM_MID, 60, -32);

    lv_obj_t* cancel = makeButton(scr_veggie_result, "Cancel",
                                   lv_color_hex(0x333333), vrCancelCb);
    lv_obj_set_size(cancel, 100, 52);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, 240, -32);
}

// Re-themes the veggie-result screen for the latest payload.
// Called from BLE event handler under the lvgl lock.
static void showVeggieResult(const String& label, float conf, const String& purpose,
                              bool known) {
    vr_pending_label   = label;
    vr_pending_purpose = purpose;
    vr_known           = known;

    if (known) {
        lv_label_set_text(vr_name_label, label.c_str());
        char b[64];
        snprintf(b, sizeof(b), "Confidence: %.0f%%", conf * 100.0f);
        lv_label_set_text(vr_conf_label, b);
        lv_label_set_text(vr_status_label, LV_SYMBOL_OK " Detected");
        lv_obj_set_style_text_color(vr_status_label, COLOR_ACCENT, 0);

        if (purpose == "out") {
            lv_label_set_text(vr_action_btn_lbl, LV_SYMBOL_UPLOAD " Take Out");
            lv_obj_set_style_bg_color(vr_action_btn, COLOR_ACCENT2, 0);
            lv_obj_remove_event_cb(vr_action_btn, vrConfirmInCb);
            lv_obj_remove_event_cb(vr_action_btn, vrConfirmOutCb);
            lv_obj_add_event_cb(vr_action_btn, vrConfirmOutCb, LV_EVENT_CLICKED, NULL);
        } else {
            lv_label_set_text(vr_action_btn_lbl, LV_SYMBOL_DOWNLOAD " Add to Fridge");
            lv_obj_set_style_bg_color(vr_action_btn, COLOR_ACCENT, 0);
            lv_obj_remove_event_cb(vr_action_btn, vrConfirmInCb);
            lv_obj_remove_event_cb(vr_action_btn, vrConfirmOutCb);
            lv_obj_add_event_cb(vr_action_btn, vrConfirmInCb, LV_EVENT_CLICKED, NULL);
        }
        lv_obj_clear_flag(vr_action_btn, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_label_set_text(vr_name_label, "Unknown");
        lv_label_set_text(vr_conf_label, "Couldn't recognise the item");
        lv_label_set_text(vr_status_label, LV_SYMBOL_WARNING " Unknown");
        lv_obj_set_style_text_color(vr_status_label, COLOR_ACCENT2, 0);
        lv_obj_add_flag(vr_action_btn, LV_OBJ_FLAG_HIDDEN);
    }
    gUiState = UI_VEGGIE_RESULT;
    switchScreen(scr_veggie_result);
}

// ============================================================================
//                          SCREEN: receipt result
// ============================================================================

static void rcpItemToggleCb(lv_event_t* e) {
    int idx = (int)(intptr_t)lv_event_get_user_data(e);
    if (idx < 0 || idx >= (int)rcp_items.size()) return;
    lv_obj_t* cb = (lv_obj_t*)lv_event_get_target(e);
    rcp_items[idx].checked = lv_obj_has_state(cb, LV_STATE_CHECKED);
}

static void rcpConfirmCb(lv_event_t* /*e*/) {
    int added = 0;
    for (auto& it : rcp_items) {
        if (it.checked && it.name.length()) {
            inventoryAdd(it.name);
            added++;
        }
    }
    Serial.printf("[receipt] added %d items to fridge\n", added);
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void rcpCancelCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildReceiptResult() {
    scr_receipt_result = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_receipt_result);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* htitle = makeLabel(header, LV_SYMBOL_FILE " Receipt items",
                                  &lv_font_montserrat_22, COLOR_ACCENT2);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);
    lv_obj_t* sub = makeLabel(header, "Tick the items you put in the fridge",
                              &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_RIGHT_MID, -24, 0);

    rcp_list_obj = lv_obj_create(scr_receipt_result);
    lv_obj_set_size(rcp_list_obj, 760, 320);
    lv_obj_align(rcp_list_obj, LV_ALIGN_TOP_MID, 0, 84);
    lv_obj_set_style_bg_color(rcp_list_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(rcp_list_obj, 0, 0);
    lv_obj_set_style_pad_all(rcp_list_obj, 8, 0);
    lv_obj_set_style_pad_row(rcp_list_obj, 6, 0);
    lv_obj_set_flex_flow(rcp_list_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* confirm = makeButton(scr_receipt_result, LV_SYMBOL_OK " Confirm",
                                    COLOR_ACCENT, rcpConfirmCb);
    lv_obj_set_size(confirm, 200, 50);
    lv_obj_align(confirm, LV_ALIGN_BOTTOM_MID, 110, -16);

    lv_obj_t* cancel = makeButton(scr_receipt_result, "Cancel",
                                   lv_color_hex(0x333333), rcpCancelCb);
    lv_obj_set_size(cancel, 160, 50);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, -120, -16);
}

static void showReceiptResult(const String& json) {
    rcp_items.clear();
    DynamicJsonDocument doc(8192);
    DeserializationError err = deserializeJson(doc, json);
    if (err) {
        Serial.printf("[receipt] json err: %s\n", err.c_str());
        gUiState = UI_ACTION_SELECT;
        switchScreen(scr_action);
        return;
    }
    JsonArray items = doc["items"].as<JsonArray>();
    for (JsonObject it : items) {
        ReceiptItem r;
        r.name = (const char*)(it["name"] | "");
        r.needs_refrig = it["needs_refrigeration"] | false;
        r.checked = it["checked"] | r.needs_refrig;
        if (r.name.length()) rcp_items.push_back(r);
    }

    // Repopulate list.
    lv_obj_clean(rcp_list_obj);
    for (size_t i = 0; i < rcp_items.size(); i++) {
        lv_obj_t* row = lv_obj_create(rcp_list_obj);
        lv_obj_set_size(row, 720, 50);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, lv_color_hex(0x333333), 0);
        lv_obj_set_style_border_width(row, 1, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_pad_all(row, 8, 0);

        lv_obj_t* cb = lv_checkbox_create(row);
        lv_checkbox_set_text(cb, rcp_items[i].name.c_str());
        if (rcp_items[i].checked) lv_obj_add_state(cb, LV_STATE_CHECKED);
        lv_obj_set_style_text_color(cb, COLOR_TEXT, 0);
        lv_obj_set_style_text_font(cb, &lv_font_montserrat_16, 0);
        lv_obj_align(cb, LV_ALIGN_LEFT_MID, 0, 0);
        lv_obj_add_event_cb(cb, rcpItemToggleCb, LV_EVENT_VALUE_CHANGED,
                            (void*)(intptr_t)i);

        if (rcp_items[i].needs_refrig) {
            lv_obj_t* tag = makeLabel(row, LV_SYMBOL_OK " fridge",
                                       &lv_font_montserrat_12, COLOR_ACCENT);
            lv_obj_align(tag, LV_ALIGN_RIGHT_MID, -8, 0);
        }
    }

    gUiState = UI_RECEIPT_RESULT;
    switchScreen(scr_receipt_result);
}

// ============================================================================
//                          SCREEN: recipe prompt + result
// ============================================================================

static void rpYesCb(lv_event_t* /*e*/) {
    String json = "{\"cmd\":\"get_recipe\",\"ingredients\":[";
    bool first = true;
    auto append = [&](const String& s){
        if (!first) json += ',';
        json += '"'; json += s; json += '"';
        first = false;
    };
    append(rp_pending_veggie);
    for (auto& f : gFridge) append(f);
    json += "]}";
    bleSendCmd(json);
    if (scan_status_label) lv_label_set_text(scan_status_label, "Cooking up a recipe…");
    gUiState = UI_SCANNING;
    startScanAnim();
    switchScreen(scr_scan);
}

static void rpNoCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildRecipePrompt() {
    scr_recipe_prompt = makeScreen();

    lv_obj_t* card = makeCard(scr_recipe_prompt, 200, 130, 400, 240);
    lv_obj_set_style_border_color(card, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(card, 2, 0);

    lv_obj_t* icon = makeLabel(card, LV_SYMBOL_OK,
                                &lv_font_montserrat_36, COLOR_ACCENT);
    lv_obj_align(icon, LV_ALIGN_TOP_MID, 0, 8);

    lv_obj_t* title = makeLabel(card, "Item taken out!",
                                 &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align_to(title, icon, LV_ALIGN_OUT_BOTTOM_MID, 0, 12);

    lv_obj_t* prompt = makeLabel(card,
        "Want a quick recipe using\nyour fridge contents?",
        &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(prompt, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(prompt, LV_ALIGN_CENTER, 0, 6);

    lv_obj_t* yes = makeButton(card, LV_SYMBOL_OK " Yes", COLOR_ACCENT, rpYesCb);
    lv_obj_set_size(yes, 150, 48);
    lv_obj_align(yes, LV_ALIGN_BOTTOM_LEFT, 12, -8);

    lv_obj_t* no = makeButton(card, "No thanks", lv_color_hex(0x444444), rpNoCb);
    lv_obj_set_size(no, 150, 48);
    lv_obj_align(no, LV_ALIGN_BOTTOM_RIGHT, -12, -8);
}

static void recDoneCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildRecipeResult() {
    scr_recipe_result = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_recipe_result);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    rec_title_lbl = makeLabel(header, "Recipe", &lv_font_montserrat_22, COLOR_ACCENT2);
    lv_obj_align(rec_title_lbl, LV_ALIGN_LEFT_MID, 24, 0);
    rec_time_lbl = makeLabel(header, "—", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(rec_time_lbl, LV_ALIGN_RIGHT_MID, -24, 0);

    rec_steps_obj = lv_obj_create(scr_recipe_result);
    lv_obj_set_size(rec_steps_obj, 720, 320);
    lv_obj_align(rec_steps_obj, LV_ALIGN_TOP_MID, 0, 90);
    lv_obj_set_style_bg_color(rec_steps_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(rec_steps_obj, 0, 0);
    lv_obj_set_style_pad_all(rec_steps_obj, 8, 0);
    lv_obj_set_style_pad_row(rec_steps_obj, 12, 0);
    lv_obj_set_flex_flow(rec_steps_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* done = makeButton(scr_recipe_result, LV_SYMBOL_OK " Done",
                                 COLOR_ACCENT, recDoneCb);
    lv_obj_set_size(done, 180, 50);
    lv_obj_align(done, LV_ALIGN_BOTTOM_MID, 0, -16);
}

static void showRecipeResult(const String& json) {
    DynamicJsonDocument doc(8192);
    DeserializationError err = deserializeJson(doc, json);
    if (err) { Serial.printf("[recipe] json err: %s\n", err.c_str()); return; }
    String title  = (const char*)(doc["title"] | "Recipe");
    int    tmin   = doc["time_min"] | 0;
    JsonArray steps = doc["steps"].as<JsonArray>();

    lv_label_set_text(rec_title_lbl, title.c_str());
    char tb[24];
    if (tmin > 0) snprintf(tb, sizeof(tb), LV_SYMBOL_REFRESH " ~%d min", tmin);
    else          snprintf(tb, sizeof(tb), "%s", "");
    lv_label_set_text(rec_time_lbl, tb);

    lv_obj_clean(rec_steps_obj);
    int i = 1;
    for (JsonVariant s : steps) {
        const char* text = s | "";
        lv_obj_t* row = lv_obj_create(rec_steps_obj);
        lv_obj_set_size(row, 690, LV_SIZE_CONTENT);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, COLOR_ACCENT2, 0);
        lv_obj_set_style_border_side(row, LV_BORDER_SIDE_LEFT, 0);
        lv_obj_set_style_border_width(row, 3, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_set_style_pad_all(row, 12, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);

        char num[8]; snprintf(num, sizeof(num), "%d.", i++);
        lv_obj_t* numLbl = makeLabel(row, num, &lv_font_montserrat_18, COLOR_ACCENT);
        lv_obj_align(numLbl, LV_ALIGN_LEFT_MID, 0, 0);

        lv_obj_t* lbl = makeLabel(row, text, &lv_font_montserrat_16, COLOR_TEXT);
        lv_label_set_long_mode(lbl, LV_LABEL_LONG_WRAP);
        lv_obj_set_width(lbl, 620);
        lv_obj_align(lbl, LV_ALIGN_LEFT_MID, 32, 0);
    }
    gUiState = UI_RECIPE_RESULT;
    switchScreen(scr_recipe_result);
}

// ============================================================================
//                              SCREEN: my fridge menu
// ============================================================================

static void menuConfirmDeleteCb(lv_event_t* /*e*/) {
    if (menu_pending_delete >= 0) {
        inventoryRemoveAt(menu_pending_delete);
        menu_pending_delete = -1;
    }
    if (menu_confirm_box) { lv_obj_del(menu_confirm_box); menu_confirm_box = nullptr; }
    rebuildMenu();
}

static void menuCancelDeleteCb(lv_event_t* /*e*/) {
    menu_pending_delete = -1;
    if (menu_confirm_box) { lv_obj_del(menu_confirm_box); menu_confirm_box = nullptr; }
}

static void showMenuConfirmDelete(int idx) {
    menu_pending_delete = idx;
    menu_confirm_box = lv_obj_create(lv_scr_act());
    lv_obj_set_size(menu_confirm_box, 400, 200);
    lv_obj_center(menu_confirm_box);
    lv_obj_set_style_bg_color(menu_confirm_box, lv_color_hex(0x222222), 0);
    lv_obj_set_style_border_color(menu_confirm_box, COLOR_DELETE, 0);
    lv_obj_set_style_border_width(menu_confirm_box, 2, 0);
    lv_obj_set_style_radius(menu_confirm_box, 14, 0);
    lv_obj_clear_flag(menu_confirm_box, LV_OBJ_FLAG_SCROLLABLE);

    char msg[80];
    snprintf(msg, sizeof(msg), "Remove \"%s\" from\nyour fridge?",
             gFridge[idx].c_str());
    lv_obj_t* lbl = makeLabel(menu_confirm_box, msg, &lv_font_montserrat_18, COLOR_TEXT);
    lv_obj_set_style_text_align(lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(lbl, LV_ALIGN_TOP_MID, 0, 24);

    lv_obj_t* yes = makeButton(menu_confirm_box, LV_SYMBOL_TRASH " Remove",
                                COLOR_DELETE, menuConfirmDeleteCb);
    lv_obj_set_size(yes, 160, 48);
    lv_obj_align(yes, LV_ALIGN_BOTTOM_LEFT, 24, -16);

    lv_obj_t* no = makeButton(menu_confirm_box, "Cancel",
                               lv_color_hex(0x444444), menuCancelDeleteCb);
    lv_obj_set_size(no, 160, 48);
    lv_obj_align(no, LV_ALIGN_BOTTOM_RIGHT, -24, -16);
}

static void menuHoldFireCb(lv_timer_t* t) {
    MenuRowData* d = (MenuRowData*)t->user_data;
    d->hold_timer = nullptr;
    lv_timer_del(t);
    showMenuConfirmDelete(d->idx);
}

static void menuRowPressedCb(lv_event_t* e) {
    MenuRowData* d = (MenuRowData*)lv_event_get_user_data(e);
    if (d->hold_timer) lv_timer_del(d->hold_timer);
    d->hold_timer = lv_timer_create(menuHoldFireCb, 3000, d);
    lv_timer_set_repeat_count(d->hold_timer, 1);
    lv_obj_set_style_bg_color(d->row, lv_color_hex(0x2A1A1A), 0);
    lv_obj_set_style_border_color(d->row, COLOR_DELETE, 0);
}

static void menuRowReleasedCb(lv_event_t* e) {
    MenuRowData* d = (MenuRowData*)lv_event_get_user_data(e);
    if (d->hold_timer) { lv_timer_del(d->hold_timer); d->hold_timer = nullptr; }
    lv_obj_set_style_bg_color(d->row, lv_color_hex(0x1E1E1E), 0);
    lv_obj_set_style_border_color(d->row, lv_color_hex(0x333333), 0);
}

static void menuBackCb(lv_event_t* /*e*/) {
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static lv_obj_t* menu_count_lbl = nullptr;

static void buildMenu() {
    scr_menu = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_menu);
    lv_obj_set_size(header, 800, 72); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* htitle = makeLabel(header, LV_SYMBOL_LIST " My Fridge",
                                  &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);
    menu_count_lbl = makeLabel(header, "0 items", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(menu_count_lbl, LV_ALIGN_RIGHT_MID, -24, 0);

    lv_obj_t* hint = makeLabel(scr_menu, "Hold a row for 3 s to remove",
                                &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(hint, LV_ALIGN_TOP_MID, 0, 82);

    menu_list_obj = lv_obj_create(scr_menu);
    lv_obj_set_size(menu_list_obj, 760, 340);
    lv_obj_align(menu_list_obj, LV_ALIGN_TOP_MID, 0, 110);
    lv_obj_set_style_bg_color(menu_list_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(menu_list_obj, 0, 0);
    lv_obj_set_style_pad_all(menu_list_obj, 0, 0);
    lv_obj_set_style_pad_row(menu_list_obj, 8, 0);
    lv_obj_set_flex_flow(menu_list_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* back = makeButton(scr_menu, LV_SYMBOL_LEFT " Back",
                                 lv_color_hex(0x333333), menuBackCb);
    lv_obj_set_size(back, 140, 44);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, 0, -10);
}

static void rebuildMenu() {
    if (!scr_menu) buildMenu();
    if (!menu_list_obj) return;
    lv_obj_clean(menu_list_obj);

    char b[32];
    snprintf(b, sizeof(b), "%d items", (int)gFridge.size());
    if (menu_count_lbl) lv_label_set_text(menu_count_lbl, b);

    int n = min((int)gFridge.size(), 40);
    for (int i = 0; i < n; i++) {
        lv_obj_t* row = lv_obj_create(menu_list_obj);
        lv_obj_set_size(row, 740, 60);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, lv_color_hex(0x333333), 0);
        lv_obj_set_style_border_width(row, 1, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_add_flag(row, LV_OBJ_FLAG_CLICKABLE);

        lv_obj_t* dot = lv_obj_create(row);
        lv_obj_set_size(dot, 8, 8);
        lv_obj_set_style_bg_color(dot, COLOR_ACCENT, 0);
        lv_obj_set_style_radius(dot, 4, 0);
        lv_obj_set_style_border_width(dot, 0, 0);
        lv_obj_align(dot, LV_ALIGN_LEFT_MID, 12, 0);

        lv_obj_t* lbl = makeLabel(row, gFridge[i].c_str(),
                                   &lv_font_montserrat_18, COLOR_TEXT);
        lv_obj_align(lbl, LV_ALIGN_LEFT_MID, 32, 0);

        lv_obj_t* hint = makeLabel(row, "hold to remove",
                                    &lv_font_montserrat_12, COLOR_SUBTEXT);
        lv_obj_align(hint, LV_ALIGN_RIGHT_MID, -12, 0);

        menu_rows[i].idx = i;
        menu_rows[i].row = row;
        menu_rows[i].hold_timer = nullptr;
        lv_obj_add_event_cb(row, menuRowPressedCb,  LV_EVENT_PRESSED,    &menu_rows[i]);
        lv_obj_add_event_cb(row, menuRowReleasedCb, LV_EVENT_RELEASED,   &menu_rows[i]);
        lv_obj_add_event_cb(row, menuRowReleasedCb, LV_EVENT_PRESS_LOST, &menu_rows[i]);
    }
}

// ============================================================================
//                       BLE event/data handlers (UI side)
// ============================================================================

// Tiny helpers (avoid pulling ArduinoJson for short messages).
static String jsonStrField(const String& json, const char* key) {
    String pat = String("\"") + key + "\":\"";
    int i = json.indexOf(pat); if (i < 0) return "";
    i += pat.length();
    int j = json.indexOf('"', i); if (j < 0) return "";
    return json.substring(i, j);
}

static float jsonNumField(const String& json, const char* key) {
    String pat = String("\"") + key + "\":";
    int i = json.indexOf(pat); if (i < 0) return 0.0f;
    return atof(json.c_str() + i + pat.length());
}

static void onEvent(const String& json) {
    String evt = jsonStrField(json, "evt");
    if (evt == "ready") return;

    if (evt == "proximity_wake") {
        if (gUiState == UI_IDLE) {
            lvgl_port_lock(-1);
            gUiState = UI_ACTION_SELECT;
            switchScreen(scr_action);
            lvgl_port_unlock();
        }
        return;
    }
    if (evt == "veggie_scanning" || evt == "receipt_capturing" ||
        evt == "receipt_uploading") {
        // Keep the scanning screen — optionally tweak status text.
        lvgl_port_lock(-1);
        if (scan_status_label) {
            const char* msg =
                evt == "receipt_capturing" ? "Capturing photo…" :
                evt == "receipt_uploading" ? "Reading receipt…" :
                                             "Looking for veggies…";
            lv_label_set_text(scan_status_label, msg);
        }
        lvgl_port_unlock();
        return;
    }
    if (evt == "veggie_result") {
        String label = jsonStrField(json, "label");
        float  conf  = jsonNumField(json, "confidence");
        String purp  = jsonStrField(json, "purpose");
        lvgl_port_lock(-1);
        stopScanAnim();
        showVeggieResult(label, conf, purp, true);
        lvgl_port_unlock();
        return;
    }
    if (evt == "veggie_unknown") {
        String purp = jsonStrField(json, "purpose");
        lvgl_port_lock(-1);
        stopScanAnim();
        showVeggieResult("", 0.0f, purp, false);
        lvgl_port_unlock();
        return;
    }
    if (evt == "veggie_cancelled") {
        // We already navigated away on cancel; nothing to do.
        return;
    }
    if (evt == "receipt_result" || evt == "recipe_result") {
        // Header — body arrives on the data characteristic. UI updates happen
        // in onData() once the full payload is reassembled.
        return;
    }
    if (evt == "receipt_error" || evt == "error") {
        String msg = jsonStrField(json, "msg");
        if (msg.length() == 0) msg = jsonStrField(json, "code");
        Serial.printf("[err] %s\n", msg.c_str());
        lvgl_port_lock(-1);
        stopScanAnim();
        if (scan_status_label) lv_label_set_text(scan_status_label,
            (String("Error: ") + msg).c_str());
        // Bounce back to action select after 2 s.
        lv_timer_t* t = lv_timer_create([](lv_timer_t* tm){
            gUiState = UI_ACTION_SELECT;
            switchScreen(scr_action);
            lv_timer_del(tm);
        }, 2000, NULL);
        lv_timer_set_repeat_count(t, 1);
        lvgl_port_unlock();
        return;
    }
}

static void onData(const String& payload) {
    // payload is a complete JSON message — figure out which screen wants it
    // by looking at the embedded "evt" field.
    String evt = jsonStrField(payload, "evt");
    lvgl_port_lock(-1);
    stopScanAnim();
    if (evt == "receipt_result") {
        showReceiptResult(payload);
    } else if (evt == "recipe_result") {
        showRecipeResult(payload);
    } else {
        Serial.printf("[data] unknown payload, evt=%s\n", evt.c_str());
    }
    lvgl_port_unlock();
}

// ============================================================================
//                              setup() / loop()
// ============================================================================

void setup() {
    Serial.begin(115200);
    Serial.print("MAC: "); Serial.println(WiFi.macAddress());
    Serial.println("Initializing board");

    Board* board = new Board();
    board->init();

#if LVGL_PORT_AVOID_TEARING_MODE
    auto lcd = board->getLCD();
    lcd->configFrameBufferNumber(LVGL_PORT_DISP_BUFFER_NUM);
#if ESP_PANEL_DRIVERS_BUS_ENABLE_RGB && CONFIG_IDF_TARGET_ESP32S3
    auto lcd_bus = lcd->getBus();
    if (lcd_bus->getBasicAttributes().type == ESP_PANEL_BUS_TYPE_RGB) {
        static_cast<BusRGB*>(lcd_bus)->configRGB_BounceBufferSize(
            lcd->getFrameWidth() * 10);
    }
#endif
#endif
    assert(board->begin());

    Serial.println("Initializing LVGL");
    lvgl_port_init(board->getLCD(), board->getTouch());

    inventoryLoad();
    Serial.printf("[inv] %d items loaded\n", (int)gFridge.size());

    Serial.println("Building UI");
    lvgl_port_lock(-1);
    buildIdle();
    buildAction();
    buildSubmode();
    buildScan();
    buildVeggieResult();
    buildReceiptResult();
    buildRecipePrompt();
    buildRecipeResult();
    buildMenu();
    lv_scr_load(scr_idle);
    lvgl_port_unlock();

    bleInit();
    Serial.println("UI ready");
}

void loop() {
    bleTick();
    lvgl_port_lock(-1);
    updateConnectionDot();
    lvgl_port_unlock();
    delay(50);
}

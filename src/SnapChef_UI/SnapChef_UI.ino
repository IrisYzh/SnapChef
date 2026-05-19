/*
 * SnapChef — Display device (Waveshare ESP32-S3-Touch-LCD-4.3)
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
#include <WiFi.h>

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

static NimBLEClient*               bleCli       = nullptr;
static NimBLERemoteCharacteristic* bleCmd       = nullptr;
static NimBLERemoteCharacteristic* bleEvt       = nullptr;
static NimBLERemoteCharacteristic* bleData      = nullptr;
static volatile bool               bleConnected = false;
static volatile bool               bleShouldScan = true;
static NimBLEAddress               bleTargetAddr;
static volatile bool               bleHasTarget = false;

static String dataAccum;
static int    dataExpectedSeq   = 1;
static int    dataExpectedTotal = 0;

static void onEvent(const String& json);
static void onData(const String& payload);

static void resetDataAccum() {
    dataAccum = "";
    dataExpectedSeq   = 1;
    dataExpectedTotal = 0;
}

static void evtNotifyCb(NimBLERemoteCharacteristic*, uint8_t* data, size_t len, bool) {
    String s; s.reserve(len);
    for (size_t i = 0; i < len; i++) s += (char)data[i];
    Serial.printf("[ble:evt<-] %s\n", s.c_str());
    onEvent(s);
}

static void dataNotifyCb(NimBLERemoteCharacteristic*, uint8_t* data, size_t len, bool) {
    int sep1 = -1, sep2 = -1;
    for (size_t i = 0; i < len; i++) {
        if (data[i] == '/' && sep1 < 0) sep1 = i;
        else if (data[i] == '|' && sep1 >= 0) { sep2 = i; break; }
    }
    if (sep1 < 0 || sep2 < 0) {
        Serial.printf("[ble:data] bad framing, len=%d\n", (int)len);
        return;
    }
    int seq   = atoi((const char*)data);
    int total = atoi((const char*)data + sep1 + 1);
    Serial.printf("[ble:data] seq=%d/%d len=%d\n", seq, total, (int)len);
    if (seq == 1) {
        resetDataAccum();
        dataExpectedTotal = total;
        // Pre-reserve to avoid heap thrash on large payloads (JPEGs).
        dataAccum.reserve((size_t)total * SNAPCHEF_DATA_FRAG_MAX);
    }
    if (seq != dataExpectedSeq || total != dataExpectedTotal) {
        resetDataAccum(); return;
    }
    int ps = sep2 + 1;
    dataAccum.concat((const char*)(data + ps), len - ps);
    dataExpectedSeq++;
    if (seq == total) { onData(dataAccum); resetDataAccum(); }
}

class CliCb : public NimBLEClientCallbacks {
    void onConnect(NimBLEClient*) override { Serial.println("[ble] connected"); }
    void onDisconnect(NimBLEClient*, int) override {
        bleConnected = false; bleShouldScan = true;
        Serial.println("[ble] disconnected");
    }
};

class ScanCb : public NimBLEScanCallbacks {
    void onResult(const NimBLEAdvertisedDevice* dev) override {
        if (dev->isAdvertisingService(NimBLEUUID(SNAPCHEF_SVC_UUID))) {
            NimBLEDevice::getScan()->stop();
            bleTargetAddr = dev->getAddress();
            bleHasTarget  = true;
        }
    }
    void onScanEnd(const NimBLEScanResults&, int) override {
        if (!bleConnected && !bleHasTarget) bleShouldScan = true;
    }
};

static void bleSendCmd(const String& json) {
    if (!bleConnected || !bleCmd) return;
    bleCmd->writeValue((uint8_t*)json.c_str(), json.length(), false);
}

static bool bleConnectTo(const NimBLEAddress& addr) {
    if (!bleCli) { bleCli = NimBLEDevice::createClient(); bleCli->setClientCallbacks(new CliCb(), false); }
    if (!bleCli->connect(addr)) return false;
    NimBLERemoteService* svc = bleCli->getService(SNAPCHEF_SVC_UUID);
    if (!svc) { bleCli->disconnect(); return false; }
    bleCmd  = svc->getCharacteristic(SNAPCHEF_CHR_CMD_UUID);
    bleEvt  = svc->getCharacteristic(SNAPCHEF_CHR_EVT_UUID);
    bleData = svc->getCharacteristic(SNAPCHEF_CHR_DATA_UUID);
    if (!bleCmd || !bleEvt || !bleData) { bleCli->disconnect(); return false; }
    if (bleEvt->canNotify())  bleEvt->subscribe(true,  evtNotifyCb);
    if (bleData->canNotify()) bleData->subscribe(true, dataNotifyCb);
    bleConnected = true;
    return true;
}

static void bleInit() {
    NimBLEDevice::init("SnapChef-Display");
    NimBLEDevice::setMTU(247);
    NimBLEDevice::setPower(9);
    NimBLEScan* scan = NimBLEDevice::getScan();
    scan->setScanCallbacks(new ScanCb(), false);
    scan->setActiveScan(true);
    scan->setInterval(80);
    scan->setWindow(40);
}

static void bleTick() {
    if (bleConnected) return;
    if (bleHasTarget) {
        NimBLEAddress addr = bleTargetAddr; bleHasTarget = false;
        if (!bleConnectTo(addr)) bleShouldScan = true;
        return;
    }
    if (bleShouldScan) {
        bleShouldScan = false;
        NimBLEDevice::getScan()->start(3000, false, true);
    }
}

// ============================================================================
//                                 UI globals
// ============================================================================

enum DisplayState {
    UI_IDLE, UI_ACTION_SELECT, UI_SUBMODE_SELECT,
    UI_SCANNING, UI_VEGGIE_RESULT, UI_RECEIPT_RESULT,
    UI_RECIPE_PROMPT, UI_RECIPE_RESULT, UI_MENU,
};

static DisplayState gUiState  = UI_IDLE;
static String       gPurpose  = "in";
static String       gScanKind = "veggie";

static lv_obj_t* scr_idle             = nullptr;
static lv_obj_t* scr_action           = nullptr;
static lv_obj_t* scr_submode          = nullptr;
static lv_obj_t* scr_scan             = nullptr;
static lv_obj_t* scr_veggie_result    = nullptr;
static lv_obj_t* scr_receipt_prep         = nullptr;
static lv_obj_t* scr_receipt_countdown    = nullptr;
static lv_obj_t* scr_receipt_photo_taken  = nullptr;
static lv_obj_t* scr_receipt_result       = nullptr;
static lv_obj_t* scr_recipe_prompt    = nullptr;
static lv_obj_t* scr_recipe_result    = nullptr;
static lv_obj_t* scr_menu             = nullptr;

static lv_obj_t*  scan_status_label = nullptr;
static lv_obj_t*  scan_label_dots   = nullptr;
static lv_timer_t* scan_anim_timer  = nullptr;
static int        scan_dots_phase   = 0;

static lv_obj_t* connection_dot = nullptr;

// Veggie result
static lv_obj_t* vr_name_label     = nullptr;
static lv_obj_t* vr_conf_label     = nullptr;
static lv_obj_t* vr_status_label   = nullptr;
static lv_obj_t* vr_action_btn     = nullptr;
static lv_obj_t* vr_action_btn_lbl = nullptr;
static lv_obj_t* vr_success_box    = nullptr;   // success overlay
static String    vr_pending_label;
static String    vr_pending_purpose;
static bool      vr_known = false;

// Take-out result (split screen)
static lv_obj_t* to_name_lbl    = nullptr;
static lv_obj_t* to_conf_lbl    = nullptr;
static lv_obj_t* to_recipe_list = nullptr;
static lv_obj_t* to_status_lbl  = nullptr;
static lv_obj_t* to_remove_btn  = nullptr;
static String    to_pending_name;

// Recipe result
static lv_obj_t* rec_title_lbl = nullptr;
static lv_obj_t* rec_time_lbl  = nullptr;
static lv_obj_t* rec_steps_obj = nullptr;

// Receipt countdown
static lv_obj_t*   countdown_num_lbl = nullptr;
static lv_timer_t* countdown_timer   = nullptr;
static int         countdown_value   = 3;

// Receipt result
struct ReceiptItem { String name; bool needs_refrig; bool checked; };
static std::vector<ReceiptItem> rcp_items;
static lv_obj_t* rcp_list_obj = nullptr;

// Menu
static lv_obj_t* menu_list_obj     = nullptr;
static lv_obj_t* menu_confirm_box  = nullptr;
static lv_obj_t* menu_count_lbl    = nullptr;
static int       menu_pending_delete = -1;


// ============================================================================
//                              UI helpers
// ============================================================================

static lv_obj_t* makeScreen() {
    lv_obj_t* s = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(s, COLOR_BG, 0);
    lv_obj_set_style_bg_opa(s, LV_OPA_COVER, 0);
    lv_obj_clear_flag(s, LV_OBJ_FLAG_SCROLLABLE);
    return s;
}

static lv_obj_t* makeCard(lv_obj_t* p, int x, int y, int w, int h) {
    lv_obj_t* c = lv_obj_create(p);
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

static lv_obj_t* makeLabel(lv_obj_t* p, const char* text,
                            const lv_font_t* font, lv_color_t color) {
    lv_obj_t* l = lv_label_create(p);
    lv_label_set_text(l, text);
    lv_obj_set_style_text_font(l, font, 0);
    lv_obj_set_style_text_color(l, color, 0);
    return l;
}

static lv_obj_t* makeButton(lv_obj_t* p, const char* text, lv_color_t bg,
                             lv_event_cb_t cb) {
    lv_obj_t* b = lv_btn_create(p);
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

// Standard header bar used by most screens
static lv_obj_t* makeHeader(lv_obj_t* scr, const char* title,
                             const char* subtitle, lv_color_t accent) {
    lv_obj_t* h = lv_obj_create(scr);
    lv_obj_set_size(h, 800, 72);
    lv_obj_set_pos(h, 0, 0);
    lv_obj_set_style_bg_color(h, COLOR_CARD, 0);
    lv_obj_set_style_radius(h, 0, 0);
    lv_obj_set_style_border_side(h, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(h, accent, 0);
    lv_obj_set_style_border_width(h, 2, 0);
    lv_obj_clear_flag(h, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* t = makeLabel(h, title, &lv_font_montserrat_28, accent);
    lv_obj_align(t, LV_ALIGN_LEFT_MID, 24, 0);
    if (subtitle && subtitle[0]) {
        lv_obj_t* s = makeLabel(h, subtitle, &lv_font_montserrat_14, COLOR_SUBTEXT);
        lv_obj_align(s, LV_ALIGN_RIGHT_MID, -24, 0);
    }
    return h;
}

static void switchScreen(lv_obj_t* target) {
    lv_scr_load_anim(target, LV_SCR_LOAD_ANIM_FADE_ON, 200, 0, false);
}

// Forward declarations
static void buildIdle();
static void buildAction();
static void buildSubmode();
static void buildScan();
static void buildVeggieResult();
static void buildTakeOutResult();
static void buildReceiptPrep();
static void buildReceiptCountdown();
static void buildReceiptPhotoTaken();
static void startReceiptCountdown();
static void stopReceiptCountdown();
static void buildReceiptResult();
static void buildRecipePrompt();
static void buildRecipeResult();
static void buildMenu();
static void rebuildMenu();
static void startScanAnim();
static void stopScanAnim();
static void showVeggieResult(const String&, float, const String&, bool);
static void showTakeOutResult(const String&, float);
static void showReceiptResult(const String&);
static void showRecipeResult(const String&);

// ============================================================================
//                              SCREEN: idle
// ============================================================================

static void onIdleStartCb(lv_event_t*) {
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
    lv_obj_align(title, LV_ALIGN_CENTER, 0, -80);

    lv_obj_t* tag = makeLabel(scr_idle, "Smart Fridge Assistant",
                              &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align_to(tag, title, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);

    lv_obj_t* div = lv_obj_create(scr_idle);
    lv_obj_set_size(div, 100, 2);
    lv_obj_set_style_bg_color(div, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(div, 0, 0);
    lv_obj_set_style_radius(div, 0, 0);
    lv_obj_align_to(div, tag, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);

    lv_obj_t* prompt = makeLabel(scr_idle,
        "Hold an ingredient near the sensor\nor tap to get started",
        &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(prompt, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(prompt, div, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);

    lv_obj_t* btn = makeButton(scr_idle, "Get Started", COLOR_ACCENT, onIdleStartCb);
    lv_obj_set_size(btn, 220, 52);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -48);

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

static void onActionPutInCb(lv_event_t*)   { gPurpose = "in";  gUiState = UI_SUBMODE_SELECT; switchScreen(scr_submode); }
static void onActionMenuCb(lv_event_t*)    { gUiState = UI_MENU; rebuildMenu(); switchScreen(scr_menu); }
static void onActionBackCb(lv_event_t*)    { gUiState = UI_IDLE; switchScreen(scr_idle); }

static void onActionTakeOutCb(lv_event_t*) {
    gPurpose = "out"; gScanKind = "veggie";
    gUiState = UI_SCANNING;
    bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"out\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Show the ingredient to remove");
    startScanAnim();
    switchScreen(scr_scan);
}

static void buildAction() {
    scr_action = makeScreen();
    makeHeader(scr_action, "SnapChef", "Choose an action", COLOR_ACCENT);

    // Three equal cards centred on screen
    const int cw = 210, ch = 290, gap = 28, cy = 96;
    const int total_w = cw * 3 + gap * 2;
    const int sx = (800 - total_w) / 2;

    struct { const char* icon; const char* title; const char* desc;
             const char* btn;  lv_color_t  col;   lv_event_cb_t cb; } cards[] = {
        { LV_SYMBOL_DOWNLOAD, "Put In\nFridge",     "Add items or\nscan a receipt",
          "Start",  COLOR_ACCENT,              onActionPutInCb },
        { LV_SYMBOL_UPLOAD,   "Take Out\nof Fridge","Scan to remove\nand get a recipe",
          "Scan",   COLOR_ACCENT2,             onActionTakeOutCb },
        { LV_SYMBOL_LIST,     "My\nFridge",         "Browse and remove\ningredients",
          "Open",   lv_color_hex(0x555555),    onActionMenuCb },
    };

    for (int i = 0; i < 3; i++) {
        lv_obj_t* c = makeCard(scr_action, sx + i * (cw + gap), cy, cw, ch);
        lv_obj_set_style_border_color(c, cards[i].col, 0);

        lv_obj_t* ic = makeLabel(c, cards[i].icon, &lv_font_montserrat_40, cards[i].col);
        lv_obj_align(ic, LV_ALIGN_TOP_MID, 0, 12);

        lv_obj_t* tt = makeLabel(c, cards[i].title, &lv_font_montserrat_20, COLOR_TEXT);
        lv_obj_set_style_text_align(tt, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_align_to(tt, ic, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);

        lv_obj_t* dd = makeLabel(c, cards[i].desc, &lv_font_montserrat_14, COLOR_SUBTEXT);
        lv_obj_set_style_text_align(dd, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_align_to(dd, tt, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

        lv_obj_t* bb = makeButton(c, cards[i].btn, cards[i].col, cards[i].cb);
        lv_obj_set_size(bb, cw - 32, 42);
        lv_obj_align(bb, LV_ALIGN_BOTTOM_MID, 0, 0);
    }

    lv_obj_t* back = makeButton(scr_action, LV_SYMBOL_LEFT " Back",
                                lv_color_hex(0x333333), onActionBackCb);
    lv_obj_set_size(back, 130, 40);
    lv_obj_align(back, LV_ALIGN_BOTTOM_LEFT, 24, -18);
}

// ============================================================================
//                            SCREEN: submode select
// ============================================================================

static void onSubmodeVeggieCb(lv_event_t*) {
    gScanKind = "veggie"; gUiState = UI_SCANNING;
    bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"in\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Hold the ingredient up to the camera");
    startScanAnim(); switchScreen(scr_scan);
}

static void onSubmodeReceiptCb(lv_event_t*) {
    gScanKind = "receipt";
    switchScreen(scr_receipt_prep);
}

static void onSubmodeBackCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }

static void buildSubmode() {
    scr_submode = makeScreen();
    makeHeader(scr_submode, "Put In Fridge", "What are you adding?", COLOR_ACCENT);

    const int cw = 270, ch = 290, gap = 40, cy = 96;
    const int sx = (800 - cw * 2 - gap) / 2;

    struct { const char* icon; const char* title; const char* desc;
             const char* btn;  lv_color_t col; lv_event_cb_t cb; } cards[] = {
        { LV_SYMBOL_EYE_OPEN, "Scan an Ingredient", "Hold one item up\nto the camera",
          "Scan",    COLOR_ACCENT,  onSubmodeVeggieCb },
        { LV_SYMBOL_FILE,     "Scan a Receipt",     "Point at the receipt\nand capture",
          "Capture", COLOR_ACCENT2, onSubmodeReceiptCb },
    };

    for (int i = 0; i < 2; i++) {
        lv_obj_t* c = makeCard(scr_submode, sx + i * (cw + gap), cy, cw, ch);
        lv_obj_set_style_border_color(c, cards[i].col, 0);

        lv_obj_t* ic = makeLabel(c, cards[i].icon, &lv_font_montserrat_40, cards[i].col);
        lv_obj_align(ic, LV_ALIGN_TOP_MID, 0, 12);

        lv_obj_t* tt = makeLabel(c, cards[i].title, &lv_font_montserrat_20, COLOR_TEXT);
        lv_obj_set_style_text_align(tt, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_align_to(tt, ic, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);

        lv_obj_t* dd = makeLabel(c, cards[i].desc, &lv_font_montserrat_14, COLOR_SUBTEXT);
        lv_obj_set_style_text_align(dd, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_align_to(dd, tt, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

        lv_obj_t* bb = makeButton(c, cards[i].btn, cards[i].col, cards[i].cb);
        lv_obj_set_size(bb, cw - 32, 42);
        lv_obj_align(bb, LV_ALIGN_BOTTOM_MID, 0, 0);
    }

    lv_obj_t* back = makeButton(scr_submode, LV_SYMBOL_LEFT " Back",
                                lv_color_hex(0x333333), onSubmodeBackCb);
    lv_obj_set_size(back, 130, 40);
    lv_obj_align(back, LV_ALIGN_BOTTOM_LEFT, 24, -18);
}

// ============================================================================
//                              SCREEN: scanning
// ============================================================================

static void scanAnimCb(lv_timer_t*) {
    if (!scan_label_dots) return;
    scan_dots_phase = (scan_dots_phase + 1) % 4;
    const char* dots[] = { "", ".", "..", "..." };
    lv_label_set_text(scan_label_dots, dots[scan_dots_phase]);
}

static void startScanAnim() {
    scan_dots_phase = 0;
    if (scan_anim_timer) lv_timer_del(scan_anim_timer);
    scan_anim_timer = lv_timer_create(scanAnimCb, 500, NULL);
}

static void stopScanAnim() {
    if (scan_anim_timer) { lv_timer_del(scan_anim_timer); scan_anim_timer = nullptr; }
}

static void onScanCancelCb(lv_event_t*) {
    bleSendCmd("{\"cmd\":\"cancel\"}");
    stopScanAnim();
    gUiState = UI_ACTION_SELECT;
    switchScreen(scr_action);
}

static void buildScan() {
    scr_scan = makeScreen();

    lv_obj_t* icon = makeLabel(scr_scan, LV_SYMBOL_REFRESH, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(icon, LV_ALIGN_CENTER, 0, -90);

    // "Scanning" and animated dots on same line
    lv_obj_t* lbl = makeLabel(scr_scan, "Scanning", &lv_font_montserrat_28, COLOR_TEXT);
    lv_obj_align_to(lbl, icon, LV_ALIGN_OUT_BOTTOM_MID, -20, 20);

    scan_label_dots = makeLabel(scr_scan, "", &lv_font_montserrat_28, COLOR_ACCENT);
    lv_obj_align_to(scan_label_dots, lbl, LV_ALIGN_OUT_RIGHT_MID, 4, 0);

    scan_status_label = makeLabel(scr_scan, "Please hold still",
                                  &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(scan_status_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(scan_status_label, LV_ALIGN_CENTER, 0, 10);

    lv_obj_t* tip = makeLabel(scr_scan, "Up to 20 seconds",
                              &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(tip, LV_ALIGN_CENTER, 0, 40);

    lv_obj_t* cancel = makeButton(scr_scan, "Cancel", lv_color_hex(0x444444), onScanCancelCb);
    lv_obj_set_size(cancel, 180, 50);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, 0, -36);
}

// ============================================================================
//  SCREEN: veggie result  (PUT IN flow — shows name + confidence + success box)
// ============================================================================

// Success overlay callbacks
static void vrSuccessBackCb(lv_event_t*) {
    if (vr_success_box) { lv_obj_del(vr_success_box); vr_success_box = nullptr; }
    gUiState = UI_ACTION_SELECT; switchScreen(scr_action);
}
static void vrSuccessMenuCb(lv_event_t*) {
    if (vr_success_box) { lv_obj_del(vr_success_box); vr_success_box = nullptr; }
    gUiState = UI_MENU; rebuildMenu(); switchScreen(scr_menu);
}

static void showVeggieSuccessBox(const String& name) {
    vr_success_box = lv_obj_create(scr_veggie_result);
    lv_obj_set_size(vr_success_box, 480, 220);
    lv_obj_center(vr_success_box);
    lv_obj_set_style_bg_color(vr_success_box, lv_color_hex(0x182E28), 0);
    lv_obj_set_style_border_color(vr_success_box, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(vr_success_box, 2, 0);
    lv_obj_set_style_radius(vr_success_box, 14, 0);
    lv_obj_clear_flag(vr_success_box, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* ic = makeLabel(vr_success_box, LV_SYMBOL_OK, &lv_font_montserrat_36, COLOR_ACCENT);
    lv_obj_align(ic, LV_ALIGN_TOP_MID, 0, 20);

    char msg[64];
    snprintf(msg, sizeof(msg), "%s added to fridge!", name.c_str());
    lv_obj_t* lbl = makeLabel(vr_success_box, msg, &lv_font_montserrat_20, COLOR_TEXT);
    lv_obj_set_style_text_align(lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(lbl, ic, LV_ALIGN_OUT_BOTTOM_MID, 0, 12);

    // Two buttons centred with equal spacing
    lv_obj_t* back = makeButton(vr_success_box, LV_SYMBOL_LEFT " Back",
                                lv_color_hex(0x444444), vrSuccessBackCb);
    lv_obj_set_size(back, 180, 46);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, -100, -16);

    lv_obj_t* menu = makeButton(vr_success_box, LV_SYMBOL_LIST " My Fridge",
                                COLOR_ACCENT, vrSuccessMenuCb);
    lv_obj_set_size(menu, 180, 46);
    lv_obj_align(menu, LV_ALIGN_BOTTOM_MID, 100, -16);
}

static void vrRetryCb(lv_event_t*) {
    if (gPurpose == "out") {
        bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"out\"}");
        if (scan_status_label) lv_label_set_text(scan_status_label, "Show the ingredient to remove");
    } else {
        bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"in\"}");
        if (scan_status_label) lv_label_set_text(scan_status_label, "Hold the ingredient up to the camera");
    }
    gUiState = UI_SCANNING; startScanAnim(); switchScreen(scr_scan);
}

static void vrCancelCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }

// Add to fridge and show success overlay
static void vrConfirmInCb(lv_event_t*) {
    inventoryAdd(vr_pending_label);
    showVeggieSuccessBox(vr_pending_label);
}

static void buildVeggieResult() {
    scr_veggie_result = makeScreen();

    vr_status_label = makeLabel(scr_veggie_result, LV_SYMBOL_OK " Ingredient Detected",
                                &lv_font_montserrat_22, COLOR_ACCENT);
    lv_obj_align(vr_status_label, LV_ALIGN_TOP_LEFT, 24, 18);

    // Divider line
    lv_obj_t* line = lv_obj_create(scr_veggie_result);
    lv_obj_set_size(line, 752, 2);
    lv_obj_set_pos(line, 24, 52);
    lv_obj_set_style_bg_color(line, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(line, 0, 0);
    lv_obj_set_style_radius(line, 0, 0);

    // Centred result card
    lv_obj_t* card = makeCard(scr_veggie_result, 250, 78, 300, 240);
    lv_obj_set_style_border_color(card, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(card, 2, 0);

    vr_name_label = makeLabel(card, "—", &lv_font_montserrat_36, COLOR_TEXT);
    lv_obj_set_width(vr_name_label, LV_PCT(100));
    lv_obj_set_style_text_align(vr_name_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(vr_name_label, LV_ALIGN_CENTER, 0, -16);

    vr_conf_label = makeLabel(card, "", &lv_font_montserrat_18, COLOR_SUBTEXT);
    lv_obj_set_width(vr_conf_label, LV_PCT(100));
    lv_obj_set_style_text_align(vr_conf_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(vr_conf_label, LV_ALIGN_CENTER, 0, 30);

    // Three bottom buttons with equal spacing
    const int bw = 190, bh = 50, by = -28;
    vr_action_btn = makeButton(scr_veggie_result, "Add to Fridge", COLOR_ACCENT, vrConfirmInCb);
    lv_obj_set_size(vr_action_btn, bw, bh);
    lv_obj_align(vr_action_btn, LV_ALIGN_BOTTOM_MID, -(bw + 16), by);
    vr_action_btn_lbl = lv_obj_get_child(vr_action_btn, 0);

    lv_obj_t* retry = makeButton(scr_veggie_result, LV_SYMBOL_REFRESH " Retry",
                                  lv_color_hex(0x444444), vrRetryCb);
    lv_obj_set_size(retry, bw, bh);
    lv_obj_align(retry, LV_ALIGN_BOTTOM_MID, 0, by);

    lv_obj_t* cancel = makeButton(scr_veggie_result, "Cancel",
                                   lv_color_hex(0x333333), vrCancelCb);
    lv_obj_set_size(cancel, bw, bh);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, (bw + 16), by);
}

static void showVeggieResult(const String& label, float conf,
                              const String& purpose, bool known) {
    vr_pending_label   = label;
    vr_pending_purpose = purpose;
    vr_known           = known;

    if (known) {
        lv_label_set_text(vr_name_label, label.c_str());
        char b[48]; snprintf(b, sizeof(b), "Confidence: %.0f%%", conf * 100.0f);
        lv_label_set_text(vr_conf_label, b);
        lv_label_set_text(vr_status_label, LV_SYMBOL_OK " Ingredient Detected");
        lv_obj_set_style_text_color(vr_status_label, COLOR_ACCENT, 0);

        // Put In flow: "Add to Fridge" button
        lv_label_set_text(vr_action_btn_lbl, LV_SYMBOL_DOWNLOAD " Add to Fridge");
        lv_obj_set_style_bg_color(vr_action_btn, COLOR_ACCENT, 0);
        lv_obj_remove_event_cb(vr_action_btn, vrConfirmInCb);
        lv_obj_add_event_cb(vr_action_btn, vrConfirmInCb, LV_EVENT_CLICKED, NULL);
        lv_obj_clear_flag(vr_action_btn, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_label_set_text(vr_name_label, "Unknown");
        lv_label_set_text(vr_conf_label, "Could not recognise the item");
        lv_label_set_text(vr_status_label, LV_SYMBOL_WARNING " Not Recognised");
        lv_obj_set_style_text_color(vr_status_label, COLOR_ACCENT2, 0);
        lv_obj_add_flag(vr_action_btn, LV_OBJ_FLAG_HIDDEN);
    }
    gUiState = UI_VEGGIE_RESULT;
    switchScreen(scr_veggie_result);
}

// ============================================================================
//  SCREEN: take-out result  (split: left = scan result, right = recipe list)
// ============================================================================

static void toBackCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }
static void toRetryCb(lv_event_t*) {
    bleSendCmd("{\"cmd\":\"start_veggie_scan\",\"purpose\":\"out\"}");
    if (scan_status_label) lv_label_set_text(scan_status_label, "Show the ingredient to remove");
    gUiState = UI_SCANNING; startScanAnim(); switchScreen(scr_scan);
}

static void toRemoveCb(lv_event_t*) {
    if (!to_pending_name.length()) return;
    inventoryRemove(to_pending_name);
    if (to_status_lbl) {
        lv_label_set_text(to_status_lbl, LV_SYMBOL_OK " Removed from fridge");
        lv_obj_set_style_text_color(to_status_lbl, COLOR_OK, 0);
    }
    if (to_remove_btn) lv_obj_add_flag(to_remove_btn, LV_OBJ_FLAG_HIDDEN);
    to_pending_name = "";
}

// Tapping a recipe row on the right panel — fetch steps
static void toRecipeTapCb(lv_event_t* e) {
    const char* name = (const char*)lv_event_get_user_data(e);
    // Ask XIAO to generate full recipe steps for this dish + current fridge
    String cmd = "{\"cmd\":\"get_recipe_steps\",\"dish\":\"";
    cmd += name; cmd += "\",\"ingredients\":[";
    bool first = true;
    for (auto& f : gFridge) {
        if (!first) cmd += ','; cmd += '"'; cmd += f; cmd += '"'; first = false;
    }
    cmd += "]}";
    bleSendCmd(cmd);
    if (scan_status_label) lv_label_set_text(scan_status_label, "Getting recipe steps");
    gUiState = UI_SCANNING; startScanAnim(); switchScreen(scr_scan);
}

static lv_obj_t* scr_takeout_result = nullptr;

static void buildTakeOutResult() {
    scr_takeout_result = makeScreen();

    // Header
    lv_obj_t* h = lv_obj_create(scr_takeout_result);
    lv_obj_set_size(h, 800, 56); lv_obj_set_pos(h, 0, 0);
    lv_obj_set_style_bg_color(h, COLOR_CARD, 0);
    lv_obj_set_style_radius(h, 0, 0);
    lv_obj_set_style_border_side(h, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(h, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(h, 2, 0);
    lv_obj_clear_flag(h, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* ht = makeLabel(h, LV_SYMBOL_OK " Ingredient Detected",
                              &lv_font_montserrat_20, COLOR_ACCENT);
    lv_obj_align(ht, LV_ALIGN_LEFT_MID, 20, 0);

    // Left panel: scan result (smaller proportions)
    lv_obj_t* left = makeCard(scr_takeout_result, 60, 70, 280, 250);
    lv_obj_set_style_border_color(left, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(left, 2, 0);

    lv_obj_t* det = makeLabel(left, "Detected", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(det, LV_ALIGN_TOP_MID, 0, 0);

    lv_obj_t* img_icon = makeLabel(left, LV_SYMBOL_IMAGE,
                                    &lv_font_montserrat_28, COLOR_ACCENT);
    lv_obj_align(img_icon, LV_ALIGN_CENTER, 0, -50);

    to_name_lbl = makeLabel(left, "—", &lv_font_montserrat_28, COLOR_TEXT);
    lv_obj_set_width(to_name_lbl, LV_PCT(100));
    lv_obj_set_style_text_align(to_name_lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(to_name_lbl, LV_ALIGN_CENTER, 0, 0);

    to_conf_lbl = makeLabel(left, "", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_width(to_conf_lbl, LV_PCT(100));
    lv_obj_set_style_text_align(to_conf_lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(to_conf_lbl, LV_ALIGN_CENTER, 0, 32);

    to_status_lbl = makeLabel(left, "", &lv_font_montserrat_12, COLOR_SUBTEXT);
    lv_obj_set_width(to_status_lbl, LV_PCT(100));
    lv_obj_set_style_text_align(to_status_lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(to_status_lbl, LV_ALIGN_BOTTOM_MID, 0, -2);

    // Right panel: recipe suggestions
    lv_obj_t* right = makeCard(scr_takeout_result, 360, 70, 380, 250);
    lv_obj_set_style_border_color(right, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(right, 2, 0);
    lv_obj_set_style_pad_all(right, 12, 0);

    lv_obj_t* rh = makeLabel(right, LV_SYMBOL_OK " Try tonight",
                              &lv_font_montserrat_16, COLOR_ACCENT2);
    lv_obj_align(rh, LV_ALIGN_TOP_MID, 0, 0);

    to_recipe_list = lv_obj_create(right);
    lv_obj_set_size(to_recipe_list, 354, 192);
    lv_obj_align(to_recipe_list, LV_ALIGN_TOP_MID, 0, 26);
    lv_obj_set_style_bg_color(to_recipe_list, lv_color_hex(0x1A1A1A), 0);
    lv_obj_set_style_border_width(to_recipe_list, 0, 0);
    lv_obj_set_style_pad_all(to_recipe_list, 4, 0);
    lv_obj_set_style_pad_row(to_recipe_list, 6, 0);
    lv_obj_set_flex_flow(to_recipe_list, LV_FLEX_FLOW_COLUMN);

    // Bottom buttons
    to_remove_btn = makeButton(scr_takeout_result, LV_SYMBOL_TRASH " Remove",
                                COLOR_DELETE, toRemoveCb);
    lv_obj_set_size(to_remove_btn, 180, 46);
    lv_obj_align(to_remove_btn, LV_ALIGN_BOTTOM_MID, -210, -14);

    lv_obj_t* retry = makeButton(scr_takeout_result, LV_SYMBOL_REFRESH " Retry",
                                  lv_color_hex(0x444444), toRetryCb);
    lv_obj_set_size(retry, 160, 46);
    lv_obj_align(retry, LV_ALIGN_BOTTOM_MID, 0, -14);

    lv_obj_t* back = makeButton(scr_takeout_result, LV_SYMBOL_LEFT " Back",
                                 lv_color_hex(0x333333), toBackCb);
    lv_obj_set_size(back, 160, 46);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, 210, -14);
}

// dish_names: pipe-separated list, e.g. "Caprese Salad|Tomato Omelette|Pasta Pomodoro"
static void populateTakeOutRecipes(const String& dish_names) {
    lv_obj_clean(to_recipe_list);
    int start = 0;
    while (start <= (int)dish_names.length()) {
        int sep = dish_names.indexOf('|', start);
        if (sep < 0) sep = dish_names.length();
        String dish = dish_names.substring(start, sep);
        dish.trim();
        if (dish.length()) {
            lv_obj_t* row = lv_obj_create(to_recipe_list);
            lv_obj_set_size(row, 344, 56);
            lv_obj_set_style_bg_color(row, lv_color_hex(0x222222), 0);
            lv_obj_set_style_border_color(row, lv_color_hex(0x444444), 0);
            lv_obj_set_style_border_side(row, LV_BORDER_SIDE_LEFT, 0);
            lv_obj_set_style_border_width(row, 3, 0);
            lv_obj_set_style_radius(row, 8, 0);
            lv_obj_set_style_pad_all(row, 10, 0);
            lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
            lv_obj_add_flag(row, LV_OBJ_FLAG_CLICKABLE);
            lv_obj_set_style_bg_color(row, lv_color_hex(0x2A2A2A), LV_STATE_PRESSED);

            // Copy dish name to heap so user_data pointer stays valid
            char* name_copy = (char*)malloc(dish.length() + 1);
            strcpy(name_copy, dish.c_str());

            lv_obj_t* lbl = makeLabel(row, name_copy, &lv_font_montserrat_18, COLOR_TEXT);
            lv_obj_align(lbl, LV_ALIGN_LEFT_MID, 0, 0);

            lv_obj_t* arr = makeLabel(row, LV_SYMBOL_RIGHT, &lv_font_montserrat_16, COLOR_ACCENT2);
            lv_obj_align(arr, LV_ALIGN_RIGHT_MID, -4, 0);

            lv_obj_add_event_cb(row, toRecipeTapCb, LV_EVENT_CLICKED, (void*)name_copy);
        }
        start = sep + 1;
    }
}

// "Loading recipes..." placeholder while waiting for the LLM list to arrive.
static void showTakeOutRecipesLoading() {
    if (!to_recipe_list) return;
    lv_obj_clean(to_recipe_list);
    lv_obj_t* lbl = makeLabel(to_recipe_list, "Loading recipes...",
                               &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(lbl, LV_ALIGN_CENTER, 0, 0);
}

// Ask main to generate a dish list for the just-detected veggie + fridge.
// Result arrives on the data char as a recipe_list event (see onData).
static void requestTakeOutRecipes(const String& trigger) {
    String cmd = "{\"cmd\":\"get_recipe_list\",\"trigger\":\"";
    cmd += trigger;
    cmd += "\",\"ingredients\":[";
    bool first = true;
    // Include the trigger so the LLM knows what's on the counter, plus the
    // fridge contents (skip dup if already inventoried).
    cmd += '"'; cmd += trigger; cmd += '"'; first = false;
    for (auto& f : gFridge) {
        if (f.equalsIgnoreCase(trigger)) continue;
        if (!first) cmd += ',';
        cmd += '"'; cmd += f; cmd += '"';
        first = false;
    }
    cmd += "]}";
    bleSendCmd(cmd);
}

static void showTakeOutResult(const String& label, float conf) {
    to_pending_name = label;

    lv_label_set_text(to_name_lbl, label.c_str());
    char b[48]; snprintf(b, sizeof(b), "Confidence: %.0f%%", conf * 100.0f);
    lv_label_set_text(to_conf_lbl, b);

    // Reset status and Remove button for a fresh result
    if (to_status_lbl) {
        lv_label_set_text(to_status_lbl, "Tap Remove to take it out");
        lv_obj_set_style_text_color(to_status_lbl, COLOR_SUBTEXT, 0);
    }
    if (to_remove_btn) lv_obj_clear_flag(to_remove_btn, LV_OBJ_FLAG_HIDDEN);

    // Show the loading state in the right panel, then kick off the LLM call.
    showTakeOutRecipesLoading();
    requestTakeOutRecipes(label);

    gUiState = UI_VEGGIE_RESULT;   // reuse state slot
    switchScreen(scr_takeout_result);
}

// ============================================================================
//                          SCREEN: receipt prep
//   Tells the user how to position the receipt before the camera fires.
//   Capture button sends the actual capture_receipt command.
// ============================================================================

static void onPrepCaptureCb(lv_event_t*) {
    // Run the 3-2-1 countdown first; capture_receipt is only sent when the
    // countdown reaches 0 (see countdownTickCb).
    switchScreen(scr_receipt_countdown);
    startReceiptCountdown();
}

static void onPrepCancelCb(lv_event_t*) {
    // If we landed here from review (Retake), main is holding a frame — drop it.
    bleSendCmd("{\"cmd\":\"cancel\"}");
    gUiState = UI_SUBMODE_SELECT;
    switchScreen(scr_submode);
}

static void buildReceiptPrep() {
    scr_receipt_prep = makeScreen();

    // Header
    lv_obj_t* h = lv_obj_create(scr_receipt_prep);
    lv_obj_set_size(h, 800, 56); lv_obj_set_pos(h, 0, 0);
    lv_obj_set_style_bg_color(h, COLOR_CARD, 0);
    lv_obj_set_style_radius(h, 0, 0);
    lv_obj_set_style_border_side(h, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(h, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(h, 2, 0);
    lv_obj_clear_flag(h, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* ht = makeLabel(h, LV_SYMBOL_FILE " Scan a Receipt",
                              &lv_font_montserrat_22, COLOR_ACCENT2);
    lv_obj_align(ht, LV_ALIGN_LEFT_MID, 20, 0);

    // Big camera/photo icon
    lv_obj_t* icon = makeLabel(scr_receipt_prep, LV_SYMBOL_IMAGE,
                                &lv_font_montserrat_48, COLOR_ACCENT2);
    lv_obj_align(icon, LV_ALIGN_CENTER, 0, -110);

    // Main instruction (centred)
    lv_obj_t* instr = makeLabel(scr_receipt_prep,
        "Place the receipt about 10 cm\nin front of the camera lens",
        &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(instr, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(instr, LV_ALIGN_CENTER, 0, -20);

    // Sub-hint
    lv_obj_t* hint = makeLabel(scr_receipt_prep,
        "Hold steady, then tap Capture",
        &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(hint, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(hint, LV_ALIGN_CENTER, 0, 70);

    // Buttons
    lv_obj_t* cancel = makeButton(scr_receipt_prep, "Cancel",
                                   lv_color_hex(0x444444), onPrepCancelCb);
    lv_obj_set_size(cancel, 180, 50);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, -110, -16);

    lv_obj_t* cap = makeButton(scr_receipt_prep, LV_SYMBOL_OK " Capture",
                                COLOR_ACCENT2, onPrepCaptureCb);
    lv_obj_set_size(cap, 180, 50);
    lv_obj_align(cap, LV_ALIGN_BOTTOM_MID, 110, -16);
}

// ============================================================================
//                       SCREEN: receipt countdown (3,2,1)
//   Visual countdown that runs *before* the camera fires. capture_receipt is
//   only sent once the countdown hits 0, then we switch to the photo-taken
//   transition screen.
// ============================================================================

static void stopReceiptCountdown() {
    if (countdown_timer) { lv_timer_del(countdown_timer); countdown_timer = nullptr; }
}

static void countdownTickCb(lv_timer_t* t) {
    countdown_value--;
    if (countdown_value > 0) {
        char b[4]; snprintf(b, sizeof(b), "%d", countdown_value);
        if (countdown_num_lbl) lv_label_set_text(countdown_num_lbl, b);
        return;
    }
    // Done — fire the camera and transition.
    stopReceiptCountdown();
    bleSendCmd("{\"cmd\":\"capture_receipt\"}");
    switchScreen(scr_receipt_photo_taken);
}

static void startReceiptCountdown() {
    countdown_value = 3;
    if (countdown_num_lbl) lv_label_set_text(countdown_num_lbl, "3");
    stopReceiptCountdown();
    countdown_timer = lv_timer_create(countdownTickCb, 1000, NULL);
}

static void onCountdownCancelCb(lv_event_t*) {
    stopReceiptCountdown();
    switchScreen(scr_receipt_prep);
}

static void buildReceiptCountdown() {
    scr_receipt_countdown = makeScreen();

    // Header
    lv_obj_t* h = lv_obj_create(scr_receipt_countdown);
    lv_obj_set_size(h, 800, 56); lv_obj_set_pos(h, 0, 0);
    lv_obj_set_style_bg_color(h, COLOR_CARD, 0);
    lv_obj_set_style_radius(h, 0, 0);
    lv_obj_set_style_border_side(h, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(h, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(h, 2, 0);
    lv_obj_clear_flag(h, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* ht = makeLabel(h, LV_SYMBOL_FILE " Get Ready",
                              &lv_font_montserrat_22, COLOR_ACCENT2);
    lv_obj_align(ht, LV_ALIGN_LEFT_MID, 20, 0);

    // Upper hint
    lv_obj_t* hint = makeLabel(scr_receipt_countdown,
        "Hold the receipt steady",
        &lv_font_montserrat_20, COLOR_TEXT);
    lv_obj_align(hint, LV_ALIGN_CENTER, 0, -120);

    // "Capturing in" sub-label
    lv_obj_t* sub = makeLabel(scr_receipt_countdown, "Capturing in",
                               &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_CENTER, 0, -50);

    // Big countdown number (scaled up via transform for more impact)
    countdown_num_lbl = makeLabel(scr_receipt_countdown, "3",
                                    &lv_font_montserrat_48, COLOR_ACCENT2);
    lv_obj_set_style_transform_zoom(countdown_num_lbl, 512, 0);  // 2x
    lv_obj_set_style_transform_pivot_x(countdown_num_lbl, 0, 0);
    lv_obj_set_style_transform_pivot_y(countdown_num_lbl, 0, 0);
    lv_obj_align(countdown_num_lbl, LV_ALIGN_CENTER, 0, 30);

    // Cancel button (in case the user changes their mind in 3 s)
    lv_obj_t* cancel = makeButton(scr_receipt_countdown, "Cancel",
                                   lv_color_hex(0x444444), onCountdownCancelCb);
    lv_obj_set_size(cancel, 180, 50);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, 0, -20);
}

// ============================================================================
//                  SCREEN: photo taken (post-capture transition)
//   Shown after the countdown fires capture_receipt, until the result data
//   arrives. Tells the user the camera has done its bit and they can put the
//   receipt down while the OCR call runs.
// ============================================================================

static void buildReceiptPhotoTaken() {
    scr_receipt_photo_taken = makeScreen();

    // Header
    lv_obj_t* h = lv_obj_create(scr_receipt_photo_taken);
    lv_obj_set_size(h, 800, 56); lv_obj_set_pos(h, 0, 0);
    lv_obj_set_style_bg_color(h, COLOR_CARD, 0);
    lv_obj_set_style_radius(h, 0, 0);
    lv_obj_set_style_border_side(h, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(h, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(h, 2, 0);
    lv_obj_clear_flag(h, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* ht = makeLabel(h, LV_SYMBOL_OK " Photo Taken",
                              &lv_font_montserrat_22, COLOR_ACCENT);
    lv_obj_align(ht, LV_ALIGN_LEFT_MID, 20, 0);

    // Big checkmark
    lv_obj_t* icon = makeLabel(scr_receipt_photo_taken, LV_SYMBOL_OK,
                                &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_set_style_transform_zoom(icon, 384, 0);   // 1.5x
    lv_obj_set_style_transform_pivot_x(icon, 0, 0);
    lv_obj_set_style_transform_pivot_y(icon, 0, 0);
    lv_obj_align(icon, LV_ALIGN_CENTER, 0, -70);

    // Headline
    lv_obj_t* title = makeLabel(scr_receipt_photo_taken, "Photo Taken",
                                 &lv_font_montserrat_28, COLOR_TEXT);
    lv_obj_align(title, LV_ALIGN_CENTER, 0, 20);

    // Sub-message
    lv_obj_t* sub = makeLabel(scr_receipt_photo_taken,
        "You can put the receipt down\nReading receipt...",
        &lv_font_montserrat_18, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(sub, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(sub, LV_ALIGN_CENTER, 0, 90);
}

// ============================================================================
//                          SCREEN: receipt result
// ============================================================================

static void rcpItemToggleCb(lv_event_t* e) {
    int idx = (int)(intptr_t)lv_event_get_user_data(e);
    if (idx < 0 || idx >= (int)rcp_items.size()) return;
    rcp_items[idx].checked = lv_obj_has_state((lv_obj_t*)lv_event_get_target(e), LV_STATE_CHECKED);
}

static void rcpConfirmCb(lv_event_t*) {
    int added = 0;
    for (auto& it : rcp_items) if (it.checked && it.name.length()) { inventoryAdd(it.name); added++; }
    Serial.printf("[receipt] added %d items\n", added);
    gUiState = UI_ACTION_SELECT; switchScreen(scr_action);
}

static void rcpCancelCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }

static void rcpRetakeCb(lv_event_t*) {
    // Send the user back to the prep screen so they can reposition the
    // receipt and tap Capture again, instead of firing the camera immediately.
    switchScreen(scr_receipt_prep);
}

static void buildReceiptResult() {
    scr_receipt_result = makeScreen();
    makeHeader(scr_receipt_result, LV_SYMBOL_FILE " Receipt Items",
               "Tick what you put in the fridge", COLOR_ACCENT2);

    rcp_list_obj = lv_obj_create(scr_receipt_result);
    lv_obj_set_size(rcp_list_obj, 752, 310);
    lv_obj_align(rcp_list_obj, LV_ALIGN_TOP_MID, 0, 82);
    lv_obj_set_style_bg_color(rcp_list_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(rcp_list_obj, 0, 0);
    lv_obj_set_style_pad_all(rcp_list_obj, 6, 0);
    lv_obj_set_style_pad_row(rcp_list_obj, 6, 0);
    lv_obj_set_flex_flow(rcp_list_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* cancel = makeButton(scr_receipt_result, "Cancel",
                                   lv_color_hex(0x333333), rcpCancelCb);
    lv_obj_set_size(cancel, 160, 48);
    lv_obj_align(cancel, LV_ALIGN_BOTTOM_MID, -210, -16);

    lv_obj_t* retake = makeButton(scr_receipt_result,
                                   LV_SYMBOL_REFRESH " Retake",
                                   lv_color_hex(0x555555), rcpRetakeCb);
    lv_obj_set_size(retake, 160, 48);
    lv_obj_align(retake, LV_ALIGN_BOTTOM_MID, 0, -16);

    lv_obj_t* confirm = makeButton(scr_receipt_result, LV_SYMBOL_OK " Confirm",
                                    COLOR_ACCENT, rcpConfirmCb);
    lv_obj_set_size(confirm, 180, 48);
    lv_obj_align(confirm, LV_ALIGN_BOTTOM_MID, 210, -16);
}

static void showReceiptResult(const String& json) {
    rcp_items.clear();
    DynamicJsonDocument doc(8192);
    if (deserializeJson(doc, json)) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); return; }
    for (JsonObject it : doc["items"].as<JsonArray>()) {
        ReceiptItem r;
        r.name        = (const char*)(it["name"] | "");
        r.needs_refrig = it["needs_refrigeration"] | false;
        r.checked      = it["checked"] | r.needs_refrig;
        if (r.name.length()) rcp_items.push_back(r);
    }
    lv_obj_clean(rcp_list_obj);
    for (size_t i = 0; i < rcp_items.size(); i++) {
        lv_obj_t* row = lv_obj_create(rcp_list_obj);
        lv_obj_set_size(row, 730, 50);
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
        lv_obj_add_event_cb(cb, rcpItemToggleCb, LV_EVENT_VALUE_CHANGED, (void*)(intptr_t)i);

        if (rcp_items[i].needs_refrig) {
            lv_obj_t* tag = makeLabel(row, LV_SYMBOL_OK " fridge",
                                       &lv_font_montserrat_12, COLOR_ACCENT);
            lv_obj_align(tag, LV_ALIGN_RIGHT_MID, -8, 0);
        }
    }
    gUiState = UI_RECEIPT_RESULT; switchScreen(scr_receipt_result);
}

// ============================================================================
//                    SCREEN: recipe prompt (unused in new flow but kept)
// ============================================================================

static void buildRecipePrompt() { scr_recipe_prompt = makeScreen(); }

// ============================================================================
//                          SCREEN: recipe result (steps)
// ============================================================================

static void recDoneCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }

static void buildRecipeResult() {
    scr_recipe_result = makeScreen();
    makeHeader(scr_recipe_result, "Recipe", "", COLOR_ACCENT2);

    rec_title_lbl = makeLabel(scr_recipe_result, "Recipe", &lv_font_montserrat_22, COLOR_ACCENT2);
    lv_obj_align(rec_title_lbl, LV_ALIGN_TOP_LEFT, 24, 18);

    rec_time_lbl = makeLabel(scr_recipe_result, "", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(rec_time_lbl, LV_ALIGN_TOP_RIGHT, -24, 22);

    rec_steps_obj = lv_obj_create(scr_recipe_result);
    lv_obj_set_size(rec_steps_obj, 752, 320);
    lv_obj_align(rec_steps_obj, LV_ALIGN_TOP_MID, 0, 80);
    lv_obj_set_style_bg_color(rec_steps_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(rec_steps_obj, 0, 0);
    lv_obj_set_style_pad_all(rec_steps_obj, 6, 0);
    lv_obj_set_style_pad_row(rec_steps_obj, 10, 0);
    lv_obj_set_flex_flow(rec_steps_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* done = makeButton(scr_recipe_result, LV_SYMBOL_OK " Done",
                                 COLOR_ACCENT, recDoneCb);
    lv_obj_set_size(done, 200, 50);
    lv_obj_align(done, LV_ALIGN_BOTTOM_MID, 0, -16);
}

static void showRecipeResult(const String& json) {
    DynamicJsonDocument doc(8192);
    if (deserializeJson(doc, json)) return;
    String title = (const char*)(doc["title"] | "Recipe");
    int    tmin  = doc["time_min"] | 0;

    lv_label_set_text(rec_title_lbl, title.c_str());
    char tb[24];
    snprintf(tb, sizeof(tb), tmin > 0 ? LV_SYMBOL_REFRESH " ~%d min" : "", tmin);
    lv_label_set_text(rec_time_lbl, tb);

    lv_obj_clean(rec_steps_obj);
    int i = 1;
    for (JsonVariant s : doc["steps"].as<JsonArray>()) {
        const char* text = s | "";
        lv_obj_t* row = lv_obj_create(rec_steps_obj);
        lv_obj_set_size(row, 730, LV_SIZE_CONTENT);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, COLOR_ACCENT2, 0);
        lv_obj_set_style_border_side(row, LV_BORDER_SIDE_LEFT, 0);
        lv_obj_set_style_border_width(row, 3, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_set_style_pad_all(row, 12, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);

        char num[8]; snprintf(num, sizeof(num), "%d.", i++);
        lv_obj_t* nl = makeLabel(row, num, &lv_font_montserrat_18, COLOR_ACCENT);
        lv_obj_align(nl, LV_ALIGN_LEFT_MID, 0, 0);

        lv_obj_t* tl = makeLabel(row, text, &lv_font_montserrat_16, COLOR_TEXT);
        lv_label_set_long_mode(tl, LV_LABEL_LONG_WRAP);
        lv_obj_set_width(tl, 650);
        lv_obj_align(tl, LV_ALIGN_LEFT_MID, 32, 0);
    }
    gUiState = UI_RECIPE_RESULT; switchScreen(scr_recipe_result);
}

// ============================================================================
//                    SCREEN: my fridge menu  (with hold progress bar)
// ============================================================================

static void menuConfirmDeleteCb(lv_event_t*) {
    if (menu_pending_delete >= 0) { inventoryRemoveAt(menu_pending_delete); menu_pending_delete = -1; }
    if (menu_confirm_box) { lv_obj_del(menu_confirm_box); menu_confirm_box = nullptr; }
    rebuildMenu();
}

static void menuCancelDeleteCb(lv_event_t*) {
    menu_pending_delete = -1;
    if (menu_confirm_box) { lv_obj_del(menu_confirm_box); menu_confirm_box = nullptr; }
}

static void showMenuConfirmDelete(int idx) {
    menu_pending_delete = idx;
    menu_confirm_box = lv_obj_create(lv_scr_act());
    lv_obj_set_size(menu_confirm_box, 420, 200);
    lv_obj_center(menu_confirm_box);
    lv_obj_set_style_bg_color(menu_confirm_box, lv_color_hex(0x1E1E1E), 0);
    lv_obj_set_style_border_color(menu_confirm_box, COLOR_DELETE, 0);
    lv_obj_set_style_border_width(menu_confirm_box, 2, 0);
    lv_obj_set_style_radius(menu_confirm_box, 14, 0);
    lv_obj_clear_flag(menu_confirm_box, LV_OBJ_FLAG_SCROLLABLE);

    char msg[80];
    snprintf(msg, sizeof(msg), "Remove \"%s\" from\nyour fridge?", gFridge[idx].c_str());
    lv_obj_t* lbl = makeLabel(menu_confirm_box, msg, &lv_font_montserrat_18, COLOR_TEXT);
    lv_obj_set_style_text_align(lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(lbl, LV_ALIGN_TOP_MID, 0, 24);

    lv_obj_t* yes = makeButton(menu_confirm_box, LV_SYMBOL_TRASH " Remove",
                                COLOR_DELETE, menuConfirmDeleteCb);
    lv_obj_set_size(yes, 170, 48);
    lv_obj_align(yes, LV_ALIGN_BOTTOM_LEFT, 24, -16);

    lv_obj_t* no = makeButton(menu_confirm_box, "Cancel",
                               lv_color_hex(0x444444), menuCancelDeleteCb);
    lv_obj_set_size(no, 170, 48);
    lv_obj_align(no, LV_ALIGN_BOTTOM_RIGHT, -24, -16);
}

static void menuRowDeleteCb(lv_event_t* e) {
    int idx = (int)(intptr_t)lv_event_get_user_data(e);
    showMenuConfirmDelete(idx);
}

static void menuBackCb(lv_event_t*) { gUiState = UI_ACTION_SELECT; switchScreen(scr_action); }

static void buildMenu() {
    scr_menu = makeScreen();

    lv_obj_t* header = lv_obj_create(scr_menu);
    lv_obj_set_size(header, 800, 64); lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* ht = makeLabel(header, LV_SYMBOL_LIST " My Fridge",
                              &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align(ht, LV_ALIGN_LEFT_MID, 20, 0);
    menu_count_lbl = makeLabel(header, "0 items", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(menu_count_lbl, LV_ALIGN_RIGHT_MID, -20, 0);

    lv_obj_t* hint = makeLabel(scr_menu, "Tap the trash icon to remove an item",
                                &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(hint, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(hint, LV_ALIGN_TOP_MID, 0, 74);

    menu_list_obj = lv_obj_create(scr_menu);
    lv_obj_set_size(menu_list_obj, 752, 340);
    lv_obj_align(menu_list_obj, LV_ALIGN_TOP_MID, 0, 100);
    lv_obj_set_style_bg_color(menu_list_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(menu_list_obj, 0, 0);
    lv_obj_set_style_pad_all(menu_list_obj, 4, 0);
    lv_obj_set_style_pad_row(menu_list_obj, 12, 0);
    lv_obj_set_flex_flow(menu_list_obj, LV_FLEX_FLOW_COLUMN);

    lv_obj_t* back = makeButton(scr_menu, LV_SYMBOL_LEFT " Back",
                                 lv_color_hex(0x333333), menuBackCb);
    lv_obj_set_size(back, 160, 44);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, 0, -10);
}

static void rebuildMenu() {
    if (!scr_menu) buildMenu();
    if (!menu_list_obj) return;
    lv_obj_clean(menu_list_obj);

    char b[32]; snprintf(b, sizeof(b), "%d items", (int)gFridge.size());
    if (menu_count_lbl) lv_label_set_text(menu_count_lbl, b);

    int n = min((int)gFridge.size(), 40);
    for (int i = 0; i < n; i++) {
        lv_obj_t* row = lv_obj_create(menu_list_obj);
        lv_obj_set_size(row, 740, 56);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, lv_color_hex(0x333333), 0);
        lv_obj_set_style_border_width(row, 1, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_CLICKABLE);

        // Green dot
        lv_obj_t* dot = lv_obj_create(row);
        lv_obj_set_size(dot, 8, 8);
        lv_obj_set_style_bg_color(dot, COLOR_ACCENT, 0);
        lv_obj_set_style_radius(dot, 4, 0);
        lv_obj_set_style_border_width(dot, 0, 0);
        lv_obj_align(dot, LV_ALIGN_LEFT_MID, 12, 0);

        lv_obj_t* lbl = makeLabel(row, gFridge[i].c_str(),
                                   &lv_font_montserrat_18, COLOR_TEXT);
        lv_obj_align(lbl, LV_ALIGN_LEFT_MID, 30, 0);

        // Separated trash button — clearly distinct from item content
        lv_obj_t* del = makeButton(row, LV_SYMBOL_TRASH, COLOR_DELETE, NULL);
        lv_obj_set_size(del, 60, 36);
        lv_obj_align(del, LV_ALIGN_RIGHT_MID, -10, 0);
        lv_obj_add_event_cb(del, menuRowDeleteCb, LV_EVENT_CLICKED,
                            (void*)(intptr_t)i);
    }
}

// ============================================================================
//                       BLE event / data handlers
// ============================================================================

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
            gUiState = UI_ACTION_SELECT; switchScreen(scr_action);
            lvgl_port_unlock();
        }
        return;
    }

    if (evt == "veggie_scanning" || evt == "receipt_capturing" || evt == "receipt_uploading") {
        lvgl_port_lock(-1);
        if (scan_status_label) {
            const char* msg =
                evt == "receipt_capturing" ? "Reading receipt\nPhoto taken — you can put it down" :
                evt == "receipt_uploading" ? "Reading receipt\nPlease wait"                       :
                                             "Looking for ingredients";
            lv_label_set_text(scan_status_label, msg);
        }
        lvgl_port_unlock();
        return;
    }

    if (evt == "veggie_result") {
        String label = jsonStrField(json, "label");
        float  conf  = jsonNumField(json, "confidence");
        String purp  = jsonStrField(json, "purpose");
        // dishes field: pipe-separated dish names for take-out flow
        String dishes = jsonStrField(json, "dishes");

        // Stop the camera so it does not keep scanning after a result is shown
        bleSendCmd("{\"cmd\":\"cancel\"}");

        lvgl_port_lock(-1);
        stopScanAnim();
        if (purp == "out") {
            showTakeOutResult(label, conf);
            if (dishes.length()) populateTakeOutRecipes(dishes);
        } else {
            showVeggieResult(label, conf, purp, true);
        }
        lvgl_port_unlock();
        return;
    }

    if (evt == "veggie_unknown") {
        String purp = jsonStrField(json, "purpose");
        bleSendCmd("{\"cmd\":\"cancel\"}");
        lvgl_port_lock(-1);
        stopScanAnim();
        showVeggieResult("", 0.0f, purp, false);
        lvgl_port_unlock();
        return;
    }

    if (evt == "receipt_result" || evt == "recipe_result") return;  // data arrives on data char

    // DEBUG: streamed receipts. `receipt_test_begin` is always sent first
    // (even when total==0), so screen-switch and list-clear happen exactly
    // once and don't depend on any item ever arriving. Each subsequent
    // `receipt_item` carries one item; we just append a row.
    if (evt == "receipt_test_begin") {
        int total = (int)jsonNumField(json, "total");
        Serial.printf("[receipt_test_begin] total=%d\n", total);
        lvgl_port_lock(-1);
        stopScanAnim();
        rcp_items.clear();
        lv_obj_clean(rcp_list_obj);
        gUiState = UI_RECEIPT_RESULT;
        switchScreen(scr_receipt_result);
        lvgl_port_unlock();
        return;
    }

    if (evt == "receipt_item") {
        int idx   = (int)jsonNumField(json, "idx");
        int total = (int)jsonNumField(json, "total");
        String name = jsonStrField(json, "name");
        String pat  = "\"needs_refrigeration\":";
        int i = json.indexOf(pat);
        bool needs_refrig = false;
        if (i >= 0) {
            i += pat.length();
            while (i < (int)json.length() && (json[i] == ' ' || json[i] == '\t')) i++;
            needs_refrig = (json.substring(i, i + 4) == "true");
        }
        Serial.printf("[receipt_item] %d/%d name='%s' refrig=%d\n",
                      idx, total, name.c_str(), (int)needs_refrig);

        lvgl_port_lock(-1);
        ReceiptItem r;
        r.name = name;
        r.needs_refrig = needs_refrig;
        r.checked = needs_refrig;
        rcp_items.push_back(r);

        lv_obj_t* row = lv_obj_create(rcp_list_obj);
        lv_obj_set_size(row, 730, 50);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, lv_color_hex(0x333333), 0);
        lv_obj_set_style_border_width(row, 1, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_pad_all(row, 8, 0);

        lv_obj_t* cb = lv_checkbox_create(row);
        lv_checkbox_set_text(cb, name.c_str());
        if (needs_refrig) lv_obj_add_state(cb, LV_STATE_CHECKED);
        lv_obj_set_style_text_color(cb, COLOR_TEXT, 0);
        lv_obj_set_style_text_font(cb, &lv_font_montserrat_16, 0);
        lv_obj_align(cb, LV_ALIGN_LEFT_MID, 0, 0);
        lv_obj_add_event_cb(cb, rcpItemToggleCb, LV_EVENT_VALUE_CHANGED,
                            (void*)(intptr_t)(rcp_items.size() - 1));

        if (needs_refrig) {
            lv_obj_t* tag = makeLabel(row, LV_SYMBOL_OK " fridge",
                                       &lv_font_montserrat_12, COLOR_ACCENT);
            lv_obj_align(tag, LV_ALIGN_RIGHT_MID, -8, 0);
        }
        lvgl_port_unlock();
        return;
    }

    if (evt == "receipt_error" || evt == "error") {
        String msg = jsonStrField(json, "msg");
        if (!msg.length()) msg = jsonStrField(json, "code");
        lvgl_port_lock(-1);
        stopScanAnim();
        if (scan_status_label) lv_label_set_text(scan_status_label,
            (String("Error: ") + msg).c_str());
        lv_timer_t* t = lv_timer_create([](lv_timer_t* tm){
            gUiState = UI_ACTION_SELECT; switchScreen(scr_action); lv_timer_del(tm);
        }, 2000, NULL);
        lv_timer_set_repeat_count(t, 1);
        lvgl_port_unlock();
        return;
    }
}

static void onData(const String& payload) {
    String evt = jsonStrField(payload, "evt");
    Serial.printf("[onData] evt=%s len=%d\n", evt.c_str(), (int)payload.length());
    lvgl_port_lock(-1);
    stopScanAnim();
    if (evt == "receipt_result")     showReceiptResult(payload);
    else if (evt == "recipe_result") showRecipeResult(payload);
    else if (evt == "recipe_list") {
        // Pipe-joined dish list from main; populate the take-out panel.
        String dishes = jsonStrField(payload, "dishes");
        populateTakeOutRecipes(dishes);
    }
    lvgl_port_unlock();
}

// ============================================================================
//                              setup() / loop()
// ============================================================================

void setup() {
    Serial.begin(115200);
    Serial.print("MAC: "); Serial.println(WiFi.macAddress());

    Board* board = new Board();
    board->init();

#if LVGL_PORT_AVOID_TEARING_MODE
    auto lcd = board->getLCD();
    lcd->configFrameBufferNumber(LVGL_PORT_DISP_BUFFER_NUM);
#if ESP_PANEL_DRIVERS_BUS_ENABLE_RGB && CONFIG_IDF_TARGET_ESP32S3
    auto lcd_bus = lcd->getBus();
    if (lcd_bus->getBasicAttributes().type == ESP_PANEL_BUS_TYPE_RGB)
        static_cast<BusRGB*>(lcd_bus)->configRGB_BounceBufferSize(lcd->getFrameWidth() * 10);
#endif
#endif
    assert(board->begin());

    lvgl_port_init(board->getLCD(), board->getTouch());
    inventoryLoad();

    lvgl_port_lock(-1);
    buildIdle();
    buildAction();
    buildSubmode();
    buildScan();
    buildVeggieResult();
    buildTakeOutResult();
    buildReceiptPrep();
    buildReceiptCountdown();
    buildReceiptPhotoTaken();
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

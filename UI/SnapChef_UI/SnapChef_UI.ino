#define BOARD_WAVESHARE_ESP32_S3_TOUCH_LCD_4_3 1
#define LV_USE_PRIVATE_API 1
#include <Arduino.h>
#include <esp_display_panel.hpp>
#include <lvgl.h>
#include "lvgl_v8_port.h"
#include <WiFi.h>

using namespace esp_panel::drivers;
using namespace esp_panel::board;

// ─── Screen handles ───────────────────────────────────────────────────────────
static lv_obj_t *scr_idle    = NULL;
static lv_obj_t *scr_mode    = NULL;
static lv_obj_t *scr_scan    = NULL;
static lv_obj_t *scr_result  = NULL;
static lv_obj_t *scr_menu    = NULL;

// ─── Color palette ────────────────────────────────────────────────────────────
#define COLOR_BG        lv_color_hex(0x0D0D0D)   // near-black
#define COLOR_CARD      lv_color_hex(0x1A1A1A)   // dark card
#define COLOR_ACCENT    lv_color_hex(0x00C896)   // mint green
#define COLOR_ACCENT2   lv_color_hex(0xFF6B35)   // warm orange
#define COLOR_TEXT      lv_color_hex(0xF0F0F0)   // off-white
#define COLOR_SUBTEXT   lv_color_hex(0x888888)   // grey
#define COLOR_DELETE    lv_color_hex(0xFF3B3B)   // red

// ─── Mock inventory ───────────────────────────────────────────────────────────
static const char *ingredients[] = {
    "Tomato", "Mozzarella", "Basil", "Eggs", "Mushrooms",
    "Garlic", "Olive Oil", "Pasta", "Parmesan", "Onion"
};
static int ingredient_count = 10;

// ─── Scan animation timer ─────────────────────────────────────────────────────
static lv_timer_t *scan_timer = NULL;
static int scan_dots = 0;
static lv_obj_t *scan_label_dots = NULL;

// ─── Forward declarations ─────────────────────────────────────────────────────
void build_idle_screen();
void build_mode_screen();
void build_scan_screen();
void build_result_screen();
void build_menu_screen();
void switch_screen(lv_obj_t *target);
void start_scan_animation();
void stop_scan_animation();

// ─── Helper: create a styled screen ──────────────────────────────────────────
static lv_obj_t* make_screen() {
    lv_obj_t *scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, COLOR_BG, 0);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);
    return scr;
}

// ─── Helper: create a card panel ─────────────────────────────────────────────
static lv_obj_t* make_card(lv_obj_t *parent, int x, int y, int w, int h) {
    lv_obj_t *card = lv_obj_create(parent);
    lv_obj_set_size(card, w, h);
    lv_obj_set_pos(card, x, y);
    lv_obj_set_style_bg_color(card, COLOR_CARD, 0);
    lv_obj_set_style_bg_opa(card, LV_OPA_COVER, 0);
    lv_obj_set_style_border_color(card, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(card, 1, 0);
    lv_obj_set_style_radius(card, 12, 0);
    lv_obj_set_style_pad_all(card, 16, 0);
    lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);
    return card;
}

// ─── Helper: accent label ─────────────────────────────────────────────────────
static lv_obj_t* make_label(lv_obj_t *parent, const char *text, const lv_font_t *font, lv_color_t color) {
    lv_obj_t *lbl = lv_label_create(parent);
    lv_label_set_text(lbl, text);
    lv_obj_set_style_text_font(lbl, font, 0);
    lv_obj_set_style_text_color(lbl, color, 0);
    return lbl;
}

// ─── Helper: styled button ────────────────────────────────────────────────────
static lv_obj_t* make_button(lv_obj_t *parent, const char *text, lv_color_t bg, lv_event_cb_t cb) {
    lv_obj_t *btn = lv_btn_create(parent);
    lv_obj_set_style_bg_color(btn, bg, 0);
    lv_obj_set_style_bg_color(btn, lv_color_darken(bg, 40), LV_STATE_PRESSED);
    lv_obj_set_style_radius(btn, 10, 0);
    lv_obj_set_style_border_width(btn, 0, 0);
    lv_obj_set_style_shadow_width(btn, 0, 0);
    lv_obj_add_event_cb(btn, cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *lbl = lv_label_create(btn);
    lv_label_set_text(lbl, text);
    lv_obj_set_style_text_font(lbl, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(lbl, COLOR_TEXT, 0);
    lv_obj_center(lbl);

    return btn;
}

// ─── Screen switching helper ──────────────────────────────────────────────────
void switch_screen(lv_obj_t *target) {
    lv_scr_load_anim(target, LV_SCR_LOAD_ANIM_FADE_ON, 200, 0, false);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCREEN 1 — IDLE
// ═══════════════════════════════════════════════════════════════════════════════

static void idle_btn_cb(lv_event_t *e) {
    switch_screen(scr_mode);
}

void build_idle_screen() {
    scr_idle = make_screen();

    // Top accent bar
    lv_obj_t *bar = lv_obj_create(scr_idle);
    lv_obj_set_size(bar, 800, 4);
    lv_obj_set_pos(bar, 0, 0);
    lv_obj_set_style_bg_color(bar, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(bar, 0, 0);
    lv_obj_set_style_radius(bar, 0, 0);

    // Logo / title
    lv_obj_t *title = make_label(scr_idle, "SnapChef", &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(title, LV_ALIGN_CENTER, 0, -80);

    // Tagline
    lv_obj_t *tag = make_label(scr_idle, "Smart Fridge Assistant", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align_to(tag, title, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

    // Divider
    lv_obj_t *div = lv_obj_create(scr_idle);
    lv_obj_set_size(div, 120, 2);
    lv_obj_set_style_bg_color(div, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(div, 0, 0);
    lv_obj_set_style_radius(div, 0, 0);
    lv_obj_align_to(div, tag, LV_ALIGN_OUT_BOTTOM_MID, 0, 20);

    // Prompt
    lv_obj_t *prompt = make_label(scr_idle,
        "Hold an ingredient near the sensor\nor tap the button to get started",
        &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(prompt, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(prompt, div, LV_ALIGN_OUT_BOTTOM_MID, 0, 20);

    // Start button
    lv_obj_t *btn = make_button(scr_idle, "Get Started", COLOR_ACCENT, idle_btn_cb);
    lv_obj_set_size(btn, 200, 52);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -48);

    // Bottom accent bar
    lv_obj_t *bar2 = lv_obj_create(scr_idle);
    lv_obj_set_size(bar2, 800, 4);
    lv_obj_set_pos(bar2, 0, 476);
    lv_obj_set_style_bg_color(bar2, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(bar2, 0, 0);
    lv_obj_set_style_radius(bar2, 0, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCREEN 2 — MODE SELECT
// ═══════════════════════════════════════════════════════════════════════════════

static void mode_ingredient_cb(lv_event_t *e) {
    start_scan_animation();
    switch_screen(scr_scan);
    // After 3 seconds, show result
    lv_timer_t *t = lv_timer_create([](lv_timer_t *timer) {
        stop_scan_animation();
        switch_screen(scr_result);
        lv_timer_del(timer);
    }, 3000, NULL);
    lv_timer_set_repeat_count(t, 1);
}

static void mode_receipt_cb(lv_event_t *e) {
    start_scan_animation();
    switch_screen(scr_scan);
    lv_timer_t *t = lv_timer_create([](lv_timer_t *timer) {
        stop_scan_animation();
        switch_screen(scr_menu);
        lv_timer_del(timer);
    }, 3000, NULL);
    lv_timer_set_repeat_count(t, 1);
}

static void mode_menu_cb(lv_event_t *e) {
    switch_screen(scr_menu);
}

static void back_to_idle_cb(lv_event_t *e) {
    switch_screen(scr_idle);
}

void build_mode_screen() {
    scr_mode = make_screen();

    // Header
    lv_obj_t *header = lv_obj_create(scr_mode);
    lv_obj_set_size(header, 800, 72);
    lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_border_width(header, 0, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t *title = make_label(header, "SnapChef", &lv_font_montserrat_30, COLOR_ACCENT);
    lv_obj_align(title, LV_ALIGN_LEFT_MID, 24, 0);

    lv_obj_t *sub = make_label(header, "What would you like to do?", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_RIGHT_MID, -24, 0);

    // Cards row — 3 mode cards
    int card_y = 110;
    int card_h = 300;
    int card_w = 220;
    int gap    = 30;
    int total  = card_w * 3 + gap * 2;
    int start_x = (800 - total) / 2;

    // Card 1: Ingredient Scan
    lv_obj_t *c1 = make_card(scr_mode, start_x, card_y, card_w, card_h);
    lv_obj_set_style_border_color(c1, COLOR_ACCENT, 0);
    lv_obj_t *ic1 = make_label(c1, LV_SYMBOL_EYE_OPEN, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(ic1, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t *t1 = make_label(c1, "Ingredient\nScan", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t1, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t1, ic1, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t *d1 = make_label(c1, "Hold an item up\nto the camera", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d1, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d1, t1, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t *b1 = make_button(c1, "Scan", COLOR_ACCENT, mode_ingredient_cb);
    lv_obj_set_size(b1, card_w - 32, 44);
    lv_obj_align(b1, LV_ALIGN_BOTTOM_MID, 0, 0);

    // Card 2: Receipt Scan
    lv_obj_t *c2 = make_card(scr_mode, start_x + card_w + gap, card_y, card_w, card_h);
    lv_obj_set_style_border_color(c2, COLOR_ACCENT2, 0);
    lv_obj_t *ic2 = make_label(c2, LV_SYMBOL_FILE, &lv_font_montserrat_48, COLOR_ACCENT2);
    lv_obj_align(ic2, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t *t2 = make_label(c2, "Receipt\nScan", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t2, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t2, ic2, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t *d2 = make_label(c2, "Hold a grocery\nreceipt to camera", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d2, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d2, t2, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t *b2 = make_button(c2, "Scan", COLOR_ACCENT2, mode_receipt_cb);
    lv_obj_set_size(b2, card_w - 32, 44);
    lv_obj_align(b2, LV_ALIGN_BOTTOM_MID, 0, 0);

    // Card 3: Edit Ingredients
    lv_obj_t *c3 = make_card(scr_mode, start_x + (card_w + gap) * 2, card_y, card_w, card_h);
    lv_obj_set_style_border_color(c3, COLOR_SUBTEXT, 0);
    lv_obj_t *ic3 = make_label(c3, LV_SYMBOL_LIST, &lv_font_montserrat_48, COLOR_SUBTEXT);
    lv_obj_align(ic3, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_t *t3 = make_label(c3, "Edit\nIngredients", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_set_style_text_align(t3, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(t3, ic3, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);
    lv_obj_t *d3 = make_label(c3, "Browse and remove\nfridge items", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_set_style_text_align(d3, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align_to(d3, t3, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_t *b3 = make_button(c3, "Open", lv_color_hex(0x444444), mode_menu_cb);
    lv_obj_set_size(b3, card_w - 32, 44);
    lv_obj_align(b3, LV_ALIGN_BOTTOM_MID, 0, 0);

    // Back button
    lv_obj_t *back = make_button(scr_mode, LV_SYMBOL_LEFT " Back", lv_color_hex(0x333333), back_to_idle_cb);
    lv_obj_set_size(back, 120, 40);
    lv_obj_align(back, LV_ALIGN_BOTTOM_LEFT, 24, -20);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCREEN 3 — SCANNING
// ═══════════════════════════════════════════════════════════════════════════════

static void scan_timer_cb(lv_timer_t *timer) {
    if (!scan_label_dots) return;
    scan_dots = (scan_dots + 1) % 4;
    char dots[5] = "    ";
    for (int i = 0; i < scan_dots; i++) dots[i] = '.';
    dots[scan_dots] = '\0';
    lv_label_set_text(scan_label_dots, dots);
}

void start_scan_animation() {
    scan_dots = 0;
    if (scan_timer) lv_timer_del(scan_timer);
    scan_timer = lv_timer_create(scan_timer_cb, 400, NULL);
}

void stop_scan_animation() {
    if (scan_timer) {
        lv_timer_del(scan_timer);
        scan_timer = NULL;
    }
}

void build_scan_screen() {
    scr_scan = make_screen();

    lv_obj_t *icon = make_label(scr_scan, LV_SYMBOL_REFRESH, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align(icon, LV_ALIGN_CENTER, 0, -60);

    lv_obj_t *lbl = make_label(scr_scan, "Scanning", &lv_font_montserrat_30, COLOR_TEXT);
    lv_obj_align_to(lbl, icon, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);

    scan_label_dots = make_label(scr_scan, "   ", &lv_font_montserrat_30, COLOR_ACCENT);
    lv_obj_align_to(scan_label_dots, lbl, LV_ALIGN_OUT_RIGHT_MID, 4, 0);

    lv_obj_t *sub = make_label(scr_scan, "Please hold still", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(sub, LV_ALIGN_CENTER, 0, 40);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCREEN 4 — RECIPE RESULT
// ═══════════════════════════════════════════════════════════════════════════════

static void result_back_cb(lv_event_t *e) {
    switch_screen(scr_mode);
}

void build_result_screen() {
    scr_result = make_screen();

    // Header
    lv_obj_t *header = lv_obj_create(scr_result);
    lv_obj_set_size(header, 800, 72);
    lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_border_width(header, 0, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t *htitle = make_label(header, LV_SYMBOL_OK " Ingredient Detected", &lv_font_montserrat_22, COLOR_ACCENT);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);

    // Detected ingredient card
    lv_obj_t *det = make_card(scr_result, 40, 96, 340, 340);
    lv_obj_set_style_border_color(det, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(det, 2, 0);

    lv_obj_t *det_title = make_label(det, "Detected", &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(det_title, LV_ALIGN_TOP_MID, 0, 0);

    lv_obj_t *det_icon = make_label(det, LV_SYMBOL_IMAGE, &lv_font_montserrat_48, COLOR_ACCENT);
    lv_obj_align_to(det_icon, det_title, LV_ALIGN_OUT_BOTTOM_MID, 0, 16);

    lv_obj_t *det_name = make_label(det, "Tomato", &lv_font_montserrat_36, COLOR_TEXT);
    lv_obj_align_to(det_name, det_icon, LV_ALIGN_OUT_BOTTOM_MID, 0, 12);

    lv_obj_t *det_conf = make_label(det, "Confidence: 94%", &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align_to(det_conf, det_name, LV_ALIGN_OUT_BOTTOM_MID, 0, 8);

    lv_obj_t *det_added = make_label(det, LV_SYMBOL_OK " Added to inventory", &lv_font_montserrat_14, COLOR_ACCENT);
    lv_obj_align(det_added, LV_ALIGN_BOTTOM_MID, 0, 0);

    // Recipe suggestions panel
    lv_obj_t *rec_panel = make_card(scr_result, 420, 96, 340, 340);
    lv_obj_set_style_border_color(rec_panel, COLOR_ACCENT2, 0);
    lv_obj_set_style_border_width(rec_panel, 2, 0);

    lv_obj_t *rec_title = make_label(rec_panel, LV_SYMBOL_OK " Try tonight", &lv_font_montserrat_16, COLOR_ACCENT2);
    lv_obj_align(rec_title, LV_ALIGN_TOP_MID, 0, 0);

    const char *dishes[] = {"Caprese Salad", "Tomato Omelette", "Pasta Pomodoro"};
    lv_color_t dish_colors[] = {COLOR_ACCENT, COLOR_ACCENT2, lv_color_hex(0x7B9EFF)};

    for (int i = 0; i < 3; i++) {
        lv_obj_t *dish_card = lv_obj_create(rec_panel);
        lv_obj_set_size(dish_card, 308, 72);
        lv_obj_set_style_bg_color(dish_card, lv_color_hex(0x252525), 0);
        lv_obj_set_style_border_color(dish_card, dish_colors[i], 0);
        lv_obj_set_style_border_width(dish_card, 1, 0);
        lv_obj_set_style_radius(dish_card, 8, 0);
        lv_obj_set_style_border_side(dish_card, LV_BORDER_SIDE_LEFT, 0);
        lv_obj_set_style_border_width(dish_card, 3, 0);
        lv_obj_align(dish_card, LV_ALIGN_TOP_MID, 0, 36 + i * 86);
        lv_obj_clear_flag(dish_card, LV_OBJ_FLAG_SCROLLABLE);

        lv_obj_t *dish_lbl = make_label(dish_card, dishes[i], &lv_font_montserrat_18, COLOR_TEXT);
        lv_obj_align(dish_lbl, LV_ALIGN_LEFT_MID, 12, 0);

        lv_obj_t *dish_arrow = make_label(dish_card, LV_SYMBOL_RIGHT, &lv_font_montserrat_16, dish_colors[i]);
        lv_obj_align(dish_arrow, LV_ALIGN_RIGHT_MID, -8, 0);
    }

    // Back button
    lv_obj_t *back = make_button(scr_result, LV_SYMBOL_LEFT " Back", lv_color_hex(0x333333), result_back_cb);
    lv_obj_set_size(back, 140, 44);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, 0, -16);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCREEN 5 — INGREDIENT MENU
// ═══════════════════════════════════════════════════════════════════════════════

static lv_obj_t *delete_confirm_box = NULL;
static int pending_delete_index = -1;
static lv_obj_t *ingredient_list_obj = NULL;

static void confirm_delete_cb(lv_event_t *e) {
    if (pending_delete_index >= 0 && pending_delete_index < ingredient_count) {
        // Shift array
        for (int i = pending_delete_index; i < ingredient_count - 1; i++) {
            ingredients[i] = ingredients[i + 1];
        }
        ingredient_count--;
    }
    lv_obj_del(delete_confirm_box);
    delete_confirm_box = NULL;

    // Rebuild list
    lv_obj_clean(ingredient_list_obj);
    // Re-populate (rebuild full menu from scratch is simpler)
    lv_obj_del(scr_menu);
    scr_menu = NULL;
    build_menu_screen();
    switch_screen(scr_menu);
}

static void cancel_delete_cb(lv_event_t *e) {
    if (delete_confirm_box) {
        lv_obj_del(delete_confirm_box);
        delete_confirm_box = NULL;
    }
}

static void show_delete_confirm(int index) {
    pending_delete_index = index;

    delete_confirm_box = lv_obj_create(lv_scr_act());
    lv_obj_set_size(delete_confirm_box, 400, 200);
    lv_obj_center(delete_confirm_box);
    lv_obj_set_style_bg_color(delete_confirm_box, lv_color_hex(0x222222), 0);
    lv_obj_set_style_border_color(delete_confirm_box, COLOR_DELETE, 0);
    lv_obj_set_style_border_width(delete_confirm_box, 2, 0);
    lv_obj_set_style_radius(delete_confirm_box, 14, 0);
    lv_obj_clear_flag(delete_confirm_box, LV_OBJ_FLAG_SCROLLABLE);

    char msg[64];
    snprintf(msg, sizeof(msg), "Remove \"%s\" from\nyour inventory?", ingredients[index]);
    lv_obj_t *lbl = make_label(delete_confirm_box, msg, &lv_font_montserrat_18, COLOR_TEXT);
    lv_obj_set_style_text_align(lbl, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(lbl, LV_ALIGN_TOP_MID, 0, 24);

    lv_obj_t *btn_del = make_button(delete_confirm_box, LV_SYMBOL_TRASH " Remove", COLOR_DELETE, confirm_delete_cb);
    lv_obj_set_size(btn_del, 160, 48);
    lv_obj_align(btn_del, LV_ALIGN_BOTTOM_LEFT, 24, -16);

    lv_obj_t *btn_cancel = make_button(delete_confirm_box, "Cancel", lv_color_hex(0x444444), cancel_delete_cb);
    lv_obj_set_size(btn_cancel, 160, 48);
    lv_obj_align(btn_cancel, LV_ALIGN_BOTTOM_RIGHT, -24, -16);
}

typedef struct {
    int index;
    lv_obj_t *row;
    lv_timer_t *hold_timer;
    bool holding;
} IngredientRowData;

static IngredientRowData row_data[20];

static void hold_timer_fire_cb(lv_timer_t *timer) {
    IngredientRowData *data = (IngredientRowData*)timer->user_data;
    data->holding = false;
    lv_timer_del(data->hold_timer);
    data->hold_timer = NULL;
    // Show delete on the currently active screen
    lv_obj_t *active = lv_scr_act();
    show_delete_confirm(data->index);
    // Reattach confirm box to active screen
    if (delete_confirm_box) {
        lv_obj_set_parent(delete_confirm_box, active);
        lv_obj_center(delete_confirm_box);
    }
}

static void row_pressed_cb(lv_event_t *e) {
    IngredientRowData *data = (IngredientRowData*)lv_event_get_user_data(e);
    data->holding = true;
    if (data->hold_timer) lv_timer_del(data->hold_timer);
    data->hold_timer = lv_timer_create(hold_timer_fire_cb, 3000, data);
    lv_timer_set_repeat_count(data->hold_timer, 1);

    // Visual feedback
    lv_obj_set_style_bg_color(data->row, lv_color_hex(0x2A1A1A), 0);
    lv_obj_set_style_border_color(data->row, COLOR_DELETE, 0);
}

static void row_released_cb(lv_event_t *e) {
    IngredientRowData *data = (IngredientRowData*)lv_event_get_user_data(e);
    data->holding = false;
    if (data->hold_timer) {
        lv_timer_del(data->hold_timer);
        data->hold_timer = NULL;
    }
    lv_obj_set_style_bg_color(data->row, lv_color_hex(0x1E1E1E), 0);
    lv_obj_set_style_border_color(data->row, lv_color_hex(0x333333), 0);
}

static void menu_back_cb(lv_event_t *e) {
    switch_screen(scr_mode);
}

void build_menu_screen() {
    scr_menu = make_screen();

    // Header
    lv_obj_t *header = lv_obj_create(scr_menu);
    lv_obj_set_size(header, 800, 72);
    lv_obj_set_pos(header, 0, 0);
    lv_obj_set_style_bg_color(header, COLOR_CARD, 0);
    lv_obj_set_style_border_width(header, 0, 0);
    lv_obj_set_style_radius(header, 0, 0);
    lv_obj_set_style_border_side(header, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header, COLOR_ACCENT, 0);
    lv_obj_set_style_border_width(header, 2, 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t *htitle = make_label(header, LV_SYMBOL_LIST " My Fridge", &lv_font_montserrat_22, COLOR_TEXT);
    lv_obj_align(htitle, LV_ALIGN_LEFT_MID, 24, 0);

    char count_str[32];
    snprintf(count_str, sizeof(count_str), "%d items", ingredient_count);
    lv_obj_t *hcount = make_label(header, count_str, &lv_font_montserrat_16, COLOR_SUBTEXT);
    lv_obj_align(hcount, LV_ALIGN_RIGHT_MID, -24, 0);

    // Hint label
    lv_obj_t *hint = make_label(scr_menu, "Hold an item for 3 seconds to remove it",
                                 &lv_font_montserrat_14, COLOR_SUBTEXT);
    lv_obj_align(hint, LV_ALIGN_TOP_MID, 0, 82);

    // Scrollable list container
    ingredient_list_obj = lv_obj_create(scr_menu);
    lv_obj_set_size(ingredient_list_obj, 760, 340);
    lv_obj_align(ingredient_list_obj, LV_ALIGN_TOP_MID, 0, 110);
    lv_obj_set_style_bg_color(ingredient_list_obj, COLOR_BG, 0);
    lv_obj_set_style_border_width(ingredient_list_obj, 0, 0);
    lv_obj_set_style_pad_all(ingredient_list_obj, 0, 0);
    lv_obj_set_style_pad_row(ingredient_list_obj, 8, 0);
    lv_obj_set_flex_flow(ingredient_list_obj, LV_FLEX_FLOW_COLUMN);

    for (int i = 0; i < ingredient_count; i++) {
        lv_obj_t *row = lv_obj_create(ingredient_list_obj);
        lv_obj_set_size(row, 740, 60);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x1E1E1E), 0);
        lv_obj_set_style_border_color(row, lv_color_hex(0x333333), 0);
        lv_obj_set_style_border_width(row, 1, 0);
        lv_obj_set_style_radius(row, 8, 0);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_add_flag(row, LV_OBJ_FLAG_CLICKABLE);

        lv_obj_t *dot = lv_obj_create(row);
        lv_obj_set_size(dot, 8, 8);
        lv_obj_set_style_bg_color(dot, COLOR_ACCENT, 0);
        lv_obj_set_style_radius(dot, 4, 0);
        lv_obj_set_style_border_width(dot, 0, 0);
        lv_obj_align(dot, LV_ALIGN_LEFT_MID, 12, 0);

        lv_obj_t *lbl = make_label(row, ingredients[i], &lv_font_montserrat_18, COLOR_TEXT);
        lv_obj_align(lbl, LV_ALIGN_LEFT_MID, 32, 0);

        lv_obj_t *hint_lbl = make_label(row, "hold to remove", &lv_font_montserrat_12, COLOR_SUBTEXT);
        lv_obj_align(hint_lbl, LV_ALIGN_RIGHT_MID, -12, 0);

        row_data[i].index = i;
        row_data[i].row = row;
        row_data[i].hold_timer = NULL;
        row_data[i].holding = false;

        lv_obj_add_event_cb(row, row_pressed_cb, LV_EVENT_PRESSED, &row_data[i]);
        lv_obj_add_event_cb(row, row_released_cb, LV_EVENT_RELEASED, &row_data[i]);
        lv_obj_add_event_cb(row, row_released_cb, LV_EVENT_PRESS_LOST, &row_data[i]);
    }

    // Back button
    lv_obj_t *back = make_button(scr_menu, LV_SYMBOL_LEFT " Back", lv_color_hex(0x333333), menu_back_cb);
    lv_obj_set_size(back, 140, 44);
    lv_obj_align(back, LV_ALIGN_BOTTOM_MID, 0, -10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETUP & LOOP
// ═══════════════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);

    // 在 Serial.begin(115200) 之后加：
    Serial.print("MAC Address: ");
    Serial.println(WiFi.macAddress());

    Serial.println("Initializing board");
    Board *board = new Board();
    board->init();

#if LVGL_PORT_AVOID_TEARING_MODE
    auto lcd = board->getLCD();
    lcd->configFrameBufferNumber(LVGL_PORT_DISP_BUFFER_NUM);
#if ESP_PANEL_DRIVERS_BUS_ENABLE_RGB && CONFIG_IDF_TARGET_ESP32S3
    auto lcd_bus = lcd->getBus();
    if (lcd_bus->getBasicAttributes().type == ESP_PANEL_BUS_TYPE_RGB) {
        static_cast<BusRGB *>(lcd_bus)->configRGB_BounceBufferSize(lcd->getFrameWidth() * 10);
    }
#endif
#endif
    assert(board->begin());

    Serial.println("Initializing LVGL");
    lvgl_port_init(board->getLCD(), board->getTouch());

    Serial.println("Building SnapChef UI");
    lvgl_port_lock(-1);

    build_idle_screen();
    build_mode_screen();
    build_scan_screen();
    build_result_screen();
    build_menu_screen();

    lv_scr_load(scr_idle);

    lvgl_port_unlock();
    Serial.println("UI ready");
}

void loop() {
    delay(10);
}


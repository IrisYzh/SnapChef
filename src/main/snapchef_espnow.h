#pragma once

// Shared ESP-NOW protocol constants for SnapChef.
// Mirror copy lives in src/SnapChef_UI/snapchef_espnow.h — keep in sync.

// 1-byte tag at offset 0 of every ESP-NOW frame demultiplexes the payload.
#define SNAPCHEF_MSG_HELLO  'H'   // display -> main (broadcast): discovery
#define SNAPCHEF_MSG_READY  'R'   // main    -> display (unicast):  discovery ack
#define SNAPCHEF_MSG_CMD    'C'   // display -> main:    JSON command, single frame
#define SNAPCHEF_MSG_EVT    'E'   // main    -> display: JSON event,   single frame
#define SNAPCHEF_MSG_DATA   'D'   // main    -> display: "<seq>/<total>|<frag>"

// ESP-NOW max single payload is 250 B. We reserve 1 B for the tag and ~12 B
// for the "<seq>/<total>|" framing header on data fragments. The remainder
// is the chunk of the original payload that fits in one frame.
static const int SNAPCHEF_DATA_FRAG_MAX = 200;

// SSID of the AP main joins. Display scans the air for this SSID so it can
// lock its radio to the same channel as main's ESP-NOW socket.
#define SNAPCHEF_PEER_WIFI_SSID "UW MPSK"

// Channel used when SSID isn't visible (e.g. main offline at display boot) —
// must match the channel main ends up on for first-contact to succeed.
static const int SNAPCHEF_ESPNOW_CHANNEL_FALLBACK = 1;

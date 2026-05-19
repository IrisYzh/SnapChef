#pragma once

// Shared ESP-NOW protocol constants for SnapChef.
// Mirror copy lives in src/main/snapchef_espnow.h — keep in sync.

#define SNAPCHEF_MSG_HELLO  'H'
#define SNAPCHEF_MSG_READY  'R'
#define SNAPCHEF_MSG_CMD    'C'
#define SNAPCHEF_MSG_EVT    'E'
#define SNAPCHEF_MSG_DATA   'D'

static const int SNAPCHEF_DATA_FRAG_MAX = 200;

#define SNAPCHEF_PEER_WIFI_SSID "UW MPSK"
static const int SNAPCHEF_ESPNOW_CHANNEL_FALLBACK = 1;

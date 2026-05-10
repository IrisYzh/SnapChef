#pragma once

// Shared BLE protocol constants for SnapChef.
// Mirror copy lives in src/main/snapchef_ble.h — keep in sync.

#define SNAPCHEF_BLE_NAME        "SnapChef-Main"

#define SNAPCHEF_SVC_UUID        "6d6e6750-7363-4865-6643-686566303031"
#define SNAPCHEF_CHR_CMD_UUID    "6d6e6750-7363-4865-6643-686566303032"
#define SNAPCHEF_CHR_EVT_UUID    "6d6e6750-7363-4865-6643-686566303033"
#define SNAPCHEF_CHR_DATA_UUID   "6d6e6750-7363-4865-6643-686566303034"

static const int SNAPCHEF_DATA_FRAG_MAX = 180;

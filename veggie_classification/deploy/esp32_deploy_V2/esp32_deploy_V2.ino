/*
 * 蔬菜分类器 - Grove Vision AI V2 + XIAO ESP32S3
 * 
 * 接线：XIAO ESP32S3 直插 Grove Vision AI V2 背面排针
 *       USB 线接 XIAO 的 Type-C 口
 * 
 * 依赖库：Seeed_Arduino_SSCMA
 * 
 * Board 设置：
 *   - Board: XIAO_ESP32S3
 *   - USB CDC On Boot: Enabled   <-- 重要！
 *   - PSRAM: OPI PSRAM
 */

#include <Wire.h>
#include <Seeed_Arduino_SSCMA.h>

SSCMA AI;

const char* vegetableNames[] = {
  "Bellpeper",    
  "Broccoli",      
  "Cabbage",    
  "Carrot",   
  "Cucumber", 
  "Eggplant",   
  "Garlic",      
  "Onion",  
  "Potato",  
  "Tomato" 
};

const int NUM_CLASSES = sizeof(vegetableNames) / sizeof(vegetableNames[0]);
const int CONFIDENCE_THRESHOLD = 90;  // 0.9 -> 90
const unsigned long INVOKE_INTERVAL = 500;

unsigned long lastInvokeTime = 0;

void setup() {
  Serial.begin(115200);

  // WiseEye2 冷启 + 加载模型通常 3–5 s,给足时间再握手
  delay(5000);

  Wire.begin();

  // I2C 扫描:V2 的 SSCMA 地址是 0x62。如果扫不到,说明没到握手这一步。
  Serial.println("I2C scan (looking for V2 @ 0x62):");
  int found = 0;
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  device at 0x%02X\n", addr);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  (no I2C devices — V2 offline, check power/seating)");
  }

  Serial.println("Calling AI.begin() ...");
  bool ok = AI.begin();
  Serial.printf("AI.begin() returned %s\n", ok ? "true" : "false");

  // V2 握手信息(ID/name 在 begin() 内部已被填充)
  Serial.print("Module ID:   "); Serial.println(AI.ID());
  Serial.print("Module name: "); Serial.println(AI.name());
  Serial.print("Module info: "); Serial.println(AI.info());

  Serial.println("========================================");
  Serial.println("  Vegetable Classifier Ready!");
  Serial.println("  Confidence Threshold: 90%");
  Serial.println("========================================");
}

void loop() {
  if (millis() - lastInvokeTime < INVOKE_INTERVAL) {
    return;
  }
  lastInvokeTime = millis();

  // 调用推理 (show=true,让 V2 通过 EVENT 回分类结果)
  int ret = AI.invoke(1, false, true);
  if (ret == 0) {

    Serial.print("[Perf] pre=");
    Serial.print(AI.perf().prepocess);
    Serial.print("ms, infer=");
    Serial.print(AI.perf().inference);
    Serial.print("ms, post=");
    Serial.print(AI.perf().postprocess);
    Serial.println("ms");

    if (AI.classes().size() > 0) {

      int bestScore = 0;
      int bestTarget = -1;

      for (int i = 0; i < AI.classes().size(); i++) {
        if (AI.classes()[i].score > bestScore) {
          bestScore = AI.classes()[i].score;
          bestTarget = AI.classes()[i].target;
        }
      }

      if (bestScore >= CONFIDENCE_THRESHOLD && bestTarget >= 0) {
        Serial.print("[Result] ");
        if (bestTarget < NUM_CLASSES) {
          Serial.print(vegetableNames[bestTarget]);
        } else {
          Serial.print("Class_");
          Serial.print(bestTarget);
        }
        Serial.print(" (");
        Serial.print(bestScore);
        Serial.println("%)");
      } else {
        Serial.print("[Result] Object Not Known (best: ");
        Serial.print(bestScore);
        Serial.print("%, maybe ");
        if (bestTarget >= 0 && bestTarget < NUM_CLASSES) {
          Serial.print(vegetableNames[bestTarget]);
        }
        Serial.println(")");
      }

    } else {
      Serial.println("[Result] No classification data");
    }

  } else {
    // 0=OK, 1=AGAIN, 2=ELOG, 3=ETIMEDOUT, 4=EIO, 5=EINVAL, 6=ENOMEM,
    // 7=EBUSY, 8=ENOTSUP, 9=EPERM, 10=EUNKNOWN  (见 SSCMA .h CMD_* 枚举)
    Serial.printf("[Error] Invoke failed, ret=%d\n", ret);
  }

  Serial.println("----------------------------------------");
}

# Snapchef Backend API（ESP32S3 接入文档）

## 基本信息

| 项 | 值 |
|---|---|
| Base URL | `https://snapchef-production.up.railway.app` |
| 协议 | HTTPS（TLS 1.2+，证书由 Let's Encrypt 签发）|
| 认证方式 | HTTP Header `X-API-Key` |
| 内容类型 | `multipart/form-data` |
| 字符编码 | UTF-8 |

**API Key**：固件中以常量形式烧录（或写在 NVS / 配网时下发）。当前测试 key 由后端管理员单独发给固件负责人，**不要硬编码到公开仓库**。

---

## 端点一览

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| GET | `/healthz` | 健康检查，连通性自检 | 否 |
| POST | `/receipts/analyze` | 上传小票图片，返回结构化商品清单 | 是 |
| POST | `/produce/recognize` | 上传果蔬图片（百度识别），返回英文名称（非果蔬返回 `Uncertain`） | 是 |
| POST | `/produce/recognize-llm` | 同上，识别改用 Claude 视觉模型（更适配塑料模型/手持场景） | 是 |
| POST | `/recipes/list` | 根据主料 + 冰箱清单，返回可做的菜名列表 | 是 |
| POST | `/recipes/steps` | 根据选定菜品 + 冰箱清单，返回步骤、所用/缺少的食材 | 是 |

---

## 1. 健康检查 `GET /healthz`

用于设备开机自检 / 网络可达性检测。

**请求**：无参数，无 header 要求。

**响应** `200 OK`：
```json
{ "status": "ok" }
```

设备端建议：开机或 Wi-Fi 连上后调一次，超时 5s，失败重试 3 次。

---

## 2. 上传小票 `POST /receipts/analyze`

### 请求

**Headers**：
| Header | 必填 | 说明 |
|---|---|---|
| `X-API-Key` | ✅ | 固定 token，后端比对 |
| `Content-Type` | ✅ | `multipart/form-data; boundary=...`（HTTP 库会自动加） |

**Body**：multipart 表单，**只有一个字段**：

| 字段名 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `image` | 文件 | ✅ | `image/jpeg` 或 `image/png`；≤ **8 MB**；不能为空 |

> ⚠️ 字段名必须是 `image`，写成 `file` / `photo` 都会被 FastAPI 拒绝（422）。

**示例 curl**：
```bash
curl -X POST https://snapchef-production.up.railway.app/receipts/analyze \
  -H "X-API-Key: <your-key>" \
  -F "image=@receipt.jpg;type=image/jpeg"
```

### 响应 `200 OK`

```json
{
  "receipt_id": "68456e95-145e-409f-bc50-b35fe5246061",
  "items": [
    {
      "id": "0",
      "raw_name": "STRAWBERRIES 1 LB",
      "name": "Strawberries 1 lb",
      "quantity": 1.0,
      "unit_price": 3.99,
      "total_price": 3.99,
      "category": "produce",
      "needs_refrigeration": true,
      "checked": true
    }
  ],
  "totals": {
    "subtotal": 22.47,
    "tax": 1.57,
    "total": 24.04
  },
  "classification_warning": null
}
```

### 响应字段详解

#### 顶层

| 字段 | 类型 | 说明 |
|---|---|---|
| `receipt_id` | string (UUID) | 服务端生成，每次请求唯一；**后端不持久化**，仅用于日志关联 |
| `items` | array | 商品列表，按小票顺序 |
| `totals` | object | 小票汇总金额 |
| `classification_warning` | string \| null | 非 null 时表示 AI 分类失败，所有 `category=other`、`needs_refrigeration=false`，但 Textract 数据仍可信 |

#### `items[]`

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 数组索引字符串化（"0","1",...），**UI 用作 key** |
| `raw_name` | string | 小票上的原始字符（多为大写英文）|
| `name` | string | AI 规范化后的可读名（首字母大写、修正拼写） |
| `quantity` | number | 数量，默认 `1.0`，可能是小数（如按重量计）|
| `unit_price` | number \| null | 单价，缺失时为 `null` |
| `total_price` | number \| null | 小计，缺失时为 `null` |
| `category` | enum | 见下方 Category 枚举 |
| `needs_refrigeration` | bool | 是否需冷藏 |
| `checked` | bool | UI 默认勾选状态，**等于 `needs_refrigeration`** |

#### `totals`

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtotal` | number \| null | 税前合计 |
| `tax` | number \| null | 税额 |
| `total` | number \| null | 实付金额 |

> 三个字段都可能为 `null`（小票上没识别到对应文本时）。

#### `Category` 枚举

固件如需本地分类显示图标，按下表映射：

| 值 | 含义 |
|---|---|
| `produce` | 果蔬 |
| `dairy` | 乳制品 |
| `meat_seafood` | 肉类/海鲜 |
| `frozen` | 冷冻品 |
| `bakery` | 烘焙 |
| `pantry` | 干货/调味 |
| `beverage` | 饮料 |
| `snack` | 零食 |
| `household` | 日用品 |
| `other` | 其他 |

> 枚举值未来可能扩展，固件遇到未知值时按 `other` 处理，**不要崩溃**。

### 错误响应

所有错误统一格式：
```json
{ "detail": "<错误描述>" }
```

| HTTP | 触发条件 | 设备处理建议 |
|---|---|---|
| **400** | 文件类型不是 jpeg/png；文件为空；超过 8MB | 提示用户重拍 / 检查图像导出格式 |
| **401** | `X-API-Key` 错误 | 致命错误，固件 bug，不应重试 |
| **422** | `image` 字段缺失；或小票上识别不到任何商品 | 提示"未识别到商品，请重拍清晰一些" |
| **502** | AWS Textract 调用失败 | 提示网络问题，可重试 1-2 次 |
| **5xx** | 服务端异常（Railway 重启等）| 指数退避重试（1s, 3s, 8s）|

> **注意**：Claude 分类失败**不会**返回 5xx。Textract 识别成功就是 200，但 `classification_warning` 会非空，且所有 `category=other`。固件应正常显示这类响应。

---

## 3. 果蔬识别 `POST /produce/recognize`

上传一张果蔬照片，后端先调用百度果蔬识别，再用 Claude 把中文名翻译成英文返回。**若图片不是果蔬，返回 `name = "Uncertain"`。**

### 请求

**Headers**：
| Header | 必填 | 说明 |
|---|---|---|
| `X-API-Key` | ✅ | 固定 token，后端比对 |
| `Content-Type` | ✅ | `multipart/form-data; boundary=...`（HTTP 库会自动加） |

**Body**：multipart 表单，**只有一个字段**：

| 字段名 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `image` | 文件 | ✅ | `image/jpeg` 或 `image/png`；不能为空；原图 ≤ **8 MB**，且 base64 编码后 ≤ **4 MB**（即原图约 ≤ 3 MB）|

> ⚠️ 字段名必须是 `image`（同 `/receipts/analyze`）。百度对编码后大小限制 4MB，所以单张果蔬图建议控制在 3MB 以内。

**示例 curl**：
```bash
curl -X POST https://snapchef-production.up.railway.app/produce/recognize \
  -H "X-API-Key: <your-key>" \
  -F "image=@tomato.jpg;type=image/jpeg"
```

### 响应 `200 OK`

识别为果蔬时：
```json
{
  "name": "Tomato",
  "raw_name": "西红柿",
  "confidence": 0.98
}
```

判定为非果蔬 / 未识别 / 置信度过低时：
```json
{
  "name": "Uncertain",
  "raw_name": null,
  "confidence": null
}
```

### 响应字段详解

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 英文果蔬名（首字母大写、单数），或固定字符串 `"Uncertain"` |
| `raw_name` | string \| null | 百度返回的中文名；`Uncertain` 时为 `null`，仅用于调试 |
| `confidence` | number \| null | 百度置信度 0~1；`Uncertain` 时为 `null` |

> 设备端逻辑：先判断 `name == "Uncertain"`，是则提示"无法识别，请对准果蔬重拍"；否则直接展示 `name`。

**`Uncertain` 触发条件**（任一满足，**不会**调用 Claude 翻译）：
- 百度未识别出任何结果
- 最高项被百度标记为"非果蔬食材"
- 最高项置信度低于阈值（默认 `0.5`，由后端 `BAIDU_PRODUCE_MIN_SCORE` 配置）

### 错误响应

| HTTP | 触发条件 | 设备处理建议 |
|---|---|---|
| **400** | 文件类型不是 jpeg/png；文件为空；原图超 8MB；或编码后超百度 4MB 上限 | 提示用户重拍 / 压缩图像 |
| **401** | `X-API-Key` 错误 | 致命错误，不应重试 |
| **422** | `image` 字段缺失 | 固件 bug，检查字段名 |
| **502** | 百度识别调用失败，或 Claude 翻译失败 | 提示网络问题，可重试 1-2 次 |

> 注意：与 `/receipts/analyze` 不同，本接口翻译失败会返回 **502**（没有合理的降级英文名），不会返回 200。

---

## 4. 果蔬识别（LLM 视觉）`POST /produce/recognize-llm`

**与 `/produce/recognize` 请求和响应完全一致**，区别在于识别过程：不调用百度，而是把图片直接交给 Claude 视觉模型，由其一次完成"识别 + 英文命名"。

使用场景已写入模型提示词：**一个人手持单个蔬菜/水果正对镜头**，模型只识别手中物体、忽略人脸/手/背景。**塑料果蔬模型也按真实果蔬识别**（塑料番茄 → `Tomato`），便于测试。

### 何时用哪个

| | `/produce/recognize`（百度） | `/produce/recognize-llm`（Claude） |
|---|---|---|
| 识别真实果蔬 | 准 | 准 |
| 塑料模型 / 手持近距 / 构图差 | 容易判为 `Uncertain` | 更稳 |
| 延迟 | 较低 | 略高（视觉推理）|
| 成本 | 低 | 较高 |

> 实测：手持塑料玉米的设备图，百度返回 `Uncertain`（最高分仅 0.37），LLM 返回 `Corn` (0.95)。**测试阶段建议用本接口。**

### 请求

与 `/produce/recognize` 相同（multipart 单字段 `image`，`image/jpeg` 或 `image/png`）。

> 大小约束略有不同：原图 ≤ **8 MB**，且 base64 编码后 ≤ **5 MB**（Claude 单图上限，约原图 ≤ 3.75 MB）。

**示例 curl**：
```bash
curl -X POST https://snapchef-production.up.railway.app/produce/recognize-llm \
  -H "X-API-Key: <your-key>" \
  -F "image=@corn.jpg;type=image/jpeg"
```

### 响应 `200 OK`

字段与 `/produce/recognize` 完全相同；`raw_name` 在本接口恒为 `null`（无中文中间结果）。

识别为果蔬：
```json
{ "name": "Corn", "raw_name": null, "confidence": 0.95 }
```

非果蔬 / 未识别：
```json
{ "name": "Uncertain", "raw_name": null, "confidence": null }
```

`confidence` 为模型自评置信度（0~1），仅供参考。`Uncertain` 由模型判定手持物不是果蔬时返回。

> **受控词表**：`name` 取值被限定在后端维护的固定果蔬词表内（通过 tool schema 的 `enum` 强约束），保证输出规范、便于下游匹配——例如永远是 `Tomato`，不会出现 `tomatoes` / `Roma Tomato` 等变体。若手持物是果蔬但不在词表内，同样返回 `Uncertain`。需要新增可识别品类时，在后端 `app/schemas.py` 的 `PRODUCE_VOCAB` 里加一项即可。

### 错误响应

| HTTP | 触发条件 | 设备处理建议 |
|---|---|---|
| **400** | 文件类型不是 jpeg/png；文件为空；原图超 8MB；或编码后超 Claude 5MB 上限 | 提示用户重拍 / 压缩图像 |
| **401** | `X-API-Key` 错误 | 致命错误，不应重试 |
| **422** | `image` 字段缺失 | 固件 bug，检查字段名 |
| **502** | Claude 视觉调用失败 | 提示网络问题，可重试 1-2 次 |

---

## 5. 推荐菜名 `POST /recipes/list`

根据一个"主料"（`trigger`，用户想用的核心食材）和冰箱现有食材，返回若干可做的菜名。

### 请求

**Headers**：`X-API-Key`（必填）；`Content-Type: application/json`。

**Body**（JSON）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `trigger` | string | ✅ | 核心食材 / 想做的方向，非空 |
| `fridge` | string[] | 否 | 冰箱现有食材，默认 `[]` |

```bash
curl -X POST https://snapchef-production.up.railway.app/recipes/list \
  -H "X-API-Key: <your-key>" -H "Content-Type: application/json" \
  -d '{"trigger":"Tomato","fridge":["Tomato","Onion","Mozzarella","Egg"]}'
```

### 响应 `200 OK`

```json
{ "dishes": ["Caprese Salad", "Tomato Omelette", "Pasta Pomodoro"] }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `dishes` | string[] | 3~6 个菜名，每个 ≤ 15 字符，均突出 `trigger`，优先使用冰箱食材 |

> 菜名是后续 `/recipes/steps` 的入参 `dish`，原样回传即可。

### 错误响应

| HTTP | 触发条件 |
|---|---|
| **401** | `X-API-Key` 错误 |
| **422** | `trigger` 缺失或为空 |
| **502** | Claude 调用失败，或未生成任何菜名 |

---

## 6. 生成步骤 `POST /recipes/steps`

根据选定菜品和冰箱清单，返回做菜步骤，并**额外告知用到了冰箱里的哪些食材、还缺哪些需要购买**。

### 请求

**Headers**：`X-API-Key`（必填）；`Content-Type: application/json`。

**Body**（JSON）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `dish` | string | ✅ | 菜名（通常取自 `/recipes/list` 的某一项），非空 |
| `fridge` | string[] | 否 | 冰箱现有食材，默认 `[]`；用于判定 used / missing |

```bash
curl -X POST https://snapchef-production.up.railway.app/recipes/steps \
  -H "X-API-Key: <your-key>" -H "Content-Type: application/json" \
  -d '{"dish":"Tomato Egg Stir-fry","fridge":["Tomato","Egg","Scallion","Mozzarella"]}'
```

### 响应 `200 OK`

```json
{
  "title": "Tomato Egg Stir-fry",
  "time_min": 15,
  "steps": [
    "Cut the tomatoes into wedges and thinly slice the scallion.",
    "Beat the eggs with a pinch of salt.",
    "Scramble the eggs, then set aside.",
    "Stir-fry the tomatoes until softened, return the eggs, toss together."
  ],
  "used_fridge_items": ["Tomato", "Egg", "Scallion"],
  "missing_items": []
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `title` | string | 菜品规范名 |
| `time_min` | int | 预计总耗时（分钟，整数）|
| `steps` | string[] | 3~8 条有序操作步骤，每条一个动作 |
| `used_fridge_items` | string[] | 这道菜**用到的、且确实在 `fridge` 里**的食材；回显 `fridge` 中的写法 |
| `missing_items` | string[] | 菜谱需要但**冰箱没有、需购买**的食材；**不含**盐/油/糖/面粉等常备调料 |

> **服务端会校对** `used_fridge_items` / `missing_items`：`used` 只保留与 `fridge` 大小写不敏感匹配的项，`missing` 会剔除其实已在 `fridge` 中的项，避免模型误报。两者都可能为空数组。

### 错误响应

| HTTP | 触发条件 |
|---|---|
| **401** | `X-API-Key` 错误 |
| **422** | `dish` 缺失或为空 |
| **502** | Claude 调用失败，或返回缺少 title/steps 等必要字段 |

---

## 性能与超时建议

| 指标 | 数值 |
|---|---|
| 端到端延迟 | **典型 4-5 秒**（Textract ~2s + Claude ~2-3s）|
| 建议 HTTP 超时 | **15-20 秒** |
| 上传带宽 | 1-3 MB JPEG，512 Kbps Wi-Fi 可用 |

固件端 **必须显示加载动画**（建议旋转图标 + 文字"识别中…"），否则用户会以为设备死机。

---

## ESP-IDF 接入示例

> 用 `esp_http_client` 走 multipart。CA 证书走 ESP-IDF 内置 `bundle`（开 `CONFIG_MBEDTLS_CERTIFICATE_BUNDLE=y`），无需手动嵌入。

```c
#include "esp_http_client.h"
#include "esp_crt_bundle.h"

#define API_URL "https://snapchef-production.up.railway.app/receipts/analyze"
#define API_KEY "<your-key>"
#define BOUNDARY "----snapchef32boundary"

esp_err_t upload_receipt(const uint8_t *jpeg, size_t jpeg_len, char *resp_buf, size_t resp_cap)
{
    // 拼 multipart body（前缀 + 图片 + 后缀）
    char prefix[256];
    int plen = snprintf(prefix, sizeof(prefix),
        "--" BOUNDARY "\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"r.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n");
    const char *suffix = "\r\n--" BOUNDARY "--\r\n";
    size_t slen = strlen(suffix);
    size_t total = plen + jpeg_len + slen;

    uint8_t *body = malloc(total);
    if (!body) return ESP_ERR_NO_MEM;
    memcpy(body, prefix, plen);
    memcpy(body + plen, jpeg, jpeg_len);
    memcpy(body + plen + jpeg_len, suffix, slen);

    esp_http_client_config_t cfg = {
        .url = API_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 20000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t cli = esp_http_client_init(&cfg);

    esp_http_client_set_header(cli, "X-API-Key", API_KEY);
    esp_http_client_set_header(cli, "Content-Type",
        "multipart/form-data; boundary=" BOUNDARY);
    esp_http_client_set_post_field(cli, (const char *)body, total);

    esp_err_t err = esp_http_client_perform(cli);
    int status = esp_http_client_get_status_code(cli);
    int rlen = esp_http_client_read_response(cli, resp_buf, resp_cap - 1);
    if (rlen >= 0) resp_buf[rlen] = '\0';

    esp_http_client_cleanup(cli);
    free(body);

    if (err != ESP_OK) return err;
    if (status != 200) return ESP_FAIL;
    return ESP_OK;
}
```

> 用 cJSON 解析返回体；遍历 `items` 数组，根据 `checked` 字段决定 UI 默认勾选。

---

## 联调清单（device dev 自检）

- [ ] `/healthz` 能返回 200
- [ ] 不带 `X-API-Key` → 422，固件能识别并提示"配置错误"
- [ ] 带错误 `X-API-Key` → 401
- [ ] 上传一张真实小票 JPEG → 200，能解析出 `items` 数组
- [ ] 上传 < 100 字节的伪图 → 400 或 422，固件不崩溃
- [ ] 故意传 `image/bmp` → 400
- [ ] 模拟网络断开 → 客户端超时不超过 20s，UI 能恢复
- [ ] 返回 `classification_warning != null` 时，UI 仍正常显示商品（只是没分类）
- [ ] `/produce/recognize` 上传一张果蔬图 → 200，`name` 为英文名
- [ ] `/produce/recognize` 上传非果蔬图（如人脸/桌子）→ 200，`name == "Uncertain"`

---

## 变更与联系

- 新增 / 修改 category 枚举值会**提前一周通知**
- 接口本身字段只增不减；如需破坏性改动，会给新版本路径（如 `/v2/receipts/analyze`）
- 问题反馈：直接联系后端负责人（@menglh）

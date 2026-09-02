# Qwen3-TTS 语音克隆服务 · 内网部署手册

本文档说明如何把本项目打包成 Docker 镜像、部署到**内网 GPU 服务器**，
并兼容 **T4（16G）** 与 **A10（24G）** 显卡。

- 模型权重（`Qwen3-TTS-12Hz-1.7B-Base`）**外挂挂载**，不写入镜像；
- 音色库数据目录**外挂持久化**，容器重启/升级后音色不丢失；
- 全程离线推理（容器内禁止任何联网下载）。

---

## 1. 架构总览

```
浏览器(Web UI) ──┐
                 ▼
             ┌───────────────────────────────┐
外部程序 ────►│  FastAPI  /api/v1             │
(curl/脚本)   │   ├─ voices 音色 CRUD          │──►  GPU 模型引擎（Qwen3TTS 1.7B-Base）
             │   ├─ tts / tts/clone 合成      │        （T4: fp16+eager / A10: fp16+sdpa或FA2）
             │   └─ models/info / 健康检查     │
             └───────────────┬───────────────┘
                             │
     /models(ro): 模型权重    │    /data(rw): 音色库 voices/<voice_id>/
```

| 挂载 | 容器路径 | 说明 |
|---|---|---|
| `MODEL_DIR`（宿主） | `/models`（只读） | 模型目录，内含 `speech_tokenizer/` |
| `DATA_DIR`（宿主） | `/data`（读写） | 音色库、日志等，**务必放到持久盘** |

---

## 2. 宿主前置条件

1. Linux + NVIDIA 驱动 **≥ 525**（CUDA 12.x）
2. Docker ≥ 24 + compose v2 插件
3. [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 已安装，验证：
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
4. 磁盘：模型 ~4GB、数据目录视音色规模而定

---

## 3. 在有外网的构建机上准备

### 3.1 下载模型权重（约 4GB）

```bash
git clone <本项目> && cd <本项目>
# 方式一：优先（国内网络推荐，需 modelscope）
pip install modelscope
bash scripts/download_models.sh
# 方式二：HuggingFace（需代理/外网）
MODEL_REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base bash scripts/download_models.sh
```
产物目录：
```
models/Qwen3-TTS-12Hz-1.7B-Base/
    ├── speech_tokenizer/        # 音频编解码器（模型必需）
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors …
    └── tokenizer files
```
> 注意：请勿删除 `speech_tokenizer/` 子目录，否则服务启动会失败。

### 3.2 构建镜像并导出

```bash
bash scripts/save_image.sh 0.1.1
# 产物：dist/qwen3-tts-server-0.1.1.tar.gz
```

> 可选：若 A10 上需要 flash-attention-2，镜像默认不内置（T4 也不支持）。
> 需要时请以 `nvidia/cuda:12.4.1-devel-ubuntu22.04` 自行编译 flash-attn，
> 或 pip 安装预编译 wheel，再打补丁镜像。**不装也不影响功能**，`auto` 会自动回退 eager。

---

## 4. 拷入内网并启动

### 4.1 导入镜像 + 放置模型

```bash
docker load -i dist/qwen3-tts-server-0.1.1.tar.gz
# 把构建机下载的 models/ 整个目录拷贝到内网主机，例如 /data/models
mkdir -p /data/models /data/ttsdata
# rsync/scp … 将 models/* 传到 /data/models/
```

### 4.2 配置 .env

```bash
cp .env.example .env
vim .env
```
关键项：
```ini
MODEL_DIR=/data/models        # 指向含 Qwen3-TTS-12Hz-1.7B-Base 的父目录
DATA_DIR=/data/ttsdata        # 音色持久化目录
SERVICE_PORT=8000
DEVICE=cuda:0
MODEL_DTYPE=auto              # T4 会自动 fp16；A10 可手动 bf16
ATTN_IMPL=auto                # 见下表
API_KEY=                      # 留空不鉴权；可设置任意字符串开启鉴权
# 若模型子目录名不同（如 0.6B），再改 MODEL_PATH=/models/<实际目录名>
```

### 4.3 T4 / A10 兼容对照

| 显卡 | 架构/算力 | `MODEL_DTYPE` | `ATTN_IMPL` 行为 |
|---|---|---|---|
| **T4 16G** | Turing · sm75 | `auto`→**fp16**（bf16 强制回退 fp16） | `auto`→**eager**（无 FA2） |
| **A10 24G** | Ampere · sm86 | `auto`→fp16；可设 `bf16` | `auto`→FA2(已安装时)/否则 **eager**（最稳，兼容 T4） |

### 4.4 启动

```bash
docker compose up -d
docker compose ps                 # 等待 healthy（首次加载模型约 1-3 分钟）
docker compose logs -f tts        # 查看启动日志（含 runtime 决策）
```

浏览器打开：`http://<内网IP>:8000/`
Swagger 文档：`http://<内网IP>:8000/docs`

> **重启容器音色不丢**：音色保存在 `DATA_DIR=/data/ttsdata/voices/`，
> 只要该目录不被删除，`docker compose restart / down+up` 后自动重建音色列表。

---

## 5. 使用说明

### 5.1 Web UI（推荐）

打开首页后：
1. **上传新音色**：名称 + 参考音频（wav/mp3/flac/ogg/m4a）+ 参考文本（ICL 模式必填），
   或勾选“仅使用说话人向量（x-vector）”免文本（效果有限）。
   保存后声纹落盘到 `/data/voices/<id>/`，**无需再上传**。
2. **音色库**：搜索 / 选中 / 删除音色。
3. **合成**：Tab1 选已存音色，或 Tab2 一次性上传克隆；输入文本与语种即可，
   结果可在线试听与下载 WAV。

### 5.2 REST API

统一前缀 `/api/v1`；若配置了 `API_KEY`，所有请求需带请求头 `X-API-Key: <key>`。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/voices` | multipart：`file,name,ref_text?,x_vector_only?` → 保存音色返回 `voice_id` |
| GET | `/api/v1/voices` | 音色列表 |
| GET | `/api/v1/voices/{id}` | 音色详情 |
| DELETE | `/api/v1/voices/{id}` | 删除音色 |
| POST | `/api/v1/tts` | JSON `{voice_id,text,language?,temperature?,top_p?,top_k?,repetition_penalty?,max_new_tokens?}` → **WAV** |
| POST | `/api/v1/tts/clone` | multipart：`file,text,ref_text?,language?,x_vector_only?` → 一次性克隆 **WAV**（不落库） |
| GET | `/api/v1/models/info` | 模型/设备/dtype/语种信息 |
| GET | `/healthz` | 存活探针；`/readyz` 模型就绪探针 |

curl 示例：

```bash
# 1) 保存音色（上传一次，之后复用）
curl -s -X POST http://127.0.0.1:8000/api/v1/voices \
  -H "X-API-Key: $KEY" \
  -F "file=@sample.wav" -F "name=主播小美" -F "ref_text=大家好，欢迎收听。" \
  | python -m json.tool
# → {"id":"3f9c…","name":"主播小美",...}

# 2) 用已保存音色合成
curl -s -X POST http://127.0.0.1:8000/api/v1/tts \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"voice_id":"3f9c…","text":"你好，很高兴认识你","language":"Auto"}' \
  -o out.wav

# 3) 一次性上传克隆（不落库）
curl -s -X POST http://127.0.0.1:8000/api/v1/tts/clone \
  -H "X-API-Key: $KEY" \
  -F "file=@sample.wav" -F "ref_text=大家好，欢迎收听。" \
  -F "text=今天天气不错，适合出行。" -o clone.wav
```

Python 调用示例：

```bash
pip install requests
QTT_BASE=http://127.0.0.1:8000 QTT_API_KEY=xxx python scripts/client_api.py create \
    --name 主播小美 --ref-text "大家好，欢迎收听。" sample.wav
QTT_BASE=http://127.0.0.1:8000 python scripts/client_api.py synth \
    --voice-id <id> --text "你好" -o out.wav
```

### 5.3 冒烟测试

```bash
# 在有完整依赖的环境（或容器内）执行；需额外安装 pytest
pip install pytest
python -m pytest server/tests -q
```

---

## 6. 常见问题

**Q：启动即退出/日志显示模型加载失败？**
检查 `MODEL_PATH` 是否指向含 `speech_tokenizer/` 的模型目录、目录是否只读挂载成功；
`PRELOAD_MODEL=true` 时加载失败会快速退出并由 `restart: unless-stopped` 反复重试。

**Q：T4 上跑会不会崩？**
T4 不支持 bf16 与 flash-attn-2。镜像内置探测：`MODEL_DTYPE=auto`/`ATTN_IMPL=auto`
时会自动落到 **fp16 + eager**。若手动设置成 `bf16`/`flash_attention_2`，启动日志会提示并强制回退。

**Q：重启后之前上传的音色不见了？**
请确认 `DATA_DIR` 指向的宿主目录被正确持久化（未被清理/换挂载点），音色文件在
`{DATA_DIR}/voices/<voice_id>/`（`meta.json` + `voice.pt` 声纹 + `ref.wav`）。
备份时整体备份该目录即可。

**Q：并发请求会 OOM 吗？**
默认 `MAX_CONCURRENCY=1`，所有 GPU 推理串行执行。如需吞吐请横向扩容（多副本，各自挂载模型与不同数据目录）。

**Q：合成等待很久正常吗？**
首次加载模型（1-3 分钟）+ 长文本逐字生成属正常现象；可用 `X-Sample-Rate` 头校验返回。参考音频建议 3-60s、清晰单声道人声。

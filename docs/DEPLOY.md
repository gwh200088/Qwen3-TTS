# Qwen3-TTS 语音克隆服务 · 内网部署手册

本文档说明如何把本项目打包成 Docker 镜像、部署到**内网 GPU 服务器**，
并兼容 **T4（16G）** 与 **A10（24G）** 显卡。

- 模型权重（`Qwen3-TTS-12Hz-1.7B-Base`）**外挂挂载**，不写入镜像；
- 音色库数据目录**外挂持久化**，容器重启/升级后音色不丢失；
- 全程离线推理（容器内禁止任何联网下载）。

> 本文所有命令均为 **Linux** 语法，部署分两个角色：
> 1. **构建机（本地，可联网）**：只负责准备模型 + **构建并导出镜像**（第 3 章）；
> 2. **内网 Linux GPU 主机**：`docker load` + 放置模型 + `docker run` 启动（第 4 章）。
> 镜像不打包模型权重，模型/音色目录均通过 `-v` 挂载。

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

| 宿主目录（示例） | 容器路径 | 说明 |
|---|---|---|
| `/data/models` | `/models`（只读） | 模型父目录，内含 `Qwen3-TTS-12Hz-1.7B-Base/` 与 `speech_tokenizer/` |
| `/data/ttsdata` | `/data`（读写） | 音色库、日志等，**务必放到持久盘** |

---

## 2. 宿主前置条件

1. Linux + NVIDIA 驱动 **≥ 525**（CUDA 12.x；驱动过低的机器无法跑 CUDA 12.4 镜像，需先升级驱动）
2. Docker Engine **≥ 18.06**（本手册以 `docker run` 直跑为主，**无需 docker compose**）
3. GPU 运行时按 Docker 版本安装：
   - **Docker ≥ 19.03**：安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，验证：
     ```bash
     docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
     ```
   - **Docker 18.x**：安装 **nvidia-docker2**（注册 `nvidia` runtime），验证：
     ```bash
     docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
     ```
4. 磁盘：模型 ~4GB、数据目录视音色规模而定

---

## 3. 构建机准备（本地，只做两件事：准备模型 + 构建导出镜像）

### 3.1 准备模型权重（约 4GB，只需 Base 一个目录）

> 只需 `Qwen3-TTS-12Hz-1.7B-Base`（内含 `speech_tokenizer/`，为模型运行所必需，
> 12Hz 分词器已内嵌，无需单独下载 Tokenizer 目录）。
> 若你本地**已有**该目录，可跳过下载步骤，第 4 章直接拷入内网主机。

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

### 3.2 构建镜像并导出（本机唯一要做的事）

```bash
bash scripts/save_image.sh 0.1.1
# 等价于 docker build + docker save，产物：
#   dist/qwen3-tts-server-0.1.1.tar.gz
# 之后只需把这个 tar.gz 拷给内网主机，本地无需再运行容器。
```

> 可选：若 A10 上需要 flash-attention-2，镜像默认不内置（T4 也不支持）。
> 需要时请以 `nvidia/cuda:12.4.1-devel-ubuntu22.04` 自行编译 flash-attn，
> 或 pip 安装预编译 wheel，再打补丁镜像。**不装也不影响功能**：
> A10 等 Ampere+ 卡部署时直接加 `-e ATTN_IMPL=sdpa`（见 4.2/4.4），
> 无需任何编译即可获得比 eager 更快的注意力实现。

---

## 4. 内网 Linux GPU 主机部署

> 启动方式为 **`docker run` 直跑，无需 docker compose，也不使用 .env 文件**
> （模型路径、音色路径、运行参数全部通过 `-v` / `-e` 直接传）。

### 4.1 拷入镜像与模型 + 导入

在**构建机**上把镜像包和模型目录传到内网主机：

```bash
# 构建机（Linux）执行
scp dist/qwen3-tts-server-0.1.1.tar.gz user@<内网IP>:/data/
rsync -av --partial models/ user@<内网IP>:/data/models/
```

在**内网主机**上：

```bash
docker load -i /data/qwen3-tts-server-0.1.1.tar.gz
mkdir -p /data/models /data/ttsdata   # 模型目录 + 音色持久化目录（务必放持久盘）
```

### 4.2 挂载与关键参数参考

容器需挂载两个宿主目录，其余为可选参数（用 `-e` 传入）：

| 用途 | 参数 | 说明 |
|---|---|---|
| 模型目录 | `-v /data/models:/models:ro` | 目录内需含 `Qwen3-TTS-12Hz-1.7B-Base/` |
| 音色库 | `-v /data/ttsdata:/data` | 读写；重建容器前不要删除 |
| 对外端口 | `-p 8000:8000` | 左侧为宿主端口 |
| 模型子目录 | `-e MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base` | 默认即此，子目录名不同才需改 |
| 设备 | `-e DEVICE=cuda:0` | 可 `cpu:0` 纯 CPU 调试 |
| 精度 | `-e MODEL_DTYPE=auto` | T4 自动 fp16；A10 可 `bf16` |
| 注意力 | `-e ATTN_IMPL=sdpa` | A10 等 Ampere+ 卡未装 FA2 时推荐显式 `sdpa`（明显快于 eager）；拿不准用 `auto` |
| 鉴权 | `-e API_KEY=` | 留空免鉴权，设值后需 `X-API-Key` 头 |
| 并发 | `-e MAX_CONCURRENCY=1` | 默认单并发，防 OOM |

### 4.3 T4 / A10 兼容对照

| 显卡 | 架构/算力 | `MODEL_DTYPE` | `ATTN_IMPL` 行为 |
|---|---|---|---|
| **T4 16G** | Turing · sm75 | `auto`→**fp16**（bf16 强制回退 fp16） | `auto`→**eager**（无 FA2） |
| **A10 24G** | Ampere · sm86 | `auto`→fp16；可设 `bf16` | `auto`→FA2(已安装时)/否则 **eager**；未装 FA2 时推荐显式 **`sdpa`**（比 eager 快，无需编译） |

### 4.4 启动（docker run 直跑）

镜像已由构建机导出并 `docker load` 完成（见 4.1），以下命令均在内网主机执行。

#### 方式一：一键脚本（推荐）

将仓库中的 `scripts/run_server.sh` 一并拷入内网主机，它自动探测 Docker 版本并
选择 GPU 注入方式（≥19.03 用 `--gpus all`；18.x 用 `--runtime=nvidia`）：

```bash
cd <内网主机上的项目目录>
MODEL_DIR=/data/models DATA_DIR=/data/ttsdata bash scripts/run_server.sh
# 查看日志 / 健康状态：
docker logs -f qwen3-tts
```

> **A10 等 Ampere+ 卡且镜像未内置 flash-attn 时**，建议追加 `ATTN_IMPL=sdpa`
> 提速（等价于给 `docker run` 传 `-e ATTN_IMPL=sdpa`）：
> ```bash
> MODEL_DIR=/data/models DATA_DIR=/data/ttsdata ATTN_IMPL=sdpa bash scripts/run_server.sh
> ```
> T4 上保持默认即可（自动落到 fp16 + eager）。

#### 方式二：手动 docker run（无需任何仓库文件）

> **性能提示**：内网 GPU 为 **Ampere+（A10 等）** 且镜像未内置 flash-attn 时，
> 下面示例将 `ATTN_IMPL` 设为 `sdpa`（比 `auto` 回退的 eager 明显更快，且无需任何额外编译）。
> 若部署在 **T4** 上，请将其改回 `-e ATTN_IMPL=auto`（自动落到 eager，最稳）。

**情况 A：Docker ≥ 19.03 + nvidia-container-toolkit**

```bash
docker run -d --name qwen3-tts --restart unless-stopped \
  -p 8000:8000 \
  -v /data/models:/models:ro \
  -v /data/ttsdata:/data \
  -e MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base \
  -e DEVICE=cuda:0 -e MODEL_DTYPE=auto -e ATTN_IMPL=auto \
  --gpus all \
  qwen3-tts-server:0.1.1
```

**情况 B：Docker 18.x + nvidia-docker2**

Docker 18 不支持 `--gpus`，改用 `--runtime=nvidia` + NVIDIA 环境变量：

```bash
docker run -d --name qwen3-tts --restart unless-stopped \
  -p 8000:8000 \
  -v /data/models:/models:ro \
  -v /data/ttsdata:/data \
  -e MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base \
  -e DEVICE=cuda:0 -e MODEL_DTYPE=auto -e ATTN_IMPL=sdpa \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --runtime=nvidia \
  qwen3-tts-server:0.1.1
```

> 若报 `runtime nvidia not found`：说明 daemon 未注册 nvidia runtime，
> 请确认已正确安装 nvidia-docker2 并重启 docker（`systemctl restart docker`）。
> 更多可选环境变量（`MAX_CONCURRENCY`、`PRELOAD_MODEL`、`MIN/MAX_AUDIO_SECONDS` 等）见 4.2 表格，
> 用 `-e 变量=值` 追加即可；非 GPU 调试可用 `DEVICE=cpu:0`（去掉 `--gpus`/`--runtime`）。

浏览器打开：`http://<内网IP>:8000/`
Swagger 文档：`http://<内网IP>:8000/docs`

> **重启容器音色不丢**：音色保存在 `/data/ttsdata/voices/`，容器重建时仍用同一
> `-v /data/ttsdata:/data` 挂载即可；停止后再次 `docker start qwen3-tts` 也能恢复。

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

**Q：Docker 18 上启动后没有 GPU（报 cuda 不可用）？**
Docker 18 不支持 `--gpus` 参数，必须用 `--runtime=nvidia` + `NVIDIA_VISIBLE_DEVICES=all`
启动（见 4.4 情况 B，或直接用 `scripts/run_server.sh` 自动适配）；若仍无 GPU，
用 `docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
验证 runtime 是否可用。

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

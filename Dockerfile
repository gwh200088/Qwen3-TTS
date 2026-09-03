# ============================================================================
# Qwen3-TTS Voice Clone Server — 推理镜像
#
#   - 基础镜像：nvidia/cuda 12.4.1 runtime（Ubuntu 22.04 / Python 3.10）
#   - 模型权重不打包进镜像，运行期通过 -v {MODEL_DIR}:/models:ro 外挂
#   - 音色库通过 -v {DATA_DIR}:/data 外挂持久化
#   - 兼容 T4 与 A10：默认 fp16 + eager；A10 可选 bf16 / flash-attn(需自行装入)
#
# 构建（需联网）：
#   docker build -t qwen3-tts-server:0.1.1 .
#
# 在 A10 上如需 flash-attention-2，请自行构建 flash-attn wheel 后
# pip install（镜像默认不编译，未装时 auto 会稳定回退 eager）。
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- 系统依赖：python、ffmpeg/libsndfile1(音频解码)、sox(qwen_tts 导入期必需)、curl(健康检查)
# 默认官方 apt 源；国内/受限网络构建可 --build-arg APT_MIRROR=https://mirrors.aliyun.com/ubuntu 换源提速
ARG APT_MIRROR=http://archive.ubuntu.com/ubuntu
RUN sed -i "s@http://archive.ubuntu.com/ubuntu@${APT_MIRROR}@g; s@http://security.ubuntu.com/ubuntu@${APT_MIRROR}@g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true \
    && apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        python3 \
        python3-pip \
        python3-distutils \
        ffmpeg \
        libsndfile1 \
        sox \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 先装 torch（CUDA wheel，含配套 nvidia-* 运行库）----
RUN python3 -m pip install --no-cache-dir \
        torch==2.6.0 \
        torchaudio==2.6.0 \
    --index-url "${TORCH_INDEX_URL}"

# --- 复制仓库（构建上下文经 .dockerignore 已排除模型/数据/缓存）---
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY qwen_tts ./qwen_tts
COPY server ./server

# --- 安装 Python 依赖（qwen-tts 走 --no-deps，避免拖入 gradio 全家桶）---
# 注意：PyPI 的 sox 是 sdist，构建元数据需 import numpy；librosa 依赖会先拉入 numpy，
# 故 sox 须在上一批安装完成后再单独装，否则报 ModuleNotFoundError: numpy。
RUN python3 -m pip install --no-cache-dir \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        librosa \
        soundfile \
        onnxruntime \
        einops \
        "setuptools>=68" \
        wheel \
    && python3 -m pip install --no-cache-dir -r server/requirements.txt \
    && python3 -m pip install --no-cache-dir --no-build-isolation sox \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

# --- 离线保证（模型外挂，禁止容器内联网下载；缓存目录必须可写，故放 /tmp）---
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/hf-cache \
    MODELSCOPE_CACHE=/tmp/modelscope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz > /dev/null || exit 1

CMD ["python3", "-m", "uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

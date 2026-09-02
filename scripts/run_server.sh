#!/usr/bin/env bash
# ============================================================================
# Qwen3-TTS 服务 · docker run 一键启动（无需 docker compose）
#
# 用法：
#   bash scripts/run_server.sh [镜像tag]            # tag 默认 0.1.1
# 可用环境变量（对应 docs/DEPLOY.md 4.2 / .env.example）：
#   MODEL_DIR  模型父目录（含 Qwen3-TTS-12Hz-1.7B-Base/），默认 ./models
#   DATA_DIR   音色持久化目录，默认 ./data
#   SERVICE_PORT  对外端口，默认 8000
#   API_KEY / DEVICE / MODEL_DTYPE / ATTN_IMPL / MAX_CONCURRENCY / PRELOAD_MODEL ...
#
# GPU 自动适配：
#   - Docker ≥ 19.03 + nvidia-container-toolkit → --gpus all
#   - Docker 18.x + nvidia-docker2               → --runtime=nvidia + NVIDIA_VISIBLE_DEVICES
#   - 可设 GPU_ARGS 强制覆盖，例如 GPU_ARGS="" 跑纯 CPU（需配 DEVICE=cpu:0）
#
# 示例：
#   MODEL_DIR=/data/models DATA_DIR=/data/ttsdata bash scripts/run_server.sh
# ============================================================================
set -euo pipefail

TAG="${1:-0.1.1}"
IMAGE="qwen3-tts-server:${TAG}"
NAME="${CONTAINER_NAME:-qwen3-tts}"
MODEL_DIR="${MODEL_DIR:-./models}"
DATA_DIR="${DATA_DIR:-./data}"
PORT="${SERVICE_PORT:-8000}"

# ---- 前置检查 --------------------------------------------------------------
if ! docker version >/dev/null 2>&1; then
  echo "错误：docker 不可用，请先启动 docker 服务。" >&2
  exit 1
fi
if [ ! -d "${MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-Base" ]; then
  echo "错误：MODEL_DIR=${MODEL_DIR} 下未找到 Qwen3-TTS-12Hz-1.7B-Base/"
  echo "请先执行 bash scripts/download_models.sh，或设置 MODEL_DIR 指向正确目录。" >&2
  exit 1
fi
mkdir -p "${DATA_DIR}"

# ---- 组装 docker run 参数 --------------------------------------------------
run_args=(
  -d --name "${NAME}" --restart unless-stopped
  -p "${PORT}:8000"
  -v "$(cd "${MODEL_DIR}" && pwd):/models:ro"
  -v "$(cd "${DATA_DIR}" && pwd):/data"
  -e "MODEL_PATH=${MODEL_PATH:-/models/Qwen3-TTS-12Hz-1.7B-Base}"
  -e "DEVICE=${DEVICE:-cuda:0}"
  -e "MODEL_DTYPE=${MODEL_DTYPE:-auto}"
  -e "ATTN_IMPL=${ATTN_IMPL:-auto}"
  -e "API_KEY=${API_KEY:-}"
  -e "MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}"
  -e "PRELOAD_MODEL=${PRELOAD_MODEL:-true}"
  -e "MIN_AUDIO_SECONDS=${MIN_AUDIO_SECONDS:-1}"
  -e "MAX_AUDIO_SECONDS=${MAX_AUDIO_SECONDS:-60}"
  -e "MAX_UPLOAD_MB=${MAX_UPLOAD_MB:-50}"
  -e "LOG_LEVEL=${LOG_LEVEL:-info}"
)

# ---- GPU 参数（按 docker 版本自动选择） --------------------------------------
if [ -z "${GPU_ARGS+x}" ]; then
  server_ver="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 0)"
  major="${server_ver%%.*}"
  rest="${server_ver#*.}"
  minor="${rest%%.*}"
  if [ "${major}" -gt 19 ] || { [ "${major}" -eq 19 ] && [ "${minor}" -ge 3 ]; }; then
    GPU_ARGS="--gpus all"
  else
    GPU_ARGS="--runtime=nvidia"
    run_args+=(
      -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
      -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    )
  fi
fi
# shellcheck disable=SC2206
run_args+=($GPU_ARGS)

echo "==> 停止并删除已存在的同名容器（如有）：${NAME}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "==> docker run ${IMAGE}"
echo "    MODEL_DIR : ${MODEL_DIR}  → /models(ro)"
echo "    DATA_DIR  : ${DATA_DIR}   → /data"
echo "    GPU       : ${GPU_ARGS}"
echo "    PORT      : ${PORT}"

docker run "${run_args[@]}" "${IMAGE}"

echo ""
echo "完成。查看： docker logs -f ${NAME}"
echo "       http://127.0.0.1:${PORT}/  (Web UI)"
echo "停止/重启： docker stop ${NAME} && docker start ${NAME}（音色不丢）"

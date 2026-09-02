#!/usr/bin/env bash
# ============================================================================
# 在“可联网的构建机”上构建镜像并导出 tar.gz，拷入内网后 docker load
#
#   bash scripts/save_image.sh [镜像tag]
#   # 产物：dist/qwen3-tts-server-<tag>.tar.gz
#
# 内网主机导入：
#   docker load -i qwen3-tts-server-<tag>.tar.gz
# ============================================================================
set -euo pipefail

TAG="${1:-0.1.1}"
IMAGE="qwen3-tts-server:${TAG}"
OUT_DIR="dist"
OUT_FILE="${OUT_DIR}/qwen3-tts-server-${TAG}.tar.gz"

mkdir -p "${OUT_DIR}"

echo "==> docker build -t ${IMAGE} ."
docker build -t "${IMAGE}" .

echo "==> docker save ${IMAGE} | gzip > ${OUT_FILE}"
docker save "${IMAGE}" | gzip > "${OUT_FILE}"

echo ""
echo "完成：${OUT_FILE}"
echo "拷贝到内网后执行： docker load -i ${OUT_FILE}"

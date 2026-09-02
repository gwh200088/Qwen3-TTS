#!/usr/bin/env bash
# ============================================================================
# 预下载 Qwen3-TTS 1.7B-Base 模型到 ./models（供内网部署外挂挂载）
#
# 用法：
#   bash scripts/download_models.sh
#   下载完成后将整个 ./models 目录拷贝/放置到内网 GPU 主机，
#   docker run 时用 -v <宿主models目录>:/models:ro 挂载即可。
#
# 依赖：python3 + huggingface_hub，或安装 modelscope CLI
# ============================================================================
set -euo pipefail

REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
OUT_DIR="${MODEL_OUT_DIR:-./models/Qwen3-TTS-12Hz-1.7B-Base}"

echo "==> 下载模型: ${REPO_ID}"
echo "==> 保存到 : ${OUT_DIR}"

mkdir -p "$(dirname "${OUT_DIR}")"

# 方式一：优先 modelscope（国内网络更快）
if command -v modelscope >/dev/null 2>&1; then
  echo "==> 使用 modelscope CLI"
  modelscope download --model "${REPO_ID}" --local_dir "${OUT_DIR}"
# 方式二：huggingface_hub snapshot_download（外网/内网有代理时）
else
  echo "==> 使用 huggingface_hub snapshot_download"
  python3 - "$REPO_ID" "$OUT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=out)
print("downloaded:", repo, "->", out)
PY
fi

echo ""
echo "完成。目录结构应为："
echo "  ${OUT_DIR}"
echo "      ├── speech_tokenizer/"
echo "      ├── config.json / generation_config.json"
echo "      └── ... 模型权重"
echo ""
echo "内网主机放置模型后，docker run 挂载该父目录即可，例如："
echo "  -v /data/models:/models:ro"
echo "例如将 ./models 放到 /data/models，则 -v /data/models:/models:ro"

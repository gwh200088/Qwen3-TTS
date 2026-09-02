# coding=utf-8
"""环境变量配置解析。

所有配置均可通过环境变量覆盖，便于容器/编排系统注入。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    # 模型
    model_path: str
    data_dir: str
    device: str
    model_dtype: str            # auto | fp16 | bf16 | fp32
    attn_impl: str              # auto | eager | sdpa | flash_attention_2

    # 服务
    host: str
    port: int
    api_key: str                # 为空则不启用鉴权
    max_concurrency: int
    preload_model: bool
    log_level: str

    # 音频校验
    min_audio_seconds: float
    max_audio_seconds: float
    max_upload_mb: int

    @property
    def voices_dir(self) -> str:
        return os.path.join(self.data_dir, "voices")

    @property
    def has_auth(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model_path=_get_str(
            "MODEL_PATH", "/models/Qwen3-TTS-12Hz-1.7B-Base"),
        data_dir=_get_str("DATA_DIR", "/data"),
        device=_get_str("DEVICE", "cuda:0"),
        model_dtype=_get_str("MODEL_DTYPE", "auto").lower(),
        attn_impl=_get_str("ATTN_IMPL", "auto").lower(),
        host=_get_str("HOST", "0.0.0.0"),
        port=_get_int("PORT", 8000),
        api_key=_get_str("API_KEY", ""),
        max_concurrency=max(1, _get_int("MAX_CONCURRENCY", 1)),
        preload_model=_get_bool("PRELOAD_MODEL", True),
        log_level=_get_str("LOG_LEVEL", "info").upper(),
        min_audio_seconds=_get_float("MIN_AUDIO_SECONDS", 1.0),
        max_audio_seconds=_get_float("MAX_AUDIO_SECONDS", 60.0),
        max_upload_mb=max(1, _get_int("MAX_UPLOAD_MB", 50)),
    )

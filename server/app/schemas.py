# coding=utf-8
"""Pydantic 响应/请求模型。"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VoiceMeta(BaseModel):
    id: str
    name: str
    mode: str                       # "icl" | "xvec"
    ref_text: Optional[str] = None
    duration_seconds: float = 0.0
    sample_rate: int = 0
    source_filename: str = ""
    created_at: float = 0.0
    size_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class VoiceListResponse(BaseModel):
    voices: List[VoiceMeta]
    total: int


class TTSRequest(BaseModel):
    """使用已保存音色合成。"""

    voice_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    language: str = "Auto"

    # 生成参数（可选；缺省时使用模型 generation_config）
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_temperature: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: str = "base"
    model_size: Optional[str] = None
    device: str = ""
    dtype: str = ""
    attn_implementation: Optional[str] = None
    cuda_capability: Optional[List[int]] = None
    languages: List[str] = Field(default_factory=list)
    ready: bool = False


class CloneFormDefaults:
    """multipart 表单字段说明（FastAPI Form 参数已在 api.py 中直接声明）。"""

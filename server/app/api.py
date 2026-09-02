# coding=utf-8
"""/api/v1 REST 路由。

端点：
  - GET    /api/v1/models/info        模型信息
  - POST   /api/v1/voices             创建/保存音色（上传音频）
  - GET    /api/v1/voices             音色列表
  - GET    /api/v1/voices/{id}        音色详情
  - DELETE /api/v1/voices/{id}        删除音色
  - POST   /api/v1/tts                使用已保存音色合成
  - POST   /api/v1/tts/clone          一次性上传音频克隆（不落库）
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from .config import Settings
from .engine import GEN_PARAM_KEYS, EngineNotReadyError, VoiceEngine
from .schemas import ModelInfo, TTSRequest, VoiceListResponse, VoiceMeta
from .voices import VoiceStore, decode_audio_bytes

router = APIRouter(prefix="/api/v1", tags=["api"])


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> VoiceEngine:
    return request.app.state.engine


def get_store(request: Request) -> VoiceStore:
    return request.app.state.store


def http_400(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def http_503(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message)


def _require_engine(request: Request) -> VoiceEngine:
    engine: VoiceEngine = get_engine(request)
    if engine is None:
        raise http_503("模型引擎尚未就绪")
    if not engine.is_ready:
        # PRELOAD_MODEL=false 时在首个请求触发懒加载；失败则 503 并记录日志
        try:
            engine.ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            raise http_503(f"模型加载失败: {exc}") from exc
    return engine


def _require_store(request: Request) -> VoiceStore:
    store: VoiceStore = get_store(request)
    if store is None:
        raise http_503("音色库尚未就绪")
    return store


# ----------------------------------------------------------------------
# 模型信息
# ----------------------------------------------------------------------
@router.get("/models/info", response_model=ModelInfo, summary="模型信息")
def models_info(request: Request) -> ModelInfo:
    engine: VoiceEngine = get_engine(request)
    settings: Settings = get_settings(request)
    base = ModelInfo(
        name="",
        path=settings.model_path,
        ready=False,
        device="",
        dtype="",
        languages=[],
    )
    if not engine.is_ready:
        return base
    info = engine.info()
    return ModelInfo(
        name=info.get("name") or "",
        path=info.get("path") or settings.model_path,
        model_type=info.get("model_type") or "base",
        model_size=info.get("model_size"),
        device=info.get("device") or "",
        dtype=info.get("dtype") or "",
        attn_implementation=info.get("attn_implementation"),
        cuda_capability=info.get("cuda_capability"),
        languages=info.get("languages") or [],
        ready=True,
    )


# ----------------------------------------------------------------------
# 音色 CRUD
# ----------------------------------------------------------------------
@router.post("/voices", response_model=VoiceMeta, status_code=201,
             summary="上传参考音频并保存为音色（返回 voice_id）")
def create_voice(
    request: Request,
    file: UploadFile = File(..., description="参考音频（wav/mp3/flac/ogg/m4a…）"),
    name: str = Form(..., min_length=1, max_length=50, description="音色名称（唯一）"),
    ref_text: Optional[str] = Form(None, description="参考音频文本（ICL 模式必填）"),
    x_vector_only: bool = Form(False, description="true=仅说话人向量（无需文本，效果有限）；false=ICL"),
) -> VoiceMeta:
    engine = _require_engine(request)
    store = _require_store(request)
    xvec = bool(x_vector_only)
    ref_text = (ref_text or "").strip() or None
    if not xvec and not ref_text:
        raise http_400("ICL 模式必须提供参考音频对应的参考文本 ref_text；如需免文本请设置 x_vector_only=true")
    try:
        raw = file.file.read()
    except Exception as exc:  # noqa: BLE001
        raise http_400(f"读取上传文件失败: {exc}") from exc
    try:
        return store.create(
            engine=engine,
            raw_audio=raw,
            source_filename=file.filename or "upload",
            name=name,
            ref_text=ref_text,
            x_vector_only=xvec,
        )
    except EngineNotReadyError as exc:
        raise http_503(str(exc)) from exc


@router.get("/voices", response_model=VoiceListResponse, summary="音色列表")
def list_voices(request: Request) -> VoiceListResponse:
    store = _require_store(request)
    voices = store.list_voices()
    return VoiceListResponse(voices=voices, total=len(voices))


@router.get("/voices/{voice_id}", response_model=VoiceMeta, summary="音色详情")
def get_voice(voice_id: str, request: Request) -> VoiceMeta:
    store = _require_store(request)
    return store.get_meta(voice_id)


@router.delete("/voices/{voice_id}", summary="删除音色")
def delete_voice(voice_id: str, request: Request):
    store = _require_store(request)
    store.delete(voice_id)
    return {"deleted": voice_id, "ok": True}


# ----------------------------------------------------------------------
# TTS 合成
# ----------------------------------------------------------------------
def _collect_gen_kwargs(payload) -> Dict[str, Any]:
    """从请求模型抽取非空生成参数（白名单）。"""
    return {k: getattr(payload, k) for k in GEN_PARAM_KEYS if getattr(payload, k, None) is not None}


def _language_choices(engine: VoiceEngine) -> list:
    try:
        return engine.info().get("languages") or []
    except Exception:  # noqa: BLE001
        return []


def _validate_language(engine: VoiceEngine, language: str) -> str:
    language = (language or "Auto").strip()
    if not language:
        language = "Auto"
    supported = _language_choices(engine)
    if supported and language.casefold() not in {str(x).casefold() for x in supported}:
        raise http_400(f"不支持的语种: {language}，可选: {', '.join(sorted(supported))}")
    return language


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    import soundfile as sf  # noqa: PLC0415

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _audio_response(wav: np.ndarray, sr: int, filename: str) -> StreamingResponse:
    data = _wav_bytes(wav, sr)
    safe_name = "".join(ch for ch in filename if ch.isalnum() or ch in "._-") or "audio"
    return StreamingResponse(
        iter([data]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.wav"',
            "X-Sample-Rate": str(int(sr)),
            "Cache-Control": "no-store",
        },
    )


@router.post("/tts", summary="使用已保存音色合成语音（返回 WAV）")
def tts_synthesize(payload: TTSRequest, request: Request) -> StreamingResponse:
    engine = _require_engine(request)
    store = _require_store(request)
    text = payload.text.strip()
    if not text:
        raise http_400("待合成文本不能为空")
    language = _validate_language(engine, payload.language)

    meta = store.get_meta(payload.voice_id)      # 404 由全局异常处理
    items = store.load_prompt_items(payload.voice_id)
    gen_kwargs = _collect_gen_kwargs(payload)

    wav, sr = engine.synthesize(
        items=items, text=text, language=language, gen_params=gen_kwargs,
    )
    return _audio_response(wav, sr, f"voice_{meta.id}")


@router.post("/tts/clone", summary="一次性上传参考音频克隆合成（不保存音色，返回 WAV）")
def tts_clone_oneshot(
    request: Request,
    file: UploadFile = File(..., description="参考音频（wav/mp3/flac/ogg/m4a…）"),
    text: str = Form(..., description="待合成文本"),
    ref_text: Optional[str] = Form(None, description="参考音频文本（ICL 模式必填）"),
    language: str = Form("Auto", description="语种"),
    x_vector_only: bool = Form(False, description="true=仅说话人向量；false=ICL"),
):
    engine = _require_engine(request)
    text = (text or "").strip()
    if not text:
        raise http_400("待合成文本不能为空")
    language = _validate_language(engine, language)
    xvec = bool(x_vector_only)
    ref_text = (ref_text or "").strip() or None
    if not xvec and not ref_text:
        raise http_400("ICL 模式必须提供参考音频对应的参考文本 ref_text；如需免文本请设置 x_vector_only=true")

    try:
        raw = file.file.read()
        wav, sr = decode_audio_bytes(raw, file.filename or "")
    except Exception as exc:  # noqa: BLE001
        raise http_400(f"参考音频解码失败: {exc}") from exc

    out_wav, out_sr = engine.one_shot_clone(
        wav=wav, sr=sr, ref_text=ref_text, x_vector_only=xvec,
        text=text, language=language, gen_params={},
    )
    return _audio_response(out_wav, out_sr, "clone_oneshot")

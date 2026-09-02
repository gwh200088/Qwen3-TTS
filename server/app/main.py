# coding=utf-8
"""FastAPI 应用入口。

启动：python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import api
from .config import Settings, get_settings
from .engine import EngineError, EngineNotReadyError, VoiceEngine
from .security import APIKeyMiddleware
from .voices import (
    InvalidAudioError,
    InvalidNameError,
    VoiceDuplicateError,
    VoiceNotFoundError,
    VoiceStore,
    VoiceStoreError,
)

logger = logging.getLogger("qwen3tts.server")

BASE_DIR = Path(__file__).resolve().parent          # server/app
STATIC_DIR = BASE_DIR.parent / "static"             # server/static


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    engine = VoiceEngine(settings)
    store = VoiceStore(settings)
    app.state.settings = settings
    app.state.engine = engine
    app.state.store = store
    logger.info("voice store ready at %s (%d voices)",
                store.root, len(store.list_voices()))

    if settings.preload_model:
        logger.info("preloading model from %s ...", settings.model_path)
        engine.load_model()          # 失败即抛出，容器 fail fast / 编排重启
        logger.info("model preloaded")
    else:
        logger.info("PRELOAD_MODEL=0，模型将在首个请求时懒加载")

    yield

    engine.unload()


app = FastAPI(
    title="Qwen3-TTS Voice Clone Server",
    description="语音克隆服务：上传参考音频保存声纹，按 voice_id 克隆合成。",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(APIKeyMiddleware, api_key=get_settings().api_key)
app.include_router(api.router)


# ----------------------------------------------------------------------
# 健康检查
# ----------------------------------------------------------------------
@app.get("/healthz", tags=["system"], include_in_schema=False)
def healthz():
    """存活探针：不依赖 GPU/模型。"""
    return {"status": "ok"}


@app.get("/readyz", tags=["system"], include_in_schema=False)
def readyz(request: Request):
    """就绪探针：模型已加载才返回 200。"""
    engine: VoiceEngine = request.app.state.engine
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": engine.load_error or "model not loaded"},
        )
    return {"status": "ready", "model": os.path.basename(engine.settings.model_path)}


# ----------------------------------------------------------------------
# 统一异常响应
# ----------------------------------------------------------------------
@app.exception_handler(EngineNotReadyError)
async def _engine_not_ready_handler(request: Request, exc: EngineNotReadyError):
    return JSONResponse(status_code=503, content={"error": "model_not_ready", "detail": str(exc)})


@app.exception_handler(EngineError)
async def _engine_error_handler(request: Request, exc: EngineError):
    logger.exception("engine error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "engine_error", "detail": str(exc)})


@app.exception_handler(InvalidNameError)
async def _invalid_name_handler(request: Request, exc: InvalidNameError):
    return JSONResponse(status_code=400, content={"error": "invalid_name", "detail": str(exc)})


@app.exception_handler(InvalidAudioError)
async def _invalid_audio_handler(request: Request, exc: InvalidAudioError):
    return JSONResponse(status_code=400, content={"error": "invalid_audio", "detail": str(exc)})


@app.exception_handler(VoiceDuplicateError)
async def _duplicate_handler(request: Request, exc: VoiceDuplicateError):
    return JSONResponse(status_code=409, content={"error": "duplicate_voice", "detail": str(exc)})


@app.exception_handler(VoiceNotFoundError)
async def _not_found_handler(request: Request, exc: VoiceNotFoundError):
    return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})


@app.exception_handler(VoiceStoreError)
async def _store_error_handler(request: Request, exc: VoiceStoreError):
    logger.exception("voice store error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "voice_store_error", "detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    logger.exception("unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# ----------------------------------------------------------------------
# 静态页（Web UI）；挂载在根路径前已注册 /api 路由，二者互不冲突
# ----------------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def _root():
        return {"name": "Qwen3-TTS Voice Clone Server", "docs": "/docs", "api": "/api/v1"}

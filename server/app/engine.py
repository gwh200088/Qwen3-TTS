# coding=utf-8
"""模型引擎单例：负责模型加载/卸载与所有模型调用。

设计要点：
  - Qwen3TTSModel 在进程内只加载一次（app.state.engine）。
  - 所有 GPU 推理调用使用 RLock 串行化，默认并发 1，防止并发 OOM；
    生成参数合并交给 qwen_tts 自身的 `_merge_generate_kwargs`（读 generate_config.json）。
  - 不记录文本/音频内容，只记录耗时与 voice_id/语种等元信息。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import gpu
from .config import Settings

logger = logging.getLogger("qwen3tts.engine")

# 允许透传到 generate_* 的生成参数白名单
GEN_PARAM_KEYS = (
    "max_new_tokens",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "subtalker_temperature",
    "subtalker_top_k",
    "subtalker_top_p",
)


class EngineError(RuntimeError):
    """引擎基础错误。"""


class EngineNotReadyError(EngineError):
    """模型尚未加载完成。"""


class VoiceEngine:
    """Qwen3TTS Base 模型封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None          # Qwen3TTSModel
        self._runtime: Optional[gpu.RuntimeConfig] = None
        self._model_lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._load_error: Optional[str] = None
        self._load_time_s: float = 0.0
        self._total_calls = 0

    # ------------------------------------------------------------------
    # 加载 / 状态
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def ensure_loaded(self) -> None:
        if self.is_ready:
            return
        self.load_model()

    def load_model(self) -> None:
        """加载模型（幂等，线程安全）。加载失败会抛出异常并记录原因。"""
        with self._load_lock:
            if self.is_ready:
                return
            settings = self.settings
            if not os.path.isdir(settings.model_path):
                raise EngineError(
                    f"模型目录不存在: {settings.model_path}（请检查 MODEL_PATH 挂载）")
            try:
                runtime = gpu.resolve_runtime(
                    device=settings.device,
                    requested_dtype=settings.model_dtype,
                    requested_attn=settings.attn_impl,
                )
                for note in runtime.notes:
                    logger.warning("runtime decision: %s", note)
                logger.info(
                    "loading model from %s (device=%s dtype=%s attn=%s)",
                    settings.model_path, runtime.device, runtime.dtype_name, runtime.attn_impl,
                )

                from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

                t0 = time.time()
                load_kwargs: Dict[str, Any] = dict(
                    device_map=runtime.device,
                    dtype=runtime.dtype,
                )
                if runtime.attn_impl:
                    load_kwargs["attn_implementation"] = runtime.attn_impl
                model = Qwen3TTSModel.from_pretrained(settings.model_path, **load_kwargs)

                t1 = time.time()
                if model.model.tts_model_type != "base":
                    raise EngineError(
                        f"当前模型类型为 {model.model.tts_model_type}，语音克隆需要 base 模型"
                    )

                self._runtime = runtime
                self._model = model
                self._load_error = None
                self._load_time_s = round(t1 - t0, 2)
                logger.info("model loaded in %.2fs", self._load_time_s)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.exception("model load failed: %s", exc)
                raise EngineError(f"模型加载失败: {exc}") from exc

    def unload(self) -> None:
        with self._load_lock:
            model = self._model
            self._model = None
            self._runtime = None
        if model is not None:
            try:
                import torch  # noqa: PLC0415
                import gc

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            logger.info("model unloaded")

    # ------------------------------------------------------------------
    # 模型调用
    # ------------------------------------------------------------------
    def _guard_call(self, fn, *args, **kwargs):
        self.ensure_loaded()
        with self._model_lock:
            self._total_calls += 1
            return fn(*args, **kwargs)

    def create_voice_prompt(
        self,
        wav: np.ndarray,
        sr: int,
        ref_text: Optional[str],
        x_vector_only: bool,
    ) -> List[Any]:
        """从参考音频波形创建声纹 prompt items（result: List[VoiceClonePromptItem]）。"""
        def _fn():
            model = self._require_model()
            return model.create_voice_clone_prompt(
                ref_audio=(wav, int(sr)),
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
            )

        return self._guard_call(_fn)

    def synthesize(
        self,
        items: List[Any],
        text: str,
        language: str,
        gen_params: Dict[str, Any],
    ) -> Tuple[np.ndarray, int]:
        """按已提取的声纹 items 合成语音。返回 (wav, sample_rate)。"""
        def _fn():
            model = self._require_model()
            return model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=items,
                **gen_params,
            )

        start = time.time()
        result = self._guard_call(_fn)
        logger.info(
            "synthesize done in %.2fs language=%s gen_params=%s",
            time.time() - start, language, gen_params,
        )
        wavs, out_sr = result
        return wavs[0], int(out_sr)

    def one_shot_clone(
        self,
        wav: np.ndarray,
        sr: int,
        ref_text: Optional[str],
        x_vector_only: bool,
        text: str,
        language: str,
        gen_params: Dict[str, Any],
    ) -> Tuple[np.ndarray, int]:
        """一次性上传音频克隆合成（不落库）。"""
        def _fn():
            model = self._require_model()
            return model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=(wav, int(sr)),
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
                **gen_params,
            )

        start = time.time()
        result = self._guard_call(_fn)
        logger.info("one-shot clone done in %.2fs", time.time() - start)
        wavs, out_sr = result
        return wavs[0], int(out_sr)

    # ------------------------------------------------------------------
    # 信息
    # ------------------------------------------------------------------
    def _require_model(self):
        if not self.is_ready:
            raise EngineNotReadyError("模型尚未加载完成，请稍后重试")
        return self._model

    def info(self) -> Dict[str, Any]:
        model = self._require_model()
        model_cls = model.model
        langs = None
        try:
            langs = model.get_supported_languages()
        except Exception:  # noqa: BLE001
            pass
        info: Dict[str, Any] = {
            "name": os.path.basename(os.path.normpath(self.settings.model_path)),
            "path": self.settings.model_path,
            "model_type": getattr(model_cls, "tts_model_type", "base"),
            "model_size": getattr(model_cls, "tts_model_size", None),
            "languages": langs or [],
            "load_time_s": self._load_time_s,
            "total_calls": self._total_calls,
            "ready": True,
        }
        if self._runtime is not None:
            info.update(self._runtime.as_dict())
        return info

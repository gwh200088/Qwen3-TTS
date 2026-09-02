# coding=utf-8
"""音色库（VoiceStore）：文件系统持久化。

目录布局（DATA_DIR 外挂持久化卷）：
    {DATA_DIR}/voices/{voice_id}/
        ├── meta.json   元信息（名称/模式/ref_text/时长/创建时间…）
        ├── voice.pt    {"items": [VoiceClonePromptItem asdict 序列]}（与官方 demo 互读）
        └── ref.wav     归一化参考音频副本（审计/复现用）
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Settings
from .schemas import VoiceMeta

logger = logging.getLogger("qwen3tts.voices")

_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff .\-()（）]{1,50}$")


class VoiceStoreError(RuntimeError):
    """音色库错误基类。"""


class VoiceDuplicateError(VoiceStoreError):
    pass


class VoiceNotFoundError(VoiceStoreError):
    pass


class InvalidAudioError(VoiceStoreError):
    pass


class InvalidNameError(VoiceStoreError):
    pass


# ----------------------------------------------------------------------
# 音频解码
# ----------------------------------------------------------------------
def _ffmpeg_probe(path: str) -> Optional[int]:
    """用 ffprobe 读取采样率；失败返回 None。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def _ffmpeg_decode(path: str) -> Tuple[np.ndarray, int]:
    sr = _ffmpeg_probe(path)
    if sr is None:
        sr = 24000
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
         "-ar", str(sr), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        capture_output=True, timeout=120, check=True,
    )
    data = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if data.size == 0:
        raise InvalidAudioError("ffmpeg 解码结果为空")
    return data, int(sr)


def decode_audio_bytes(raw: bytes, source_filename: str = "") -> Tuple[np.ndarray, int]:
    """把上传音频字节解码为 (mono float32 wav, sr)。

    依次尝试：soundfile(bytes) → 临时文件 + librosa(支持 ffmpeg/audioread) → ffmpeg 原始 PCM。
    """
    if not raw:
        raise InvalidAudioError("上传的音频为空")

    # 1) soundfile 直接解码（wav/flac/ogg/部分 mp3）
    try:
        with io.BytesIO(raw) as f:
            audio, sr = sf_read(f)
        return _to_mono_float32(audio), int(sr)
    except Exception:  # noqa: BLE001
        pass

    # 2) 临时文件 + librosa（mp3/m4a 等需要 ffmpeg/audioread 后端）
    with tempfile.NamedTemporaryFile(suffix=_safe_suffix(source_filename), delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        import librosa  # noqa: PLC0415

        try:
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            return audio.astype(np.float32), int(sr)
        except Exception:  # noqa: BLE001
            pass
        # 3) ffmpeg 兜底（任何失败统一归为“无法解析的音频”）
        try:
            audio, sr = _ffmpeg_decode(tmp_path)
            return _to_mono_float32(audio), int(sr)
        except InvalidAudioError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InvalidAudioError(f"音频解码失败，请确认上传的是有效音频文件: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def sf_read(f):
    import soundfile as sf  # noqa: PLC0415

    return sf.read(f, dtype="float32", always_2d=False)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _safe_suffix(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext and len(ext) <= 10 and ext.replace(".", "").isalnum():
        return ext
    return ".audio"


# ----------------------------------------------------------------------
# VoiceStore
# ----------------------------------------------------------------------
class VoiceStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.voices_dir
        self._lock = threading.RLock()
        os.makedirs(self.root, exist_ok=True)

    # ---- helpers -------------------------------------------------
    def _dir_for(self, voice_id: str) -> str:
        # 防止路径穿越
        if not voice_id or "/" in voice_id or "\\" in voice_id or voice_id in (".", ".."):
            raise VoiceNotFoundError(f"非法的 voice_id: {voice_id!r}")
        return os.path.join(self.root, voice_id)

    def _validate_name(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise InvalidNameError("音色名称不能为空")
        if not _NAME_RE.match(name):
            raise InvalidNameError("音色名称仅支持中英文/数字/空格及 .-() 等字符，长度 ≤ 50")
        return name

    # ---- CRUD ----------------------------------------------------
    def create(
        self,
        engine,
        raw_audio: bytes,
        source_filename: str,
        name: str,
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
    ) -> VoiceMeta:
        settings = self.settings
        if len(raw_audio) > settings.max_upload_mb * 1024 * 1024:
            raise InvalidAudioError(f"音频文件过大（超过 {settings.max_upload_mb}MB）")
        name = self._validate_name(name)
        key = name.casefold()

        with self._lock:
            if self._name_exists(key):
                raise VoiceDuplicateError(f"音色名称已存在: {name}")

            wav, sr = decode_audio_bytes(raw_audio, source_filename)
            duration = len(wav) / float(sr)
            if duration < settings.min_audio_seconds:
                raise InvalidAudioError(
                    f"参考音频过短（{duration:.2f}s），至少需要 {settings.min_audio_seconds:g}s")
            if duration > settings.max_audio_seconds:
                raise InvalidAudioError(
                    f"参考音频过长（{duration:.2f}s），超过上限 {settings.max_audio_seconds:g}s")

            # 声纹提取（GPU 推理，串行）
            items = engine.create_voice_prompt(
                wav=wav, sr=sr, ref_text=(ref_text or "").strip() or None,
                x_vector_only=bool(x_vector_only),
            )
            if not items:
                raise VoiceStoreError("声纹提取结果为空")

            voice_id = uuid.uuid4().hex[:16]
            voice_dir = self._dir_for(voice_id)
            os.makedirs(voice_dir, exist_ok=True)
            try:
                meta = VoiceMeta(
                    id=voice_id,
                    name=name,
                    mode="xvec" if x_vector_only else "icl",
                    ref_text=(ref_text or "").strip() or None,
                    duration_seconds=round(duration, 3),
                    sample_rate=int(sr),
                    source_filename=source_filename,
                    created_at=time.time(),
                    size_bytes=len(raw_audio),
                )
                self._save_payload(voice_dir, items, meta)
                self._write_ref_wav(voice_dir, wav, sr)
                self._write_meta(voice_dir, meta)
            except Exception:
                shutil.rmtree(voice_dir, ignore_errors=True)
                raise
            logger.info("voice created: id=%s name=%s mode=%s duration=%.2fs",
                        voice_id, name, meta.mode, duration)
            return meta

    def _name_exists(self, key: str) -> bool:
        for meta in self._iter_metas():
            if (meta.get("name") or "").casefold() == key:
                return True
        return False

    def _iter_metas(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not os.path.isdir(self.root):
            return results
        for entry in os.scandir(self.root):
            if not entry.is_dir():
                continue
            meta_path = os.path.join(entry.path, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception:  # noqa: BLE001
                logger.warning("skip broken voice meta: %s", meta_path, exc_info=True)
        return results

    def list_voices(self) -> List[VoiceMeta]:
        metas = sorted(self._iter_metas(), key=lambda m: m.get("created_at", 0.0), reverse=True)
        voices = []
        for m in metas:
            try:
                voices.append(VoiceMeta(**m))
            except Exception:  # noqa: BLE001
                logger.warning("invalid voice meta: %s", m.get("id"), exc_info=True)
        return voices

    def get_meta(self, voice_id: str) -> VoiceMeta:
        meta_path = os.path.join(self._dir_for(voice_id), "meta.json")
        if not os.path.isfile(meta_path):
            raise VoiceNotFoundError(f"音色不存在: {voice_id}")
        with open(meta_path, "r", encoding="utf-8") as f:
            return VoiceMeta(**json.load(f))

    def load_prompt_items(self, voice_id: str) -> List[Any]:
        """读取声纹 .pt 并还原为 List[VoiceClonePromptItem]。"""
        from qwen_tts import VoiceClonePromptItem  # noqa: PLC0415

        import torch  # noqa: PLC0415

        pt_path = os.path.join(self._dir_for(voice_id), "voice.pt")
        if not os.path.isfile(pt_path):
            raise VoiceNotFoundError(f"音色声纹文件不存在: {voice_id}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "items" not in payload:
            raise VoiceStoreError(f"音色文件格式错误: {voice_id}")
        items_raw = payload["items"]
        if not isinstance(items_raw, list) or not items_raw:
            raise VoiceStoreError(f"音色声纹为空: {voice_id}")

        items: List[VoiceClonePromptItem] = []
        for d in items_raw:
            if not isinstance(d, dict):
                raise VoiceStoreError("音色内部 item 格式错误")
            ref_code = d.get("ref_code")
            if ref_code is not None and not isinstance(ref_code, (torch.Tensor, list)):
                ref_code = torch.tensor(ref_code)
            elif isinstance(ref_code, list):
                ref_code = torch.tensor(ref_code)
            ref_spk = d.get("ref_spk_embedding")
            if ref_spk is None:
                raise VoiceStoreError("缺少 ref_spk_embedding")
            if not isinstance(ref_spk, torch.Tensor):
                ref_spk = torch.tensor(ref_spk)
            items.append(VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text"),
            ))
        return items

    def delete(self, voice_id: str) -> None:
        voice_dir = self._dir_for(voice_id)
        if not os.path.isdir(voice_dir):
            raise VoiceNotFoundError(f"音色不存在: {voice_id}")
        with self._lock:
            shutil.rmtree(voice_dir, ignore_errors=True)
        logger.info("voice deleted: id=%s", voice_id)

    # ---- 落盘 ------------------------------------------------
    def _save_payload(self, voice_dir: str, items, meta: VoiceMeta) -> None:
        import torch  # noqa: PLC0415

        payload = {"items": [asdict(it) for it in items]}
        torch.save(payload, os.path.join(voice_dir, "voice.pt"))

    def _write_meta(self, voice_dir: str, meta: VoiceMeta) -> None:
        with open(os.path.join(voice_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)

    def _write_ref_wav(self, voice_dir: str, wav: np.ndarray, sr: int) -> None:
        import soundfile as sf  # noqa: PLC0415

        sf.write(os.path.join(voice_dir, "ref.wav"), wav, sr, format="WAV", subtype="PCM_16")

# coding=utf-8
"""服务端离线冒烟测试（不需要 GPU，但需要已安装 qwen_tts + torch + soundfile + fastapi）。

运行（在安装了 server/requirements.txt 且 `pip install .` 后的环境）：
    python -m pytest server/tests -q
    # 或
    python -m pytest server/tests/test_api_smoke.py -q
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile

import numpy as np
import pytest

# 必须在导入 server 模块前配置环境变量（get_settings 带缓存）
_TMP = tempfile.mkdtemp(prefix="qwen3tts_test_")
os.environ["DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["MODEL_PATH"] = os.path.join(_TMP, "no_such_model")
os.environ["PRELOAD_MODEL"] = "false"
os.environ["DEVICE"] = "cpu"
os.environ["MODEL_DTYPE"] = "fp32"
os.environ["MIN_AUDIO_SECONDS"] = "0.1"
os.environ["MAX_AUDIO_SECONDS"] = "10"
os.environ["API_KEY"] = ""

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from qwen_tts import VoiceClonePromptItem  # noqa: E402

from server.app import main as app_main  # noqa: E402


def _make_wav_bytes(seconds: float = 1.2, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeEngine:
    """替身引擎：不加载模型，仅返回固定形状结果，便于测试 API 契约。"""

    def __init__(self) -> None:
        self.ready = True
        self.calls = 0

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def load_error(self) -> None:
        return None

    @property
    def settings(self):
        return app_main.get_settings()

    def ensure_loaded(self) -> None:
        return None

    def info(self) -> dict:
        return {
            "name": "fake",
            "path": str(self.settings.model_path),
            "model_type": "base",
            "model_size": None,
            "device": "cpu",
            "dtype": "float32",
            "attn_implementation": "eager",
            "cuda_capability": None,
            "languages": ["Auto"],
            "ready": True,
        }

    def create_voice_prompt(self, wav, sr, ref_text, x_vector_only):
        self.calls += 1
        spk = torch.zeros(512, dtype=torch.float32)
        if x_vector_only:
            return [VoiceClonePromptItem(
                ref_code=None, ref_spk_embedding=spk,
                x_vector_only_mode=True, icl_mode=False, ref_text=None)]
        code = torch.zeros((1, 8), dtype=torch.long)
        return [VoiceClonePromptItem(
            ref_code=code, ref_spk_embedding=spk,
            x_vector_only_mode=False, icl_mode=True, ref_text=ref_text)]

    def synthesize(self, items, text, language, gen_params):
        self.calls += 1
        wav = np.zeros(int(24000 * 0.5), dtype=np.float32)
        return wav, 24000

    def one_shot_clone(self, wav, sr, ref_text, x_vector_only, text, language, gen_params):
        self.calls += 1
        return self.synthesize([], text, language, gen_params)


@pytest.fixture(scope="module")
def client():
    with TestClient(app_main.app) as c:
        c.app.state.engine = FakeEngine()      # 替换为替身引擎
        c.app.state.store = app_main.app.state.store
        yield c
    shutil.rmtree(_TMP, ignore_errors=True)


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/api/v1/models/info")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True


def test_voice_crud(client):
    wav = _make_wav_bytes()

    # 创建
    r = client.post("/api/v1/voices", data={
        "name": "测试音色", "ref_text": "参考文本", "x_vector_only": "false",
    }, files={"file": ("demo.wav", wav, "audio/wav")})
    assert r.status_code == 201, r.text
    meta = r.json()
    vid = meta["id"]
    assert meta["mode"] == "icl"
    assert meta["name"] == "测试音色"

    # 重名 → 409
    r2 = client.post("/api/v1/voices", data={
        "name": "测试音色", "ref_text": "x", "x_vector_only": "false",
    }, files={"file": ("demo.wav", wav, "audio/wav")})
    assert r2.status_code == 409

    # 列表 / 详情
    assert client.get("/api/v1/voices").json()["total"] == 1
    detail = client.get(f"/api/v1/voices/{vid}").json()
    assert detail["id"] == vid

    # 合成（使用已保存音色）
    r3 = client.post("/api/v1/tts", json={"voice_id": vid, "text": "你好世界", "language": "Auto"})
    assert r3.status_code == 200, r3.text
    assert r3.headers["content-type"].startswith("audio/wav")
    assert len(r3.content) > 0

    # ICL 缺参考文本的一次性克隆 → 400
    r4 = client.post("/api/v1/tts/clone", data={"text": "hi"},
                     files={"file": ("demo.wav", wav, "audio/wav")})
    assert r4.status_code == 400

    # 一次性克隆（x-vector）
    r5 = client.post("/api/v1/tts/clone", data={"text": "hi", "x_vector_only": "true"},
                     files={"file": ("demo.wav", wav, "audio/wav")})
    assert r5.status_code == 200, r5.text
    assert r5.headers["content-type"].startswith("audio/wav")

    # 删除
    assert client.delete(f"/api/v1/voices/{vid}").status_code == 200
    assert client.get("/api/v1/voices").json()["total"] == 0
    assert client.get(f"/api/v1/voices/{vid}").status_code == 404


def test_invalid_audio(client):
    r = client.post("/api/v1/voices", data={
        "name": "坏音频", "ref_text": "x", "x_vector_only": "false",
    }, files={"file": ("bad.txt", b"not an audio", "text/plain")})
    assert r.status_code == 400

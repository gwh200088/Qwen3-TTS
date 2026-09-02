# coding=utf-8
"""GPU 能力探测与运行配置决策（T4 / A10 兼容策略）。

关键约束：
  - T4（Turing, compute capability 7.5）：不支持 bfloat16，也不支持
    官方 flash-attention-2（要求 sm80+），因此强制 fp16 + eager。
  - A10（Ampere, cc 8.6）：支持 bf16 与 flash-attention-2；默认仍用 fp16
    （对两种卡都稳妥），可通过环境变量显式选择 bf16 / flash_attention_2。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("qwen3tts.gpu")

DTYPE_ALIASES = {
    "fp16": "float16",
    "half": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}

_DEVICE_RE = re.compile(r"^cuda:(\d+)$")


def _torch():
    """延迟导入 torch（避免无 torch 环境下 import 本模块即失败）。"""
    import torch  # noqa: PLC0415

    return torch


def torch_dtype_from_name(name: str):
    """把 fp16/bf16/fp32 等字符串映射为 torch.dtype。"""
    torch = _torch()
    key = DTYPE_ALIASES.get(str(name).lower())
    if key is None:
        raise ValueError(f"MODEL_DTYPE 取值不合法: {name}，支持 fp16/bf16/fp32。")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[key]


def parse_device_index(device: str) -> Optional[int]:
    """从 'cuda' / 'cuda:0' 解析出 GPU 索引；非 cuda 设备返回 None。"""
    if not device or not str(device).startswith("cuda"):
        return None
    m = _DEVICE_RE.match(str(device))
    return int(m.group(1)) if m else 0


def cuda_capability(device: str) -> Optional[Tuple[int, int]]:
    """返回 (major, minor)；无 CUDA / 非 cuda 设备返回 None。"""
    idx = parse_device_index(device)
    if idx is None:
        return None
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(idx)
        return (int(major), int(minor))
    except Exception:  # noqa: BLE001
        return None


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def is_ampere_or_newer(cap: Optional[Tuple[int, int]]) -> bool:
    return bool(cap) and cap[0] >= 8


def choose_dtype(device: str, cap: Optional[Tuple[int, int]], requested: str) -> Tuple[str, str]:
    """返回 (dtype_name, note)。requested: auto|fp16|bf16|fp32。"""
    requested = str(requested or "auto").lower()
    on_cpu = not str(device).startswith("cuda") or cap is None

    if requested == "auto":
        if on_cpu:
            return "float32", "未检测到可用 CUDA，回退 CPU + float32"
        return "float16", "auto 策略默认 fp16"
    if requested == "bf16":
        if on_cpu:
            return "float32", "CPU 不支持 bf16，已回退 float32"
        if not is_ampere_or_newer(cap):
            return "float16", "T4(Turing) 不支持 bf16，已强制回退 fp16"
    try:
        return DTYPE_ALIASES[requested], ""
    except KeyError as e:  # noqa: BLE001
        raise ValueError(f"MODEL_DTYPE 不合法: {requested}") from e


def choose_attn(device: str, cap: Optional[Tuple[int, int]], requested: str) -> Tuple[Optional[str], str]:
    """返回 (attn_implementation, note)；None 表示交由 transformers 默认。"""
    on_cuda = str(device).startswith("cuda")
    ampere = on_cuda and is_ampere_or_newer(cap)
    requested = str(requested or "auto").lower()
    if requested == "auto":
        if ampere and flash_attn_available():
            return "flash_attention_2", "Ampere+ 且 flash-attn 可用 → flash_attention_2"
        return "eager", "未满足 FA2 条件(T4/CPU/未安装) → eager(兼容性最稳)"
    if requested in ("eager", "sdpa"):
        return requested, ""
    if requested == "flash_attention_2":
        if ampere and flash_attn_available():
            return "flash_attention_2", ""
        return "eager", "显式请求 flash_attention_2 但硬件/库不满足，已回退 eager"
    raise ValueError(f"ATTN_IMPL 不合法: {requested}，支持 auto/eager/sdpa/flash_attention_2")


class RuntimeConfig:
    """不可变运行时决策结果。"""

    __slots__ = ("device", "dtype", "dtype_name", "attn_impl", "capability", "notes")

    def __init__(
        self,
        device: str,
        dtype,
        dtype_name: str,
        attn_impl: Optional[str],
        capability: Optional[Tuple[int, int]],
        notes: List[str],
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.dtype_name = dtype_name
        self.attn_impl = attn_impl
        self.capability = capability
        self.notes = notes

    def as_dict(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "dtype": self.dtype_name,
            "attn_implementation": self.attn_impl,
            "cuda_capability": list(self.capability) if self.capability else None,
        }


def resolve_runtime(device: str, requested_dtype: str, requested_attn: str) -> RuntimeConfig:
    """决策最终的 (device, dtype, attn_impl)。

    规则：
      1. 无法使用 CUDA（无卡/驱动问题）→ 回退 CPU，警告；dtype float32，attn eager。
      2. capability < 8.0（T4）→ fp16 + eager。
      3. capability >= 8.0（A10 等）→ 默认 fp16；若 auto 且 flash_attn 可导入则 FA2。
    """
    torch = _torch()
    cap = cuda_capability(device)
    notes: List[str] = []

    if str(device).startswith("cuda") and cap is None:
        device = "cpu"
        notes.append("DEVICE=cuda 但未检测到可用 CUDA，已回退 CPU（性能极低，仅建议调试）")

    dtype_name, note = choose_dtype(device, cap, requested_dtype)
    if note:
        notes.append(note)
    attn, note2 = choose_attn(device, cap, requested_attn)
    if note2:
        notes.append(note2)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return RuntimeConfig(
        device=device,
        dtype=dtype_map[dtype_name],
        dtype_name=dtype_name,
        attn_impl=attn,
        capability=cap,
        notes=notes,
    )

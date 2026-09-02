#!/usr/bin/env python3
# coding=utf-8
"""Qwen3-TTS 语音克隆服务 — API 客户端示例（requests）。

用法示例：
    python scripts/client_api.py create --name 主播小明 --ref-text "大家晚上好" demo.wav
    python scripts/client_api.py list
    python scripts/client_api.py synth --voice-id <id> --text "你好，很高兴认识你。" -o out.wav
    python scripts/client_api.py clone --ref demo.wav --ref-text "大家晚上好" \
        --text "今天天气不错。" -o clone.wav

环境变量：QTT_BASE=http://127.0.0.1:8000   QTT_API_KEY=可选
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

BASE = os.getenv("QTT_BASE", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("QTT_API_KEY", "")


def headers() -> dict:
    h = {"Accept": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def err_and_exit(resp: requests.Response) -> None:
    try:
        body = resp.json()
        msg = body.get("detail") or body.get("error") or resp.text
    except Exception:
        msg = resp.text
    sys.exit(f"HTTP {resp.status_code}: {msg}")


def cmd_create(args: argparse.Namespace) -> None:
    with open(args.audio, "rb") as f:
        data = {"name": args.name, "x_vector_only": "true" if args.xvec else "false"}
        if args.ref_text:
            data["ref_text"] = args.ref_text
        r = requests.post(f"{BASE}/api/v1/voices", headers=headers(),
                          data=data, files={"file": (os.path.basename(args.audio), f)})
    if r.status_code != 201:
        err_and_exit(r)
    meta = r.json()
    print(f"OK 音色已保存 → id={meta['id']} name={meta['name']} mode={meta['mode']}")
    print(f"   参考时长 {meta['duration_seconds']}s，采样率 {meta['sample_rate']}Hz")
    print(f"   后续合成请携带 voice_id: {meta['id']}")


def cmd_list(_: argparse.Namespace) -> None:
    r = requests.get(f"{BASE}/api/v1/voices", headers=headers())
    if r.status_code != 200:
        err_and_exit(r)
    voices = r.json().get("voices", [])
    if not voices:
        print("（音色库为空）")
        return
    for v in voices:
        print(f"{v['id']}  {v['name']:<20} mode={v['mode']:<4} "
              f"dur={v['duration_seconds']:>6.2f}s  created={v['created_at']:.0f}")


def cmd_synth(args: argparse.Namespace) -> None:
    body = {
        "voice_id": args.voice_id,
        "text": args.text,
        "language": args.language or "Auto",
    }
    if args.max_new_tokens:
        body["max_new_tokens"] = args.max_new_tokens
    r = requests.post(f"{BASE}/api/v1/tts", headers=headers(), json=body)
    if r.status_code != 200:
        err_and_exit(r)
    out = args.output or "tts_output.wav"
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"OK 已保存 {len(r.content)} 字节 → {out}（采样率 {r.headers.get('X-Sample-Rate')}Hz）")


def cmd_clone(args: argparse.Namespace) -> None:
    if not args.ref_text and not args.xvec:
        sys.exit("ICL 模式需 --ref-text；或加 --xvec 使用仅说话人向量")
    with open(args.ref_audio, "rb") as f:
        data = {"text": args.text, "language": args.language or "Auto",
                "x_vector_only": "true" if args.xvec else "false"}
        if args.ref_text:
            data["ref_text"] = args.ref_text
        r = requests.post(f"{BASE}/api/v1/tts/clone", headers=headers(),
                          data=data, files={"file": (os.path.basename(args.ref_audio), f)})
    if r.status_code != 200:
        err_and_exit(r)
    out = args.output or "clone_output.wav"
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"OK 已保存 {len(r.content)} 字节 → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-TTS voice-clone API 示例客户端")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="上传音频保存音色")
    c.add_argument("audio")
    c.add_argument("--name", required=True)
    c.add_argument("--ref-text")
    c.add_argument("--xvec", action="store_true")
    c.set_defaults(fn=cmd_create)

    sub.add_parser("list", help="音色列表").set_defaults(fn=cmd_list)

    s = sub.add_parser("synth", help="使用已保存音色合成")
    s.add_argument("--voice-id", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--language", default="Auto")
    s.add_argument("--max-new-tokens", type=int)
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_synth)

    c2 = sub.add_parser("clone", help="一次性上传克隆合成")
    c2.add_argument("--ref-audio", required=True)
    c2.add_argument("--ref-text")
    c2.add_argument("--text", required=True)
    c2.add_argument("--language", default="Auto")
    c2.add_argument("--xvec", action="store_true")
    c2.add_argument("-o", "--output")
    c2.set_defaults(fn=cmd_clone)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

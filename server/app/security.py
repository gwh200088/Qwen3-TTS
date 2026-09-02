# coding=utf-8
"""可选 API Key 鉴权中间件。

规则：
  - 未配置 API_KEY 时不做任何校验（内网默认可用）。
  - 配置后：所有以 /api/ 开头的请求必须携带匹配的 X-API-Key 头；
    静态页面、/healthz、/readyz 保持开放（页面本身无需鉴权即可打开，
    浏览器中的 API 调用由页面携带用户输入的 key）。
"""
from __future__ import annotations

import hmac
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, api_key: Optional[str] = None) -> None:
        super().__init__(app)
        self._api_key = (api_key or "").strip() or None

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if self._api_key and path.startswith("/api/"):
            provided = request.headers.get("x-api-key") or request.headers.get("authorization")
            if provided and provided.lower().startswith("bearer "):
                provided = provided[7:]
            if not provided or not hmac.compare_digest(provided, self._api_key):
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "detail": "缺少或错误的 X-API-Key"},
                )
        return await call_next(request)

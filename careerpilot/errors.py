"""统一 API 错误响应。

目标：把项目里所有接口的错误返回格式收敛成一种：

    {"code": <HTTP状态码>, "msg": <人类可读信息>, "trace_id": <本次请求ID>}

- code    ：HTTP 状态码，方便前端按码区分错误类型。
- msg     ：给用户看的、不泄露内部信息的错误信息。
- trace_id：本次请求的唯一 ID（来自 RequestContextMiddleware 设置的
            request.state.request_id），前端/日志可拿它定位对应请求。

用法：在应用入口（main.py）里调用 register_exception_handlers(app)。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("careerpilot")


def _to_response(request: Request, status_code: int, msg: str):
    """构造统一格式的错误响应，并带上本次请求的 trace_id。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "msg": msg,
            "trace_id": getattr(request.state, "request_id", "") or "",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """给 app 注册统一异常处理器（HTTP 异常 / 参数校验 / 未捕获异常）。

    - StarletteHTTPException：route 里主动 raise 的 HTTPException（4xx/5xx）。
    - RequestValidationError：请求体/参数未通过 Pydantic 校验（默认 422）。
    - Exception：其他所有未捕获异常（默认 500），只在服务端记日志，不向客户端
      泄漏堆栈。
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return _to_response(request, exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # 取第一条校验错误拼成可读信息，避免把整个 errors 数组原样暴露
        msg = "请求参数校验失败"
        try:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", [])) or "body"
            msg = f"参数 [{'body' if loc == 'body' else loc}]：{first.get('msg', '不合法')}"
        except Exception:
            pass
        return _to_response(request, 422, msg)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # 只在服务端记录完整堆栈，响应体给统一的、不含内部细节的错误信息
        logger.error("未处理异常", exc_info=(type(exc), exc, exc.__traceback__))
        return _to_response(request, 500, "服务器内部错误")
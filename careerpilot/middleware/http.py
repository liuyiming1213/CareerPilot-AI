"""HTTP 请求中间件。

作用域说明：
- 本模块基于 Starlette 的 BaseHTTPMiddleware 实现一个 Web 中间件，
  会在每个 HTTP 请求进入路由处理前、以及响应返回客户端前"插一脚"；
- 它做三件事：为请求分配唯一 request_id、统计并回写处理耗时、把未捕获异常
  统一转成 500 JSON 响应（避免把堆栈泄漏给客户端）。
- 注意：为了让纯 Agent 单元测试不依赖 Web 框架，careerpilot/middleware/__init__.py
  不会自动导入本模块，需要显式导入。
"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RequestContextMiddleware(BaseHTTPMiddleware):
    """给每个请求补充 request_id、耗时响应头并统一保护异常。

    继承 BaseHTTPMiddleware 后，只需重写 async dispatch 方法，
    call_next 是"继续执行后续路由处理"的协程。
    """
    async def dispatch(self, request: Request, call_next):
        # 请求 ID 优先取客户端传入的 X-Request-ID（便于跨系统链路追踪），
        # 客户端没传则自己生成一个 UUID，保证每个请求都有唯一标识
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # 挂到 request.state 上，后续任何处理器（路由、其他中间件）都能通过
        # request.state.request_id 拿到同一个 ID
        request.state.request_id = request_id
        # perf_counter 是高精度计时器，用于统计整个请求的处理耗时
        started = time.perf_counter()
        try:
            # call_next(request)：把请求交给后续路由/其他中间件继续处理
            response = await call_next(request)
            # 回写响应头，方便日志/监控/前端追踪这个请求对应哪个 request_id。
            # 注意必须在 call_next 返回 response 之后再设置头部，因为响应对象
            # 只有在这个时刻才真正创建出来，之前无法修改
            response.headers["X-Request-ID"] = request_id
            # 把处理耗时（秒，保留 4 位小数）也写进响应头，便于排查性能问题
            response.headers["X-Process-Time"] = f"{time.perf_counter() - started:.4f}"
            return response
        except Exception:
            # 兜底：任何未被路由处理的异常都在这里转成统一的 500 JSON，
            # 且大概率永远到不了这里——因为 FastAPI 的 Exception handler 会优先
            # 处理路由层异常。这里作为最后一道防线，返回统一错误格式 {code,msg,trace_id}。
            # 注意：这里不打印堆栈，避免向客户端泄露内部信息。
            return JSONResponse(
                status_code=500,
                content={"code": 500, "msg": "服务器内部错误", "trace_id": request_id},
            )


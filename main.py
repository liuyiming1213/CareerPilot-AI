"""CareerPilot 应用入口。

启动命令：
    python -m uvicorn main:app --reload --port 8000

这里故意不再导入旧的订单业务路由，也不调用 drop_all/init_data。原有文件仍
保留在项目中，但 CareerPilot 的启动过程不会删除或重建任何旧数据。
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from careerpilot.db import init_database
from careerpilot.errors import register_exception_handlers
from careerpilot.middleware.http import RequestContextMiddleware
from api_router.router import router as careerpilot_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表，关闭时由连接池自动释放资源。"""
    init_database()
    yield


app = FastAPI(
    title="CareerPilot AI 求职与职业成长平台",
    description="面向应届生的多智能体、Agent Tool Calling、MCP 和 Agentic RAG 后端",
    version="1.0.0",
    lifespan=lifespan,
)

# 前端会独立开发，因此本地默认允许所有来源跨域访问。
# 生产环境务必通过环境变量 CAREERPILOT_CORS_ORIGINS 收紧为明确的前端域名
# （逗号分隔，例如 "https://app.example.com,https://admin.example.com"）。
cors_origins = [
    o.strip() for o in os.getenv("CAREERPILOT_CORS_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 请求上下文中间件：为每个请求分配唯一 request_id、统计处理耗时、
# 并把未捕获异常统一转成 500 JSON（具体实现在 careerpilot/middleware/http.py）。
app.add_middleware(RequestContextMiddleware)
# 注册 CareerPilot 的全部 API 路由（登录、简历、JD、记忆、Agent 等）。
# 每个接口的完整路径 = api_router.router 里定义的 prefix + 各自的 path。
app.include_router(careerpilot_router)

# 注册统一错误处理器：所有接口异常归一成 {code, msg, trace_id} 格式。
register_exception_handlers(app)

# 静态资源放行：把项目根目录的 static/ 挂到 /static，浏览器可直接访问其中的
# 图片/JS/CSS 等静态文件（例如 <ip>:8000/static/xxx.png）。
# 目录不存在时先自动创建空目录，避免没有静态文件时应用启动就报错。
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 前端单页入口：访问 / 时若存在 static/index.html 就返回它（登录页），
# 否则退回接口说明 JSON。static/index.html 内已实现"登录→存 token→带 Bearer 调受保护接口"。
@app.get("/", tags=["System"])
def root():
    """访问 / 优先返回前端登录页；没有则返回接口说明。"""
    index_file = Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        # 用 HTMLResponse 直接返回页面内容，避免 FileResponse 在部分中间件/测试
        # 组合下的流式问题（实际 uvicorn 下 FileResponse 也正常，但用 HTMLResponse 更稳）。
        return HTMLResponse(index_file.read_text(encoding="utf-8"), status_code=200)
    return JSONResponse({"name": "CareerPilot", "docs": "/docs", "health": "/api/careerpilot/health"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=False,
    )
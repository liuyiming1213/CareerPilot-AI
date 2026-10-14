"""真实健康检查：探测关键外部依赖是否可达。

用途：把 /health 从"硬编码一切就绪"升级为真实探测，运维/探活脚本能据此
看到哪些子系统（数据库、Redis、向量库、记忆）处于降级或故障状态，而不必
等到调用某个功能时才暴露问题。

每个探测都会被 try/except 包住、尽量短超时，保证某个依赖挂了时 /health
本身仍能快速返回——健康检查接口本身永远可用，只是通过 components 标记哪项
失败，degraded 列出所有未就绪的组件名。
"""

from sqlalchemy import text

from careerpilot.config import settings
from careerpilot.db import engine
from careerpilot.memory.manager import memory_manager
from careerpilot.rag.store import rag_store


def check_db() -> str:
    """数据库探活：能执行 SELECT 1 即视为正常。"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "error"


def check_redis() -> str:
    """Redis 探活：能 ping 通才算可用；未装 client 或连不上都视为不可用。"""
    try:
        import redis

        # 显式短超时，避免 Redis 不可达时健康检查长时间挂起
        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=1, socket_timeout=1
        )
        return "ok" if client.ping() else "unavailable"
    except Exception:
        return "unavailable"


def check_chroma() -> str:
    """Chroma 向量库探活：collection 能 count 即视为可用。"""
    try:
        if rag_store.knowledge is None:
            return "unavailable"
        rag_store.knowledge.count()
        return "ok"
    except Exception:
        return "error"


def check_memory() -> str:
    """记忆子系统探活：读一次快照不抛异常即视为可用。"""
    try:
        memory_manager.snapshot("__health__", "__health__")
        return "ok"
    except Exception:
        return "error"


def run_checks() -> dict:
    """执行全部探测，返回 {components: {...}, degraded: [...]}。

    返回:
        components: 各子系统状态字符串（ok / unavailable / error）。
        degraded:   所有"未就绪"（非 ok/ready）的组件名列表，供前端/告警展示。
    """
    probes = {
        "api": "ok",        # 能返回响应本身就说明 API 进程存活
        "graph": "ok",      # 多智能体图可编译（模块导入成功即视为就绪）
        "db": check_db(),
        "redis": check_redis(),
        "chroma": check_chroma(),
        "memory": check_memory(),
    }
    degraded = [name for name, state in probes.items() if state not in ("ok", "ready")]
    return {"components": probes, "degraded": degraded}
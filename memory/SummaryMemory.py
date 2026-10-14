"""摘要记忆：复用原项目设计，保存到 PostgreSQL。

作用域：L2 摘要记忆层。以"会话"（session_id）为粒度保存/读取对话摘要，
摘要由 SummaryAgent（LLM）生成，用于跨轮次保持对话上下文。
生产环境使用 PostgreSQL 的 conversation_summary 表；没有 PostgreSQL 时
降级到进程内字典（_fallback），方便本地调试。
"""

import os

# 尝试导入连接池库；没安装时置为 None，整个类自动走内存 fallback
try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None

# 尝试加载 .env 中的环境变量；没有 python-dotenv 时忽略，不阻塞启动
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class SummaryMemory:
    """按 conversation/session id 保存摘要。

    原实现中 save 因缩进错误变成了 load 的内部函数，外部无法调用；这里修复
    为真正的类方法。没有 PostgreSQL 时使用内存 fallback，避免影响其他模块。
    """

    # 类级别的内存降级存储：session_id -> 摘要文本
    # 用类变量让不同实例共享同一份摘要数据
    _fallback: dict[str, str] = {}

    def __init__(self):
        self.pool = None
        # 没有 PostgreSQL 时使用进程内 fallback；配置 ENABLE_POSTGRES_MEMORY=1
        # 后才创建连接池，避免 API 启动阶段等待数据库网络连接。
        if ConnectionPool is not None and os.getenv("ENABLE_POSTGRES_MEMORY", "0") == "1":
            # 从环境变量拼出 PostgreSQL 连接串（DSN），所有值都有默认值兜底
            conninfo = (
                f"host={os.getenv('DB_HOST', 'localhost')} "
                f"port={os.getenv('DB_PORT', '5432')} "
                f"dbname={os.getenv('DB_NAME', 'careerpilot')} "
                f"user={os.getenv('DB_USER', 'postgres')} "
                f"password={os.getenv('DB_PASSWORD', '')}"
            )
            try:
                self.pool = ConnectionPool(
                    conninfo=conninfo,
                    min_size=int(os.getenv("DB_MIN_SIZE", "1")),
                    max_size=int(os.getenv("DB_MAX_SIZE", "5")),
                    # open=False 表示不立即建立连接，避免启动时阻塞
                    open=False,
                )
                # wait=False：后台异步建立连接，即使数据库没起来也不卡住启动
                self.pool.open(wait=False)
                self._ensure_table()
            except Exception:
                # 连接池创建失败则禁用 PostgreSQL，继续使用内存 fallback
                self.pool = None

    def _ensure_table(self) -> None:
        """确保 conversation_summary 表存在（幂等 CREATE TABLE IF NOT EXISTS）。

        MySQL/PostgreSQL 启动时可能还没有这张表，直接插入会失败；这里在
        初始化连接池后补建表，让摘要记忆的正式持久化真正可用。
        """
        if self.pool is None:
            return
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_summary (
                        session_id    VARCHAR(255) PRIMARY KEY,
                        summary       TEXT NOT NULL,
                        update_time   TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
                conn.commit()
        except Exception:
            # 建表失败（如数据库不可用/权限不足）就静默降级，继续用进程内 fallback
            pass

    def load(self, session_id: str) -> str:
        """读取指定会话摘要。

        参数 session_id：会话标识。
        返回值：摘要文本字符串；没有摘要或读取失败时返回空串。
        """
        if self.pool:
            try:
                # 从连接池借一条连接执行查询，用完自动归还
                with self.pool.connection() as conn, conn.cursor() as cur:
                    # 使用参数化查询（%s 占位）防止 SQL 注入
                    cur.execute("SELECT summary FROM conversation_summary WHERE session_id=%s", (session_id,))
                    row = cur.fetchone()
                    return row[0] if row else ""
            except Exception:
                # 查询异常（如数据库未启动）时静默忽略，降级到内存读取
                pass
        return self._fallback.get(str(session_id), "")

    def save(self, session_id: str, summary: str) -> None:
        """写入或更新摘要。

        参数 session_id：会话标识；summary：新的摘要文本。
        先更新内存 fallback 保证立即可读，再尝试同步到数据库。
        """
        # 无论数据库是否可用，都先写进内存，保证本轮会话内能立刻读到
        self._fallback[str(session_id)] = summary
        if self.pool:
            try:
                with self.pool.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversation_summary(session_id, summary)
                        VALUES (%s, %s)
                        ON CONFLICT(session_id)
                        DO UPDATE SET summary=EXCLUDED.summary, update_time=NOW()
                        """,
                        (session_id, summary),
                    )
                    # 显式提交事务，让数据真正落库
                    conn.commit()
            except Exception:
                # 写入失败（如数据库未启动）时静默忽略，仅保留内存中的摘要
                pass

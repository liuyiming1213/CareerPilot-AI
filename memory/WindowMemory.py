"""窗口记忆：保存当前会话最近几轮对话。

这是你原项目中的 L1 记忆实现。CareerPilot 会继续使用这个类，而不是另起
一套窗口记忆。生产环境使用 Redis，Redis 不可用时降级到进程内 deque，方便
你在没有启动 Redis 的情况下调试其他 Agent。
"""

import json
import os
from collections import defaultdict, deque

# 尝试导入 redis 库；没安装时置为 None，整个类自动走内存 fallback
try:
    import redis
except ImportError:
    redis = None

# 尝试加载 .env 中的环境变量；没有 python-dotenv 时忽略，不阻塞启动
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 内存降级存储：Redis key -> 消息双端队列
# 用模块级变量 + defaultdict，不同 WindowMemory 实例间也能按 key 共享数据
_fallback_store: dict[str, deque] = defaultdict(deque)


class WindowMemory:
    """最近消息窗口。window_size 表示消息条数，而不是完整对话轮数。"""

    def __init__(self, session_id: str, window_size: int | None = None, ttl: int | None = None):
        self.session_id = str(session_id)
        # Redis key 统一加 "chat_window:" 前缀，与其它记忆的 key 命名区分
        self.key = f"chat_window:{self.session_id}"
        # 窗口大小：优先用传入参数，否则读环境变量 WINDOW_SIZE，默认 10 条消息
        self.window_size = window_size or int(os.getenv("WINDOW_SIZE", "10"))
        # 窗口存活时间（TTL）：默认 604800 秒 = 7 天，超时后整段历史自动清除
        self.window_time = ttl or int(os.getenv("WINDOW_TIME", "604800"))
        self.client = None
        # 开发环境默认不主动连接 Redis；设置 ENABLE_REDIS_MEMORY=1 后启用真实 Redis。
        # 这样没有启动基础设施时，项目仍能立即启动并使用进程内 fallback。
        if redis is not None and os.getenv("ENABLE_REDIS_MEMORY", "0") == "1":
            try:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    db=int(os.getenv("REDIS_DB", "0")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    # 让 Redis 返回 Python 字符串而非 bytes，省去手动解码
                    decode_responses=True,
                    # Redis 未启动时最多等待 0.3 秒，然后自动使用内存 fallback。
                    socket_connect_timeout=0.3,
                    socket_timeout=0.3,
                )
                # 立即 ping 一次确认连接真的可用，失败则回退到内存存储
                self.client.ping()
            except Exception:
                self.client = None
        # 用 deque 的 maxlen 实现"滑动窗口"：超过 window_size 时自动丢弃最旧消息。
        # 注意这是内存 fallback 专用的裁剪机制；Redis 分支用 ltrim 单独裁剪。
        _fallback_store[self.key] = deque(_fallback_store[self.key], maxlen=self.window_size)

    def add(self, role: str, content: str) -> None:
        """追加一条消息，并裁剪旧消息。

        参数 role：说话人（"user" 或 "assistant"）；content：消息内容。
        只会保留窗口内的最新若干条消息。
        """
        item = {"role": role, "content": content}
        if self.client:
            # Redis List 模拟队列：rpush 从右边推入新消息
            self.client.rpush(self.key, json.dumps(item, ensure_ascii=False))
            # ltrim 只保留最右边 window_size 条，多余的旧消息立即删除
            self.client.ltrim(self.key, -self.window_size, -1)
            # 每次写入都刷新 TTL，让"活跃会话"的历史一直存活
            self.client.expire(self.key, self.window_time)
        else:
            # 内存 fallback：deque 已设置 maxlen，append 会自动丢弃最旧消息
            _fallback_store[self.key].append(item)

    def load(self) -> list[dict[str, str]]:
        """读取当前窗口，统一返回 Python 字典列表。

        返回值：[{role, content}, ...] 按时间从旧到新的消息列表，
        供 PromptBuilder 逐条拼进提示词。
        """
        if self.client:
            # Redis 里存的是 JSON 字符串，取回后逐个反序列化成字典
            return [json.loads(item) for item in self.client.lrange(self.key, 0, -1)]
        # 转成普通 list 返回，避免调用方拿到 deque 引用后误改内部存储
        return list(_fallback_store[self.key])

    def clear(self) -> None:
        """清除当前会话窗口。

        删除该会话的所有历史消息（包括 Redis key 和内存 fallback 记录）。
        """
        if self.client:
            self.client.delete(self.key)
        _fallback_store.pop(self.key, None)

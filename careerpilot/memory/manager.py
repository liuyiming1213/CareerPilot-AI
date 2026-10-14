"""CareerPilot 对原有 memory 包的适配层。

重要：这里没有重新实现四层记忆。CareerPilot 会优先使用项目原来的：

- `memory.WindowMemory`
- `memory.SummaryMemory`
- `memory.LongMemory`
- `memory.ProfileMemory`

本适配层只负责把它们组合成 Graph 需要的统一接口。这样你以后学习记忆模块
时，可以从 CareerPilot Graph 追到你原来写的 memory 包。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryContext:
    """Graph 使用的统一记忆快照。"""

    window: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    long_term: list[str] = field(default_factory=list)
    profile: dict[str, str] = field(default_factory=dict)


class MemoryManager:
    """使用原有 memory 包完成四层记忆读写。

    对外暴露两组能力：
    - 读取：load() 一次性取回四层记忆（窗口/摘要/长期/画像），返回统一快照 MemoryContext；
    - 写入：append_message / save_summary / save_long_memory / update_profile
      分别写窗口、摘要、长期记忆、画像四个层。

    无论读还是写，都优先走原 memory 包（真实存储）；任何一步失败都静默降级
    到进程内 _fallback，保证 Graph 在外部服务未启动时也能调试运行。
    """

    def __init__(self):
        # 缓存表：key 是 (user_id, conversation_id) 二元组，value 是对应的 ConversationManager。
        # 这样同一个用户、同一个会话的多次请求可以复用同一个管理器对象，避免每次重建连接。
        # 注意：ConversationManager 内部维护着连接/会话状态，复用它是为了性能；
        # 但这也意味着它会随本模块的存活一直驻留内存，不适合无限增长（此处仅作演示规模使用）。
        self._sessions: dict[tuple[str, str], Any] = {}
        # 兜底容器：当外部存储（Redis/PostgreSQL/ChromaDB）不可用时，
        # 把数据先暂存在这个进程内对象里，保证 Graph 在开发阶段也能调试运行。
        # 注意：进程内内存重启即清空，只用于开发调试，绝不能当持久化存储使用。
        self._fallback = MemoryContext()

    def _session(self, user_id: str, conversation_id: str):
        """按用户和会话缓存 ConversationManager，避免每次请求重复建连接。

        - 参数：user_id 用户 ID；conversation_id 会话 ID。
        - 返回：对应 (user_id, conversation_id) 的 ConversationManager 实例。
        """
        # 统一转成字符串作为缓存的 key，避免"数字 1"和"字符串 '1'"被当成两个不同的会话
        key = (str(user_id), str(conversation_id))
        if key not in self._sessions:
            # 延迟导入：只有真正需要时才加载原 memory 包的 ConversationManager，
            # 这样即使原包或它的依赖缺失，也不会影响本模块被导入。
            from memory.ConversationManager import ConversationManager

            self._sessions[key] = ConversationManager(user_id, conversation_id)
        return self._sessions[key]

    def load(self, user_id: str, conversation_id: str, query: str = "") -> MemoryContext:
        """分别从原来的四个 Memory 类读取数据。

        - 参数：user_id 用户 ID；conversation_id 会话 ID；query 检索长期记忆用的查询词。
        - 返回：包含四层记忆的 MemoryContext 快照。
        """
        try:
            # 一次性把四层记忆全部读出来，拼成一个统一快照交给 Graph 使用；
            # 这样 Graph 节点启动时只需调一次 load() 就能拿到全部上下文
            manager = self._session(user_id, conversation_id)
            return MemoryContext(
                # 窗口记忆：取出最近几轮的对话消息
                window=manager.window.load(),
                # 摘要记忆：取出该会话的对话摘要
                summary=manager.summary.load(conversation_id),
                # 长期记忆：有 query 做语义检索；无 query 时取该用户最近几条，
                # 让"记忆快照"也能看到已沉淀的长期记忆（而非永远为空）
                long_term=(
                    manager.long_memory.load(user_id, query, limit=5)
                    if query else manager.long_memory.recent(user_id, limit=5)
                ),
                # 用户画像：取出该用户的静态特征
                profile=manager.profile.load(),
            )
        except Exception:
            # 外部服务未启动时，保留一个进程内 fallback，保证 Graph 可以调试。
            # 注意：降级后返回同一个 fallback 对象，方便后续写入时也能对上同一份数据。
            return self._fallback

    def append_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        """通过原 WindowMemory.add 写入窗口记忆。

        - 参数：role 消息角色（如 user/assistant）；content 消息内容。
        """
        try:
            manager = self._session(user_id, conversation_id)
            manager.window.add(role, content)
        except Exception:
            # 外部存储不可用时，把消息暂存到进程内 fallback，保证对话流程不中断
            self._fallback.window.append({"role": role, "content": content})

    def save_summary(self, user_id: str, conversation_id: str, summary: str, db=None) -> None:
        """通过原 SummaryMemory.save 写入摘要。

        - 参数：summary 要保存的摘要文本；db 为可选参数，仅用于兼容原有调用习惯，本适配层未使用。
        """
        try:
            self._session(user_id, conversation_id).summary.save(conversation_id, summary)
        except Exception:
            # 降级：仅存到进程内 fallback
            self._fallback.summary = summary

    def save_long_memory(self, user_id: str, content: str, memory_type: str = "other") -> None:
        """通过原 LongMemory.add_memory 写入 ChromaDB。

        - 参数：content 要沉淀的长期记忆内容；memory_type 记忆类型（默认 other）。
        """
        try:
            # 长期记忆不属于某个具体会话，所以会话 ID 用一个固定的拼接值 "memory-{user_id}"，
            # 相当于为该用户单独开一个"记忆会话"来承载他的长期记忆
            self._session(user_id, f"memory-{user_id}").long_memory.add_memory(user_id, content, memory_type)
        except Exception:
            # 降级：追加到进程内 fallback 的长期记忆列表
            self._fallback.long_term.append(content)

    def search_long_memory(self, user_id: str, query: str, limit: int = 5) -> list[str]:
        """通过原 LongMemory.load 检索用户自己的长期记忆。

        - 参数：query 查询词；limit 最多返回条数。
        - 返回：命中的长期记忆字符串列表。
        """
        try:
            return self._session(user_id, f"memory-{user_id}").long_memory.load(user_id, query, limit)
        except Exception:
            # 降级：直接从进程内 fallback 里截取前 limit 条
            return self._fallback.long_term[:limit]

    def update_profile(self, user_id: str, values: dict[str, str]) -> None:
        """通过原 ProfileMemory.batch_add 更新画像。

        - 参数：values 需要更新/新增的画像键值对。
        """
        try:
            # 画像同样不属于某个会话，使用固定的 "profile-{user_id}" 作为会话 ID
            self._session(user_id, f"profile-{user_id}").profile.batch_add(values)
        except Exception:
            # 降级：把值统一转成字符串后合并进进程内 fallback 的画像字典
            # （str(k)/str(v) 是为了保证键值一定是字符串类型，与其他层的约定保持一致）
            self._fallback.profile.update({str(k): str(v) for k, v in values.items()})

    def snapshot(self, user_id: str, conversation_id: str, query: str = "") -> dict[str, Any]:
        """返回四层记忆和实际使用的存储类型，方便前端展示。

        - 参数：同 load。
        - 返回：包含 window/summary/long_term/profile 及 source、layers 的字典。
        """
        context = self.load(user_id, conversation_id, query)
        # 判断这次读取走的是真实存储还是 fallback：
        # 如果该会话在缓存表里有对应 manager，说明用的是原 memory 包；否则是降级模式
        manager = self._sessions.get((str(user_id), str(conversation_id)))
        return {
            "window": context.window,
            "summary": context.summary,
            "long_term": context.long_term,
            "profile": context.profile,
            # source：标记数据来源，便于前端/调试区分"真实存储"和"降级内存"
            "source": "original memory package" if manager else "fallback",
            # layers：解释每层记忆实际落在哪个存储里，方便前端展示各层去向
            "layers": {
                "window": "memory.WindowMemory -> Redis",
                "summary": "memory.SummaryMemory -> PostgreSQL",
                "long_term": "memory.LongMemory -> ChromaDB",
                "profile": "memory.ProfileMemory -> Redis",
            },
        }


# 模块级单例：整个应用共享同一个 MemoryManager。
# 上层代码直接 from .manager import memory_manager 即可使用，无需自己 new 一个。
memory_manager = MemoryManager()


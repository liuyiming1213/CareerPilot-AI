"""四层记忆的会话门面。

Agent 不需要知道 Redis、PostgreSQL 和 Chroma 的细节，只通过
ConversationManager 读取和写入记忆。这也是原项目记忆代码和 CareerPilot
Graph 之间的连接点。
"""

from memory.LongMemory import LongMemory
from memory.ProfileMemory import ProfileMemory
from memory.SummaryMemory import SummaryMemory
from memory.WindowMemory import WindowMemory


class ConversationManager:
    """把四层记忆组合成一个可复用的会话对象。"""

    def __init__(self, user_id: str, session_id: str):
        # 统一把传入值转成字符串，避免不同调用方传入 int 等类型导致后面键值不匹配
        self.user_id = str(user_id)
        self.session_id = str(session_id)
        # L1 短期记忆：以"会话"为粒度，滑动窗口最多保留 10 条最近消息
        self.window = WindowMemory(self.session_id, window_size=10)
        # L2 摘要记忆：按 session_id 存储/读取对话摘要
        self.summary = SummaryMemory()
        # L3 长期记忆：按 user_id 写入/读取到 Chroma 向量库
        self.long_memory = LongMemory()
        # L4 用户画像：按 user_id 保存用户长期偏好画像
        self.profile = ProfileMemory(self.user_id)

    def save_user(self, question: str) -> None:
        """保存用户消息到 L1 窗口记忆。

        参数 question：用户刚刚说的一句话。
        role 记为 "user"，方便后续拼提示词时区分说话人。
        """
        self.window.add("user", question)

    def save_ai(self, response: str) -> None:
        """保存 AI 消息到 L1 窗口记忆。

        参数 response：助手生成的回复。
        role 记为 "assistant"。
        """
        self.window.add("assistant", response)

    def save_summary(self, summary: str) -> None:
        """保存 L2 摘要记忆。

        参数 summary：由 SummaryAgent 生成的对话摘要文本，
        按当前 session_id 写入，只与本次会话相关。
        """
        self.summary.save(self.session_id, summary)

    def save_long_memory(self, content: str, memory_type: str = "other") -> None:
        """保存 L3 长期记忆。

        参数 content：值得长期记住的事实（通常由 LongMemoryAgent 提取）。
        参数 memory_type：记忆类型标签（如偏好/计划），默认 "other"。
        写入时必须带上 user_id，保证不同用户的记忆互相隔离。
        """
        self.long_memory.add_memory(self.user_id, content, memory_type)

    def build_prompt(self, question: str) -> list[dict[str, str]]:
        """组装带四层记忆的提示词，供 LLM Agent 使用。

        参数 question：用户当前问题。
        返回值：OpenAI 风格的对话消息列表，这里只有一条 user 消息，
        所有记忆都拼在这条消息的 content 文本里。
        """
        prompt = PromptBuilder(
            self.window,
            self.summary,
            self.long_memory,
            self.profile,
            self.user_id,
            self.session_id,
        ).build(question)
        # 包装成标准的多轮对话消息格式，方便 LangChain 等框架直接调用
        return [{"role": "user", "content": prompt}]


class PromptBuilder:
    """把四层记忆转换为模型容易理解的上下文。"""

    def __init__(self, window, summary, long_memory, profile, user_id, session_id):
        self.window = window
        self.summary = summary
        self.long_memory = long_memory
        self.profile = profile
        self.user_id = user_id
        self.session_id = session_id

    def build(self, question: str) -> str:
        """把四层记忆拼成一段带结构的提示词文本。

        参数 question：用户当前问题，会被拼到提示词最后，告诉模型"当前要回答什么"。
        返回值：拼接好的提示词字符串（纯文本，尚未包装成消息格式）。
        """
        # 开场白：给模型设定"有记忆的 AI 助手"的人设
        prompt = "你是一个有记忆能力的AI助手。请根据以下记忆回答用户问题。\n\n"
        # 第一段：历史摘要。只有该会话确实存在摘要时才拼进去，避免空段落
        summary_text = self.summary.load(self.session_id)
        if summary_text:
            prompt += f"【历史摘要】\n{summary_text}\n\n"
        # 第二段：长期记忆。用当前问题做语义检索，只取最相关的 5 条；
        # 检索时内部已按 user_id 过滤，不会读到其他用户的记忆
        long_memories = self.long_memory.load(self.user_id, question, limit=5)
        if long_memories:
            prompt += "【长期记忆】\n" + "\n".join(f"- {item}" for item in long_memories) + "\n\n"
        # 第三段：用户画像。展示用户长期稳定的属性（岗位、城市、技能等）
        profile = self.profile.load()
        if profile:
            prompt += "【用户画像】\n" + "\n".join(f"- {key}: {value}" for key, value in profile.items()) + "\n\n"
        # 第四段：最近聊天记录（L1 滑动窗口），逐条拼成 "角色: 内容" 形式
        prompt += "【最近聊天记录】\n"
        prompt += "\n".join(f"{item['role']}: {item['content']}" for item in self.window.load())
        # 最后附上当前问题，让模型在记忆上下文中作答
        return f"{prompt}\nuser: {question}"


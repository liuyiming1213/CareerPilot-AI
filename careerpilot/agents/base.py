"""所有专业 Agent 共用的执行基类。

这里统一解决三个问题：
1. 有真实大模型时调用大模型。
2. 没有大模型或调用失败时使用规则型 fallback。
3. 临时错误自动重试，超过次数后返回结构化失败，而不是让整个 Graph 崩溃。

使用方式：具体的 Agent（如 ResumeAgent、JDAgent）继承 BaseAgent，只需要
实现自己的 run() 方法，并在其中用 ask_llm()/ask_structured() 调用大模型、
用 invoke_with_retry() 包裹核心逻辑，即可免费获得"重试 + 结构化失败"能力。

两个 LLM 入口如何取舍：
- ask_llm()：只需要一段自由文本时用（如面试出题、轻量问答）。
- ask_structured()：需要拿到"字段明确的 Pydantic 结构"时用（如简历/JD 解析）。
两者的共同点是：未配置 LLM（get_llm() 返回 None）时都返回 None，调用方看到
None 就应切换到规则型 fallback——这是"无 LLM 降级"的统一触发点。
"""

import time
from collections.abc import Callable
from typing import Any

from careerpilot.agents.contracts import AgentEnvelope
from careerpilot.config import settings
from careerpilot.llm import get_llm


class BaseAgent:
    """专业 Agent 的最小公共实现：提供重试、降级和 LLM 调用的通用工具方法。"""

    # 每个 Agent 子类都会覆盖这个名字，用于标识"我是谁"，会写进
    # AgentEnvelope.agent_name 以及错误信息里，便于日志排查。
    name = "base_agent"

    def __init__(self, max_retries: int = 2):
        # 允许重试的最大次数；默认 2 次（加上首次尝试，总共最多执行 3 次）。
        self.max_retries = max_retries

    def invoke_with_retry(self, work: Callable[[], AgentEnvelope], fallback: Callable[[Exception], AgentEnvelope]) -> AgentEnvelope:
        """执行 Agent 工作，失败后使用指数退避重试。

        面试时可以解释：重试只用于临时网络/模型错误，不应该对参数错误无限重试。

        参数:
            work: 真正执行 Agent 任务的函数，应返回 AgentEnvelope；
                抛出异常会被捕获并触发重试。
            fallback: 重试彻底失败后调用，接收最后一次异常并返回一个
                降级（degraded=True）的外壳，避免异常击穿整个工作流。

        返回:
            AgentEnvelope：成功时 attempts 为实际尝试次数；
            彻底失败时 attempts = max_retries + 1 且 degraded = True。
        """
        # 注意：work() 抛出的所有异常都会被这里拦截——包括 LLM 网络错误、
        # MCP 工具调用失败、模型输出格式异常等。所以调用方应把"调 LLM +
        # 规则 fallback 的完整逻辑"全部放进 work()，由本方法统一做重试兜底。
        last_error: Exception | None = None
        # range(1, max_retries + 2)：最多执行 max_retries+1 次
        # （首次尝试 + max_retries 次重试）。
        for attempt in range(1, self.max_retries + 2):
            try:
                result = work()
                result.attempts = attempt  # 记录本次实际用了几次尝试
                return result
            except Exception as exc:
                last_error = exc
                # 还没超过重试上限就等待后重试。
                if attempt <= self.max_retries:
                    # 指数退避：第 1 次失败等 0.05s，第 2 次等 0.1s，第 3 次等 0.2s……
                    # 避免在模型/网络抖动时用同一频率反复打爆服务。
                    time.sleep(0.05 * (2 ** (attempt - 1)))
        # 走到这里说明所有尝试都失败了：调用 fallback 生成降级结果。
        result = fallback(last_error or RuntimeError("unknown agent error"))
        result.attempts = self.max_retries + 1  # 标记为"已用完全部机会"
        result.degraded = True  # 明确告知下游：这份结果是降级得到的
        return result

    def ask_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        """调用可选 LLM；返回 None 表示使用本地规则 fallback。

        参数:
            system_prompt: 系统提示词，用来设定模型扮演的角色/行为准则。
            user_prompt: 用户提示词，即本次要模型处理的具体任务内容。

        返回:
            模型生成的文本；若未配置 LLM（get_llm() 返回 None）则返回 None，
            调用方据此走规则型 fallback。
        """
        # 从全局配置/工厂获取大模型实例；没有配置 API Key 时为 None。
        llm = get_llm()
        if llm is None:
            return None
        # 以 (角色, 内容) 消息列表的方式调用 LangChain 兼容的模型接口。
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        # 统一把模型返回内容转成纯字符串（某些模型的 content 可能不是字符串）。
        return response.content if isinstance(response.content, str) else str(response.content)

    def ask_structured(self, output_model, system_prompt: str, user_prompt: str):
        """让模型直接返回 Pydantic 结构，避免 Agent 输出无法解析的散文。

        参数:
            output_model: 一个 Pydantic 模型类（如 ResumeAnalysis），
                模型会按该结构的字段生成输出。
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。

        返回:
            一个 output_model 类型的实例；未配置 LLM 时返回 None。
        """
        llm = get_llm()
        if llm is None:
            return None
        # with_structured_output 让模型输出被强制约束为 Pydantic 结构，
        # 调用方无需再写"解析自由文本"的脆弱代码。
        structured_llm = llm.with_structured_output(output_model)
        # 注意：这里同样不捕获异常——真正的模型调用错误（网络/额度超限）会
        # 向上抛出，由调用方用 invoke_with_retry() 统一重试；返回 None 仅表示
        # "未配置 LLM"，此时调用方应走规则型 fallback。
        return structured_llm.invoke([("system", system_prompt), ("human", user_prompt)])

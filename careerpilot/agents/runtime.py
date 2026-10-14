"""基于 LangChain ``create_agent`` 的真实智能体运行时。

旧版本项目中的 Agent 类仍然保留，因为它们包含业务 fallback、重试和结果
封装。本文件负责新增真正的 LLM Agent：当配置了 LLM_API_KEY 时，角色 Agent
会由 ``create_agent`` 创建；没有密钥时返回 None，让业务代码进入可测试的降级
路径。

与 agents/*_agent.py 的协作关系：每个 *_agent 的 run() 先调 create_role_agent()，
非 None 就走"真实 Agent + MCP 工具"最高级路径；返回 None 时降级到 ask_structured
（仅 LLM），再不行落到关键词规则。本文件就是这条三级链路上的"第一级"开关。
"""

from typing import Any

from langchain.agents import create_agent

from careerpilot.agents.contracts import AgentEnvelope
from careerpilot.llm import get_llm
from careerpilot.tools.build_tools import build_tools


def create_role_agent(
    role_name: str,
    user_id: str,
    system_prompt: str,
    allowed_tools: set[str],
    response_format: Any | None = None,
):
    """创建一个可以自主决定是否调用工具的真实 Agent。

    ``create_agent`` 返回的是一个可调用的 LangGraph CompiledStateGraph，
    不是普通 Python 类。调用它时传入 ``{"messages": [...]}``，模型会在
    需要时生成 tool_call，框架执行 MCP Tool，再把工具结果回传给模型。
    """
    # 两个"返回 None"的触发点，恰好是业务代码切换降级路径的信号：
    # 1) 环境里根本没装 langchain（create_agent 为 None）；
    # 2) 没有配置 LLM_API_KEY（get_llm() 返回 None）。
    # 任一命中，调用方（如 ResumeAgent.run）都会走 ask_structured/规则 fallback。
    llm = get_llm()
    if llm is None:
        return None
    return create_agent(
        model=llm,
        tools=build_tools(user_id, allowed_tools),
        system_prompt=system_prompt,
        response_format=response_format,
        name=role_name,
    )


def invoke_role_agent(agent, prompt: str) -> tuple[Any, list[str]]:
    """调用 create_agent，并提取结构化响应和实际工具调用轨迹。"""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    # LangGraph 会把模型的"结构化响应"（即 response_format 指定的 Pydantic 结构）
    # 放在结果的 structured_response 键里，调用方无需再从散文里二次解析。
    structured = result.get("structured_response")
    used_tools: list[str] = []
    # 遍历模型多轮往返产生的全部消息，收集其中真实发出的 tool_call 名称
    # （去重后）填入 AgentEnvelope.used_tools，便于审计/展示实际调了哪些 MCP 工具。
    for message in result.get("messages", []):
        for tool_call in getattr(message, "tool_calls", []) or []:
            name = tool_call.get("name")
            if name and name not in used_tools:
                used_tools.append(name)
    return structured, used_tools

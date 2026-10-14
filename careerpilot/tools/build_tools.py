"""把工具包装成 LLM 可以发现和调用的 LangChain StructuredTool。

连接链（替代原先的 build_mcp_tools）：

    LLM create_agent
        -> LangChain StructuredTool
        -> ToolClient.call()（治理 + 审计）
        -> registry.handle_tool_call
        -> 真实业务实现

关键点：这里同时是"租户隔离"的入口。为某个用户 build 出来的工具，其
user_id 会被闭包写死，模型只能提交业务参数（query / 技能列表等），无法在
工具参数里伪造另一个用户的 ID——这是最小权限原则在工具层的关键实现。
"""

from typing import Any

from langchain_core.tools import StructuredTool

from careerpilot.tools.client import ToolClient


def build_tools(user_id: str, allowed_names: set[str] | None = None) -> list[StructuredTool]:
    """为某个用户创建带身份上下文的 LangChain 工具。

    - 参数：user_id 当前发起请求的用户 ID（被闭包写死）；allowed_names 可选的
      工具白名单（None 表示全部，否则只保留名字在白名单里的工具）。
    - 返回：包装好的 StructuredTool 列表，供 LLM 发现并调用。
    """
    # 每个用户独立一个 ToolClient：内部独立的频控计数与审计记录互不串扰
    client = ToolClient()

    # 租户隔离：下面每个工具函数都是闭包，在创建时把 user_id 捕获进来。
    # 于是工具签名里根本不出现 user_id 参数，LLM 看到的 schema 也没有这个入参，
    # 真正调用时 user_id 由这里写死进 arguments —— 模型改不了身份。
    def read_user_profile() -> dict[str, str]:
        """读取当前用户画像。"""
        return client.call("read_user_profile", {"user_id": user_id})

    def search_user_long_memory(query: str) -> list[str]:
        """检索当前用户的长期记忆。"""
        return client.call("search_user_long_memory", {"user_id": user_id, "query": query})

    def search_career_knowledge(query: str, limit: int = 5) -> list[str]:
        """检索公共职业知识库。"""
        # 公共知识库不分用户，无需 user_id
        return client.call("search_career_knowledge", {"query": query, "limit": limit})

    def calculate_skill_gap(required_skills: list[str], user_skills: list[str]) -> dict[str, Any]:
        """计算岗位技能差距。"""
        return client.call(
            "calculate_skill_gap",
            {"required_skills": required_skills, "user_skills": user_skills},
        )

    # 把闭包函数包装成 StructuredTool：from_function 会根据签名+docstring 生成
    # LLM 可读的工具 schema，让模型能发现并正确调用
    candidates = [
        StructuredTool.from_function(read_user_profile, name="read_user_profile"),
        StructuredTool.from_function(search_user_long_memory, name="search_user_long_memory"),
        StructuredTool.from_function(search_career_knowledge, name="search_career_knowledge"),
        StructuredTool.from_function(calculate_skill_gap, name="calculate_skill_gap"),
    ]
    # 白名单过滤：allowed_names 为 None 表示全量放行；否则只保留白名单内的工具，
    # 把无权使用的工具从 LLM 视野里直接移除（"看不见 = 调不了"）
    if allowed_names is None:
        return candidates
    return [tool for tool in candidates if tool.name in allowed_names]
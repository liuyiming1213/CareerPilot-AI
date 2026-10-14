"""工具注册表与统一分派（替代原先的 MCP server 层）。

智能体（Agent）可调用的工具统一在这里收口：
- 每个工具是一个普通的 Python 函数（业务实现）；
- `TOOL_HANDLERS` 记录 工具名 -> 函数；
- `handle_tool_call(name, arguments)` 是统一入口：按名字查到函数并调用，
  供带治理/审计的 ToolClient（tools/client.py）在真正调用时使用。

这里不再引入 MCP 协议——它是一套进程内的普通工具分派，但保留了"租户隔离、
治理、审计"这些有价值的能力（分别在 build_tools.py 与 client.py / middleware
里实现）。
"""

from typing import Any

from careerpilot.memory.manager import memory_manager
from careerpilot.rag.store import rag_store
from careerpilot.tools.career_tools import calculate_skill_gap as _calculate_skill_gap


def read_user_profile(user_id: str) -> dict[str, str]:
    """读取用户画像。

    - 参数：user_id 用户 ID。
    - 返回：该用户的画像键值对字典。
    """
    # 画像不属于任何具体会话，用固定会话 ID "tool-profile" 去加载
    return memory_manager.load(user_id, "tool-profile").profile


def search_user_long_memory(user_id: str, query: str) -> list[str]:
    """检索指定用户的长期记忆，不允许跨用户查询。

    - 参数：user_id 用户 ID（只能查自己的记忆）；query 查询词。
    - 返回：命中的长期记忆文本列表。
    """
    # 长期记忆按 user_id 过滤存储，传入同一个 user_id，机制上保证查不到别人
    return memory_manager.search_long_memory(user_id, query)


def search_career_knowledge(query: str, limit: int = 5) -> list[str]:
    """查询公共求职和技术知识库。

    - 参数：query 查询文本；limit 最多返回条数。
    - 返回：命中的知识库文本列表。
    """
    # 知识库是系统公共知识，不区分用户，无需 user_id
    return rag_store.search_knowledge(query, limit)


def calculate_skill_gap(required_skills: list[str], user_skills: list[str]) -> dict[str, Any]:
    """计算岗位技能差距。

    - 参数：required_skills 岗位要求的技能列表；user_skills 用户已具备的技能列表。
    - 返回：{"matched": 匹配项, "missing": 缺失项, "score": 匹配百分比}。
    """
    # career_tools.calculate_skill_gap 是 @tool 包装，取 .func 才是普通函数
    raw = getattr(_calculate_skill_gap, "func", _calculate_skill_gap)
    return raw(required_skills, user_skills)


# 工具注册表：工具名 -> 处理函数。handle_tool_call 按名字在这里查找并调用。
TOOL_HANDLERS = {
    "read_user_profile": read_user_profile,
    "search_user_long_memory": search_user_long_memory,
    "search_career_knowledge": search_career_knowledge,
    "calculate_skill_gap": calculate_skill_gap,
}


def handle_tool_call(name: str, arguments: dict) -> object:
    """按名字调用指定的工具（registry 的统一入口）。

    - 参数：name 工具名（TOOL_HANDLERS 的键）；arguments 传给该工具的关键字参数字典。
    - 返回：对应工具的执行结果。
    - 抛出：工具名未注册时抛 ValueError。
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"未注册的工具：{name}")
    # 把 arguments 字典按关键字展开传给处理函数
    return handler(**arguments)
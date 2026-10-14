"""Agent 工具定义（工具层）。

经整理后，本文件只保留"真实会被调用"的实现。当前唯一在用的工具是
`calculate_skill_gap`，它被 tools/registry.py 作为工具注册表里的计算实现复用。

其余原来的普通工具（get_user_profile / search_user_memory /
search_career_knowledge / save_user_memory）在 Agent 的实际路径上并未被使用
——真正的工具调用统一走 tools/build_tools.py -> tools/client.py -> registry，
工具的注册与业务实现收口在 tools/registry.py，这里只留纯计算函数，
避免"同一套功能存在两处实现"。
"""

from typing import Any

try:
    from langchain_core.tools import tool
except ImportError:  # 未装 LangChain 时退化为普通函数，工具仍可被直接调用
    def tool(func):
        return func


@tool
def calculate_skill_gap(required_skills: list[str], user_skills: list[str]) -> dict[str, Any]:
    """计算岗位技能差距，返回匹配项、缺失项和基础匹配分。

    这是纯计算：入参是已经提取好的技能列表，不做数据库/文本解析。

    - 参数：required_skills 岗位要求的技能列表；user_skills 用户已具备的技能列表。
    - 返回：{"matched": 匹配项, "missing": 缺失项, "score": 匹配百分比}。
    """
    # 统一转"去空白 + 小写"，做大小写无关、忽略首尾空格的比较
    required = {item.strip().lower() for item in required_skills if item.strip()}
    owned = {item.strip().lower() for item in user_skills if item.strip()}
    matched = sorted(required & owned)   # 匹配项 = 交集
    missing = sorted(required - owned)   # 缺失项 = 岗位要求有、用户没有的
    # 匹配分 = 命中数 / 要求数 * 100，保留两位小数；要求为空时取 0.0（避免除零）
    score = round(len(matched) / len(required) * 100, 2) if required else 0.0
    return {"matched": matched, "missing": missing, "score": score}
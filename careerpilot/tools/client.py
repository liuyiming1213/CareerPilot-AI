"""统一工具调用入口（带治理 + 审计），替代原先的 MCPToolClient。

所有工具调用都经过 ToolClient.call()：先做治理频控/权限检查（ToolGovernance），
再调用 registry.handle_tool_call 真正执行，并记录本轮用过的工具名 used_tools，
供上层写进 AgentEnvelope 做追溯。不再涉及 MCP 协议，它就是进程内的普通工具调用。
"""

from typing import Any

from careerpilot.config import settings
from careerpilot.middleware.governance import ToolGovernance


class ToolClient:
    """统一调用工具，并记录本轮用过的工具。

    职责：
    1. 收敛所有工具调用，便于统一加治理（频控）与审计；
    2. 记录本轮调用了哪些工具（used_tools），供追溯展示。
    """

    def __init__(self):
        # 治理器：限制单次任务最多能调用多少次工具，防止 Agent 失控循环调用
        self.governance = ToolGovernance(settings.max_tool_calls)
        self.call_count = 0              # 本轮已调用工具的次数（供频控判断）
        self.used_tools: list[str] = []  # 本轮实际用过的工具名列表

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用一个工具。

        先做治理检查，再调用 registry 里的对应实现。

        - 参数：name 工具名（须在 tools/registry.py 的 TOOL_HANDLERS 注册过）；
          arguments 传给该工具的关键字参数。
        - 返回：该工具的执行结果；工具未注册会抛 ValueError。
        """
        # 先做治理检查：超出频控上限会抛异常，从源头拦下失控调用
        self.governance.check(name, self.call_count)
        # 延迟导入（函数体内 import）：避免模块加载阶段形成循环依赖
        from careerpilot.tools.registry import handle_tool_call

        self.call_count += 1  # 记录本次调用，供下一次频控判断
        result = handle_tool_call(name, arguments)
        self.used_tools.append(name)  # 记录工具名，最终写进 AgentEnvelope.used_tools
        return result
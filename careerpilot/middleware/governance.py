"""Agent Tool 调用治理。
任何会访问用户数据、写入长期记忆或调用外部服务的工具，都应该经过这里的次数和权限检查。

作用域说明：
- 本模块只负责"检查"，不负责"执行"：在 Agent 真正调用工具前，由上层（如
  Graph 节点）调用 ToolGovernance.check() 做前置校验；
- 校验包含两条：一是调用总次数不能超过上限（防止 Agent 陷入死循环/失控），
  二是工具名必须在当前 Agent 的白名单内（最小权限原则，防止越权调用）。
"""
from careerpilot.config import settings


class ToolGovernance:
    """Tool 调用次数和白名单检查器。

    典型用法：
        governance = ToolGovernance()
        governance.check("search_knowledge", current_count, {"search_knowledge"})
    """
    def __init__(self, max_calls: int = settings.max_tool_calls):
        # 单次任务中允许的最大工具调用次数，默认从全局配置读取。
        # 注意：max_calls 只是"次数阈值"，真正的累计计数（call_count）由调用方
        # （例如 tools/client.py 的 ToolClient.call）在每次调用前传入；本类不维护任何可变状态，
        # 因此同一个 ToolGovernance 实例可以被多个任务、多个 Agent 安全复用
        self.max_calls = max_calls

    def check(self, tool_name: str, call_count: int, allowed_tools: set[str] | None = None) -> None:
        """执行两次前置检查，不通过则抛异常。

        - 参数：tool_name 本次要调用的工具名；call_count 目前已累计的调用次数
          （注意：是"已经发生的"次数，所以判断用 >= 保证调用前还剩至少一次额度）；
          allowed_tools 当前 Agent 允许调用的工具白名单（None 表示不限制）。
        - 返回：无。任何一项检查失败都会抛出异常（ValueError / PermissionError），
          上层应捕获并终止本次工具调用。
        """
        # 检查 1：次数上限。用 >= 而不是 >，表示"第 max_calls 次调用本身也不允许"，
        # 确保 Agent 最多只实际执行 max_calls - 1 次工具调用，留有余量。
        if call_count >= self.max_calls:
            raise ValueError("本次任务的工具调用次数已达到上限")
        # 检查 2：白名单。只有当调用方显式传入 allowed_tools 时才做此校验
        # （传 None 表示"当前场景不做白名单限制"），并且工具名必须精确匹配。
        if allowed_tools is not None and tool_name not in allowed_tools:
            raise PermissionError(f"当前 Agent 无权调用工具：{tool_name}")


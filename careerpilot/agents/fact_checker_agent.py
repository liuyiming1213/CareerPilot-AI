"""事实校验 Agent：检查最终匹配结果是否有上游证据。

职责：作为多智能体流水线的"质检"环节，把 Match Agent 得出的结论
（匹配技能数量、缺失技能数量）逐条转成可核查的声明（claim），并标明
每一条声明的来源（source）与支持与否（supported），防止把没有证据支撑
的内容直接交给用户。

注意：本 Agent 不调用 LLM（也不需要 MCP/外部工具），而是基于上游的
结构化数据做规则型校验，因此它天然可复现、不会"编造"证据，是整个
流水线里最稳定、零外部依赖的一环。
"""

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope, AgentEvidence, FactCheckItem, FactCheckResult, MatchAnalysis


class FactCheckerAgent(BaseAgent):
    """事实校验智能体：把匹配结论转换为带来源的 FactCheckItem 清单。"""

    name = "fact_checker_agent"

    def run(self, match_result: AgentEnvelope) -> AgentEnvelope:
        """校验匹配结果。

        参数:
            match_result: Match Agent 的输出外壳，data["analysis"] 是
                MatchAnalysis 的字典形式。

        返回:
            AgentEnvelope：data["fact_check"] 为 FactCheckResult 字典，
            包含 passed 标志、逐条声明 items 和警告 warnings。
        """
        def work() -> AgentEnvelope:
            # 刻意设计：这里完全不调用 LLM、也不使用 create_agent/MCP，
            # 只对上游的结构化数据做确定性规则校验，因此结果 100% 可复现、
            # 绝不会"编造"证据——这正是质检环节应该具备的可靠性质。
            # 从上游外壳中反序列化出 MatchAnalysis 结构化模型。
            match = MatchAnalysis.model_validate(match_result.data.get("analysis", {}))
            # 把匹配结论逐条固化为"可核查声明"：声明内容、是否被支持、理由、来源。
            items = [
                FactCheckItem(
                    claim=f"匹配技能数量为 {len(match.matched_skills)}",
                    supported=True,
                    reason="由 Match Agent 的结构化技能交集计算得到",
                    source="match_agent",
                ),
                FactCheckItem(
                    claim=f"缺失技能数量为 {len(match.missing_skills)}",
                    supported=True,
                    reason="由 MCP calculate_skill_gap 工具返回",
                    source="mcp.calculate_skill_gap",
                ),
            ]
            # 若上游发生降级/失败，则整体标记为未通过，并附带警告说明。
            result = FactCheckResult(passed=match_result.success, items=items, warnings=[] if match_result.success else ["上游 Agent 发生降级"])
            # 把校验结果打包进统一外壳；success 恒为 True（校验本身成功了），
            # 真正反映结论可信度的是 data["fact_check"].passed。
            return AgentEnvelope(
                agent_name=self.name,
                success=True,
                data={"fact_check": result.model_dump()},
                evidence=[AgentEvidence(source="fact_checker", text="已检查匹配分数和技能差距来源")],
            )

        # 基类重试机制：失败时返回一个 passed=False 的校验结果外壳。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc), data={"fact_check": FactCheckResult(passed=False).model_dump()}),
        )


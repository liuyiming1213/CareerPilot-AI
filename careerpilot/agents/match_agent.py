"""岗位匹配 Agent：接收 Resume Agent 和 JD Agent 的结构化真实输出。

职责：把上游两个 Agent（简历分析、JD 分析）的结构化结果拿来比较，
调用 MCP 的 calculate_skill_gap 工具计算匹配度、命中的技能和缺失的技能，
并针对缺失技能生成"补强建议"，作为后续 Fact Checker（事实校验）和
Learning Plan（学习规划）的输入。

重要特点：本 Agent 不直接处理原始文本，而是接收上游 AgentEnvelope 中的
结构化 data（这正是 contracts.py 定义契约的原因）。
"""

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope, AgentEvidence, JDAnalysis, MatchAnalysis, ResumeAnalysis
from careerpilot.agents.runtime import create_role_agent, invoke_role_agent
from careerpilot.tools.client import ToolClient


class MatchAgent(BaseAgent):
    """人岗匹配智能体：比较简历技能与岗位要求，输出匹配结果与补强建议。"""

    name = "match_agent"

    def run(self, resume_result: AgentEnvelope, jd_result: AgentEnvelope) -> AgentEnvelope:
        """执行匹配。

        参数:
            resume_result: Resume Agent 的输出外壳，其 data["analysis"] 是
                ResumeAnalysis 的字典形式。
            jd_result: JD Agent 的输出外壳，其 data["analysis"] 是
                JDAnalysis 的字典形式。

        返回:
            AgentEnvelope：data["analysis"] 为 MatchAnalysis 字典；
            data["upstream"] 保留上游两个分析的副本便于排查。
            注意 success 同时取决于上游是否都成功，否则标记 degraded。

        匹配链路按两级尝试：
        第 1 级：create_agent 真实 Agent（由模型调用 calculate_skill_gap 工具算分）；
        第 2 级：无 create_agent 时直接手工调用该 MCP 工具算分（纯规则，无 LLM）。
        """
        # ToolClient 统一封装工具调用并记录已用工具名。
        tools = ToolClient()

        def work() -> AgentEnvelope:
            # —— 第 1 步：从上游外壳中反序列化出结构化模型 ——
            # 本 Agent 不读原始文本，只吃上游 AgentEnvelope.data 里的结构化
            # dict，再用 model_validate 还原成 Pydantic 对象（dict -> 对象）。
            resume = ResumeAnalysis.model_validate(resume_result.data.get("analysis", {}))
            jd = JDAnalysis.model_validate(jd_result.data.get("analysis", {}))
            # —— 第 2 步：优先走"真实 Agent"路径 ——
            # 有密钥时创建"必须调用 calculate_skill_gap MCP 工具"的智能体，
            # 把上游两个结构化结果以 JSON 形式喂给模型，由它算分并产出 MatchAnalysis；
            # 无密钥时 create_role_agent 返回 None，进入第 3 步纯规则降级。
            runtime_agent = create_role_agent(
                self.name, "system",
                "你是人岗匹配专家。必须调用 calculate_skill_gap MCP Tool 计算分数，不要凭感觉估算，再根据证据给出建议。",
                {"calculate_skill_gap"}, MatchAnalysis,
            )
            runtime_tools: list[str] = []
            if runtime_agent is not None:
                result, runtime_tools = invoke_role_agent(
                    runtime_agent,
                    "请根据以下两个上游 Agent 的结构化结果完成匹配：\n"
                    f"简历分析：{resume.model_dump_json()}\n岗位分析：{jd.model_dump_json()}",
                )
                if result is None:
                    raise RuntimeError("create_agent 没有返回 MatchAnalysis")
            else:
                # —— 第 3 步：无 create_agent 时走纯规则降级 ——
                # 直接调用同一个 MCP 工具算分（返回 missing/matched/score），
                # 再据此拼出证据缺口和补强建议；这里完全不经过 LLM。
                gap = tools.call("calculate_skill_gap", {"required_skills": jd.required_skills, "user_skills": resume.skills})
                missing = gap["missing"]
                result = MatchAnalysis(
                    score=gap["score"], matched_skills=gap["matched"], missing_skills=missing,
                    evidence_gaps=[f"缺少 {skill} 的项目证据" for skill in missing],
                    recommendations=[f"针对 {skill} 准备一个可讲清楚的实践案例" for skill in missing],
                )
            # —— 第 4 步：打包结果 ——
            # 把上游两份证据拼接保留，作为下游 Fact Checker 的核查来源；
            # success 取决于上游是否全部成功，任一上游降级则自身也标记 degraded。
            return AgentEnvelope(
                agent_name=self.name,
                success=resume_result.success and jd_result.success,
                data={"analysis": result.model_dump(), "upstream": {"resume": resume.model_dump(), "jd": jd.model_dump()}},
                evidence=resume_result.evidence + jd_result.evidence,
                used_tools=runtime_tools or tools.used_tools,
                degraded=not (resume_result.success and jd_result.success),
            )

        # 基类重试机制：异常自动重试，最终失败返回 success=False 外壳，
        # 避免异常击穿工作流。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc), data={"analysis": MatchAnalysis().model_dump()}),
        )

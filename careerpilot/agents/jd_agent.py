"""岗位 JD 分析 Agent。

职责：把招聘岗位描述（JD，Job Description）文本解析成结构化信息
（岗位名称、职责、硬技能、加分项、面试重点），供 Match Agent（人岗匹配）
和 Interview Agent（面试模拟）使用。

调用 LLM 的方式：优先用基类的 ask_structured() 让大模型直接输出
JDAnalysis 结构；未配置 LLM 或调用失败时，退回到基于关键词的规则型
fallback（按行找“负责/设计/开发/构建”等职责行、用 extract_skills 提取技能）。
"""

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope, AgentEvidence, JDAnalysis
from careerpilot.service import extract_skills
from careerpilot.tools.client import ToolClient
from careerpilot.agents.runtime import create_role_agent, invoke_role_agent


class JDAgent(BaseAgent):
    """岗位分析智能体：把 JD 文本解析为结构化的 JDAnalysis。"""

    name = "jd_agent"

    def run(self, jd_text: str) -> AgentEnvelope:
        """分析岗位 JD。

        参数:
            jd_text: 招聘岗位的原始描述文本。

        返回:
            AgentEnvelope：成功时 data["analysis"] 是 JDAnalysis 的字典形式，
            data["knowledge"] 是 MCP 查回的公共知识；失败时 success=False。

        解析链路按三级降级依次尝试：
        第 1 级：create_agent 真实 Agent（可调用职业知识库 MCP）；
        第 2 级：仅 LLM（ask_structured 输出 JDAnalysis 结构）；
        第 3 级：无 LLM 时用关键词规则提取。
        """
        # ToolClient 统一封装工具调用并记录已用工具名。
        tools = ToolClient()

        def work() -> AgentEnvelope:
            # —— 第 1 级：优先走"真实 Agent"路径 ——
            # 有密钥时创建可自主调用"职业知识库"MCP 工具的智能体（用于补充
            # 行业背景）；无密钥时 create_role_agent 返回 None，进入第 2 级。
            runtime_agent = create_role_agent(
                self.name, "system",
                "你是招聘岗位分析专家，请严格根据 JD 提取要求，不要编造技术栈；需要行业背景时调用职业知识库 MCP Tool。",
                {"search_career_knowledge"}, JDAnalysis,
            )
            runtime_tools: list[str] = []
            knowledge = []
            if runtime_agent is not None:
                analysis, runtime_tools = invoke_role_agent(runtime_agent, f"请结构化分析以下岗位 JD：\n{jd_text}")
                if analysis is None:
                    raise RuntimeError("create_agent 没有返回 JDAnalysis")
            else:
                # —— 第 2 级：create_agent 不可用时降级 ——
                # 先手工调用职业知识库 MCP，再用 ask_structured 让 LLM 输出
                # JDAnalysis 结构；无 LLM 时 analysis 为 None，进入第 3 级。
                knowledge = tools.call("search_career_knowledge", {"query": "AI应用开发岗位能力", "limit": 5})
                analysis = self.ask_structured(
                    JDAnalysis,
                    "你是招聘岗位分析专家，请严格根据 JD 提取要求，不要补充原文没有的技术栈。",
                    f"请结构化分析以下岗位 JD：\n{jd_text}",
                )
            # —— 第 3 级：LLM 不可用或调用失败（analysis 为 None）时走规则型 fallback ——
            # 岗位名取第一行，职责行按关键词过滤，硬技能与面试重点复用技能提取，
            # 保证没有模型也能产出可用的 JDAnalysis。
            if analysis is None:
                skills = extract_skills(jd_text)
                analysis = JDAnalysis(
                    role_title=jd_text.splitlines()[0][:100] if jd_text.splitlines() else "目标岗位",
                    responsibilities=[line.strip() for line in jd_text.splitlines() if any(x in line for x in ["负责", "设计", "开发", "构建"])][:10],
                    required_skills=skills,
                    preferred_skills=[],
                    interview_focus=skills[:8],
                )
            # —— 第 4 步：把分析结果、公共知识与证据一起打包进统一外壳返回 ——
            # used_tools 同样优先取 create_agent 的工具轨迹，否则取手工调用的记录。
            return AgentEnvelope(
                agent_name=self.name,
                success=True,
                data={"analysis": analysis.model_dump(), "knowledge": knowledge},
                evidence=[AgentEvidence(source="jd", text=jd_text[:500])],
                used_tools=runtime_tools or tools.used_tools,
            )

        # 基类重试机制：临时异常自动重试，最终失败返回 success=False 外壳，
        # 保证异常不会击穿整个工作流。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc), data={"analysis": JDAnalysis().model_dump()}),
        )

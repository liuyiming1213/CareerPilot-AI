"""简历分析 Agent。

职责：把用户粘贴的原始简历文本解析成结构化信息（候选人简介、技能列表、
项目经历、教育经历、证据文本），供下游的 Match Agent（岗位匹配）、
Interview Agent（面试模拟）等环节使用。

调用 LLM 的方式：优先使用基类 BaseAgent 提供的 ask_structured()，让大模型
直接输出 Pydantic 结构 ResumeAnalysis（避免解析自由文本）；当没有配置
LLM 或调用失败（返回 None）时，退回到基于关键词的规则型 fallback，
保证即使没有大模型也能跑通整个流程。
"""

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope, AgentEvidence, ResumeAnalysis
from careerpilot.service import extract_skills
from careerpilot.tools.client import ToolClient
from careerpilot.agents.runtime import create_role_agent, invoke_role_agent


class ResumeAgent(BaseAgent):
    """简历分析智能体：把原始简历文本解析为结构化的 ResumeAnalysis。"""

    name = "resume_agent"

    def run(self, user_id: str, resume_text: str) -> AgentEnvelope:
        """解析简历。

        参数:
            user_id: 用户 ID，用于通过 MCP 工具读取该用户已有的画像信息。
            resume_text: 用户粘贴的原始简历文本。

        返回:
            AgentEnvelope：统一外壳。成功时 data["analysis"] 是 ResumeAnalysis
            的字典形式，data["profile"] 是用户画像；多次重试仍失败时
            success=False，并携带一个空的 ResumeAnalysis。

        解析链路按三级降级依次尝试：
        第 1 级：LangChain create_agent 真实 Agent（模型可自主调用 MCP 工具）；
        第 2 级：仅 LLM（ask_structured 直接输出 ResumeAnalysis 结构）；
        第 3 级：无 LLM 时用关键词规则提取（保证离线也能跑通）。
        """
        # ToolClient 负责以统一方式调用工具，并记录本轮实际用了哪些工具。
        tools = ToolClient()

        def work() -> AgentEnvelope:
            # —— 第 1 级：优先走"真实 Agent"路径 ——
            # 只要配置了 LLM_API_KEY，create_role_agent 就用 LangChain create_agent
            # 创建"能自主决定何时调用 MCP 工具"的智能体（如读用户画像、查长期记忆），
            # 由模型在回答时自行决定是否发起 tool_call；没有密钥时它返回 None，
            # 于是进入下面的第 2 级降级。
            runtime_agent = create_role_agent(
                self.name, user_id,
                "你是简历分析专家，只能根据简历原文提取信息，不允许编造经历。需要补充用户画像时调用 MCP 工具。",
                {"read_user_profile", "search_user_long_memory"}, ResumeAnalysis,
            )
            runtime_tools: list[str] = []
            profile = {}
            if runtime_agent is not None:
                analysis, runtime_tools = invoke_role_agent(runtime_agent, f"请结构化分析以下简历：\n{resume_text}")
                if analysis is None:
                    raise RuntimeError("create_agent 没有返回 ResumeAnalysis")
            else:
                # —— 第 2 级：create_agent 不可用（没装 langchain 或没密钥）时降级 ——
                # 先手工调用 MCP 工具读取用户画像，再用 ask_structured 让 LLM
                # 直接输出 ResumeAnalysis 结构；若连 LLM 都没有，analysis 会是 None。
                profile = tools.call("read_user_profile", {"user_id": user_id})
                analysis = self.ask_structured(
                    ResumeAnalysis,
                    "你是简历分析专家，只能根据原文提取信息，不允许编造经历。",
                    f"请结构化分析以下简历：\n{resume_text}",
                )
            # —— 第 3 级：LLM 不可用或调用失败（analysis 为 None）时走规则型 fallback ——
            # 这里刻意不用模型：按行扫描原文，用关键词找出项目/教育相关行，
            # 技能交给 extract_skills 提取，证据取原文前 500 字符，保证离线可跑。
            if analysis is None:
                projects = [line.strip() for line in resume_text.splitlines() if "项目" in line or "RAG" in line]
                analysis = ResumeAnalysis(
                    candidate_summary=f"候选人简历共 {len(resume_text)} 个字符，具备 {len(extract_skills(resume_text))} 项可识别技术能力。",
                    skills=extract_skills(resume_text),
                    projects=projects[:10],
                    education=[line.strip() for line in resume_text.splitlines() if "大学" in line or "本科" in line][:5],
                    evidence_texts=[resume_text[:500]],
                )
            # —— 第 4 步：把分析结果、用户画像、证据和用过的工具名打包进统一外壳返回 ——
            # used_tools 优先取 create_agent 实际发出的工具轨迹；第 2/3 级则取
            # ToolClient 手工调用时记录的工具名。
            return AgentEnvelope(
                agent_name=self.name,
                success=True,
                data={"analysis": analysis.model_dump(), "profile": profile},
                evidence=[AgentEvidence(source="resume", text=resume_text[:500])],
                used_tools=runtime_tools or tools.used_tools,
            )

        # 使用基类的重试机制：work() 抛异常时自动指数退避重试，最终仍失败则
        # 返回 success=False 的外壳（data 里放一个空 ResumeAnalysis），
        # 而不是让异常向上抛出导致整个工作流（Graph）崩溃。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc), data={"analysis": ResumeAnalysis().model_dump()}),
        )

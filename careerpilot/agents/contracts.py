"""多智能体之间传递的结构化数据契约。

为什么要单独定义契约？
如果 Agent 之间只传字符串，Match Agent 很难知道哪些内容是技能、哪些内容是
证据，也无法让 Fact Checker 精确检查。使用 Pydantic 模型后，每个 Agent 的
输入和输出都有明确字段，面试时也可以展示"结构化消息传递"。

本文件只定义"数据结构"，不包含任何业务逻辑。契约分两层：
- 外壳层（AgentEnvelope）：所有 Agent 统一的进出接口，携带成功/重试/降级
  状态、证据清单和用过的工具名，让上层 Graph 只依赖这一种类型就能串联所有 Agent。
- 业务层（ResumeAnalysis / JDAnalysis / MatchAnalysis / FactCheckResult）：
  各 Agent 真正的业务载荷，塞在 envelope.data["analysis"] 里，dict 与
  Pydantic 之间用 model_dump()（出）与 model_validate()（入）互转。

AgentEvidence 是关键：它把"结论"和"结论依据"绑在一起，
FactCheckerAgent 正是遍历 evidence 判断结论是否可信，而不是听信 Agent 自述。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentEvidence(BaseModel):
    """Agent 做出结论时引用的原始证据。

    目的：让"结论"可被追溯/核查。Fact Checker 正是靠 evidence 来判断
    某个结论是否有依据，而不是只凭 Agent 自说自话。
    """

    # 证据来源，例如 resume（简历）、jd（岗位描述）、mcp_tool（工具返回）。
    source: str = Field(description="证据来源，例如 resume、jd、mcp_tool")
    text: str = Field(description="证据原文或证据摘要")
    # 置信度，默认 1.0（完全可信），范围限制在 [0, 1]（ge=大于等于, le=小于等于）。
    confidence: float = Field(default=1.0, ge=0, le=1)


class AgentEnvelope(BaseModel):
    """
    所有 Agent 的统一外壳。

    每个 Agent 的 run() 都返回这个结构，上层（Graph / Supervisor）只依赖
    这一种类型即可串联所有 Agent。

    success=False 时，Graph 不会直接崩溃，而是由 Supervisor 决定是否重试、
    降级或让 Fact Checker 标记结果不完整。
    """

    agent_name: str  # 哪个 Agent 产出的（对应 BaseAgent.name）
    success: bool  # 本次执行是否成功（fail 后靠 fallback 产出时也是 False）
    attempts: int = 1  # 实际尝试了几次（配合基类重试机制）
    data: dict[str, Any] = Field(default_factory=dict)  # 各 Agent 自己的业务数据（如 analysis 字典）
    evidence: list[AgentEvidence] = Field(default_factory=list)  # 结论引用的原始证据清单
    used_tools: list[str] = Field(default_factory=list)  # 本轮实际调用过的 MCP 工具名
    error: str | None = None  # 失败原因（成功时为 None）
    degraded: bool = False  # 是否为降级结果（重试用尽或依赖上游降级）


class ResumeAnalysis(BaseModel):
    """简历 Agent 的标准输出。

    由 ResumeAgent 产出：把原始简历文本整理成下面的结构化字段。
    """

    candidate_summary: str = ""  # 候选人一句话简介
    skills: list[str] = Field(default_factory=list)  # 从简历中提取的技能列表
    projects: list[str] = Field(default_factory=list)  # 项目经历列表
    education: list[str] = Field(default_factory=list)  # 教育经历列表
    evidence_texts: list[str] = Field(default_factory=list)  # 支撑结论的原文片段


class JDAnalysis(BaseModel):
    """JD Agent 的标准输出。

    由 JDAgent 产出：把岗位描述整理成下面的结构化字段。
    """

    role_title: str = ""  # 岗位名称
    responsibilities: list[str] = Field(default_factory=list)  # 岗位职责列表
    required_skills: list[str] = Field(default_factory=list)  # 硬技能（必备技能）
    preferred_skills: list[str] = Field(default_factory=list)  # 加分项/加分技能
    interview_focus: list[str] = Field(default_factory=list)  # 面试应重点考察的方向


class MatchAnalysis(BaseModel):
    """Match Agent 的标准输出。

    由 MatchAgent 产出：比较简历技能与岗位要求后的匹配结论。
    """

    score: float = Field(default=0, ge=0, le=100)  # 匹配度分数，0~100 分
    matched_skills: list[str] = Field(default_factory=list)  # 命中的技能（双方都有）
    missing_skills: list[str] = Field(default_factory=list)  # 缺失技能（岗位要求但没有）
    evidence_gaps: list[str] = Field(default_factory=list)  # 证据缺口（有技能但缺项目佐证）
    recommendations: list[str] = Field(default_factory=list)  # 给用户的补强建议


class FactCheckItem(BaseModel):
    """事实校验中的单条声明。

    把某个结论固化为"可核查的断言"：声明内容 + 是否被支持 + 理由 + 来源。
    """

    claim: str  # 声明内容，例如"匹配技能数量为 3"
    supported: bool  # 该声明是否有证据支持
    reason: str  # 支持/不支持的理由
    source: str = ""  # 结论来源（如 match_agent、mcp.calculate_skill_gap）


class FactCheckResult(BaseModel):
    """事实校验结果。

    由 FactCheckerAgent 产出：整体是否通过 + 逐条声明 + 警告。
    """

    passed: bool  # 整体是否通过（上游降级/失败时为 False）
    items: list[FactCheckItem] = Field(default_factory=list)  # 逐条核查的声明清单
    warnings: list[str] = Field(default_factory=list)  # 警告信息（如"上游 Agent 发生降级"）


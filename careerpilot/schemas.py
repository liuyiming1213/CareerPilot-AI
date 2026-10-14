"""CareerPilot API 和 LangGraph 使用的结构化数据模型。

把输入、输出模型单独放在 schemas.py，可以避免路由文件里混杂业务逻辑，
也方便未来让 Vue3 前端直接依据 OpenAPI 文档对接。

说明：这里既包含 FastAPI 的 HTTP 请求/响应模型（RegisterRequest、ChatRequest、
TokenResponse 等），也被 LangGraph / Agent 层复用（CareerPilotChatResponse 等），
所以统一放在一起。用 Pydantic 后，字段校验（如必填、长度、取值枚举）在进入
路由函数之前就已完成，非法请求会直接得到 422 校验错误。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field



class CareerPilotChatRequest(BaseModel):
    """一次非流式对话请求。

    Field(..., min_length=1) 表示必填且非空字符串；`...` 代表"该字段没有默认值"。
    """

    # user_id 由请求体直接传入（当前聊天接口不校验 JWT）。
    user_id: str = Field(..., min_length=1, description="用户唯一标识")
    # conversation_id 决定这次对话归入哪段会话（用于记忆续接）。
    conversation_id: str = Field(..., min_length=1, description="会话唯一标识")
    message: str = Field(..., min_length=1, description="用户问题")


class CareerPilotChatResponse(BaseModel):
    """非流式返回的统一结构。

    当前 skeleton 先返回可解释的执行信息；后续接入真实 LLM 后，仍建议保留
    used_agents、used_tools 和 memory_updated，便于前端展示和面试演示。
    """

    conversation_id: str
    answer: str
    # intent：本次请求识别出的意图（job_match / mock_interview / learning_plan / general_qa）。
    intent: str
    # 以下两个列表带默认空列表：default_factory=list 让每个实例各自持有新列表，
    # 避免多个实例共享同一个可变列表的经典 Pydantic 陷阱。
    used_agents: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    # 本次对话是否已更新记忆（图流程默认都会写回窗口记忆，所以路由里固定置 True）。
    memory_updated: bool = False
    # trace_id：全链路追踪 id，可用于把一次请求的日志/记忆串起来排查。
    trace_id: str | None = None


class MemoryWriteRequest(BaseModel):
    """手动写入长期记忆的调试接口参数。"""

    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    # Literal 枚举：只允许这五种取值，其他值会在校验阶段被 Pydantic 拒绝。
    memory_type: Literal["project", "skill", "preference", "interview", "other"] = "other"


class HealthResponse(BaseModel):
    """健康检查接口的响应结构：服务状态 + 各子模块状态 + 降级组件清单。"""
    status: str
    # components：每个子系统的真实状态（ok / unavailable / error）。
    components: dict[str, str]
    # degraded：所有未就绪（非 ok/ready）的组件名列表，方便前端/告警一键识别。
    degraded: list[str] = Field(default_factory=list)


class RegisterRequest(BaseModel):
    """注册请求：用户名必填（2~64位）、密码必填（6~128位）、邮箱可选。"""
    username: str = Field(..., min_length=2, max_length=64)
    # 密码长度校验只是"请求格式"层面的约束；真正的安全保证是入库前的哈希化。
    password: str = Field(..., min_length=6, max_length=128)
    email: str | None = None


class LoginRequest(BaseModel):
    """登录请求：用户名 + 密码。"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """登录成功后的响应：JWT 令牌 + 令牌类型 + 用户 id。"""
    # JWT 字符串，客户端后续请求需放在 Authorization: Bearer <token> 头里。
    access_token: str
    token_type: str = "bearer"
    user_id: int


class TextCreateRequest(BaseModel):
    """新增简历/JD 的通用请求：标题 + 正文，公司名可选。"""
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)
    company: str | None = None


class MatchRequest(BaseModel):
    """岗位匹配请求：指定要匹配的简历与岗位 id。"""
    resume_id: int
    job_id: int


class LearningPlanRequest(BaseModel):
    """学习计划请求：学习目标 + 需补强的技能清单。"""
    goal: str = Field(..., min_length=1)
    missing_skills: list[str] = Field(default_factory=list)


class InterviewStartRequest(BaseModel):
    """开始面试请求：可选传入岗位/简历 id 作为出题上下文。"""
    job_id: int | None = None
    resume_id: int | None = None


class InterviewAnswerRequest(BaseModel):
    """面试回答请求：用户的回答内容。"""
    answer: str = Field(..., min_length=1)


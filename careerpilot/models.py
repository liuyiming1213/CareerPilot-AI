"""CareerPilot 的关系型数据模型。

关系型数据库负责保存”需要精确查询、事务和分页”的数据：用户、简历、JD、
消息、摘要、面试记录和学习任务。长期语义记忆不放在这里，而放进 ChromaDB。

表结构速览（每张表都用 cp_ 前缀与其他模块的表区分开）：
- cp_users：用户账号 + 密码哈希
- cp_resumes / cp_job_descriptions：简历与岗位 JD 的原文 + 解析后的结构化技能
- cp_conversations / cp_messages / cp_conversation_summaries：会话、消息与摘要
- cp_interview_sessions / cp_interview_turns：模拟面试场次与”提问-回答-点评”回合
- cp_learning_plans / cp_learning_tasks：学习计划主表与任务明细

共同约定：
- id 为主键（字符串表如会话/面试用业务生成的 id，数字表用自增 id）；
- 凡是”某条数据属于哪个用户”的表，都有带索引的 user_id 外键，路由层据此
  做所有权校验（越权访问时找不到数据）；
- created_at 用 server_default=func.now()，即让数据库在插入时生成时间戳，
  而不是在 Python 里填时间。
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from careerpilot.db import Base


class User(Base):
    """用户账号表：保存账号信息与登录用的密码哈希。"""

    __tablename__ = "cp_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # unique=True 加唯一索引：用户名不允许重复，注册时的重复检查依赖它兜底。
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 只存哈希后的密码（由 security.hash_password 生成），绝不存明文。
    password_hash: Mapped[str] = mapped_column(String(256))
    # email 也可选且唯一；nullable=True 表示该字段允许为 NULL（不填）。
    email: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    # server_default=func.now()：由数据库在插入时生成当前时间戳。
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Resume(Base):
    """简历表：保存简历原文与解析出的结构化技能信息。"""

    __tablename__ = "cp_resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 外键指向 cp_users.id：简历归属哪个用户；index=True 加速按用户查询。
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    # 简历原文，Text 类型可存长文本；规则式技能提取读取的就是它。
    raw_text: Mapped[str] = mapped_column(Text)
    # parse_document 的结果以 JSON 字符串形式存这里（default="{}" 兜底空对象）。
    structured_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class JobDescription(Base):
    """岗位 JD 表：保存岗位描述原文与解析出的技能要求。"""

    __tablename__ = "cp_job_descriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    # 公司名可选，不填则为 NULL。
    company: Mapped[str | None] = mapped_column(String(200), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    # 同上：岗位要求的技能解析结果存这里，供匹配/学习计划参考。
    structured_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Conversation(Base):
    """会话表：一个会话（conversation_id）对应一段连续对话。"""

    __tablename__ = "cp_conversations"

    # 会话 id 由业务侧传入（如 UUID），所以是字符串主键而非自增数字。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    """消息表：保存会话里的每条用户/助手消息。"""

    __tablename__ = "cp_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("cp_conversations.id"), index=True)
    # role 取值约定：user（用户消息）/ assistant（助手/模型消息）。
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ConversationSummary(Base):
    """会话摘要表：存放每个会话的累计摘要（用于长期上下文）。"""

    __tablename__ = "cp_conversation_summaries"

    # 以 conversation_id 作为复合主键的一部分：一个会话最多对应一条摘要。
    conversation_id: Mapped[str] = mapped_column(ForeignKey("cp_conversations.id"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    # onupdate=func.now()：每次 UPDATE 该行时数据库自动刷新为当前时间。
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class InterviewSession(Base):
    """面试场次表：一场模拟面试的会话记录。"""

    __tablename__ = "cp_interview_sessions"

    # 面试场次 id 由 router 层用 UUID 生成，作为字符串主键。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    # 出题所用的岗位 JD（可选）；面试答题时还会把结果沉淀为长期记忆。
    job_id: Mapped[int | None] = mapped_column(ForeignKey("cp_job_descriptions.id"), nullable=True)
    # 场次状态，当前默认 "active"（进行中），预留给后续"已结束"等状态。
    status: Mapped[str] = mapped_column(String(32), default="active")
    # 整场面试的累计/最终分数，可空。
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class InterviewTurn(Base):
    """面试回合表：某一轮"提问-回答-点评"的记录。"""

    __tablename__ = "cp_interview_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 属于哪场面试（外键指向 cp_interview_sessions.id）。
    session_id: Mapped[str] = mapped_column(ForeignKey("cp_interview_sessions.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    # 用户在 /interviews/{id}/answer 里提交的回答，提交前为 NULL。
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)


class LearningPlan(Base):
    """学习计划主表：一行 = 一份个性化学习计划。"""

    __tablename__ = "cp_learning_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("cp_users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    # 用户设定的学习/职业目标（service.generate_learning_plan 的原样记录）。
    goal: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LearningTask(Base):
    """学习任务表：学习计划下的每一条具体学习任务。"""

    __tablename__ = "cp_learning_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 属于哪份计划（外键指向 cp_learning_plans.id）。
    plan_id: Mapped[int] = mapped_column(ForeignKey("cp_learning_plans.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    # 优先级：数字越小越优先（LearningPlanAgent 按此排序返回）。
    priority: Mapped[int] = mapped_column(Integer, default=3)
    # 是否已完成，默认未完成；预留给后续任务勾选功能。
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


"""CareerPilot 领域服务。

路由只负责接收 HTTP 参数和返回响应，真正的业务逻辑集中在这里。这样未来
前端改成 Vue3、CLI 或测试脚本时，都可以复用同一套服务。

本文件承载"纯业务"：从简历/JD 文本里提取技能、解析文档结构、保存简历与岗位、
计算岗位匹配、生成学习计划、生成面试问题等。注意：这里不直接接触 HTTP，
所有函数都以 (db: Session, user_id, ...) 的形式接收依赖，便于复用与测试。

函数地图（按"被谁调用"理解依赖方向）：
- extract_skills / parse_document：纯函数，规则式技能提取与文本结构化，被下面
  所有函数共用，也是"无 LLM 也能跑"的兜底基线。
- create_resume / create_job：被 router 的 POST /resumes、/jobs 调用，落库。
- generate_learning_plan：被 router 的 POST /learning-plans 调用；内部调
  LearningPlanAgent 生成任务列表。
- make_interview_question：规则兜底出题函数；只有 InterviewAgent 无 LLM 时
  才会被 router 的 /interviews/start 用到。

说明：岗位匹配（/jobs/match）走的是 graph.run_match_workflow 多智能体版本
（含 Fact Checker、LLM），不在本 service 里实现；本文件只承载"无 LLM 也能跑"
的确定性业务与规则兜底。
"""

import json
import re
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

# 需要用到的 ORM 模型（要读写哪几张表，就在这导入哪几个）。
from careerpilot.models import JobDescription, LearningPlan, LearningTask, Resume
# rag_store：RAG（检索增强生成）知识库的读写入口（长期记忆/知识沉淀用）。
# 注意：本文件当前还没有真正调用它（保留为后续功能预留的导入位），
# 现有的记忆读写实际走 memory.manager 与 rag.store 的入口。
from careerpilot.rag.store import rag_store
# 注意：不在这里 `from careerpilot.agents.learning_plan_agent import LearningPlanAgent`——
# 那会在顶层形成循环导入（agents/__init__ -> jd_agent -> service）。改为在
# generate_learning_plan() 函数体内延迟导入，见下面的写法。


# 服务层内置的技能词表：用于从文本中做"关键词级"的技能识别。
# 它刻意保持简单、可解释，作为稳定的 baseline，而不是依赖昂贵的 LLM。
SKILL_WORDS = [
    "Python", "Java", "C++", "FastAPI", "Django", "LangChain", "LangGraph",
    "RAG", "Agent", "PyTorch", "TensorFlow", "Redis", "PostgreSQL", "MySQL",
    "ChromaDB", "Docker", "Linux", "Vue3", "SQLAlchemy", "BM25", "JWT", "SSE",
]


def extract_skills(text: str) -> list[str]:
    """从文本中提取已知技术词。

    这不是最终的 NLP 方案，而是一个稳定的 baseline：遍历 SKILL_WORDS 词表，
    凡词表里的技能名出现在文本中就算命中（大小写不敏感）。后续可以把它替换成
    LLM 结构化抽取，但建议保留这个规则结果用于兜底和评测对比。

    Args:
        text: 简历或 JD 的原文文本。

    Returns:
        list[str]：命中词表的技能名列表，保持词表顺序；未命中时为空列表。
    """
    # 把文本整体小写化，再逐个检查词表里的技能是否出现在文本中（大小写不敏感匹配）。
    lowered = text.lower()
    return [skill for skill in SKILL_WORDS if skill.lower() in lowered]


def parse_document(text: str) -> dict:
    """把简历/JD 文本整理成前端可展示的结构。

    Args:
        text: 简历或 JD 的原文文本。

    Returns:
        dict：包含 skills（识别出的技能）、sections（按行拆出的正文小节，
        最多 30 行）、text_length（文本总长度）。该字典会以 JSON 字符串
        形式存进数据表的 structured_json 字段。
    """
    return {
        "skills": extract_skills(text),                                          # 识别出的技能清单
        "sections": [line.strip() for line in text.splitlines() if line.strip()][:30],  # 按行拆出的正文小节（最多 30 行）
        "text_length": len(text),                                                # 文本总长度
    }


def create_resume(db: Session, user_id: int, title: str, raw_text: str) -> Resume:
    """保存简历，并把简历内容同步到用户长期记忆/知识处理中。

    Args:
        db: 数据库 Session。
        user_id: 简历所属的用户 id。
        title: 简历标题。
        raw_text: 简历原文文本。

    Returns:
        Resume：已落库（含数据库生成的自增 id）的 Resume ORM 对象。
    """
    # 先解析出结构化信息，再以 JSON 字符串形式存进 structured_json 字段。
    structured = parse_document(raw_text)
    resume = Resume(user_id=user_id, title=title, raw_text=raw_text, structured_json=json.dumps(structured, ensure_ascii=False))
    db.add(resume)
    db.commit()           # 提交事务落库
    db.refresh(resume)    # 刷新对象，拿到数据库生成的自增 id
    return resume


def create_job(db: Session, user_id: int, title: str, raw_text: str, company: str | None = None) -> JobDescription:
    """保存岗位 JD，并记录结构化技能。

    Args:
        db: 数据库 Session。
        user_id: JD 所属的用户 id。
        title: 岗位名称。
        raw_text: JD 原文文本。
        company: 公司名，可选。

    Returns:
        JobDescription：已落库（含自增 id）的 JobDescription ORM 对象。
    """
    job = JobDescription(
        user_id=user_id,
        title=title,
        company=company,
        raw_text=raw_text,
        structured_json=json.dumps(parse_document(raw_text), ensure_ascii=False),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def generate_learning_plan(db: Session, user_id: int, goal: str, missing_skills: list[str]) -> dict:
    """根据技能差距生成学习计划并保存到 PostgreSQL/SQLite。

    这是 POST /learning-plans 的真实业务。关键点：学习任务不是写死的，而是
    交给 LearningPlanAgent（careerpilot.agents.learning_plan_agent）生成——
    有 LLM 时由模型结合"目标 + 缺失技能"产出有针对性、可执行的任务列表；
    无 LLM 或调用失败时，Agent 内部已降级为规则式任务生成，因此这里
    `agent_result.get("tasks")` 的 or 分支只是"二次兜底"，正常情况下走不到。

    Args:
        db: 数据库 Session。
        user_id: 计划所属的用户 id。
        goal: 用户的学习/职业目标。
        missing_skills: 需补强的技能清单（来自岗位匹配的缺失项）。

    Returns:
        dict：plan_id（新建计划的自增 id）、goal（目标）、tasks（生成的任务
        列表，每项含 title/description/priority）。
    """
    # 先建一张学习计划主表（一行 = 一个计划）。
    plan = LearningPlan(user_id=user_id, title="CareerPilot 个性化学习计划", goal=goal)
    db.add(plan)
    db.flush()  # 先刷出 plan.id，供下面的任务表外键引用（但暂不提交）
    # 延迟导入 LearningPlanAgent：避免顶层 import 导致循环导入（见模块头注释）。
    from careerpilot.agents.learning_plan_agent import LearningPlanAgent
    # 用 LearningPlanAgent 生成有针对性、可执行的任务列表（无 LLM 时内部降级为规则）。
    agent_result = LearningPlanAgent().run(missing_skills, goal).data
    tasks = agent_result.get("tasks") or [
        {"title": f"掌握 {skill} 基础并完成一个小练习", "priority": index,
         "description": f"围绕目标“{goal}”学习 {skill}，完成文档阅读、代码实践和面试表达准备。"}
        for index, skill in enumerate(missing_skills, start=1)
    ]
    # 把每个任务保存为一条 LearningTask（priority 越靠前越优先）。
    for task in tasks:
        db.add(LearningTask(
            plan_id=plan.id,
            title=task.get("title", "学习任务"),
            description=task.get("description", ""),
            priority=task.get("priority", 1),
        ))
    db.commit()
    db.refresh(plan)
    return {"plan_id": plan.id, "goal": goal, "tasks": tasks}


def make_interview_question(job_text: str, resume_text: str = "") -> str:
    """生成一个可追问的面试问题，后续由 Interview Agent 替换。

    这是 InterviewAgent 无 LLM 时的"规则兜底"出题函数：router 的
    /interviews/start 在 InterviewAgent 没给出问题时调用它。

    Args:
        job_text: 岗位 JD 文本，优先作为提问依据。
        resume_text: 简历文本，JD 没有可用技能时的备选依据。

    Returns:
        str：一个要求候选人结合真实项目、讲清"问题-方案-结果"的追问式问题。
    """
    # 优先从 JD 文本里挑第一个技能作为提问焦点；JD 没识别到技能就用简历，都没有则落到"你的核心项目"。
    skills = extract_skills(job_text or resume_text)
    focus = skills[0] if skills else "你的核心项目"
    return f"请介绍一个你实际使用 {focus} 的项目，并说明你遇到的一个问题、解决方案和最终结果。"

"""CareerPilot 全部 HTTP 接口。

接口层只做三件事：接收参数、调用服务、组织响应。数据库查询和 Agent 流程
分别放在 service.py 与 graph.py，便于你逐层学习。

本文件是 HTTP 层的唯一路由文件，这里注册了健康检查、注册/登录、简历与岗位的
增查、岗位匹配、统一聊天、记忆读写、学习计划和模拟面试等接口。每个接口函数都
很薄，只负责：1) 从请求体/路径/当前登录用户中取出参数；2) 调用 service 层或
graph 层完成真正的业务；3) 把结果拼装成响应返回给前端。

接口速查表（"需登录"指请求头必须带 JWT，由 Depends(current_user_id) 解析）：
    GET  /health                 健康检查            不需登录
    POST /auth/register          注册                不需登录
    POST /auth/login             登录（返回 JWT）     不需登录
    POST /resumes                新增简历            需登录
    GET  /resumes                简历列表            需登录
    POST /jobs                   新增 JD             需登录
    GET  /jobs                   JD 列表             需登录
    POST /jobs/match             岗位匹配            需登录，走多智能体 LangGraph
    POST /chat                   统一聊天            当前不校验 JWT（user_id 由请求体传入）
    GET  /memory                 记忆快照            需登录
    POST /memory/long-term       写长期记忆          需登录
    POST /learning-plans         学习计划            需登录，调 LearningPlanAgent
    POST /interviews/start       开始面试            需登录，调 InterviewAgent
    POST /interviews/{id}/answer 提交面试回答        需登录，调 InterviewAgent

"Agent / MCP / 记忆"在本层是如何体现的：
- Agent：岗位匹配走 graph.run_match_workflow（Resume/JD/Match/FactChecker 四个
  智能体组成的 LangGraph 有向图）；统一聊天走 graph.run_careerpilot_graph
  （先做意图分类，再按意图路由到不同流程）；学习计划与模拟面试分别调用
  LearningPlanAgent、InterviewAgent。
- MCP：各 Agent 在有真实 LLM 时，会经由 MCP Client 调用 read_user_profile /
  search_user_long_memory / search_career_knowledge / calculate_skill_gap 等工具，
  实际用到的工具名会出现在响应的 used_tools 字段里（本文件不直接触碰 MCP）。
- 记忆：memory_manager 统一读写四层记忆——窗口记忆（Redis）、摘要（PostgreSQL）、
  长期记忆（ChromaDB）、画像（Redis）；匹配/聊天流程运行前后会自动读写记忆，
  面试答题后会沉淀一条"interview"长期记忆。
"""

import json
import os
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

# 以下导入按职责理解：
# - auth.current_user_id：从 JWT 解析"当前登录用户 id"的 FastAPI 依赖
# - db.get_db：提供数据库 Session 的依赖
# - graph.flow：LangGraph 工作流入口（多智能体匹配 / 统一聊天）
# - memory.manager：窗口记忆与长期记忆的读写入口
# - models：SQLAlchemy ORM 模型（User / Resume / JobDescription 等）
# - schemas：Pydantic 请求/响应数据模型
# - security：密码哈希与 JWT 生成/校验工具
# - service：真正的业务逻辑函数
from careerpilot.auth import current_user_id
from careerpilot.db import get_db
from careerpilot.graph.flow import (
    classify_intent,
    run_careerpilot_graph,
    run_careerpilot_graph_stream,
    run_match_workflow,
)
from careerpilot.health import run_checks
from careerpilot.memory.manager import memory_manager
from careerpilot.models import JobDescription, Resume, User
from careerpilot.schemas import (
    CareerPilotChatRequest, CareerPilotChatResponse, HealthResponse, InterviewAnswerRequest,
    InterviewStartRequest, LearningPlanRequest, LoginRequest, MatchRequest, MemoryWriteRequest,
    RegisterRequest, TextCreateRequest, TokenResponse,
)
from careerpilot.agents import InterviewAgent
from careerpilot.pdf import PDFOcrUnavailableError, PDFParseError, extract_pdf_text
from careerpilot.security import create_access_token, hash_password, verify_password
from careerpilot.service import create_job, create_resume, generate_learning_plan, make_interview_question


# 创建 APIRouter 实例：
# - prefix="/api/careerpilot"：所有接口都以该前缀开头，如 /api/careerpilot/chat
# - tags=["CareerPilot"]：在自动生成的 OpenAPI 文档里把接口归到同一个分组
router = APIRouter(prefix="/api/careerpilot", tags=["CareerPilot"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查，不要求登录，方便 Docker/Kubernetes 探活。

    返回的是各子系统的真实探测结果（数据库/Redis/向量库/记忆），
    而不是硬编码的"一切就绪"；status 恒为 ok（进程活着能响应即 ok），
    具体哪些依赖掉线会写进 components 与 degraded，供告警判断。

    Args:
        无参数。

    Returns:
        HealthResponse：status="ok"，components 为各子系统实时状态，
        degraded 为未就绪的组件名列表。
    """
    result = run_checks()
    return HealthResponse(status="ok", components=result["components"], degraded=result["degraded"])


@router.post("/auth/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    """注册用户，并保存哈希后的密码。

    公开接口，无需 JWT。新用户入库前会把密码哈希化，绝不保存明文。

    Args:
        request: RegisterRequest，含 username（必填，2~64 位）、password（必填，
            6~128 位）、email（可选）。
        db: 数据库 Session，由 FastAPI 依赖注入。

    Returns:
        dict：新用户的自增 id 与用户名。

    Raises:
        HTTPException 409：用户名已存在。
    """
    # 先查一次用户名是否已存在（数据库的 unique 约束是兜底，这里给出更友好的提示）。
    if db.scalar(select(User).where(User.username == request.username)):
        raise HTTPException(status_code=409, detail="用户名已存在")
    # 密码绝不明文入库，而是用 security.hash_password 哈希后再存储。
    user = User(username=request.username, email=request.email, password_hash=hash_password(request.password))
    db.add(user)          # 把新用户加入 Session（此刻尚未写入数据库）
    db.commit()           # 提交事务，真正落库
    db.refresh(user)      # 重新读取一次，让数据库生成的自增 id 等字段可用
    return {"user_id": user.id, "username": user.username}


@router.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """登录并返回 JWT access token。

    公开接口，无需 JWT。验证通过后，以用户 id 为载荷签发 JWT，客户端之后
    请求受保护接口时都要在 Authorization 头带上该令牌。

    Args:
        request: LoginRequest，含 username、password。
        db: 数据库 Session。

    Returns:
        TokenResponse：JWT access_token + 令牌类型（bearer）+ user_id。

    Raises:
        HTTPException 401：用户名不存在或密码错误。为安全起见，两种失败
        原因对外统一提示"用户名或密码错误"，不暴露具体是哪一种。
    """
    # 按用户名查用户；无论"用户不存在"还是"密码错误"都统一返回 401，不暴露具体原因。
    user = db.scalar(select(User).where(User.username == request.username))
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    # 验证通过后，以用户 id 为载荷签发一个 JWT；客户端后续凭它访问受保护接口。
    return TokenResponse(access_token=create_access_token(user.id), user_id=user.id)


@router.post("/resumes")
def create_resume_api(request: TextCreateRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """新增一份简历：调用 service 层保存并解析。

    需登录：Depends(current_user_id) 会从请求头 Authorization 里的 JWT
    解析出当前登录用户 id，简历会归属到这个用户。

    Args:
        request: TextCreateRequest，含 title（简历标题）、text（简历正文）。
            company 字段对本接口无意义，可忽略。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：新简历的 id、title，以及解析后的结构化信息
        （structured_json 是 JSON 字符串，这里反序列化为 dict 方便前端）。
    """
    # Depends(current_user_id)：从请求头里的 JWT 解析出当前登录用户 id。
    resume = create_resume(db, user_id, request.title, request.text)
    # structured_json 字段存的是 JSON 字符串，返回前反序列化成字典更方便前端使用。
    return {"id": resume.id, "title": resume.title, "structured": json.loads(resume.structured_json)}


@router.post("/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict:
    """上传一份 PDF 简历：后端解析成文本入库，再走与文本版一致的分析。

    需登录。该接口解决"简历大多是 PDF 文件"的问题：把 PDF 内容抽成纯文本
    存入 resume.raw_text，下游（结构化解析 / ResumeAgent / 岗位匹配 / 面试）
    无需改动即可直接使用。

    Args:
        file: 上传的 PDF 文件（multipart/form-data，字段名为 "file"）。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：新简历的 id、title（取文件名，去掉扩展名）和结构化信息。

    Raises:
        HTTPException 400：文件为空 / 不是合法 PDF / 扫描件但未配置 OCR。
    """
    # 读取上传的 PDF 字节；为空则直接报错
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 解析 PDF 为文本：电子版直接提取，扫描件走 OCR；解析失败/无 OCR 时给清晰提示
    try:
        text = extract_pdf_text(data, file.filename or "resume.pdf")
    except (PDFParseError, PDFOcrUnavailableError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 标题取文件名（去掉 .pdf 扩展名），与文本版接口一致地入库并解析
    filename = file.filename or "resume"
    title = os.path.splitext(filename)[0]
    resume = create_resume(db, user_id, title, text)
    return {"id": resume.id, "title": resume.title, "structured": json.loads(resume.structured_json)}


@router.get("/resumes")
def list_resumes(user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> list[dict]:
    """列出当前用户的所有简历，只返回列表页需要的字段。

    需登录，仅能看到自己名下的简历（按 user_id 过滤）。

    Args:
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        list[dict]：每个元素含 id、title、created_at，不含简历正文
        （raw_text 较大，列表页不需要，避免浪费带宽）。
    """
    # 按 user_id 过滤并倒序（最新在前）；只取 id/title/created_at 三个字段，避免返回大段正文。
    rows = db.scalars(select(Resume).where(Resume.user_id == user_id).order_by(Resume.id.desc())).all()
    return [{"id": row.id, "title": row.title, "created_at": row.created_at} for row in rows]


@router.post("/jobs")
def create_job_api(request: TextCreateRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """新增一份岗位 JD。

    需登录，JD 会归属到当前用户。保存时同步解析出岗位要求的技能，
    供后续岗位匹配接口使用。

    Args:
        request: TextCreateRequest，含 title（岗位名称）、text（JD 正文）、
            company（公司名，可选）。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：新 JD 的 id、title、company，以及解析后的结构化技能信息。
    """
    # request.company 是可选的（由 TextCreateRequest 的 company 字段承载）。
    job = create_job(db, user_id, request.title, request.text, request.company)
    return {"id": job.id, "title": job.title, "company": job.company, "structured": json.loads(job.structured_json)}


@router.get("/jobs")
def list_jobs(user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> list[dict]:
    """列出当前用户的所有岗位 JD。

    需登录，仅返回当前用户名下的岗位。

    Args:
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        list[dict]：每个元素含 id、title、company、created_at，不含 JD 正文。
    """
    rows = db.scalars(select(JobDescription).where(JobDescription.user_id == user_id).order_by(JobDescription.id.desc())).all()
    return [{"id": row.id, "title": row.title, "company": row.company, "created_at": row.created_at} for row in rows]


@router.post("/jobs/match")
def match_job(request: MatchRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """对指定简历与岗位执行多智能体匹配分析。

    需登录，且简历/岗位必须属于当前用户。

    这是"多智能体 + 记忆"的核心接口：内部调用 graph.run_match_workflow
    跑一张 LangGraph 有向图——Resume Agent 与 JD Agent 先并行分析，
    再把结果汇给 Match Agent 计算匹配度，最后由 FactChecker Agent 校验结论
    是否有证据。图中的 prepare_context / finalize 节点会在流程前后自动读写
    用户的窗口记忆与长期记忆；各 Agent 在有真实 LLM 时还可能经由 MCP Client
    调用检索/画像类工具。

    Args:
        request: MatchRequest，含 resume_id、job_id。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：answer（整理给用户的人话总结）、match_analysis / fact_check
        （各 Agent 的结构化结果）、used_agents / used_tools（本次用到的
        智能体与工具名）、trace_id（全链路追踪 id，用于排错）。

    Raises:
        HTTPException 404：简历或岗位不存在，或不属于当前用户。
    """
    try:
        # 先做所有权校验：简历和岗位都必须属于当前登录用户，防止越权访问别人的数据。
        resume = db.scalar(select(Resume).where(Resume.id == request.resume_id, Resume.user_id == user_id))
        job = db.scalar(select(JobDescription).where(JobDescription.id == request.job_id, JobDescription.user_id == user_id))
        if not resume or not job:
            raise LookupError("简历或岗位不存在，或不属于当前用户")
        # 这里进入真正的多智能体 LangGraph 工作流，而不是简单的集合求交。
        # conversation_id 用"match-简历id-岗位id"固定：同一对简历/岗位的多次匹配会落到
        # 同一个 LangGraph 线程（thread），checkpointer 可据此做状态续接。
        result = run_match_workflow(
            user_id=str(user_id),
            conversation_id=f"match-{request.resume_id}-{request.job_id}",
            message="请分析这份简历与目标岗位的匹配度",
            resume_text=resume.raw_text,
            jd_text=job.raw_text,
        )
        match_result = result["match_result"]
        fact_result = result["fact_check_result"]
        # 把各 Agent 的结构化结果、执行元数据和 trace_id 一并返回，方便前端展示与调试。
        return {
            "answer": result["answer"],
            "match_analysis": match_result.model_dump(),
            "fact_check": fact_result.model_dump(),
            "used_agents": result.get("used_agents", []),
            "used_tools": result.get("used_tools", []),
            "trace_id": result.get("trace_id"),
        }
    except LookupError as exc:
        # 资源不存在类错误统一转成 404；`from exc` 保留原始异常作为原因链，便于排查。
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat", response_model=CareerPilotChatResponse)
def chat(request: CareerPilotChatRequest) -> CareerPilotChatResponse:
    """非流式统一聊天接口。

    注意：该接口当前不校验 JWT，user_id 由请求体直接传入（docstring 里已提示
    后续可改为从 JWT 自动注入，补上身份校验）。

    内部走 graph 层统一入口 run_careerpilot_graph：先做规则式意图分类
    （job_match / mock_interview / learning_plan / general_qa），再按意图路由
    到对应的多智能体流程或直接问答；流程前后会自动读写窗口记忆与长期记忆。

    Args:
        request: CareerPilotChatRequest，含 user_id、conversation_id、message。

    Returns:
        CareerPilotChatResponse：统一结构——answer（回答）、intent（识别出的
        意图）、used_agents / used_tools（本次用到的智能体与工具）、
        memory_updated（是否已更新记忆）、trace_id。
    """
    # 通过 graph 层统一入口分发：先做意图分类，再按意图走对应流程。
    result = run_careerpilot_graph(request.user_id, request.conversation_id, request.message)
    # 用响应模型把结果规范成统一结构，前端只需要解析这一种格式。
    return CareerPilotChatResponse(
        conversation_id=request.conversation_id,
        answer=result["answer"],
        intent=result["intent"],
        used_agents=result["used_agents"],
        used_tools=result["used_tools"],
        memory_updated=True,
        trace_id=result["trace_id"],
    )


def _chat_sse_events(request: CareerPilotChatRequest):
    """把一次对话的流式回复包装成 SSE 事件序列（data: ...）。

    事件格式（每行 data: 后跟一个 JSON，以空行结尾）：
        {"type":"start","intent":...}   对话开始，告知前端识别到的意图
        {"type":"chunk","content":...}  一段模型文本（通用问答为真实 token 流）
        {"type":"error","msg":...}      中途出错
        {"type":"done"}                 对话结束
    """
    yield "data: " + json.dumps({"type": "start", "intent": classify_intent(request.message)}, ensure_ascii=False) + "\n\n"
    try:
        # 逐个把流式文本片段作为 chunk 事件发给前端
        for text in run_careerpilot_graph_stream(
            request.user_id, request.conversation_id, request.message
        ):
            yield "data: " + json.dumps({"type": "chunk", "content": text}, ensure_ascii=False) + "\n\n"
    except Exception as exc:
        # 中途异常不直接掐断连接，而是发一个 error 事件再正常结束
        yield "data: " + json.dumps({"type": "error", "msg": str(exc)}, ensure_ascii=False) + "\n\n"
    yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"


@router.post("/chat/stream")
def chat_stream(request: CareerPilotChatRequest):
    """统一聊天的流式接口（SSE / Server-Sent Events）。

    与 /chat 的区别：这里返回 text/event-stream，前端用 EventSource（或 fetch
    流式读取）逐段拿到模型生成的文本，实现"打字机"效果。默认走真实 LLM 的
    llm.stream()，未配 LLM 或非问答意图时一次给一段文案。

    Args:
        request: CareerPilotChatRequest，含 user_id、conversation_id、message。

    Returns:
        StreamingResponse：text/event-stream，包含 start / chunk… / done 事件。
    """
    # StreamingResponse 会在线程池里迭代这个同步生成器，不会阻塞事件循环；
    # X-Accel-Buffering: no 和 Cache-Control: no-cache 关闭中间层对 SSE 的缓冲，
    # 保证文本能实时推给前端。
    return StreamingResponse(
        _chat_sse_events(request),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.get("/memory")
def memory_snapshot(conversation_id: str, user_id: int = Depends(current_user_id)) -> dict:
    """查看某会话当前的记忆快照，供调试使用。

    需登录。返回的是四层记忆（窗口/摘要/长期/画像）的完整快照，直接来自
    memory_manager.snapshot，不读数据库。

    Args:
        conversation_id: 路径查询参数，要查看的会话 id。
        user_id: 从 JWT 解析出的当前登录用户 id。

    Returns:
        dict：window（窗口记忆）、summary（摘要记忆）、long_term（长期记忆）、
        profile（画像），以及 source（真实存储 or 降级内存）与 layers 说明。
    """
    # 快照内容来自 memory_manager（窗口记忆 + ChromaDB 长期记忆）。
    return memory_manager.snapshot(str(user_id), conversation_id)


@router.post("/memory/long-term")
def write_long_memory(request: MemoryWriteRequest, user_id: int = Depends(current_user_id)) -> dict:
    """手动写入一条长期记忆，便于调试记忆系统。

    需登录。长期记忆存放在 ChromaDB（由 memory_manager.save_long_memory 写入），
    会在后续对话/画像中被检索复用。

    Args:
        request: MemoryWriteRequest，含 user_id（声称的写入对象）、content
            （记忆内容）、memory_type（记忆类型，如 project/skill/preference/
            interview/other）。
        user_id: 从 JWT 解析出的当前登录用户 id。

    Returns:
        dict：固定返回 {"success": True}。

    Raises:
        HTTPException 403：请求体里的 user_id 与 JWT 解析出的当前用户不一致时，
        拒绝写入，防止伪造他人身份污染别人的记忆。
    """
    # 安全检查：请求里声称的 user_id 必须与 JWT 解析出的当前用户一致，防止写别人的记忆。
    if str(user_id) != request.user_id:
        raise HTTPException(status_code=403, detail="不能写入其他用户的记忆")
    # memory_type 决定该条记忆归入哪个长期记忆集合（project/skill/preference 等）。
    memory_manager.save_long_memory(request.user_id, request.content, request.memory_type)
    return {"success": True}


@router.post("/learning-plans")
def learning_plan(request: LearningPlanRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """根据用户目标与技能差距生成学习计划并入库。

    需登录。真正的业务在 service.generate_learning_plan，它调用
    LearningPlanAgent（careerpilot.agents.learning_plan_agent）把每个缺失技能
    拆成按优先级排序的学习任务：有 LLM 时由模型生成有针对性、可执行的任务，
    未配置 LLM 或调用失败时内部降级为规则式任务生成（不会报错）。任务最终
    保存进 LearningPlan + LearningTask 两张表。

    Args:
        request: LearningPlanRequest，含 goal（学习目标）、missing_skills
            （需补强的技能清单，可为空列表）。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：plan_id（新建计划的 id）、goal（目标）、tasks（生成的任务列表，
        每项含 title/description/priority）。
    """
    # generate_learning_plan 内部会实例化并调用 LearningPlanAgent（见 service.py）。
    return generate_learning_plan(db, user_id, request.goal, request.missing_skills)


@router.post("/interviews/start")
def start_interview(request: InterviewStartRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """开始一场模拟面试：基于简历/JD 生成第一道面试题。

    需登录。流程：
    1. 校验请求里的 job_id / resume_id 归属当前用户，并读取其原文作为出题
       上下文（两者都可省略，省略时 InterviewAgent 只拿到空文本）；
    2. 生成一个 UUID 作为面试场次 id，新建 InterviewSession 并入库；
    3. 调用 InterviewAgent().run(job_text, resume_text) 出第一道题——有 LLM 时
       由模型基于 JD/简历生成能深挖项目经验的问题，无 LLM 或调用失败时
       .data 里没有 question，随即用规则函数 make_interview_question 兜底；
    4. 把第一道题存为一条 InterviewTurn，随会话一起提交事务。

    Args:
        request: InterviewStartRequest，含可选的 job_id、resume_id。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：session_id（面试场次 id，供后续 /answer 接口使用）与第一道
        面试题 question。
    """
    # 延迟导入：InterviewSession/InterviewTurn 只在这个接口用到，按需加载可加快启动速度。
    from careerpilot.models import InterviewSession, InterviewTurn
    job_text = ""
    resume_text = ""
    # 若传了 job_id / resume_id 且属于当前用户，就读取其原文作为出题上下文；否则留空。
    if request.job_id:
        job = db.scalar(select(JobDescription).where(JobDescription.id == request.job_id, JobDescription.user_id == user_id))
        job_text = job.raw_text if job else ""
    if request.resume_id:
        resume = db.scalar(select(Resume).where(Resume.id == request.resume_id, Resume.user_id == user_id))
        resume_text = resume.raw_text if resume else ""
    # 会话 id 用 UUID 保证全局唯一，作为面试会话表的主键。
    session_id = str(uuid4())
    session = InterviewSession(id=session_id, user_id=user_id, job_id=request.job_id)
    db.add(session)
    db.flush()  # 先刷出主键等字段，但还不提交事务；下面还要往同一事务里加数据
    # 调用 InterviewAgent 出题：Agent 的 run() 返回一个带 .data 属性的结果外壳
    # （AgentEnvelope），.data 是结构化字典。有 LLM 时 data["question"] 是模型生成的
    # 真实题目；无 LLM 或调用失败时没有该键，于是用 or 右边的规则函数兜底出题。
    interview_data = InterviewAgent().run(job_text, resume_text).data
    question = interview_data.get("question") or make_interview_question(job_text, resume_text)
    # 把第一道题作为首轮 InterviewTurn 一起保存，随会话一起提交。
    db.add(InterviewTurn(session_id=session_id, question=question))
    db.commit()
    return {"session_id": session_id, "question": question}


@router.post("/interviews/{session_id}/answer")
def answer_interview(session_id: str, request: InterviewAnswerRequest, user_id: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict:
    """提交面试回答并保存，同时给出当前 MVP 版本的评分。

    需登录。流程：
    1. 校验面试场次存在且属于当前用户；
    2. 取该场次最近一轮（尚未回答）作为当前待答问题；
    3. 调用 InterviewAgent().run("", "", answer=...) 评分并点评——有 LLM 时基于
       STAR、技术深度和证据给出 0-100 分数与反馈；无 LLM 或解析失败时降级为
       启发式评分（按回答长度折算，兜底到 [20, 100]）；
    4. 把回答、分数、点评写回该轮并提交事务；
    5. 把"面试问题 + 得分"沉淀为一条 interview 类型长期记忆（存 ChromaDB），
       供后续对话/画像复用。

    Args:
        session_id: 路径参数，面试场次 id（来自 /interviews/start 的返回值）。
        request: InterviewAnswerRequest，含用户的回答 answer。
        user_id: 从 JWT 解析出的当前登录用户 id。
        db: 数据库 Session。

    Returns:
        dict：score（0-100 分数）、feedback（点评）、next_question（固定提示语，
        当前 MVP 版本未真正追问下一题）。

    Raises:
        HTTPException 404：面试场次不存在 / 不属于当前用户，或当前没有待回答的问题。
    """
    from careerpilot.models import InterviewSession, InterviewTurn
    # 先确认会话存在且属于当前用户，否则返回 404。
    session = db.scalar(select(InterviewSession).where(InterviewSession.id == session_id, InterviewSession.user_id == user_id))
    if not session:
        raise HTTPException(status_code=404, detail="面试场次不存在")
    # 取出该会话最近的一轮（按 id 倒序取第一条），它就是当前待回答的问题。
    turn = db.scalar(select(InterviewTurn).where(InterviewTurn.session_id == session_id).order_by(InterviewTurn.id.desc()))
    if not turn:
        raise HTTPException(status_code=404, detail="当前没有待回答问题")
    # 调用 InterviewAgent 评分并点评：answer 参数非空时，Agent 进入 "feedback"
    # 阶段。同样的降级约定——.data 里有 score/feedback 就用真实结果，否则
    # 用 or 右边按回答长度折算的启发式分数兜底（保证 MVP 也能给出评分）。
    feedback_data = InterviewAgent().run("", "", answer=request.answer).data
    score = feedback_data.get("score") or min(100, max(20, len(request.answer) // 2))
    turn.answer = request.answer
    turn.score = score
    turn.feedback = feedback_data.get("feedback", "回答已保存。请补充具体数据、技术取舍和结果。")
    db.commit()
    # 把本次面试问答沉淀为一条长期记忆，供后续对话/画像复用。
    memory_manager.save_long_memory(str(user_id), f"面试问题：{turn.question}；本次回答得分：{score}", "interview")
    return {"score": score, "feedback": turn.feedback, "next_question": "请继续说明你在项目中的技术取舍。"}

"""CareerPilot 多智能体 LangGraph 工作流。

岗位匹配流程是真正的多智能体闭环：

prepare_context
       ├───────────────┐
       ↓               ↓
ResumeAgent       JDAgent        # 两个节点并行执行
       └───────┬───────┘
               ↓
          MatchAgent       # 接收两个上游结构化结果
               ↓
       FactCheckerAgent    # 校验结果是否有证据
               ↓
             finalize

并行的关键是：Resume Agent 和 JD Agent 只读各自输入，互相没有依赖，因此
LangGraph 可以在同一 super-step 中执行它们；Match Agent 必须等两者都完成。

本文件是"编排层"：它不实现 Agent 的具体智能，只负责把若干 Agent 节点组织成
一张有向图，并驱动它们的执行顺序与数据传递。图里流动的数据全部放在
CareerPilotState 中：每个节点读取自己需要的字段、写回自己负责的字段，
节点之间通过这份共享状态协作。
"""

import operator
from typing import Annotated, TypedDict
from uuid import uuid4

# 四个 Agent 各自封装"单步智能"（分析简历 / 分析 JD / 匹配 / 事实校验）；
# memory_manager 负责在流程前后读写用户的窗口记忆与长期记忆。
from careerpilot.agents.contracts import AgentEnvelope
from careerpilot.agents.fact_checker_agent import FactCheckerAgent
from careerpilot.agents.interview_agent import InterviewAgent
from careerpilot.agents.jd_agent import JDAgent
from careerpilot.agents.learning_plan_agent import LearningPlanAgent
from careerpilot.agents.match_agent import MatchAgent
from careerpilot.agents.resume_agent import ResumeAgent
from careerpilot.llm import get_llm
from careerpilot.memory.manager import memory_manager

# 把 LangGraph 的导入包在 try/except 里：如果环境没装 LangGraph（例如只跑规则逻辑
# 的测试环境），下面的代码仍能正常导入，并自动退化为顺序执行的 fallback 版本。
try:
    try:
        # 新版本 LangGraph 把内存态 checkpointer 移到 langgraph.checkpoint.memory 的
        # InMemorySaver；旧版本里叫 MemorySaver。两个名字都试一次以兼容不同版本。
        from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError:
    # LangGraph 不可用时的占位值，保证后续代码不会因 ImportError 直接崩溃。
    MemorySaver = None
    StateGraph = None
    START, END = "__start__", "__end__"


class CareerPilotState(TypedDict, total=False):
    """所有节点共享的状态；Agent 之间通过这些结构化字段传递结果。"""
    # total=False 表示这些字段都是可选的（并非每个节点都会用到全部字段）。

    user_id: str          # 当前用户唯一标识
    conversation_id: str  # 当前会话唯一标识
    message: str          # 用户输入的消息
    resume_text: str      # 简历原文，供 Resume Agent 读取
    jd_text: str          # 岗位描述原文，供 J.D Agent 读取
    intent: str           # 意图分类结果（job_match / mock_interview / ...）
    resume_result: AgentEnvelope    # Resume Agent 的结构化输出
    jd_result: AgentEnvelope        # JD Agent 的结构化输出
    match_result: AgentEnvelope     # Match Agent 的结构化输出
    fact_check_result: AgentEnvelope  # FactChecker Agent 的结构化输出
    answer: str           # 最终整理给用户的回答文本
    # Annotated[list, operator.add]：当多个节点都向同一字段"追加"时，LangGraph
    # 会用 operator.add 自动合并，而不是覆盖——用来累计用过的 Agent / 工具。
    used_agents: Annotated[list[str], operator.add]
    used_tools: Annotated[list[str], operator.add]
    trace_id: str         # 一次运行的全链路追踪 id


def classify_intent(message: str) -> str:
    """根据用户问题路由到不同业务流程。"""
    # 规则式意图分类：先把消息转小写，再用关键词做包含匹配，命中即返回对应流程。
    text = message.lower()
    # 命中"简历/岗位/匹配"等词 → 走岗位匹配流程
    if any(word in text for word in ["简历", "resume", "岗位", "jd", "匹配"]):
        return "job_match"
    # 命中"面试/回答/追问"等词 → 走模拟面试流程
    if any(word in text for word in ["面试", "回答", "追问"]):
        return "mock_interview"
    # 命中"学习/技能差距/计划"等词 → 走学习计划流程
    if any(word in text for word in ["学习", "技能差距", "计划"]):
        return "learning_plan"
    # 兜底：其余问题当作通用问答处理
    return "general_qa"


def build_match_graph():
    """构建岗位匹配多智能体图。"""
    # 环境没有 LangGraph 时返回 None，调用方据此走顺序 fallback。
    if StateGraph is None:
        return None

    #用CareerPilotState 作为整张图的"状态类型"。
    builder = StateGraph(CareerPilotState)

    # 下面是图中各节点。每个节点函数都接收当前 state，返回一个 dict：
    # 返回的字段会被合并进状态（键名相同则覆盖写；operator.add 字段则累加）。
    def prepare_context(state: CareerPilotState) -> dict:
        """读取窗口/画像等上下文，并初始化流程元数据。"""
        # 流程开始前，把该会话的历史记忆加载进上下文，供后续 Agent 感知。
        memory_manager.load(state["user_id"], state["conversation_id"], state.get("message", ""))
        # 标记本次流程的意图，并记录"总控"（supervisor）已参与。
        return {"intent": "job_match", "used_agents": ["supervisor"]}

    def resume_analysis(state: CareerPilotState) -> dict:
        """Resume Agent 节点，只处理简历。"""
        result = ResumeAgent().run(state["user_id"], state.get("resume_text", ""))
        # 把 Agent 的结构化结果和它用过的工具记进状态。
        return {"resume_result": result, "used_agents": ["resume_agent"], "used_tools": result.used_tools}

    def jd_analysis(state: CareerPilotState) -> dict:
        """JD Agent 节点，只处理岗位描述。"""
        result = JDAgent().run(state.get("jd_text", ""))
        return {"jd_result": result, "used_agents": ["jd_agent"], "used_tools": result.used_tools}

    def match_analysis(state: CareerPilotState) -> dict:
        """Match Agent 节点：这里明确接收两个上游 Agent 的真实输出。"""
        # 直接取上游两个 Agent 的产物作为输入，体现"多智能体协作"。
        result = MatchAgent().run(state["resume_result"], state["jd_result"])
        return {"match_result": result, "used_agents": ["match_agent"], "used_tools": result.used_tools}

    def fact_check(state: CareerPilotState) -> dict:
        """Fact Checker 节点：检查 Match Agent 的结论和证据来源。"""
        result = FactCheckerAgent().run(state["match_result"])
        return {"fact_check_result": result, "used_agents": ["fact_checker_agent"], "used_tools": result.used_tools}

    def finalize(state: CareerPilotState) -> dict:
        """把结构化结果整理成前端可以展示的 JSON 字符串。"""
        # 从 Match / FactChecker 的结构化数据里取出关键字段，拼成给用户的最终回答。
        match_data = state["match_result"].data.get("analysis", {})
        fact_data = state["fact_check_result"].data.get("fact_check", {})
        answer = (
            f"岗位匹配度：{match_data.get('score', 0)}\n"
            f"匹配技能：{', '.join(match_data.get('matched_skills', [])) or '暂无'}\n"
            f"缺失技能：{', '.join(match_data.get('missing_skills', [])) or '暂无'}\n"
            f"事实校验：{'通过' if fact_data.get('passed') else '存在风险'}\n"
            f"建议：{'；'.join(match_data.get('recommendations', [])) or '继续补充项目证据'}"
        )
        # 把用户提问与最终回答写回记忆，让后续对话拥有上下文。
        memory_manager.append_message(state["user_id"], state["conversation_id"], "user", state.get("message", ""))
        memory_manager.append_message(state["user_id"], state["conversation_id"], "assistant", answer)
        # 每次运行生成一个新的 trace_id，用于串联日志、定位问题。
        return {"answer": answer, "trace_id": str(uuid4())}

    # ---- 注册节点：把上面定义的函数挂到图上，并指定字符串名 ----
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("resume_analysis", resume_analysis)
    builder.add_node("jd_analysis", jd_analysis)
    builder.add_node("match_analysis", match_analysis)
    builder.add_node("fact_check", fact_check)
    builder.add_node("finalize", finalize)

    # ---- 连边：定义节点之间的流转关系 ----
    builder.add_edge(START, "prepare_context")
    # 两条边从同一个节点出发，Resume Agent 和 JD Agent 会并行执行。
    builder.add_edge("prepare_context", "resume_analysis")
    builder.add_edge("prepare_context", "jd_analysis")
    # 这个 join edge 会等待两个上游节点都完成。
    # add_edge 的起点传入节点列表表示"汇聚"：等 resume_analysis 与 jd_analysis
    # 都执行完后，才进入 match_analysis（LangGraph 自动处理这个同步点）。
    builder.add_edge(["resume_analysis", "jd_analysis"], "match_analysis")
    builder.add_edge("match_analysis", "fact_check")
    builder.add_edge("fact_check", "finalize")
    builder.add_edge("finalize", END)
    # compile 生成可执行图；checkpointer 负责记录每一步状态（线程级续接）。
    # 没有 LangGraph 时 checkpointer 为 None，此时 careerpilot_graph 会是 None，
    # 调用方会自动走顺序兜底逻辑。
    return builder.compile(checkpointer=MemorySaver() if MemorySaver else None)


# 模块加载时一次性构建并编译好图，供后续直接 invoke。
careerpilot_graph = build_match_graph()


def run_match_workflow(user_id: str, conversation_id: str, message: str, resume_text: str, jd_text: str) -> dict:
    """运行岗位匹配主流程，优先使用 LangGraph，缺少依赖时顺序 fallback。"""
    # 组装初始状态：把外部传入的参数放进 CareerPilotState，供图读取。
    initial: CareerPilotState = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message": message,
        "resume_text": resume_text,
        "jd_text": jd_text,
        "used_agents": [],
        "used_tools": [],
    }
    if careerpilot_graph is not None:
        # 用 conversation_id 作为 LangGraph 的 thread_id，让同一会话的多次调用
        # 共享同一个线程的状态历史（checkpointer 据此续接）。
        config = {"configurable": {"thread_id": conversation_id}}
        # invoke 会从 START 一路执行到 END，返回最终状态（含 answer / trace_id 等）。
        return careerpilot_graph.invoke(initial, config=config)

    # 没有安装 LangGraph 时，仍按同样的 Agent 依赖顺序执行，结果结构与图版本保持一致。
    resume_result = ResumeAgent().run(user_id, resume_text)
    jd_result = JDAgent().run(jd_text)
    match_result = MatchAgent().run(resume_result, jd_result)
    fact_result = FactCheckerAgent().run(match_result)
    match_data = match_result.data.get("analysis", {})
    fact_data = fact_result.data.get("fact_check", {})
    answer = f"岗位匹配度：{match_data.get('score', 0)}；事实校验：{'通过' if fact_data.get('passed') else '存在风险'}"
    return {
        **initial,
        "resume_result": resume_result,
        "jd_result": jd_result,
        "match_result": match_result,
        "fact_check_result": fact_result,
        "answer": answer,
        # 顺序执行时没有真正跑图，这里手工记录调用过的 Agent 与工具。
        "used_agents": ["supervisor", "resume_agent", "jd_agent", "match_agent", "fact_checker_agent"],
        "used_tools": resume_result.used_tools + jd_result.used_tools + match_result.used_tools,
        "trace_id": str(uuid4()),
    }


def _memory_messages(user_id: str, conversation_id: str, message: str) -> list:
    """把该会话/该用户的记忆拼进提示词，让模型"记得"之前的对话。

    读取四层记忆（窗口历史 / 用户画像 / 会话摘要 / 相关长期记忆），作为上下文
    与当前问题一起喂给模型——这样"先告诉它名字、过会儿再问"才能答得出来。

    注意：应在把本轮用户消息写入窗口记忆"之前"调用，避免历史里重复出现当前这句。
    """
    try:
        ctx = memory_manager.load(user_id, conversation_id, query=message)
    except Exception:
        ctx = None

    parts: list[str] = []
    if ctx is not None:
        if ctx.profile:
            parts.append("已知用户画像：" + "；".join(f"{k}={v}" for k, v in ctx.profile.items()))
        if ctx.summary:
            parts.append("本会话摘要：" + ctx.summary)
        if ctx.long_term:
            parts.append("与该用户相关的长期记忆：" + "；".join(ctx.long_term))
        history = ctx.window or []
        if history:
            hist = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
                for m in history
            )
            parts.append("【历史对话】\n" + hist)

    system = "你是一个贴心的职业成长助手 CareerPilot，请用简体中文，简洁、真诚地回答。"
    if parts:
        # 把记忆追加进 system，要求模型据此保持连贯、不要忘记用户已告知的信息
        system += "\n\n以下是你的记忆上下文，请在回答时充分利用、保持连贯：\n" + "\n".join(parts)
    return [("system", system), ("human", message)]


def run_careerpilot_graph(user_id: str, conversation_id: str, message: str) -> dict:
    """统一聊天入口；有真实 LLM 时让模型回答，否则返回可继续追问的通用结果。"""
    # 先做意图分类，决定走哪条流程。
    intent = classify_intent(message)
    if intent == "job_match":
        # 统一聊天入口拿不到简历/JD 原文，这里把用户消息同时当作 resume 和 jd 传入，
        # 让岗位匹配流程仍能给出"可继续追问"的结果（属占位式实现）。
        return run_match_workflow(user_id, conversation_id, message, message, message)

    # 面试意图：用 InterviewAgent 生成第一道面试题（含记忆上下文）
    if intent == "mock_interview":
        memory_manager.append_message(user_id, conversation_id, "user", message)
        question = InterviewAgent().run("", "").data.get("question", "请介绍一下你自己。")
        memory_manager.append_message(user_id, conversation_id, "assistant", question)
        return {"intent": intent, "answer": question, "used_agents": ["interview_agent"], "used_tools": [], "trace_id": str(uuid4())}

    # 学习计划意图：用 LearningPlanAgent 根据消息中的技能差距生成学习计划
    if intent == "learning_plan":
        memory_manager.append_message(user_id, conversation_id, "user", message)
        from careerpilot.service import extract_skills
        skills = extract_skills(message)
        plan = LearningPlanAgent().run(skills, message).data
        tasks = plan.get("tasks", [])
        answer = "已为您生成学习计划：\n" + "\n".join(f"{i}. {t.get('title')}" for i, t in enumerate(tasks, 1))
        memory_manager.append_message(user_id, conversation_id, "assistant", answer)
        return {"intent": intent, "answer": answer, "used_agents": ["learning_plan_agent"], "used_tools": [], "trace_id": str(uuid4())}

    # 非匹配类意图：先读记忆上下文（此时窗口还不含本条消息），再写入本轮用户消息
    messages = _memory_messages(user_id, conversation_id, message)
    memory_manager.append_message(user_id, conversation_id, "user", message)

    llm = get_llm()
    if llm is not None:
        # 有真实 LLM：带上记忆上下文让模型回答。异常时降级为引导性文案，保证不崩。
        try:
            reply = llm.invoke(messages)
            answer = reply.content if isinstance(reply.content, str) else str(reply.content)
        except Exception:
            answer = "已识别为" + intent + "任务。请通过岗位匹配、面试或学习计划接口提供更完整上下文。"
    else:
        # 没有 LLM：返回引导性文案
        answer = "已识别为" + intent + "任务。请通过岗位匹配、面试或学习计划接口提供更完整上下文。"

    memory_manager.append_message(user_id, conversation_id, "assistant", answer)
    return {"intent": intent, "answer": answer, "used_agents": ["supervisor"], "used_tools": [], "trace_id": str(uuid4())}


def run_careerpilot_graph_stream(user_id: str, conversation_id: str, message: str):
    """统一聊天入口的流式版本：逐段 yield 最终回答文本（用于 SSE/流式返回）。

    与 run_careerpilot_graph 的区别：
    - 通用问答（general_qa）：用 llm.stream() 真正逐 token 产出文本片段。
    - 匹配/面试/学习计划意图：这些是结构化智能体先算好完整结果，这里把
      最终 answer 文本作为一段产出（它们本身不是 token 流）。
    无论哪种路径，都在产出完毕后把完整回答写回记忆，保证记忆与流式结果一致。
    """
    intent = classify_intent(message)

    # 岗位匹配：跑完整多智能体流程后，产出其最终 answer
    if intent == "job_match":
        result = run_match_workflow(user_id, conversation_id, message, message, message)
        yield result["answer"]
        return

    # 面试：用 InterviewAgent 生成第一道题（真实 LLM 文本）
    if intent == "mock_interview":
        memory_manager.append_message(user_id, conversation_id, "user", message)
        question = InterviewAgent().run("", "").data.get("question", "请介绍一下你自己。")
        memory_manager.append_message(user_id, conversation_id, "assistant", question)
        yield question
        return

    # 学习计划：LearningPlanAgent 生成任务列表，拼成回答文本
    if intent == "learning_plan":
        memory_manager.append_message(user_id, conversation_id, "user", message)
        from careerpilot.service import extract_skills

        skills = extract_skills(message)
        plan = LearningPlanAgent().run(skills, message).data
        tasks = plan.get("tasks", [])
        answer = "已为您生成学习计划：\n" + "\n".join(f"{i}. {t.get('title')}" for i, t in enumerate(tasks, 1))
        memory_manager.append_message(user_id, conversation_id, "assistant", answer)
        yield answer
        return

    # 通用问答：真正流式产出模型 token（带记忆上下文，模型才能"记得"之前的对话）
    messages = _memory_messages(user_id, conversation_id, message)
    memory_manager.append_message(user_id, conversation_id, "user", message)
    llm = get_llm()
    chunks: list[str] = []
    if llm is not None:
        try:
            # llm.stream() 逐块返回模型输出，这里把每块 content 交给 SSE
            for chunk in llm.stream(messages):
                content = getattr(chunk, "content", "")
                if content:
                    chunks.append(content)
                    yield content
        except Exception:
            # 流式失败：退回一次性引导文案，保证前端能拿到收尾
            fallback = "已识别为" + intent + "任务，但对话暂时不可用，请稍后重试。"
            chunks[:] = [fallback]
            yield fallback
    else:
        # 没有 LLM：一次性给出引导文案
        fallback = "已识别为" + intent + "任务。请通过岗位匹配、面试或学习计划接口提供更完整上下文。"
        chunks[:] = [fallback]
        yield fallback
    memory_manager.append_message(user_id, conversation_id, "assistant", "".join(chunks))

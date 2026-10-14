"""CareerPilot 多智能体角色实现。

与 agents/*_agent.py 中"类"形式的 Agent 不同，本文件提供一套更轻量的
"角色"入口：每个角色只是不同 system_prompt + 同一个执行函数 run_role_agent。

每个角色都使用同一个 run_role_agent 入口，但 system_prompt 不同。这样做的
好处是：模型客户端、异常处理和日志可以统一，而每个 Agent 的职责仍然清晰。
没有配置模型时，函数返回可解释的 fallback，保证你可以先调试 Graph 和 API。

用途场景：适合在不需要结构化输出、只需一问一答的场景（如聊天、演示）里
快速驱动任意角色；需要结构化数据时请用 agents/ 下的专用 Agent 类。

与 agents/*_agent.py 的区别：本文件没有重试、证据收集和 AgentEnvelope 外壳，
只做"一句 prompt 换一句回答"的最薄封装；返回类型是 str 而非结构化对象，
降级也只是一段带 [role fallback] 前缀的说明文字。
"""

from careerpilot.llm import get_llm


# 角色名 -> 系统提示词 的映射表。系统提示词决定模型"扮演谁、按什么原则工作"。
# 键名与 agents/base.py 里各 Agent 的 name 保持一致，方便统一理解。
ROLE_PROMPTS = {
    "resume_agent": "你是简历分析专家，只提取简历中真实出现的技能、项目、教育经历和证据，不允许编造经历。",
    "jd_agent": "你是招聘岗位分析专家，请提取岗位职责、硬技能、软技能、加分项和面试重点。",
    "match_agent": "你是岗位匹配专家，请比较岗位要求和候选人证据，输出匹配项、缺失项和补强建议。",
    "interview_agent": "你是严格但友好的技术面试官，要根据简历和岗位进行提问、追问和评分。",
    "learning_plan_agent": "你是学习规划专家，要把技能差距拆成按优先级排序的可执行任务。",
    "fact_checker_agent": "你是事实校验专家，只确认有证据支持的内容，并指出可能的编造或证据不足。",
}


def run_role_agent(role: str, task: str, context: str = "") -> str:
    """执行一个领域 Agent。

    真实模型调用使用 LangChain ChatOpenAI 兼容客户端；调用失败时不把异常
    原样返回给用户，而是返回明确的降级结果，便于开发阶段继续调试。

    参数:
        role: 角色名（须为 ROLE_PROMPTS 的键）；未知角色用通用助手提示词。
        task: 交给该角色的具体任务描述。
        context: 可选的上下文信息（如简历/JD 文本），会拼进用户消息里。

    返回:
        str：模型回答文本；未配置 LLM 或调用失败时返回带 [role fallback]
        前缀的降级文案（而非抛出异常），保证调用方永远拿到字符串。
    """
    # 从全局工厂取大模型实例；未配置 API Key 时为 None。
    llm = get_llm()
    # 按角色取系统提示词；未知角色回退到通用助手提示词，保证不抛 KeyError。
    system = ROLE_PROMPTS.get(role, "你是 CareerPilot 的通用助手。")
    # 没有大模型时给出可解释的降级文案，便于开发阶段先跑通流程。
    # 注意：这段文案带固定的 "[role fallback]" 前缀，测试/前端可据此判断
    # 是否真的走过了模型，而不会误把降级结果当成真实模型回答。
    if llm is None:
        return f"[{role} fallback] 已接收任务：{task}。请配置 LLM_API_KEY 后启用真实 Agent。"
    try:
        # 以 (角色, 内容) 消息列表调用 LangChain 兼容接口；
        # 把上下文和任务拼成一条用户消息发给模型。
        response = llm.invoke([("system", system), ("human", f"上下文：{context}\n任务：{task}")])
        # 统一把模型返回转成纯字符串（content 可能是非字符串类型）。
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        # 捕获异常并降级为文案，而不是把堆栈抛给用户；
        # 仍保留 [role fallback] 前缀，让调用方一眼看出这不是真实模型输出。
        return f"[{role} fallback] 模型暂时不可用：{type(exc).__name__}。"


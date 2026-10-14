"""学习规划 Agent。

职责：接收 Match Agent 给出的"缺失技能"清单和用户的职业目标，把每个缺失技能
拆解成一条按优先级排序的可执行学习任务（标题、优先级、说明），输出给用户作为
提升计划。

有真实 LLM 时让模型根据目标与技能差距生成有针对性、可执行的任务列表；未配置
LLM（或调用失败）时回退到规则型任务生成（对每个缺失技能生成一条任务，
优先级按列表顺序递增）。

降级开关：_llm_tasks() 的返回值。只要它返回空列表，run() 就认为"模型没有
产出可用任务"，自动切换到规则型生成——保证没有 LLM 也能给出可执行计划。
"""

import json
import re

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope
from careerpilot.llm import get_llm


class LearningPlanAgent(BaseAgent):
    """学习规划智能体：把技能差距转化为可执行的学习任务列表。"""

    name = "learning_plan_agent"

    def run(self, missing_skills: list[str], goal: str) -> AgentEnvelope:
        """制定学习计划。

        参数:
            missing_skills: 需要补强的技能列表（来自 Match Agent 的缺失项）。
            goal: 用户的职业/学习目标，用于让任务描述更有针对性。

        返回:
            AgentEnvelope：data["goal"] 为用户目标，data["tasks"] 为按优先级
            排序的任务字典列表（每项含 title/description/priority）。
        """
        def work() -> AgentEnvelope:
            # —— 第 1 步：优先让 LLM 生成结构化任务列表 ——
            # _llm_tasks 内部自含降级：无 LLM、调用异常或解析不出 JSON 时
            # 返回空列表，用空列表作为"该走规则兜底"的信号。
            tasks = self._llm_tasks(missing_skills, goal)
            if not tasks:
                # —— 第 2 步：LLM 不可用/未产出时，退到规则型任务生成 ——
                # 对每个缺失技能生成一条任务，优先级随列表顺序递增（越靠前越优先）。
                tasks = [
                    {"title": f"学习 {skill}", "priority": index,
                     "description": f"围绕{goal}完成{skill}文档阅读、代码练习和面试表达。"}
                    for index, skill in enumerate(missing_skills, start=1)
                ]
            return AgentEnvelope(agent_name=self.name, success=True, data={"goal": goal, "tasks": tasks})

        # 基类重试机制：异常自动重试，最终失败返回 success=False 外壳。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc)),
        )

    # 让 LLM 生成任务数组；无 LLM、调用异常或解析失败时一律返回空列表，
    # 空列表会在 run() 里触发规则型任务兜底（这是本 Agent 的降级开关）。
    def _llm_tasks(self, missing_skills: list[str], goal: str) -> list[dict]:
        llm = get_llm()
        if llm is None:
            return []
        try:
            prompt = (
                f"学习目标：{goal}\n"
                f"缺失技能：{', '.join(missing_skills) if missing_skills else '（未指定，请给出通用入门建议）'}\n"
                '请输出一个 JSON 数组作为学习计划，每项是 {"title": "任务标题", "description": "具体可执行说明", "priority": 序号}，'
                "按优先级从高到低排序。只输出 JSON 数组，不要任何多余文字。"
            )
            reply = llm.invoke(
                [("system", "你是学习规划专家，把技能差距拆成按优先级排序的可执行任务，只输出 JSON 数组。"),
                 ("human", prompt)]
            )
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            # 用正则抓取文本里的 JSON 数组段（re.S 让 . 能匹配换行），
            # 抓不到或解析失败都返回 []，交给 run() 走规则兜底。
            match = re.search(r"\[.*\]", text, re.S)
            if not match:
                return []
            tasks = json.loads(match.group(0))
            return tasks if isinstance(tasks, list) else []
        except Exception:
            return []
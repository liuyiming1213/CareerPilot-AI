"""模拟面试 Agent。

职责：模拟技术面试官与候选人进行多轮问答。第一轮根据岗位 JD 和简历生成
一道能深挖真实项目经验的问题（stage=question）；候选人给出答案后，调用
大模型基于 STAR、技术深度和证据给出评分与反馈（stage=feedback）。

有真实 LLM 时用模型生成提问/点评，未配置 LLM（或调用失败）时自动回退到
内置固定问题 + 启发式评分，保证没有模型也能跑通整个面试流程。
"""

import json
import re

from careerpilot.agents.base import BaseAgent
from careerpilot.agents.contracts import AgentEnvelope, AgentEvidence
from careerpilot.llm import get_llm


class InterviewAgent(BaseAgent):
    """模拟面试智能体：负责出题、接收回答并给出评分与点评。"""

    name = "interview_agent"

    def run(self, job_text: str, resume_text: str = "", answer: str | None = None) -> AgentEnvelope:
        """执行一轮模拟面试。

        参数:
            job_text: 岗位 JD 文本（出题/点评的背景）。
            resume_text: 候选人简历文本（出题依据）。
            answer: 候选人对上一轮问题的回答；为 None 时表示第一轮，需要先出题。

        返回:
            AgentEnvelope：data["stage"] 区分阶段——"question" 表示本轮给出问题；
            "feedback" 表示本轮给出 score 与 feedback。

        流程由 stage 字段驱动：第一轮（answer 为空）走 _ask_llm 出题并返回
        stage=question；前端把用户的回答传回（answer 非空）时走 _evaluate
        评分点评并返回 stage=feedback。任何一步没有 LLM 都会回退到
        内置固定问题 + 启发式评分，保证无模型也能完成整轮面试。
        """
        def work() -> AgentEnvelope:
            if not answer:
                # —— 第一轮：出题（真实 LLM 优先，固定问题兜底）——
                # _ask_llm 在未配置 LLM 或调用异常时返回空串 ""，
                # 于是这里用内置固定问题兜底，保证一定有题可问。
                question = self._ask_llm(
                    "你是严格但友好的技术面试官，针对候选人的简历和岗位提出一道能深挖真实项目经验的问题。只输出问题本身。",
                    f"岗位描述：{job_text}\n候选人简历：{resume_text}",
                )
                # 没配 LLM 或调用失败（question 为空串）时，用内置固定问题兜底。
                if not question:
                    question = "请介绍一个你实际完成的 AI 应用项目，并说明其中一次技术取舍。"
                return AgentEnvelope(
                    agent_name=self.name, success=True,
                    data={"question": question, "stage": "question"},
                    evidence=[AgentEvidence(source="interview", text=job_text)],
                )
            # —— 已有回答：进入评分 + 点评 ——
            # _evaluate 内部自带"LLM 出 JSON -> 解析失败回退启发式评分"的降级，
            # 因此这里不需要再判断 LLM 是否可用。
            score, feedback = self._evaluate(answer, job_text, resume_text)
            return AgentEnvelope(
                agent_name=self.name, success=True,
                data={"score": score, "feedback": feedback, "stage": "feedback"},
                evidence=[AgentEvidence(source="interview", text=answer)],
            )

        # 基类重试机制：异常自动重试，最终失败返回 success=False 外壳。
        return self.invoke_with_retry(
            work,
            lambda exc: AgentEnvelope(agent_name=self.name, success=False, error=str(exc)),
        )

    # 调一次真实 LLM，返回去空白后的纯文本；无 LLM 或异常时返回空串 ""。
    # 与基类 ask_llm 的差异：这里连"调用异常"也吞掉并返回空串，
    # 由调用方统一判断（空串即代表"没有可用模型，请走规则兜底"）。
    def _ask_llm(self, system: str, user: str) -> str:
        llm = get_llm()
        if llm is None:
            return ""
        try:
            reply = llm.invoke([("system", system), ("human", user)])
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            return text.strip()
        except Exception:
            return ""

    # 让 LLM 输出 JSON {score, feedback}（先在 prompt 里硬性要求，再用正则
    # 容错解析）；模型没配/报错/输出不合法时，回退到启发式评分。
    def _evaluate(self, answer: str, job_text: str, resume_text: str) -> tuple[int, str]:
        raw = self._ask_llm(
            "你是技术面试评分官，请基于 STAR 结构、技术深度和证据给出 0-100 的整数评分和简短反馈。"
            "只输出严格 JSON：{\"score\": 数字, \"feedback\": \"字符串\"}",
            f"问题背景：{job_text}\n候选人回答：{answer}",
        )
        score_match = re.search(r"\{\s*\"score\"\s*:\s*(\d+)", raw)
        if score_match:
            score = max(0, min(100, int(score_match.group(1))))
            fb = re.search(r"\"feedback\"\s*:\s*\"(.*?)(?:\"|$)", raw)
            feedback = fb.group(1) if fb else "已由面试官评分，建议补充具体数据、技术取舍和结果。"
            return score, feedback
        # 兜底：启发式评分（不依赖模型）——
        # 分数 = 回答长度折算（每 2 字符得 1 分）+ 提到"问题/难点"等深度信号加 20 分，
        # 最后用 min/max 裁到 [20, 100] 区间，避免出现离谱分数。
        score = min(100, max(20, len(answer) // 2 + (20 if ("问题" in answer or "难点" in answer) else 0)))
        return score, "回答已评分。建议补充具体数据、技术取舍和结果。"
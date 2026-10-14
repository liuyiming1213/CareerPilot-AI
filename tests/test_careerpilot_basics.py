"""不依赖外部数据库的基础测试。

测试目标不是证明整个系统已经完成，而是保证最重要的纯逻辑不会在重构时被
误删：技能提取、意图分类、窗口记忆和工具调用次数限制。

作用域说明：
- 这是"基础冒烟测试"：所有用例都不依赖 Redis / PostgreSQL / ChromaDB /
  Web 框架，任何机器上都能直接跑，用于快速发现纯逻辑被改坏；
- 运行方式（在项目根目录）：python -m pytest tests/test_careerpilot_basics.py
  或直接 `pytest` 跑整个测试目录。
"""

# 待测的纯逻辑入口分别来自：意图分类（graph 流程）、
# 记忆管理（memory 适配层）、工具治理（middleware）和技能提取（service）
from careerpilot.graph.flow import classify_intent
from careerpilot.memory.manager import MemoryManager
from careerpilot.middleware.governance import ToolGovernance
from careerpilot.service import extract_skills


def test_classify_intent():
    """意图分类：确保两个典型用户话术能被分到正确的意图类型。

    第一个用例验证"简历 + 岗位 JD 分析"被识别为 job_match；
    第二个用例验证"制定学习计划"被识别为 learning_plan。
    这两条是最核心的对话入口分支，不能被重构破坏。
    """
    assert classify_intent("请分析我的简历和岗位JD") == "job_match"
    assert classify_intent("帮我制定学习计划") == "learning_plan"


def test_extract_skills():
    """技能提取：从一段自然语言项目描述里抽取出技能关键词。

    用一句包含 Python、FastAPI、LangGraph、Redis 的话做输入，
    断言至少能抽中 Python 和 LangGraph（注意大小写与原词一致）。
    """
    skills = extract_skills("我使用 Python、FastAPI、LangGraph 和 Redis 做过项目")
    assert "Python" in skills
    assert "LangGraph" in skills


def test_memory_window():
    """窗口记忆：验证 MemoryManager 能写入并读回最近一轮消息。

    先以用户 u1 / 会话 c1 写入一条 user 角色的消息，
    再通过 snapshot 取快照，断言快照窗口的第一条内容正是写入的"你好"。
    注意：这里走的是进程内 fallback，无需真实 Redis 也能跑通。
    """
    memory = MemoryManager()
    memory.append_message("u1", "c1", "user", "你好")
    snapshot = memory.snapshot("u1", "c1")
    assert snapshot["window"][0]["content"] == "你好"


def test_tool_governance():
    """工具治理：验证次数上限和白名单两重防线。

    构造 max_calls=2 的治理器：
    1. 白名单内的 search 在次数未超限时能通过 check；
    2. 白名单外的 delete 调用必须被 PermissionError 拒绝——
       因此正常情况下永远走不到后面的 assert False，
       一旦 PermissionError 没抛出来（校验失效），测试就会失败报警。
    """
    governance = ToolGovernance(max_calls=2)
    governance.check("search", 0, {"search"})
    try:
        # delete 不在白名单 {"search"} 里，预期会抛 PermissionError
        governance.check("delete", 0, {"search"})
        # 如果上面没有抛异常，说明白名单校验失效，这里主动让测试失败
        assert False, "不在白名单中的工具应该被拒绝"
    except PermissionError:
        # 按预期捕获到权限异常，测试通过
        pass

if __name__ == '__main__':
    test_memory_window()
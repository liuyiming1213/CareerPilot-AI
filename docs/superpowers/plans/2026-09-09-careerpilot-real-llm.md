# CareerPilot Backend 落地：真实 LLM 优先

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 CareerPilot 后端能在本机启动，并使聊天/匹配流程在配置了 `DASHSCOPE_API_KEY` 时真正调用你的 MaaS 大模型（含 MCP 工具），未配置时回退到现有规则逻辑，全程可测试。

**Architecture:** 分三层实现：先把 `main.py` 的坏导入修好让服务能启动（P0）；再修 `careerpilot/llm.py` 的 `get_llm()` 使其读取 `.env` 的真实 base_url/model/thinking 配置（真实 LLM 的关键）；最后把该真实 LLM 接入 `graph/flow.py` 的聊天/匹配流程，并用 `runtime.py` 的 `create_role_agent` + MCP 工具接入，保留无 key 时的规则 fallback。

**Tech Stack:** FastAPI, Uvicorn, langchain, langchain_openai, langgraph, pydantic, mcp, chromadb, redis, sqlalchemy, Jinja…（后端）

## Global Constraints

- 使用交互解释器：`D:/agent_env/Scripts/python.exe`（本会话可用的 venv）。
- API Key 从环境变量 `DASHSCOPE_API_KEY` 读取，**绝不写入代码或 .env**。
- `.env` 已配置 `LLM_BASE_URL`（MaaS）、`LLM_MODEL=qwen3.8-flash`、`LLM_ENABLE_THINKING=false`；`get_llm()` 必须改用这些值而非硬编码。
- 所有改动保持"无 key 时正常降级"的既有约定：能跑规则，绝不因 LLM 缺失而崩溃。
- 只改后端，不碰前端、不打包部署。
- 每个任务都要能独立验证（启动 / 单测）。

---

### Task 0: 修复 main.py 导入并启动服务

**Files:**
- Modify: `D:/order_sever/main.py:17`

**Interfaces:**
- Consumes: `api_router/router.py` 导出的 `router`（APIRouter 实例）
- Produces: 可启动的 `main:app`（FastAPI 实例，含 health/lifespan）

- [ ] **Step 1:** 把第 17 行的错误导入改为从 `api_router.router` 导入

```python
from api_router.router import router as careerpilot_router
```

- [ ] **Step 2:** 验证服务能启动、`/health` 通

Run:
```bash
cd D:/order_sever
"D:/agent_env/Scripts/python.exe" -c "import main; print('main OK', main.app.title)"
```
Expected: `main OK CareerPilot AI 求职与职业成长平台`（无 `ModuleNotFoundError`）

- [ ] **Step 3:** 用 TestClient 冒烟 health

```python
# tests/test_health.py
from fastapi.testclient import TestClient
from main import app

def test_health():
    r = TestClient(app).get("/api/careerpilot/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
```

Run: `D:/agent_env/Scripts/python.exe -m pytest tests/test_health.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

---

### Task 1: 让 get_llm() 读取真实配置（关键）

**Files:**
- Modify: `D:/order_sever/careerpilot/llm.py`（整体重写 `get_llm`）
- Modify: `D:/order_sever/careerpilot/config.py`（`Settings` 增加 `llm_enable_thinking`）

**Interfaces:**
- Consumes: `careerpilot.config.settings`（`llm_api_key/llm_base_url/llm_model`）
- Produces: `get_llm() -> ChatOpenAI | None`——有 key 时返回按 `.env` 配置的真实客户端，否则 None

- [ ] **Step 1:** 在 `config.py` 的 `Settings` 中加 `enable_thinking` 字段（在 `llm_model` 之后）

```python
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    llm_enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "false").lower() in ("1", "true")
```

- [ ] **Step 2:** 重写 `careerpilot/llm.py`

```python
from careerpilot.config import settings


def get_llm():
    """返回按 .env 配置的真实 OpenAI 兼容客户端；没有 API Key 时返回 None。"""
    if not settings.llm_api_key:
        return None
    extra: dict = {}
    if settings.llm_enable_thinking:
        extra["enable_thinking"] = True
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.2,
            streaming=True,
            model_kwargs=extra,
        )
    except ImportError:
        return None
```

- [ ] **Step 3:** 验证

Run:
```bash
"D:/agent_env/Scripts/python.exe" -c "from careerpilot.llm import get_llm; llm=get_llm(); print('llm:', getattr(llm,'model_name',None), '| base_url:', getattr(llm,'openai_api_base', None) if llm else None)"
```
Expected: 打印 `llm: qwen3.8-flash` 且 base_url 以 `...maas.aliyuncs.com` 开头（不是硬编码的 `dashscope.aliyuncs.com`）

- [ ] **Step 4: Commit**

---

### Task 2: 聊天流程接入真实 LLM（带规则 fallback）

**Files:**
- Modify: `D:/order_sever/careerpilot/graph/flow.py`（`run_careerpilot_graph` 非匹配分支）

**Interfaces:**
- Consumes: `careerpilot.llm.get_llm()`，`memory_manager.append_message`
- Produces: `run_careerpilot_graph(user_id, conversation_id, message) -> dict`，`answer` 在 LLM 可用时为真实模型回复，否则为原固定文案

- [ ] **Step 1:** 重写 `run_careerpilot_graph` 的非匹配分支，让 LLM 可用时走真实模型

```python
def run_careerpilot_graph(user_id: str, conversation_id: str, message: str) -> dict:
    """统一聊天入口；有真实 LLM 时让模型回答，否则返回可继续追问的通用结果。"""
    intent = classify_intent(message)
    if intent == "job_match":
        return run_match_workflow(user_id, conversation_id, message, message, message)

    # 先把用户消息写入窗口记忆
    memory_manager.append_message(user_id, conversation_id, "user", message)

    llm = get_llm()
    if llm is not None:
        try:
            system = "你是一个贴心的职业成长助手 CareerPilot，请用简体中文，简洁、真诚地回答。"
            reply = llm.invoke([("system", system), ("human", message)])
            answer = reply.content if isinstance(reply.content, str) else str(reply.content)
        except Exception:
            answer = "已识别为" + intent + "任务。请通过岗位匹配、面试或学习计划接口提供更完整上下文。"
    else:
        answer = "已识别为" + intent + "任务。请通过岗位匹配、面试或学习计划接口提供更完整上下文。"

    memory_manager.append_message(user_id, conversation_id, "assistant", answer)
    return {"intent": intent, "answer": answer, "used_agents": ["supervisor"], "used_tools": [], "trace_id": str(uuid4())}
```

（顶部补 `from careerpilot.llm import get_llm`）

- [ ] **Step 2:** 写无 key 的降级测试（不依赖真实模型/网络）

```python
# tests/test_chat_fallback.py（仅验证规则路径）
from careerpilot.graph.flow import run_careerpilot_graph

def test_chat_fallback_without_llm():
    # 未配置 key 时 get_llm 返回 None，答案应为固定兜底文案且含意图
    import os
    os.environ.pop("DASHSCOPE_API_KEY", None)
    from careerpilot.llm import get_llm
    assert get_llm() is None
    out = run_careerpilot_graph("u1", "c1", "你好呀")
    assert out["answer"]  # 非空
    assert "intent" in out
```

Run: `D:/agent_env/Scripts/python.exe -m pytest tests/test_chat_fallback.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

---

### Task 3: 匹配流程接入真实 LLM 角色 Agent + MCP 工具

**Files:**
- Modify: `D:/order_sever/careerpilot/graph/flow.py`（`run_match_workflow`）
- Modify: require `mcp`（install）

**Interfaces:**
- Consumes: `runtime.create_role_agent/invoke_role_agent`，`build_mcp_tools`，规则 Agent（fallback）
- Produces: `run_match_workflow(...)->dict`；有 LLM 时走 `create_agent`+MCP 工具，否则规则

- [ ] **Step 1:** 安装 MCP 依赖并加入 requirements

```bash
"D:/agent_env/Scripts/python.exe" -m pip install mcp
```
（把 `mcp` 追加进 `D:/order_sever/requirements-careerpilot.txt`）

- [ ] **Step 2:** 在 `run_match_workflow` 里，LLM 可用时用 `create_role_agent` 构建真实角色 Agent 执行，否则保留现有规则分支

```python
def run_match_workflow(user_id, conversation_id, message, resume_text, jd_text) -> dict:
    from careerpilot.llm import get_llm
    from careerpilot.agents.runtime import create_role_agent, invoke_role_agent

    llm = get_llm()
    if llm is not None:
        supervisor = create_role_agent(
            role_name="match_supervisor", user_id=user_id,
            system_prompt="你是求职匹配助手，基于简历与JD分析匹配度并给出建议。",
            allowed_tools={"search_career_knowledge", "calculate_skill_gap", "search_user_long_memory"},
        )
        if supervisor is not None:
            prompt = f"简历：\n{resume_text}\n\n岗位JD：\n{jd_text}\n\n请给出匹配度、命中技能、缺失技能和建议。"
            structured, used_tools = invoke_role_agent(supervisor, prompt)
            answer = structured or "已生成匹配分析。"
            return {
                "intent": "job_match", "answer": answer, "used_agents": ["supervisor"],
                "used_tools": used_tools, "trace_id": str(uuid4()),
            }
    # —— 规则 fallback（沿用现逻辑，保持不变）——
    resume_result = ResumeAgent().run(user_id, resume_text)
    jd_result = JDAgent().run(jd_text)
    match_result = MatchAgent().run(resume_result, jd_result)
    fact_result = FactCheckerAgent().run(match_result)
    match_data = match_result.data.get("analysis", {})
    fact_data = fact_result.data.get("fact_check", {})
    answer = f"岗位匹配度：{match_data.get('score', 0)}；事实校验：{'通过' if fact_data.get('passed') else '存在风险'}"
    return {
        "intent": "job_match", "answer": answer,
        "match_result": match_result, "fact_check_result": fact_result,
        "used_agents": ["supervisor", "resume_agent", "jd_agent", "match_agent", "fact_checker_agent"],
        "used_tools": resume_result.used_tools + jd_result.used_tools,
        "trace_id": str(uuid4()),
    }
```

- [ ] **Step 3:** 验证降级（无 key）仍走规则、不出错

Run:
```bash
"D:/agent_env/Scripts/python.exe" -c "
import os; os.environ.pop('DASHSCOPE_API_KEY', None)
from careerpilot.graph.flow import run_match_workflow
r = run_match_workflow('u1','c','m','用Python做过项目','需要Python和Redis')
print('fallback answer ok:', bool(r['answer']))
"
```
Expected: `fallback answer ok: True`

- [ ] **Step 4: Commit**

---

### Task 4: 修复 ProfileAgent 缩进 bug（被记忆模块触发的潜在 NameError）

**Files:**
- Modify: `D:/order_sever/memoryagent/ProfileAgent.py`

**Interfaces:**
- Consumes: 无（独立修正）
- Produces: `ProfileAgent.get_agent/get_prompt/update` 三个方法位于类体内，实例化不再 NameError

- [ ] **Step 1:** 把模块层级错误缩进的 `get_agent`/`get_prompt`/`update` 三个方法移入 `ProfileAgent` 类内（缩进 +4），不改任何逻辑。
- [ ] **Step 2:** 验证实例化与 py_compile

Run:
```bash
"D:/agent_env/Scripts/python.exe" -m py_compile memoryagent/ProfileAgent.py
"D:/agent_env/Scripts/python.exe" -c "from memoryagent.ProfileAgent import ProfileAgent; ProfileAgent(); print('ProfileAgent OK')"
```
Expected: `ProfileAgent OK`

- [ ] **Step 3: Commit**

---

## Self-Review

- **Spec coverage:** Task 0 = 能启动（用户主诉求之一）；Task 1 = 真实 LLM 读取真实配置（核心）；Task 2 = 聊天走真实 LLM + fallback；Task 3 = 匹配走 create_agent + MCP + fallback；Task 4 = 修复一个会炸的记忆 bug。覆盖"能跑 + LLM 优先 + 不因缺 key 崩溃"。
- **Placeholder:** 无 TBD/占位；每步给全代码与命令。
- **类型一致:** `get_llm()` 返回 `ChatOpenAI|None`，Task 2/3 均按该契约判断；`create_role_agent/invoke_role_agent` 签名取自 `runtime.py` 现状一致。
# CareerPilot AI 求职与职业成长多智能体平台

这是一个不包含前端的后端项目，前端可以使用 Vue3 独立开发。

## 已实现模块

- FastAPI：认证、简历、岗位、匹配、聊天、面试、学习计划、记忆接口
- LangGraph：load memory、意图识别、Supervisor、专业 Agent、结果汇总
- Agent Tool Calling：用户画像、长期记忆、公共知识库、技能差距、记忆写入
- MCP Server：`python -m careerpilot.mcp_server`
- 四层记忆：Redis Window/Profile、SQLAlchemy Summary、Chroma LongMemory
- Agentic RAG 基础：公共知识库独立 collection、检索与文档入库
- Middleware：request_id、耗时、异常保护、工具调用白名单和调用次数控制
- SQLite fallback：本地没有 PostgreSQL、Redis、Chroma 时仍可启动并学习流程
- 多智能体岗位匹配闭环：Resume Agent 与 JD Agent 并行，Match Agent 汇总，Fact Checker 校验
- Agent 失败自动重试，超过次数后返回 degraded 结果，不拖垮整条 Graph
- Resume Agent、JD Agent、Match Agent 通过 MCP Client 实际调用 MCP Tools

## 后端目录职责

```text
careerpilot/
├─ agents/       # 专业智能体：简历、JD、匹配、面试、学习规划
├─ api/          # FastAPI 路由，只负责 HTTP 参数和响应
├─ graph/        # LangGraph 状态、节点和 Supervisor 编排
├─ memory/       # Window/Summary/Long/Profile 四层记忆
├─ middleware/   # HTTP 中间件和 Tool 权限治理
├─ tools/        # Agent Tool 定义
├─ mcp/          # MCP Server 和 MCP Tool 注册
├─ rag/          # Agentic RAG 与两个 Chroma collection
├─ config.py     # 环境变量配置
├─ db.py         # SQLAlchemy engine、Base、Session
├─ models.py     # 用户、简历、岗位、会话、面试、学习计划表
├─ schemas.py    # API 请求/响应模型
├─ service.py    # 业务服务
└─ security.py   # 密码哈希和 JWT
```

建议的依赖关系是：

```text
api → service / graph / memory
graph → agents / tools / memory / rag
agents → llm
tools → memory / rag
mcp → tools / memory
```

底层模块不要反向依赖 API，这样可以避免循环导入。

## 依赖安装

项目原来的 `.venv` 指向已经不存在的 `D:\agent_env`，建议重新创建：

```powershell
cd D:\order_sever
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-careerpilot.txt
```

## 启动 API

```powershell
cd D:\order_sever
python -m uvicorn main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000/docs`。

## 启动 MCP Server

```powershell
cd D:\order_sever
python -m careerpilot.mcp.server
```

MCP Server 默认使用 stdio。主 API 服务和 MCP Server 可以作为两个独立进程运行。

## 接口顺序

1. `POST /api/careerpilot/auth/register`
2. `POST /api/careerpilot/auth/login`
3. 使用返回的 Bearer Token 调用简历和岗位接口
4. `POST /api/careerpilot/resumes`
5. `POST /api/careerpilot/jobs`
6. `POST /api/careerpilot/jobs/match`
7. `POST /api/careerpilot/chat`
8. `POST /api/careerpilot/interviews/start`
9. `POST /api/careerpilot/interviews/{session_id}/answer`

## 数据库配置

默认使用 SQLite：

```text
DATABASE_URL=sqlite:///D:/order_sever/data/careerpilot.db
```

生产环境使用 PostgreSQL：

```text
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/careerpilot
REDIS_URL=redis://localhost:6379/0
CHROMA_DB_PATH=D:/order_sever/chromadb
```

## 代码阅读顺序

建议先读 `main.py`，再读 `careerpilot/api/router.py`，然后依次读
`service.py`、`memory/manager.py`、`tools/career_tools.py`、
`agents/roles.py` 和 `graph/flow.py`。最后阅读 `rag/store.py` 与
`mcp/server.py`，理解公共知识库和 MCP 工具如何接入。

## 多智能体岗位匹配流程

调用 `POST /api/careerpilot/jobs/match` 时，实际执行：

```text
prepare_context
      ├───────────────┐
      ↓               ↓
ResumeAgent       JDAgent
      └───────┬───────┘
              ↓
         MatchAgent
              ↓
       FactCheckerAgent
              ↓
           finalize
```

Resume Agent 会通过 MCP 调用 `read_user_profile`，JD Agent 会通过 MCP 调用
`search_career_knowledge`，Match Agent 会通过 MCP 调用 `calculate_skill_gap`。
接口响应中的 `used_agents` 和 `used_tools` 可以直接用于前端展示，也可以在面试
时证明每个 Agent 和 MCP Tool 确实参与了本次任务。

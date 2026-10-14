# CareerPilot — AI 求职与职业成长平台（后端）

面向应届生的多智能体、Agent Tool Calling + Agentic RAG 后端。提供简历/JD 的结构化解析、岗位匹配、
统一聊天（含流式）、四层记忆、模拟面试、学习计划等能力；本地加载 BGE 向量化与重排模型，无外部网络依赖。

## 技术栈
- **Web**: FastAPI + Uvicorn
- **Agent 编排**: LangGraph（多智能体有向图）+ LangChain `create_agent`（Agent Tool Calling）
- **工具层**: 进程内工具注册表 + 统一工具调用（ToolClient：租户隔离 / 频控 / 审计）
- **RAG**: ChromaDB（向量）+ BM25（jieba/rank_bm25）+ RRF 融合 + 本地 `bge-reranker-large_v1` 精排
- **记忆**: 四层记忆（窗口/摘要/长期/画像），长期用 Chroma
- **向量化**: 本地 `paraphrase-multilingual-MiniLM-L12-v2`（384 维），离线加载

> 说明：项目不使用外部 MCP 协议，工具调用是进程内普通函数式实现，保持了治理、租户隔离与审计。

## 目录结构
```
main.py                        应用入口（挂载 /static，注册路由/错误处理/健康检查）
api_router/router.py          全部 HTTP 接口（统一错误 {code,msg,trace_id}）
careerpilot/
  agents/                     多智能体（Resume/JD/Match/FactChecker/Interview/LearningPlan）
  graph/flow.py               LangGraph 编排：build_match_graph / run_match_workflow / 流式聊天
  rag/store.py                混合检索：向量 + BM25 + RRF + 本地重排
  memory/manager.py           四层记忆适配层
  tools/                      工具层（registry/client/build_tools/career_tools）
  config.py / db.py / llm.py / errors.py ...
memory/                       原记忆模块（Window/Summary/Long/Profile）
static/index.html             极简登录页（演示"登录→存token→带Bearer调用"）
```

## 快速开始
1. 准备 Python 3.12 环境，安装依赖：
   ```bash
   python -m pip install -r requirements-careerpilot.txt
   ```
2. 准备本地模型目录（可选，缺了会用哈希降级）：
   - `models/embedding_model`（向量化）
   - `models/bge-reranker-large_v1`（重排）
3. 配置环境：
   ```bash
   cp .env.example .env   # 按需修改；SECRET_KEY 务必改成长随机串
   ```
   大模型 Key 从系统环境变量注入：`export DASHSCOPE_API_KEY=你的key`
4. 启动：
   ```bash
   python main.py          # 或 python -m uvicorn main:app --reload --port 8000
   ```
5. 浏览器打开 `http://localhost:8000/` 进入登录页；接口文档 `/docs`。

## 接口速览（前缀 `/api/careerpilot`）
| 方法+路径 | 需登录 | 说明 |
|---|---|---|
| GET /health | 否 | 健康检查（实时探测 DB/Redis/Chroma/记忆）|
| POST /auth/register | 否 | 注册 |
| POST /auth/login | 否 | 登录 → 返回 JWT access_token |
| POST /resumes · GET /resumes | 是 | 新增/列表简历 |
| POST /resumes/upload | 是 | 上传 PDF 简历（电子版直提文本，扫描件走 OCR 兜底）|
| POST /jobs · GET /jobs | 是 | 新增/列表岗位 JD |
| POST /jobs/match | 是 | 多智能体岗位匹配 |
| POST /chat/stream | — | 统一聊天（SSE 流式，意图自动分发）|
| GET /memory | 是 | 记忆快照 |
| POST /memory/long-term | 是 | 写长期记忆（user_id 须与 JWT 一致）|
| POST /learning-plans | 是 | 生成学习计划 |
| POST /interviews/start · /{id}/answer | 是 | 模拟面试 |

## 认证 & 前端 401 注意
- 登录接口返回：`{"access_token": "...", "token_type": "bearer", "user_id": n}`
- **前端必须**：在登录成功后把 `access_token` 存起来（如 `localStorage`），
  并在每个受保护接口的请求头加上 **`Authorization: Bearer <token>`**。
  **没带 token 或带了无效 token → 401**（这是"登录成功但页面不跳转/后续接口全 401"的常见原因）。
- 参考 `static/index.html`：它实现了"登录→存 token→带 Bearer 调 /resumes"的完整闭环。

## 安全
- JWT 用 HS256，`SECRET_KEY` 须 ≥32 字节（`.env` 已配置强密钥；启动时会校验长度并告警）。
- 统一错误响应 `{code, msg, trace_id}`；`trace_id` 用于全链路追踪。
- 工具层实现租户隔离（user_id 闭包写死）、频控（ToolGovernance）与审计。
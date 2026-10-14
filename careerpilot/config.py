"""项目配置。

初学者可以把配置理解为"程序启动时要读取的一组开关"。例如：

- DATABASE_URL：PostgreSQL 连接地址；本地没有 PostgreSQL 时默认使用 SQLite。
- REDIS_URL：窗口记忆和画像记忆使用的 Redis 地址。
- CHROMA_DB_PATH：ChromaDB 本地持久化目录。
- LLM_API_KEY：模型 API 密钥；没有密钥时，项目会使用可演示的规则型 fallback。

这样设计的好处是：你可以先把业务流程跑通，再逐步启动 PostgreSQL、Redis、
ChromaDB 和真实大模型，而不是一开始就被环境问题卡住。

实现上：用一个 dataclass 集中存放所有配置，业务代码统一通过 settings 读取，
而不是到处直接 os.getenv。未显式设置的值都有合理的本地默认值，保证项目
"开箱即跑"。
"""

import os
from dataclasses import dataclass
from pathlib import Path

# 尝试读取 .env 文件（本地把密钥/连接串放在 .env 里，避免写死在代码中）。
# 如果没安装 python-dotenv，就用一个空实现兜底，保证纯逻辑测试仍能运行。
try:
    from dotenv import load_dotenv
except ImportError:  # 依赖尚未安装时，仍允许纯逻辑测试运行
    def load_dotenv() -> None:
        return None

# 把 .env 里的键值对加载进环境变量，之后 os.getenv 就能读到它们。
load_dotenv()


# 项目根目录：本文件位于 <根>/careerpilot/config.py，向上两级即项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 本地数据目录：SQLite 数据库默认放在这里。
DATA_DIR = PROJECT_ROOT / "data"
# 启动时确保 data 目录存在，避免首次写库时因目录不存在而报错。
DATA_DIR.mkdir(exist_ok=True)


def _resolve_database_url() -> str:
    """解析数据库连接串：只接受 SQLAlchemy 能识别的 URL，否则回落到本地 SQLite。

    - 未配置 DATABASE_URL，或配置成了 JDBC 格式（jdbc:postgresql://...，
      那是 Java 的连接串，SQLAlchemy 解析不了）时，都当作"未配置"处理，
      回落到 data/careerpilot.db，保证项目开箱即跑。
    - 生产环境只要设置标准 SQLAlchemy URL（如
      postgresql+psycopg://user:pwd@host:port/db）即可切到 PostgreSQL。
    """
    raw = os.getenv("DATABASE_URL", "").strip()
    if raw and not raw.startswith("jdbc:"):
        return raw
    return f"sqlite:///{DATA_DIR / 'careerpilot.db'}"


@dataclass(frozen=True)
class Settings:
    """集中保存所有配置，业务代码不直接到处读取 os.getenv。"""
    # frozen=True：创建后不可修改，避免运行期配置被意外改动。

    # --- 应用基本信息 ---
    app_name: str = os.getenv("CAREERPILOT_APP_NAME", "CareerPilot")
    environment: str = os.getenv("CAREERPILOT_ENV", "development")
    # --- 数据库：生产设 DATABASE_URL 指向 PostgreSQL，本地默认 SQLite 文件 ---
    database_url: str = _resolve_database_url()
    # --- 记忆用 Redis（窗口记忆 / 画像记忆） ---
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # --- ChromaDB：本地持久化目录 + 两个集合（知识库 / 长期记忆） ---
    chroma_path: str = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chromadb"))
    rag_collection: str = os.getenv("CAREERPILOT_RAG_COLLECTION", "careerpilot_knowledge")
    long_memory_collection: str = os.getenv(
        "CAREERPILOT_LONG_MEMORY_COLLECTION", "careerpilot_long_memory"
    )
    # --- 本地向量化模型路径：RAG / 长期记忆把文本转成向量时用到的模型目录 ---
    embedding_model_path: str = os.getenv(
        # PROJECT_ROOT/models/embedding_model 即 D:/order_sever/models/embedding_model
        "EMBEDDING_MODEL_PATH", str(PROJECT_ROOT / "models" / "embedding_model")
    )
    # --- RAG 混合检索：本地重排模型路径 + 召回/融合数量 ---
    reranker_model_path: str = os.getenv(
        "RERANKER_MODEL_PATH", str(PROJECT_ROOT / "models" / "bge-reranker-large_v1")
    )
    rag_recall_k: int = int(os.getenv("CAREERPILOT_RAG_RECALL_K", "20"))     # 向量/BM25 各召回多少
    rag_fuse_k: int = int(os.getenv("CAREERPILOT_RAG_FUSE_K", "10"))        # RRF 融合后取前几名进重排
    # --- 大模型：优先读 DASHSCOPE_API_KEY（阿里云百炼），兼容 OPENAI_API_KEY ---
    llm_api_key: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL", "https://ws-10wk3k5xp7l1fl01.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    llm_model: str = os.getenv("LLM_MODEL", "qwen3.8-flash")
    # qwen3 系列在 thinking 模式下不允许 create_agent 的结构化 tool_choice。
    # 默认关闭，保证 Agent Tool Calling 与 response_format 兼容。
    llm_enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
    # --- 认证：JWT 签名密钥、签名算法、令牌有效期（分钟） ---
    # 默认值也是一个足够长（≥32 字节）的弱默认，防止没配 SECRET_KEY 时又冒出
    # InsecureKeyLengthWarning；生产务必用环境变量 SECRET_KEY 覆盖成你自己的随机串。
    jwt_secret: str = os.getenv(
        "SECRET_KEY", "careerpilot-dev-secret-change-me-please-set-a-longer-secret"
    )
    jwt_algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_minutes: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "120"))
    # --- 记忆 / 工具参数：窗口大小、TTL、单轮最多工具调用次数 ---
    window_size: int = int(os.getenv("CAREERPILOT_WINDOW_SIZE", "10"))
    window_ttl_seconds: int = int(os.getenv("CAREERPILOT_WINDOW_TTL", "604800"))
    max_tool_calls: int = int(os.getenv("CAREERPILOT_MAX_TOOL_CALLS", "8"))


# 模块级单例：全项目统一使用这一个配置对象。
settings = Settings()

# 启动时校验 JWT 密钥强度：HS256 要求密钥 ≥32 字节，太短会在签发/校验时告警，
# 也更容易被暴力破解。这里只打印警告，不影响启动。
if len(settings.jwt_secret) < 32:
    import warnings

    warnings.warn(
        "SECRET_KEY 过短（<32 字节），存在安全风险且会产生 InsecureKeyLengthWarning。"
        "请为 .env 设置一个随机的长密钥。",
        RuntimeWarning,
    )

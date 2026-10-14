"""Agentic RAG 基础能力。

知识库和长期记忆必须是两个 Chroma collection：
- careerpilot_knowledge：技术、面试和职业知识，属于系统公共知识。
- careerpilot_long_memory：用户自己的经历和历史，必须按 user_id 过滤。

本模块提供文档入库、查询和简单的“是否需要继续检索”判断。真正的 Agentic
RAG 会在此基础上增加查询改写、重排序和证据充分性判断。
"""

from pathlib import Path
from typing import Any

from careerpilot.config import settings


class HashEmbeddingFunction:
    """无外部下载的开发期 Embedding。

    这不是最终语义模型，而是一个确定性的向量 fallback。它的价值是让本地
    Chroma 流程可运行；生产环境应替换为你原项目的 BGE Embedding。

    实现方式：把文本转成 128 维的"词袋式"统计向量（字符哈希散列），
    再做 L2 归一化。因为同样的文本永远得到同样的向量，所以可复现、无需联网
    下载模型，适合开发和测试阶段跑通整套 RAG 流程。
    """

    def __init__(self, dimension: int = 128):
        # 向量维度，默认 128 维
        self.dimension = dimension

    def _hash_one(self, text) -> list[float]:
        # 单条文本 -> 一个 128 维、L2 归一化的向量（字符特征散列）
        vector = [0.0] * self.dimension
        # 遍历每个字符，用 (码点 + 位置下标) 对维度取模累加，
        # 让不同文本大致产生不同的向量分布
        for index, char in enumerate(str(text)):
            vector[(ord(char) + index) % self.dimension] += 1.0
        # L2 范数；`or 1.0` 防全零向量除 0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        # chroma 入库时调用：一批文本 -> 一批向量
        return [self._hash_one(t) for t in input]

    def embed_query(self, input) -> list[list[float]]:
        # chroma 检索时调用：要求返回"向量列表"（即使只有一条查询也要再套一层列表，
        # 否则下游把单条向量当 Embeddings 解析会报 float 无法转 Sequence）。
        items = list(input) if isinstance(input, (list, tuple)) else [input]
        return [self._hash_one(t) for t in items]

    def __call__(self, input):
        # 兼容旧接口：等价于 embed_documents
        return self.embed_documents(input)

    def name(self) -> str:
        """Chroma 需要的 embedding 标识；返回 'default' 以跳过配置冲突校验。

        chroma 在 get_or_create_collection 命中已存在 collection 时会调用
        embedding_function.name() 做配置冲突判断；本项目始终用同一个函数，
        返回 'default' 让其短路校验，同时不影响真实的 __call__ 向量化。
        """
        return "default"


class LocalEmbeddingFunction:
    """用你项目的本地 BGE 向量模型（models/embedding_model）做向量化。

    设计要点：
    - 懒加载：第一次真正需要向量化时才加载本地模型，避免拖慢应用启动（模型较大）。
    - 降级：本地模型缺失或加载失败时，透明退回 HashEmbeddingFunction（词袋式哈希），
      保证即使没有本地模型，RAG 流程也能跑通。
    - 兼容：对外暴露和 HashEmbeddingFunction 一样的 __call__(input) 接口，
      Chroma 写入/查询的调用方式完全不变。
    """

    def __init__(self):
        self._local = None     # 本地 BGE 模型实例（懒加载，未加载为 None）
        self._active = True    # 是否还允许尝试加载；失败后置 False，避免反复加载
        self._hash = HashEmbeddingFunction()  # 兜底向量器

    def name(self) -> str:
        """Chroma 需要的 embedding 标识；与本地模型同用 'default'，跳过冲突校验。"""
        return "default"

    def _ensure(self):
        # 懒加载本地 BGE 模型：第一次真正需要向量化时才加载，避免拖慢启动；
        # 加载失败则置 _active=False，之后不再反复尝试。
        if self._local is None and self._active:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._local = HuggingFaceEmbeddings(model_name=settings.embedding_model_path)
                # 用一条探测文本真正触发权重加载，确认模型可出向量
                self._local.embed_query("_probe_")
            except Exception:
                self._local = None
                self._active = False  # 加载失败：之后直接走哈希兜底
        return self._local

    def embed_documents(self, input) -> list[list[float]]:
        # 入库：一批文本 -> 一批语义向量；本地模型不可用时走哈希兜底
        model = self._ensure()
        if model is not None:
            return model.embed_documents(list(input))
        return self._hash.embed_documents(list(input))

    def embed_query(self, input) -> list[list[float]]:
        # 检索：返回"向量列表"（单条查询也要套一层列表，符合 chroma 的 Embeddings 约定）
        items = list(input) if isinstance(input, (list, tuple)) else [input]
        model = self._ensure()
        if model is not None:
            # 传入文本列表，模型返回 list[list] 语义向量
            return model.embed_documents(items)
        return self._hash.embed_query(items)

    def __call__(self, input):
        # 兼容旧接口：等价于 embed_documents
        return self.embed_documents(input)


class RerankerUnavailableError(RuntimeError):
    """本地重排模型不可用（未安装依赖或加载失败）。"""


class BM25Search:
    """BM25 词法检索：用 rank_bm25 + jieba 对语料建索引并检索。

    与向量检索互补：向量懂"语义换措辞"，BM25 精确匹配关键词。两者结果
    再经 RRF 融合，兼顾召回与精确命中。
    """

    def __init__(self):
        self._bm25 = None        # BM25Okapi 索引实例（fit 后才有）
        self._ids: list[str] = []  # 与索引对齐的文档 id 列表

    def fit(self, texts: list[str], ids: list[str]) -> None:
        """用语料文本重建 BM25 索引。"""
        import jieba
        from rank_bm25 import BM25Okapi

        # 空语料无法建索引（BM25Okapi 会除零），置为"无索引"状态即可
        if not texts:
            self._bm25 = None
            self._ids = []
            return
        # 中文用 jieba 分词（BM25 按词统计），否则整段当一个词，几乎无法命中
        tokenized = [list(jieba.cut(t)) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._ids = list(ids)

    def search(self, query: str, k: int) -> list[str]:
        """返回按 BM25 得分从高到低的前 k 个文档 id。"""
        if self._bm25 is None or not self._ids:
            return []
        import jieba

        scores = self._bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # 只返回得分 > 0（真正命中的关键词）的结果
        return [self._ids[i] for i in ranked[:k] if scores[i] > 0]


class Reranker:
    """用本地 bge-reranker-large_v1（交叉编码器）对候选做精排。

    - 硬要求：模型加载失败直接抛 RerankerUnavailableError，不静默降级。
    - 懒加载：首次真正重排时才加载（模型约 2GB），避免拖慢应用启动。
    """

    def __init__(self):
        self._ce = None  # CrossEncoder 实例（懒加载）

    def _model(self):
        if self._ce is None:
            try:
                from sentence_transformers import CrossEncoder

                self._ce = CrossEncoder(settings.reranker_model_path)
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"无法加载本地重排模型 bge-reranker-large_v1：{exc}"
                ) from exc
        return self._ce

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[str]:
        """对 (query, doc) 逐对打分，返回得分最高的前 top_n 个文档文本。"""
        if not docs:
            return []
        ce = self._model()
        try:
            # predict 接收 (query, doc) 对列表，返回每对的相关系分数
            scores = ce.predict([(query, doc) for doc in docs])
        except Exception as exc:
            raise RerankerUnavailableError(f"重排模型调用失败：{exc}") from exc
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]


def _rrf(ordered_id_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：多路排名的倒数排名融合，返回融合后的 id 有序列表。

    每个文档在其出现过的排名里，按 1/(k + 排名) 累计分数；两路都靠前的文档
    获得更高融合分。k=60 是 RRF 论文里常用的参数。
    """
    scores: dict[str, float] = {}
    for ids in ordered_id_lists:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class ChromaRAG:
    """ChromaDB 适配器；依赖未安装或服务失败时使用简单文本 fallback。

    职责：
    - 连接 ChromaDB（持久化到本地磁盘路径 settings.chroma_path）；
    - 维护两个 collection：公共知识库（knowledge）和用户长期记忆（long_memory）；
    - 提供文档入库（add_knowledge / ingest_text_file）和查询（search_knowledge）。
    """

    def __init__(self):
        # 三个属性初始为 None，表示"尚未成功连接"；
        # 一旦下面任一环节失败，后续所有方法都会走 fallback 分支
        self.client = None
        self.knowledge = None
        self.long_memory = None
        try:
            # 延迟导入 chromadb：只有真正用到时才加载，避免本模块在依赖缺失时无法导入
            import chromadb

            # PersistentClient：数据持久化到本地目录，重启后仍在
            self.client = chromadb.PersistentClient(path=settings.chroma_path)
            # 用同一个 Embedding 函数保证"写入"和"查询"走同一套向量化规则。
            # 优先用本地 BGE 模型（LocalEmbeddingFunction），模型不可用时自动退哈希。
            embedding_function = LocalEmbeddingFunction()
            # get_or_create_collection：存在则复用，不存在则新建，天然幂等
            # 公共知识库 collection：存放技术、面试和职业知识（系统公共知识）
            self.knowledge = self.client.get_or_create_collection(
                settings.rag_collection,
                metadata={"description": "CareerPilot 公共知识库"},
                embedding_function=embedding_function,
            )
            # 用户长期记忆 collection：存放用户自己的经历和历史，
            # 检索时必须在 metadata 里按 user_id 过滤，避免串用户
            # 说明：本模块只负责"建好"这个 collection 并沿用同一套 embedding，
            # 其真正的读写（save_long_memory / search_long_memory）由
            # careerpilot.memory.manager 通过原 LongMemory 完成
            self.long_memory = self.client.get_or_create_collection(
                settings.long_memory_collection,
                metadata={"description": "用户长期记忆"},
                embedding_function=embedding_function,
            )
        except Exception:
            # 只要 chromadb 未安装 / 启动失败 / 目录不可写等，就静默降级：
            # self.knowledge / self.long_memory 保持 None，后续方法走文本 fallback
            pass

    def add_knowledge(self, documents: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        """把文本写入公共知识库 collection。

        - 参数：documents 待入库的文本列表；metadatas 对应的元数据列表（可为空）。
        - 返回：本次实际处理的文档数量。
        """
        if not documents:
            # 空列表直接返回 0，避免无意义调用
            return 0
        if self.knowledge is None:
            # 降级模式（Chroma 不可用）：假装入库成功，直接返回条数，
            # 让上层逻辑照常继续，不抛异常
            return len(documents)
        # 用"序号 + 文本哈希"拼出稳定且不易冲突的文档 ID。
        # abs(hash(text)) 保证 ID 为非负数（Python 的 hash 对 str 每次运行可能不同，
        # 但同一次进程内一致；且 upsert 是覆盖写，重复文本会更新而非报错）
        ids = [f"knowledge-{i}-{abs(hash(text))}" for i, text in enumerate(documents)]
        # upsert：存在则更新、不存在则插入，比 add 更宽容
        self.knowledge.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(documents)

    def search_knowledge(self, query: str, limit: int = 5) -> list[str]:
        """混合检索：向量召回 + BM25 召回 + RRF 融合 + 本地重排。

        流程：
        1. 向量召回：用与入库相同的 embedding 把 query 转向量，在 Chroma 内做
           最近邻检索，取前 rag_recall_k。
        2. BM25 召回：用 rank_bm25+jieba 对全量语料做词法检索，取前 rag_recall_k。
        3. RRF 融合：两路结果按倒数排名融合、去重，取前 rag_fuse_k 进重排。
        4. 重排：用 bge-reranker-large_v1 对 (query, 候选) 逐对打分，取前 limit。
           —— 硬要求：重排模型/依赖缺失时抛 RerankerUnavailableError，不降级。

        - 参数：query 查询文本；limit 最多返回条数。
        - 返回：重排后最相关的前 limit 条知识库文本。
        """
        if self.knowledge is None:
            # 降级模式：返回空结果，不抛异常
            return []

        recall_k = settings.rag_recall_k   # 向量/BM25 各自召回多少
        fuse_k = settings.rag_fuse_k       # RRF 融合后取前几名进重排

        # 1) 向量召回：query 向量化由 Chroma 内部用同一套 embedding 完成
        vec = self.knowledge.query(query_texts=[query], n_results=recall_k)
        vec_ids = (vec.get("ids") or [[]])[0]

        # 2) 全量语料取出，重建 BM25 索引并做词法召回
        all_data = self.knowledge.get()
        all_ids = list(all_data["ids"])
        all_texts = list(all_data["documents"])
        if not all_ids:
            # 知识库为空：没有可检索内容（BM25 也对空语料建不了索引），直接返回空
            return []
        bm = BM25Search()
        bm.fit(all_texts, all_ids)
        bm_ids = bm.search(query, recall_k)

        # 3) RRF 融合两路排名的文档 id
        fused_ids = _rrf([vec_ids, bm_ids])[:fuse_k]
        text_by_id = dict(zip(all_ids, all_texts))
        candidates = [text_by_id[i] for i in fused_ids if i in text_by_id]
        if not candidates:
            return []

        # 4) 本地重排（硬要求）：模型不可用会抛 RerankerUnavailableError
        return Reranker().rerank(query, candidates, limit)

    def ingest_text_file(self, file_path: str, category: str = "general") -> int:
        """读取 txt/md 文件并按段落切分后入库。

        - 参数：file_path 本地文件路径；category 内容分类标签（默认 general）。
        - 返回：入库的段落数量。
        """
        # 以 UTF-8 读取整个文件内容
        text = Path(file_path).read_text(encoding="utf-8")
        # 按两个连续换行（空行）切分成段落，并去掉首尾空白、跳过空段。
        # 这样每个 chunk 是一段相对独立的知识，检索时更精准
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        # 为每个 chunk 打上来源文件、分类和序号，方便以后溯源/过滤
        metadata = [{"source": str(file_path), "category": category, "chunk_index": i} for i in range(len(chunks))]
        return self.add_knowledge(chunks, metadata)


# 模块级单例：整个应用共享同一个 ChromaRAG 实例。
# 上层代码直接 from careerpilot.rag import rag_store 使用即可。
rag_store = ChromaRAG()



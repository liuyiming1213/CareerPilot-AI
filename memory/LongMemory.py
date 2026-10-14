"""长期记忆：保留原类名，并接入 CareerPilot 的 Chroma 长期记忆集合。

这里特别使用“延迟初始化”：不要在导入文件时加载本地 embedding 模型。
原因是原项目的 ``database.vector_store`` 会加载一个可能很大的本地模型，
而 API 启动、单元测试甚至只是导入 ConversationManager 都不应该被它阻塞。
真正第一次写入/检索长期记忆时，才初始化向量集合。
"""

import os
import time

from langchain_core.documents import Document


def _legacy_collection():
    """按需加载旧项目的向量集合。

    只有显式设置 USE_LEGACY_VECTOR_STORE=1 时才走旧的本地 embedding。
    默认使用 CareerPilot 自己的 Chroma 集合，它同样是 ChromaDB，且不需要
    在启动时下载 embedding 模型，更适合新手本地运行和面试演示。
    """
    if os.getenv("USE_LEGACY_VECTOR_STORE", "0") != "1":
        return None
    try:
        from database.vector_store import get_memory_collection

        return get_memory_collection()
    except Exception:
        return None


def _careerpilot_collection():
    """按需取得 CareerPilot 独立的长期记忆 collection。"""
    try:
        from careerpilot.rag.store import rag_store

        return rag_store.long_memory
    except Exception:
        return None


class LongMemory:
    """跨会话的语义记忆。

    每条记录都必须保存 user_id metadata，查询时也必须带 user_id 过滤，
    这是防止不同用户互相读到记忆的关键。
    """

    _fallback: dict[str, list[str]] = {}

    def __init__(self):
        # 构造对象时不连接数据库，避免导入项目时卡住。
        self.collection = None

    def _ensure_collection(self):
        """第一次真正读写时才选择向量集合。"""
        if self.collection is not None:
            return self.collection
        self.collection = _legacy_collection() or _careerpilot_collection()
        return self.collection

    def add_memory(self, user_id: str, content: str, memory_type: str = "other") -> None:
        """写入一条长期事实。"""
        memory_id = f"{user_id}-{time.time_ns()}"
        metadata = {"user_id": str(user_id), "memory_type": memory_type}
        collection = self._ensure_collection()
        if collection is not None and hasattr(collection, "add_documents"):
            self.collection.add_documents(
                [Document(page_content=content, metadata=metadata)],
                ids=[memory_id],
            )
        elif collection is not None:
            # CareerPilot 使用 Chroma 原生 collection，直接保存文本和 metadata。
            collection.add(ids=[memory_id], documents=[content], metadatas=[metadata])
        else:
            self._fallback.setdefault(str(user_id), []).append(content)

    def load_memory(self, user_id: str, question: str = "", limit: int = 5) -> list[str]:
        """按问题检索指定用户的长期记忆。"""
        if not question:
            return []
        collection = self._ensure_collection()
        if collection is not None and hasattr(collection, "similarity_search"):
            documents = collection.similarity_search(question, k=limit, filter={"user_id": str(user_id)})
            return [doc.page_content for doc in documents]
        if collection is not None:
            result = collection.query(
                query_texts=[question],
                n_results=limit,
                where={"user_id": str(user_id)},
            )
            return (result.get("documents") or [[]])[0]
        return self._fallback.get(str(user_id), [])[:limit]

    def load(self, user_id: str, question: str = "", limit: int = 5) -> list[str]:
        """兼容 ConversationManager 原来的 load 调用。"""
        return self.load_memory(user_id, question, limit)

    def recent(self, user_id: str, limit: int = 5) -> list[str]:
        """按 user_id 直接取最近几条长期记忆（不需要语义 query）。

        用于"记忆快照"这类场景：在没有查询词时也能展示该用户已沉淀的长期记忆。
        """
        collection = self._ensure_collection()
        if collection is not None:
            try:
                # Chroma 原生的按 metadata 过滤 + 取最新记录
                result = collection.get(where={"user_id": str(user_id)}, limit=limit)
                return (result.get("documents") or [])[:limit]
            except Exception:
                pass
        # 降级：进程内 fallback 的该用户记忆
        return self._fallback.get(str(user_id), [])[:limit]

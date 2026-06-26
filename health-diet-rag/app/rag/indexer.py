"""
ChromaDB 索引器（直接使用 chromadb 官方客户端，不经过 langchain-chroma）。

为什么绕过 langchain-chroma？
----------------------------
langchain-chroma 0.1.x 与 chromadb 1.x + numpy 2.x 不兼容。
直接用 chromadb 官方客户端反而更简单：
- 少一层抽象，更易理解
- 摆脱依赖版本冲突
- 后续切到其他向量库（FAISS / Qdrant）时迁移成本一样

核心 API
--------
- chromadb.PersistentClient(path=...) → 创建持久化客户端
- client.get_or_create_collection(name, metadata) → 获取/创建集合
- collection.add(documents, embeddings, metadatas, ids) → 写入数据
- collection.count() → 文档数
- collection.query(query_embeddings, n_results, where) → 检索
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.core.config import settings
from app.rag.embedder import get_embeddings
from app.rag.loaders import load_all


# ---------------------------------------------------------------------------
# 索引器主类
# ---------------------------------------------------------------------------

class KnowledgeIndexer:
    """把 Document 写入 ChromaDB，并管理 collection。

    用法：
        indexer = KnowledgeIndexer()
        result = indexer.index_all()
        # result = {"ingredients": 20, "recipes": 10, "guides": 7, "total": 37}
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection or settings.CHROMA_COLLECTION
        self.data_dir = data_dir or "data"
        self._embeddings = get_embeddings()

    # ----- 核心：建索引 -----

    def index_all(self) -> dict[str, int]:
        """加载全部数据并写入 ChromaDB。

        幂等：先清空再写入，保证多次执行结果一致。
        """
        docs = load_all(self.data_dir)
        counts = self._count_by_source(docs)

        # 幂等：建索引前清空旧数据
        self.clear()

        # 批量写入
        if docs:
            self._add_documents(docs)
        return counts

    def index_documents(self, docs: list[Any]) -> int:
        """增量写入 Document 列表（不重建索引）。

        docs 可以是 langchain Document 或 dict。
        """
        if not docs:
            return 0
        self._add_documents(docs)
        return len(docs)

    # ----- 查询/管理 -----

    def count(self) -> int:
        """返回当前 collection 里的文档总数。"""
        try:
            col = self._get_collection()
            return col.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """清空 collection（删除并重建）。"""
        try:
            client = self._get_client()
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass  # 不存在就跳过
            # 重建（用 cosine 相似度，比默认 L2 更适合文本）
            client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"[indexer] 清空 collection 失败：{e}")

    def is_indexed(self) -> bool:
        """判断是否已建过索引。"""
        return self.count() > 0

    # ----- 内部：底层操作 -----

    def _get_client(self) -> chromadb.api.ClientAPI:
        """获取 chromadb 持久化客户端（每次新建，chromadb 自己管理连接池）。"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=self.persist_dir)

    def _get_collection(self) -> chromadb.Collection:
        """获取已存在的 collection（不存在则创建）。"""
        client = self._get_client()
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _add_documents(self, docs: list[Any]) -> None:
        """把 LangChain Document 列表写入 Chroma。

        步骤：
        1. 从 Document.page_content 提取文本
        2. 用 embedding 客户端批量向量化
        3. 生成唯一 ID（source-index 形式）
        4. 调 chroma.add() 写入
        """
        # 兼容 LangChain Document 和 dict 两种输入
        texts = []
        metadatas = []
        sources = []
        for d in docs:
            if hasattr(d, "page_content"):  # LangChain Document
                text = d.page_content
                meta = d.metadata or {}
            else:  # dict
                text = d.get("page_content", "")
                meta = d.get("metadata", {})
            texts.append(text)
            # chromadb 要求 metadata 值是基础类型（str/int/float/bool），不能是 list
            cleaned = {k: _coerce_meta_value(v) for k, v in meta.items()}
            metadatas.append(cleaned)
            sources.append(meta.get("source", "unknown"))

        # 批量向量化（fastembed 自带 batching，比一条条快很多）
        vectors = self._embeddings.embed_documents(texts)

        # 生成稳定 ID：source-index，避免重复
        ids = [f"{src}-{i}" for i, src in enumerate(sources)]

        col = self._get_collection()
        col.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )

    @staticmethod
    def _count_by_source(docs: list[Any]) -> dict[str, int]:
        from collections import Counter
        sources = []
        for d in docs:
            if hasattr(d, "metadata"):
                sources.append(d.metadata.get("source", "unknown"))
            else:
                sources.append(d.get("metadata", {}).get("source", "unknown"))
        counts = Counter(sources)
        result = dict(counts)
        result["total"] = len(docs)
        return result


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _coerce_meta_value(v: Any) -> Any:
    """把 metadata 值强制转成 chromadb 支持的基础类型。"""
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_indexer: KnowledgeIndexer | None = None


def get_indexer() -> KnowledgeIndexer:
    """获取全局 indexer 单例。"""
    global _indexer
    if _indexer is None:
        _indexer = KnowledgeIndexer()
    return _indexer

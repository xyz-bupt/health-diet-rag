"""
向量检索器。

封装 ChromaDB 的 query 接口，提供：
- 相似度检索（默认）
- 元数据过滤（只搜特定来源）
- top-k 控制
- 距离转相似度

为什么自己包装一层？
--------------------
chromadb.query() 返回的是 dict（格式复杂、字段多）。
本检索器把它转成统一的 SearchResult 列表，方便：
- API 层直接序列化返回
- 测试断言
- 后续 LangGraph Agent 直接使用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.rag.embedder import get_embeddings
from app.rag.indexer import get_indexer


# ---------------------------------------------------------------------------
# 检索结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """单条检索结果。"""

    document: str                 # 文档原文
    metadata: dict[str, Any]      # 元数据
    score: float                  # 相似度分数（0-1，越大越相似）
    source: str = field(default="unknown")  # 来源类型

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """向量检索器。

    用法：
        retriever = KnowledgeRetriever()
        results = retriever.search("低脂高蛋白的早餐")
    """

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.RAG_TOP_K
        self._embeddings = get_embeddings()
        self._indexer = get_indexer()

    # ----- 核心检索 -----

    def search(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """语义检索：query → top-k 相关文档。

        参数：
            query: 查询文本（自然语言）
            k: 召回数量（默认走配置 RAG_TOP_K）
            source_filter: 仅返回该来源的文档（"ingredients"/"recipes"/"guides"）

        返回：
            按相似度降序的 SearchResult 列表
        """
        if not self._indexer.is_indexed():
            return []  # 未建索引时返回空，让上层决定怎么提示

        k = k or self.top_k
        query_vec = self._embeddings.embed_query(query)

        # 构造 where 过滤条件（chromadb 格式）
        where = {"source": source_filter} if source_filter else None

        col = self._indexer._get_collection()
        raw = col.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_raw_results(raw)

    # ----- 多关键词召回 + 合并（简易版 MMR 替代） -----

    def search_multiline(
        self,
        queries: list[str],
        k_per_query: int = 3,
    ) -> list[SearchResult]:
        """对多个查询分别召回，合并去重，按最高分排序。

        场景：用户输入是复合需求"低脂高蛋白早餐"
        可以拆成 ["低脂早餐", "高蛋白早餐"] 两次召回，覆盖更广。
        """
        seen: dict[str, SearchResult] = {}
        for q in queries:
            results = self.search(q, k=k_per_query)
            for r in results:
                key = r.document[:64]  # 用前 64 字符做去重 key
                if key not in seen or r.score > seen[key].score:
                    seen[key] = r
        # 按分数降序
        return sorted(seen.values(), key=lambda r: -r.score)

    # ----- 内部：解析 chromadb 原始返回 -----

    def _parse_raw_results(self, raw: dict[str, Any]) -> list[SearchResult]:
        """chromadb 返回的是 list of lists（因为支持多 query），这里取第 0 个。"""
        if not raw.get("documents") or not raw["documents"][0]:
            return []

        docs = raw["documents"][0]
        metas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(docs)
        dists = raw["distances"][0] if raw.get("distances") else [0.0] * len(docs)

        results = []
        for doc, meta, dist in zip(docs, metas, dists):
            # chromadb cosine 模式下，distance 是 1 - cosine_similarity
            # 所以相似度 = 1 - distance
            score = max(0.0, 1.0 - dist)
            results.append(SearchResult(
                document=doc,
                metadata=meta or {},
                score=round(score, 4),
                source=(meta or {}).get("source", "unknown"),
            ))
        return results


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """获取全局 retriever 单例。"""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever

"""
Embedding 工厂。

为什么不用 sentence-transformers（标准 BGE 推理库）？
-----------------------------------------------
sentence-transformers 依赖 PyTorch，但 PyTorch 在 Python 3.13 + macOS 上
还没有官方 wheel（截至 2026-06）。我们改用 `fastembed`：
- 基于 ONNX Runtime（不依赖 torch）
- 预打包主流 BGE / MiniLM 模型
- 包大小仅 ~50MB（vs sentence-transformers + torch 几个 GB）
- 在 CPU 上推理速度反而更快（ONNX 优化）

为什么用单例（lru_cache）？
--------------------------
embedding 模型加载需要解析 ONNX 文件、分配内存，单次约 2-5 秒。
频繁重新加载会让接口极慢。lru_cache 保证整个进程只加载一次。

切换 provider 怎么办？
-----------------------
本工厂预留 EMBEDDING_PROVIDER 配置项，目前只实现 fastembed，
后续可扩展 OpenAI / 智谱 embedding（如 DeepSeek 提供了 embedding）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Protocol

from app.core.config import settings


# ---------------------------------------------------------------------------
# Embedding 协议：定义"什么是 embedding 客户端"
# ---------------------------------------------------------------------------

class EmbeddingsLike(Protocol):
    """所有 embedding 客户端都该实现的接口。

    与 LangChain 的 Embeddings 抽象类对齐：
    - embed_documents(texts) → 文档向量化（用于建索引）
    - embed_query(text)      → 查询向量化（用于检索）
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Mock Embedding：模型未加载时的占位
# ---------------------------------------------------------------------------

class MockEmbeddings:
    """占位 embedding：返回固定维度随机向量。

    用途：CI 环境、首次跑测试、模型下载失败时让流程继续。
    语义检索会失效，但接口结构正确。

    注意：MockEmbeddings 之间生成的向量是确定的（基于 hash），
    保证同一查询每次返回同样的"假向量"，让测试可重现。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model = "mock-embeddings"

    def _fake_vec(self, text: str) -> list[float]:
        """基于文本 hash 生成确定性的伪向量。"""
        import hashlib

        h = hashlib.md5(text.encode("utf-8")).digest()
        # 用 hash 重复填充到目标维度
        vec = []
        for i in range(self.dim):
            vec.append((h[i % 16] / 255.0) * 2 - 1)  # 归一化到 [-1, 1]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._fake_vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._fake_vec(text)


# ---------------------------------------------------------------------------
# FastEmbed 包装：让 fastembed 适配 LangChain Embeddings 接口
# ---------------------------------------------------------------------------

class FastEmbedWrapper:
    """把 fastembed 的 TextEmbedding 包装成 LangChain Embeddings 兼容接口。

    为什么自己包一层而不直接用 langchain_community 的 FastEmbedEmbeddings？
    - 直接用 TextEmbedding 更少依赖、更少抽象层（学习项目优先）
    - 自己包可以清楚看到"embedding 客户端该满足什么接口"
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        # 提示：第一次调用会从网络下载模型（约 90MB），后续从本地缓存加载
        self._model = TextEmbedding(model_name=model_name)
        self.model = model_name
        # 取一条样本向量拿维度（ChromaDB 建表需要）
        sample = next(self._model.embed(["dim probe"]))
        self.dim = len(sample)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed 返回 numpy.float32 数组，但 chromadb 1.x 要求原生 float
        # 这里显式转 float，避免 ValueError
        return [
            [float(x) for x in vec]
            for vec in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(self._model.embed([text]))]


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingsLike:
    """获取全局 embedding 单例。

    第一次调用会触发模型下载（如使用 fastembed）。
    后续调用直接返回缓存的单例。
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    model = settings.EMBEDDING_MODEL

    if provider == "fastembed":
        try:
            return FastEmbedWrapper(model)
        except Exception as e:
            # 模型下载失败、网络问题、磁盘满等 → 降级
            print(f"[embedder] FastEmbed 加载失败，降级到 MockEmbeddings：{e}")
            return MockEmbeddings(dim=512)
    elif provider == "mock":
        return MockEmbeddings(dim=512)
    else:
        # 未支持的 provider 先降级，未来扩展再加分支
        print(f"[embedder] 未支持的 provider={provider}，降级到 MockEmbeddings")
        return MockEmbeddings(dim=512)


def is_real_embeddings() -> bool:
    """判断当前是否启用了真实 embedding（而非 Mock）。

    用于测试打标记、API 响应里告知调用方。
    """
    emb = get_embeddings()
    return not isinstance(emb, MockEmbeddings)

"""
RAG 模块测试。

分层策略：
- loaders 测试：纯函数，无 LLM/embedding 依赖，超快
- indexer 测试：用临时目录，不污染真实数据
- retriever 测试：依赖真实 embedding，可能下载模型
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from app.rag.embedder import (
    FastEmbedWrapper,
    MockEmbeddings,
    get_embeddings,
    is_real_embeddings,
)
from app.rag.indexer import KnowledgeIndexer
from app.rag.loaders import load_all, load_guides, load_ingredients, load_recipes
from app.rag.retriever import KnowledgeRetriever


# ---------------------------------------------------------------------------
# 数据文件存在性
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def test_data_files_exist():
    assert (DATA_DIR / "ingredients.json").exists()
    assert (DATA_DIR / "recipes.json").exists()
    assert (DATA_DIR / "nutrition_guides.md").exists()


# ---------------------------------------------------------------------------
# Loaders：纯函数测试
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_load_ingredients_count(self):
        docs = load_ingredients(DATA_DIR / "ingredients.json")
        assert len(docs) == 20

    def test_load_ingredients_has_nutrition_in_text(self):
        docs = load_ingredients(DATA_DIR / "ingredients.json")
        chicken = docs[0]
        assert "鸡胸肉" in chicken.page_content
        assert "蛋白质" in chicken.page_content
        assert chicken.metadata["source"] == "ingredients"
        assert chicken.metadata["name"] == "鸡胸肉"
        assert chicken.metadata["calories"] == 133

    def test_load_recipes_count(self):
        docs = load_recipes(DATA_DIR / "recipes.json")
        assert len(docs) == 10

    def test_load_recipes_has_steps(self):
        docs = load_recipes(DATA_DIR / "recipes.json")
        recipe = docs[0]
        assert "【菜谱】" in recipe.page_content
        assert "步骤" in recipe.page_content
        assert recipe.metadata["source"] == "recipes"
        assert "tags" in recipe.metadata

    def test_load_guides_count(self):
        docs = load_guides(DATA_DIR / "nutrition_guides.md")
        assert len(docs) >= 3  # 至少 3 个章节切片

    def test_load_guides_has_title(self):
        docs = load_guides(DATA_DIR / "nutrition_guides.md")
        # 至少有一个切片带 title
        titles = [d.metadata.get("title", "") for d in docs]
        assert any(t for t in titles)

    def test_load_all_total_count(self):
        docs = load_all(DATA_DIR)
        # 20 食材 + 10 菜谱 + N 指南切片
        assert len(docs) >= 30

    def test_load_all_sources_distinct(self):
        docs = load_all(DATA_DIR)
        sources = {d.metadata.get("source") for d in docs}
        assert "ingredients" in sources
        assert "recipes" in sources
        assert "guides" in sources


# ---------------------------------------------------------------------------
# Embedder：工厂 + Mock
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_get_embeddings_returns_singleton(self):
        a = get_embeddings()
        b = get_embeddings()
        assert a is b

    def test_mock_embeddings_dim(self):
        mock = MockEmbeddings(dim=64)
        v = mock.embed_query("test")
        assert len(v) == 64

    def test_mock_embeddings_deterministic(self):
        """同一文本应得到相同向量（让测试可重现）。"""
        mock = MockEmbeddings()
        v1 = mock.embed_query("鸡胸肉")
        v2 = mock.embed_query("鸡胸肉")
        assert v1 == v2

    def test_mock_embeddings_different_texts_differ(self):
        mock = MockEmbeddings()
        v1 = mock.embed_query("鸡胸肉")
        v2 = mock.embed_query("三文鱼")
        assert v1 != v2


# ---------------------------------------------------------------------------
# Indexer：用临时目录隔离
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_indexer():
    """每个测试用独立临时目录的 indexer，互不干扰。"""
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    indexer = KnowledgeIndexer(persist_dir=tmpdir, collection="test")
    yield indexer
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestIndexer:
    def test_empty_indexer_count_zero(self, tmp_indexer):
        assert tmp_indexer.count() == 0
        assert tmp_indexer.is_indexed() is False

    def test_index_all_returns_counts(self, tmp_indexer):
        result = tmp_indexer.index_all()
        assert result["total"] >= 30
        assert result["ingredients"] == 20
        assert result["recipes"] == 10
        assert result["guides"] >= 1

    def test_index_all_idempotent(self, tmp_indexer):
        """多次建索引结果应一致（幂等）。"""
        r1 = tmp_indexer.index_all()
        r2 = tmp_indexer.index_all()
        assert r1 == r2
        # 文档数不会重复累加
        assert tmp_indexer.count() == r2["total"]

    def test_is_indexed_after_build(self, tmp_indexer):
        assert tmp_indexer.is_indexed() is False
        tmp_indexer.index_all()
        assert tmp_indexer.is_indexed() is True

    def test_clear_resets_count(self, tmp_indexer):
        tmp_indexer.index_all()
        assert tmp_indexer.count() > 0
        tmp_indexer.clear()
        assert tmp_indexer.count() == 0


# ---------------------------------------------------------------------------
# Retriever：真实 embedding 检索（最慢的测试，但最重要）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_index():
    """模块级共享索引，避免每个测试都重新建索引（embedding 是瓶颈）。"""
    tmpdir = tempfile.mkdtemp(prefix="chroma_retriever_")
    indexer = KnowledgeIndexer(persist_dir=tmpdir, collection="retriever_test")
    indexer.index_all()
    yield indexer
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def shared_retriever(shared_index):
    """共享 retriever，复用上面的索引。"""
    return KnowledgeRetriever()


class TestRetriever:
    def test_unindexed_returns_empty(self):
        """没建索引时 search 应返回空列表。"""
        tmpdir = tempfile.mkdtemp(prefix="chroma_empty_")
        try:
            indexer = KnowledgeIndexer(persist_dir=tmpdir, collection="empty_test")
            retriever = KnowledgeRetriever()
            retriever._indexer = indexer  # 注入空 indexer
            results = retriever.search("anything")
            assert results == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_search_returns_results(self, shared_retriever):
        results = shared_retriever.search("低脂高蛋白的肉", k=3)
        assert len(results) <= 3
        assert len(results) >= 1
        # 每条都有必要字段
        for r in results:
            assert r.document
            assert isinstance(r.score, float)
            assert 0 <= r.score <= 1
            assert r.source in {"ingredients", "recipes", "guides"}

    def test_search_finds_chicken_for_low_fat_protein(self, shared_retriever):
        """语义检索：'低脂高蛋白的肉' 应该召回鸡胸肉。"""
        results = shared_retriever.search("低脂高蛋白的肉", k=3)
        top_doc = results[0].document
        assert "鸡胸肉" in top_doc or "虾仁" in top_doc or "瘦牛" in top_doc

    def test_search_finds_salmon_for_omega3(self, shared_retriever):
        """'Omega-3 鱼类' 应召回三文鱼。"""
        results = shared_retriever.search("富含 Omega-3 的鱼", k=3)
        top_doc = results[0].document
        assert "三文鱼" in top_doc

    def test_source_filter_works(self, shared_retriever):
        """source_filter 应只返回指定来源。"""
        results = shared_retriever.search("食物", k=10, source_filter="ingredients")
        assert all(r.source == "ingredients" for r in results)

    def test_k_parameter_limits_results(self, shared_retriever):
        r1 = shared_retriever.search("主食", k=2)
        r2 = shared_retriever.search("主食", k=5)
        assert len(r1) <= 2
        assert len(r2) <= 5
        assert len(r2) >= len(r1)

    def test_scores_are_descending(self, shared_retriever):
        """检索结果应按相似度降序。"""
        results = shared_retriever.search("健康饮食", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# API 接口测试
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestRagAPI:
    def test_index_status_endpoint(self):
        r = client.get("/api/v1/index/status")
        assert r.status_code == 200
        data = r.json()
        assert "indexed" in data
        assert "count" in data
        assert "embeddings_available" in data

    def test_search_endpoint_works(self):
        """假设已建索引（前面测试建过），搜索应返回结果。"""
        # 先确保有索引
        client.post("/api/v1/index")
        r = client.get("/api/v1/foods/search", params={"q": "低脂高蛋白", "k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "低脂高蛋白"
        assert len(data["results"]) <= 3

    def test_search_endpoint_rejects_empty_query(self):
        r = client.get("/api/v1/foods/search", params={"q": ""})
        assert r.status_code == 422  # min_length=1

    def test_search_endpoint_rejects_invalid_k(self):
        r = client.get("/api/v1/foods/search", params={"q": "test", "k": 0})
        assert r.status_code == 422  # ge=1

    def test_search_with_source_filter(self):
        r = client.get("/api/v1/foods/search",
                       params={"q": "食物", "k": 5, "source": "recipes"})
        assert r.status_code == 200
        data = r.json()
        for item in data["results"]:
            assert item["source"] == "recipes"

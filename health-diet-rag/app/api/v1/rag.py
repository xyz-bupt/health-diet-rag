"""
RAG 相关路由。

提供：
- POST /api/v1/index        ：建立向量库索引
- GET  /api/v1/foods/search ：语义检索食材/菜谱/指南
- GET  /api/v1/index/status ：查看索引状态
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.rag.embedder import is_real_embeddings
from app.rag.indexer import get_indexer
from app.rag.retriever import get_retriever

router = APIRouter()


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------

class IndexResponse(BaseModel):
    """建索引的响应。"""
    status: str
    counts: dict[str, int]
    embeddings_used: str  # fastembed / mock


class SearchResultItem(BaseModel):
    document: str
    metadata: dict
    score: float
    source: str


class SearchResponse(BaseModel):
    query: str
    k: int
    source_filter: str | None
    embeddings_used: str
    results: list[SearchResultItem]


class IndexStatus(BaseModel):
    indexed: bool
    count: int
    embeddings_available: bool


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.post("/index", response_model=IndexResponse, summary="建立向量库索引")
async def build_index() -> IndexResponse:
    """加载 data/ 目录下的全部数据，建/重建向量库索引。

    - 幂等：多次调用结果一致（先清空再写入）
    - 第一次调用会下载 embedding 模型（约 90MB）
    - 已建索引后，应用重启无需再调用
    """
    indexer = get_indexer()
    try:
        counts = indexer.index_all()
        return IndexResponse(
            status="ok",
            counts=counts,
            embeddings_used=type(get_retriever()._embeddings).__name__,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"建索引失败：{e}")


@router.get("/index/status", response_model=IndexStatus, summary="查看索引状态")
async def index_status() -> IndexStatus:
    """查询当前向量库是否已建索引、有多少条文档。"""
    indexer = get_indexer()
    return IndexStatus(
        indexed=indexer.is_indexed(),
        count=indexer.count(),
        embeddings_available=is_real_embeddings(),
    )


@router.get("/foods/search", response_model=SearchResponse, summary="语义检索")
async def search_foods(
    q: str = Query(..., min_length=1, description="自然语言查询，如'低脂高蛋白早餐'"),
    k: int = Query(default=4, ge=1, le=20, description="召回数量"),
    source: str | None = Query(
        default=None,
        description="过滤来源：ingredients / recipes / guides",
    ),
) -> SearchResponse:
    """对食材/菜谱/指南做语义检索。

    示例：
        GET /api/v1/foods/search?q=低脂高蛋白的肉&k=3
        GET /api/v1/foods/search?q=减脂早餐&source=recipes
    """
    retriever = get_retriever()
    if not retriever._indexer.is_indexed():
        raise HTTPException(
            status_code=503,
            detail="向量库未建索引，请先 POST /api/v1/index",
        )
    results = retriever.search(q, k=k, source_filter=source)
    return SearchResponse(
        query=q,
        k=k,
        source_filter=source,
        embeddings_used=type(retriever._embeddings).__name__,
        results=[SearchResultItem(**r.to_dict()) for r in results],
    )

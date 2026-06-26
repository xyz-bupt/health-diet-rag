"""
健康检查路由。

提供 /health 和 / 两个端点：
- GET /：根路径，返回项目简介（或前端 HTML）
- GET /health：健康检查，用于 Docker / 监控
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/")
async def root():
    """根路径。

    如果存在 static/index.html，返回前端页面；
    否则返回 JSON 项目简介。
    """
    static_index = Path(__file__).parent.parent.parent.parent / "static" / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    return {
        "message": f"欢迎使用 {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@router.get("/health")
async def health_check():
    """健康检查端点。

    用途：
    - Docker 的 healthcheck 可以调用它判断容器是否健康
    - Kubernetes 的 liveness/readiness probe
    - 负载均衡器判断服务是否可用
    """
    return {"status": "healthy", "env": settings.ENV}

"""
FastAPI 应用入口。

启动方式：
    uvicorn app.main:app --reload

访问：
    http://localhost:8000/         根路径（前端首页）
    http://localhost:8000/health   健康检查
    http://localhost:8000/docs     Swagger 文档（自动生成）
    http://localhost:8000/redoc    ReDoc 文档（另一种风格）
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1 import assess, diet_plan, health, rag
from app.core.config import settings
from app.core.exceptions import register_exception_handlers


# lifespan：FastAPI 推荐的应用生命周期管理方式
# 旧的 @app.on_event("startup") 已经废弃，统一用 lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭时的钩子。

    yield 之前：启动时执行（加载数据库、建索引、预热等）
    yield 之后：关闭时执行（清理连接、保存状态）
    """
    print(f"🚀 {settings.APP_NAME} 启动中...")
    print(f"   环境: {settings.ENV}")
    print(f"   文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   前端: http://{settings.HOST}:{settings.PORT}/")
    # 后续阶段会在这里：初始化向量库、预热 LLM 客户端等
    yield
    print(f"👋 {settings.APP_NAME} 已关闭")


def create_app() -> FastAPI:
    """应用工厂函数。

    用工厂函数（而不是直接 app = FastAPI()）的好处：
    1. 测试时可以创建多个独立实例
    2. 不同环境用不同配置
    3. 方便添加中间件、路由、事件钩子
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="集成 RAG 检索增强与多 Agent 协作的个性化健康饮食方案生成服务",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # 1. CORS 中间件：允许前端跨域访问
    # 开发环境允许所有来源；生产环境应限定具体域名
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # 生产环境改成 ["http://your-frontend.com"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. 注册路由
    app.include_router(health.router, tags=["基础"])
    app.include_router(assess.router, prefix="/api/v1", tags=["健康评估"])
    app.include_router(rag.router, prefix="/api/v1", tags=["RAG 检索"])
    app.include_router(diet_plan.router, prefix="/api/v1", tags=["完整方案"])

    # 3. 注册统一异常处理器
    register_exception_handlers(app)

    # 4. 挂载静态资源（前端）
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# 全局 app 实例（uvicorn 通过这个字符串找到应用：app.main:app）
app = create_app()

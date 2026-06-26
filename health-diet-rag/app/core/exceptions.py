"""
统一异常体系。

为什么需要自定义异常？
----------------------
1. 业务错误 vs 系统错误分离：
   - "向量库未建索引" 是业务错误（用户该看到）
   - "KeyError: 'foo'" 是系统错误（用户不该看到细节）

2. 统一响应格式：所有错误返回相同结构，前端易处理
3. 自动映射到合适的 HTTP 状态码

错误响应格式
------------
{
  "error": {
    "code": "INDEX_NOT_BUILT",            ← 业务错误码（大写下划线）
    "message": "向量库未建索引...",       ← 给用户看的人话消息
    "details": {...}                      ← 可选，调试信息
  }
}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 错误响应模型
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str = Field(..., description="业务错误码")
    message: str = Field(..., description="人话错误消息")
    details: dict | None = Field(default=None, description="可选调试信息")


class ErrorResponse(BaseModel):
    """统一的错误响应结构。"""
    error: ErrorDetail


# ---------------------------------------------------------------------------
# 业务异常基类
# ---------------------------------------------------------------------------

class AppException(Exception):
    """所有业务异常的基类。

    子类必须定义：
    - code: 错误码（如 INDEX_NOT_BUILT）
    - http_status: 对应的 HTTP 状态码
    - default_message: 默认消息
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    default_message: str = "服务内部错误"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.default_message
        self.details = details
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# 具体业务异常
# ---------------------------------------------------------------------------

class IndexNotBuiltError(AppException):
    code = "INDEX_NOT_BUILT"
    http_status = 503  # Service Unavailable
    default_message = "向量库未建索引，请先 POST /api/v1/index"


class WorkflowFailedError(AppException):
    code = "WORKFLOW_FAILED"
    http_status = 500
    default_message = "工作流执行失败"


class LLMUnavailableError(AppException):
    code = "LLM_UNAVAILABLE"
    http_status = 503
    default_message = "LLM 服务不可用"


class ValidationError(AppException):
    code = "VALIDATION_ERROR"
    http_status = 422
    default_message = "输入校验失败"


# ---------------------------------------------------------------------------
# 异常处理器（注册到 FastAPI）
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """把所有异常处理器注册到 FastAPI 应用。

    在 main.py 的 create_app 里调用一次即可。
    """

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException):
        """处理所有业务异常。"""
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                )
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        request: Request, exc: RequestValidationError
    ):
        """FastAPI 自带的请求校验异常（如字段缺失、类型错），统一格式。"""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="请求参数校验失败",
                    details={"errors": exc.errors()},
                )
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception):
        """兜底：所有未捕获的异常统一返回 500，不暴露 stack trace 给用户。"""
        # 实际生产环境这里要记日志 + 告警
        print(f"[unhandled] {request.method} {request.url}: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="服务内部错误，请稍后重试",
                )
            ).model_dump(),
        )

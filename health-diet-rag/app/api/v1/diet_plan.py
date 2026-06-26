"""
完整饮食方案路由（Stage 5 升级版）。

变更：
1. 路由改 async，调 arun_diet_plan / astream_diet_plan
2. 加 TTL 缓存（5 分钟内相同 profile 直接返回）
3. 用自定义异常替代 HTTPException，统一错误格式
4. 加 /api/v1/cache/stats 接口查看缓存状态
"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.cache import get_cache, profile_cache_key
from app.core.exceptions import IndexNotBuiltError, WorkflowFailedError
from app.graph.workflow import arun_diet_plan, astream_diet_plan
from app.models.diet import DietPlan
from app.models.health import HealthProfile
from app.rag.indexer import get_indexer

router = APIRouter()


# ---------------------------------------------------------------------------
# 辅助：递归把 Pydantic 模型 / dataclass / 嵌套结构转成可 JSON 序列化的纯结构
# ---------------------------------------------------------------------------

def to_serializable(obj):
    """递归转 Pydantic 模型/dict/list 为纯 JSON 兼容结构。

    用途：LangGraph 的 state 更新可能是 {"health": HealthAssessment(...)}，
    直接 json.dumps 会失败。本函数把所有嵌套的 Pydantic 模型都用 model_dump()
    展开成 dict。
    """
    # Pydantic v2 模型
    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    # dict：递归每个 value
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    # list / tuple：递归每个元素
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    # 基础类型（str/int/float/bool/None）原样返回
    return obj


# ---------------------------------------------------------------------------
# 同步生成完整方案（带缓存）
# ---------------------------------------------------------------------------

@router.post("/diet-plan", response_model=DietPlan, summary="生成完整饮食方案")
async def diet_plan(profile: HealthProfile) -> DietPlan:
    """提交健康画像，多 Agent 协作生成完整饮食方案。

    内部流程：
    1. 检查缓存（命中直接返回，省 1-3 秒）
    2. 检查 RAG 索引状态（未建则提示）
    3. 异步执行 5-Agent 工作流
    4. 写入缓存
    5. 返回最终方案

    响应字段说明：
    - health: 健康评估结果
    - nutrition: 营养规划（三餐热量/饮水/时机）
    - recipe: 三餐菜谱（含 RAG 召回的食材）
    - exercise: 每周运动计划
    - summary: 自然语言方案摘要
    - key_actions: 最优先的 3-5 条可执行行动
    - llm_used: 是否调用了真实 LLM（false 时为规则化兜底）
    """
    cache = get_cache()
    key = profile_cache_key(profile)

    # 1. 查缓存
    cached = cache.get(key)
    if cached is not None:
        return cached

    # 2. 检查 RAG 索引（菜谱 Agent 依赖它）
    if not get_indexer().is_indexed():
        raise IndexNotBuiltError()

    # 3. 异步执行工作流
    try:
        state = await arun_diet_plan(profile)
    except Exception as e:
        raise WorkflowFailedError(
            message=f"工作流执行失败：{type(e).__name__}: {e}",
            details={"exception_type": type(e).__name__},
        )

    if "final_plan" not in state:
        raise WorkflowFailedError(
            message="工作流未产出最终方案",
            details={"errors": state.get("errors", [])},
        )

    final_plan = state["final_plan"]
    # 4. 写缓存
    cache.set(key, final_plan)
    return final_plan


# ---------------------------------------------------------------------------
# SSE 流式版本（不缓存，每次都重算）
# ---------------------------------------------------------------------------

@router.post("/diet-plan/stream", summary="流式生成完整方案（SSE）")
async def diet_plan_stream(profile: HealthProfile) -> StreamingResponse:
    """SSE 流式版本：每个 Agent 完成时推送一个事件。

    每行格式：
        data: {"node": "health_node", "result": {...}}

    最后发送 `data: [DONE]` 表示完成。

    curl 示例：
        curl -N -X POST http://localhost:8000/api/v1/diet-plan/stream \\
             -H "Content-Type: application/json" \\
             -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male"}'
    """
    if not get_indexer().is_indexed():
        raise IndexNotBuiltError()

    async def event_generator():
        try:
            async for event in astream_diet_plan(profile):
                for node_name, output in event.items():
                    # LangGraph 返回的 output 可能是：
                    #   - dict（含 Pydantic 模型值），如 {"health": HealthAssessment(...)}
                    #   - 直接是 Pydantic 模型（错误时）
                    #   - 字符串
                    # 需要递归转成可 JSON 序列化的纯 dict/list/标量
                    payload = to_serializable(output)
                    yield (
                        "data: "
                        + json.dumps(
                            {"node": node_name, "result": payload},
                            ensure_ascii=False,
                            default=str,  # 兜底：未知类型转 str
                        )
                        + "\n\n"
                    )
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "code": "STREAM_FAILED",
                            "message": str(e),
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 不缓冲
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# 缓存状态查询（调试用）
# ---------------------------------------------------------------------------

@router.get("/cache/stats", summary="查看缓存统计")
async def cache_stats() -> dict:
    """返回当前缓存的命中率、大小等统计。"""
    return get_cache().stats()


@router.post("/cache/clear", summary="清空缓存")
async def clear_cache() -> dict:
    """清空缓存（管理操作）。"""
    get_cache().clear()
    return {"status": "cleared"}

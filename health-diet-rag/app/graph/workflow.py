"""
LangGraph 工作流的构建与执行。

核心 API
--------
- StateGraph(StateType)：创建一个状态图
- graph.add_node(name, fn)：注册节点
- graph.add_edge(src, dst)：添加边
- graph.add_conditional_edges(src, routing_fn, mapping)：条件路由
- graph.compile()：编译成可执行图
- compiled.invoke(initial_state)：执行

本工作流的结构
--------------
START → health → nutrition → recipe → exercise → supervisor → END

特点：
1. 顺序依赖（每个 Node 都依赖上游的输出）
2. 显式错误传递（任何 Node 失败记到 state["errors"]，不崩工作流）
3. Mock 兜底（无 LLM 时也能跑完整流程）
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    exercise_node,
    health_node,
    nutrition_node,
    recipe_node,
    supervisor_node,
)
from app.graph.state import DietPlanState
from app.models.health import HealthProfile


# ---------------------------------------------------------------------------
# 工作流构建
# ---------------------------------------------------------------------------

def build_diet_plan_graph():
    """构建完整饮食方案的多 Agent 工作流。

    返回一个 CompiledGraph，可调用 .invoke(state) / .stream(state)。
    """
    graph = StateGraph(DietPlanState)

    # 1. 注册所有节点
    # 注意：LangGraph 不允许节点名与 State 字段名相同，所以加 _node 后缀
    graph.add_node("health_node", health_node)
    graph.add_node("nutrition_node", nutrition_node)
    graph.add_node("recipe_node", recipe_node)
    graph.add_node("exercise_node", exercise_node)
    graph.add_node("supervisor_node", supervisor_node)

    # 2. 定义边：顺序流水线
    graph.add_edge(START, "health_node")
    graph.add_edge("health_node", "nutrition_node")
    graph.add_edge("nutrition_node", "recipe_node")
    graph.add_edge("recipe_node", "exercise_node")
    graph.add_edge("exercise_node", "supervisor_node")
    graph.add_edge("supervisor_node", END)

    # 3. 编译（可选：传 checkpointer 实现 checkpoint/重放）
    return graph.compile()


# ---------------------------------------------------------------------------
# 便捷入口：一步生成完整方案
# ---------------------------------------------------------------------------

# 全局单例（编译一次，多次调用）
_compiled_graph = None


def get_workflow():
    """获取编译后的全局工作流单例。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_diet_plan_graph()
    return _compiled_graph


def run_diet_plan(profile: HealthProfile) -> dict[str, Any]:
    """对一个用户画像跑完整工作流，返回最终 state。

    用法：
        state = run_diet_plan(profile)
        final_plan = state["final_plan"]

    state["errors"] 包含所有失败的 Node（如果有）。
    """
    workflow = get_workflow()
    initial_state: DietPlanState = {
        "profile": profile,
        "errors": [],
    }
    return workflow.invoke(initial_state)


# ---------------------------------------------------------------------------
# 进阶：流式执行（每个 Node 完成时 yield）
# ---------------------------------------------------------------------------

def stream_diet_plan(profile: HealthProfile):
    """流式执行工作流，每个 Node 完成时 yield 一个事件。

    用于前端逐个展示 Agent 的结果（而不是等所有都完成）。

    用法：
        for event in stream_diet_plan(profile):
            print(event)  # {"health": ...} → {"nutrition": ...} → ...
    """
    workflow = get_workflow()
    initial_state: DietPlanState = {
        "profile": profile,
        "errors": [],
    }
    yield from workflow.stream(initial_state, stream_mode="updates")


# ---------------------------------------------------------------------------
# 异步版本（Stage 5）：给 FastAPI 用，不阻塞事件循环
# ---------------------------------------------------------------------------

async def arun_diet_plan(profile: HealthProfile) -> dict[str, Any]:
    """异步执行工作流。

    实现策略：用 asyncio.to_thread 把同步 workflow.invoke 放进线程池。
    这样：
    - 内部代码不动（保持简单 + 易测试）
    - FastAPI 事件循环不被 LLM 长耗时调用阻塞
    - 多个并发请求可真正并行（不同线程）

    为什么不用 workflow.ainvoke？
    - 我们的 Node 是同步函数，ainvoke 也救不了
    - 真正的 async node 需要把所有 Agent 方法都改成 async，工程量大
    - to_thread 模式更简单，效果一样
    """
    import asyncio
    return await asyncio.to_thread(run_diet_plan, profile)


async def astream_diet_plan(profile: HealthProfile):
    """异步流式执行工作流，作为 async generator。

    用于 FastAPI 的 StreamingResponse（SSE）。
    """
    import asyncio
    workflow = get_workflow()
    initial_state: DietPlanState = {
        "profile": profile,
        "errors": [],
    }
    # workflow.astream 是 LangGraph 原生 async 流式接口
    # 即使 Node 是同步的，astream 也会异步调度
    async for event in workflow.astream(initial_state, stream_mode="updates"):
        yield event

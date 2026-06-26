"""
LangGraph 的节点函数。

每个 Node 的契约
----------------
输入：完整 State
输出：State 的**部分更新**（dict），LangGraph 会自动合并到当前 State

为什么返回 dict 而不是完整 State？
----------------------------------
- 性能：只传递变化的部分
- 解耦：Node 不需要知道其他字段
- 并行友好：多个 Node 可以同时更新不同字段

每个 Node 都用 try/except 包住，失败时记到 state["errors"]，不让工作流崩溃。
"""

from __future__ import annotations

from app.agents.exercise import ExerciseAdvisorAgent
from app.agents.health import HealthAssessmentAgent
from app.agents.nutrition import NutritionPlannerAgent
from app.agents.recipe import RecipeAgent
from app.agents.supervisor import SupervisorAgent
from app.graph.state import DietPlanState


def _on_error(state: DietPlanState, node_name: str, err: Exception) -> dict:
    """统一的错误处理：记到 errors 列表，不抛异常。"""
    msg = f"{node_name}: {type(err).__name__}: {err}"
    print(f"[graph] 节点失败 {msg}")
    errors = list(state.get("errors", []))
    errors.append(msg)
    return {"errors": errors}


# ---------------------------------------------------------------------------
# 节点 1：健康评估
# ---------------------------------------------------------------------------

def health_node(state: DietPlanState) -> dict:
    """健康评估节点：输入 profile，输出 HealthAssessment。"""
    try:
        profile = state["profile"]
        agent = HealthAssessmentAgent()
        assessment = agent.assess(profile)
        return {"health": assessment}
    except Exception as e:
        return _on_error(state, "health_node", e)


# ---------------------------------------------------------------------------
# 节点 2：营养规划
# ---------------------------------------------------------------------------

def nutrition_node(state: DietPlanState) -> dict:
    """营养规划节点：依赖 health。"""
    try:
        profile = state["profile"]
        health = state["health"]
        agent = NutritionPlannerAgent()
        plan = agent.plan(health, profile)
        return {"nutrition": plan}
    except Exception as e:
        return _on_error(state, "nutrition_node", e)


# ---------------------------------------------------------------------------
# 节点 3：菜谱生成（调用 RAG）
# ---------------------------------------------------------------------------

def recipe_node(state: DietPlanState) -> dict:
    """菜谱生成节点：依赖 nutrition + profile，并调用 RAG 检索。"""
    try:
        profile = state["profile"]
        nutrition = state["nutrition"]
        agent = RecipeAgent()
        plan = agent.generate(nutrition, profile)
        return {"recipe": plan}
    except Exception as e:
        return _on_error(state, "recipe_node", e)


# ---------------------------------------------------------------------------
# 节点 4：运动建议
# ---------------------------------------------------------------------------

def exercise_node(state: DietPlanState) -> dict:
    """运动建议节点：依赖 health + profile。"""
    try:
        profile = state["profile"]
        health = state["health"]
        agent = ExerciseAdvisorAgent()
        plan = agent.advise(health, profile)
        return {"exercise": plan}
    except Exception as e:
        return _on_error(state, "exercise_node", e)


# ---------------------------------------------------------------------------
# 节点 5：Supervisor 整合
# ---------------------------------------------------------------------------

def supervisor_node(state: DietPlanState) -> dict:
    """整合节点：读取所有上游结果，生成最终 DietPlan。"""
    try:
        profile = state["profile"]
        health = state["health"]
        nutrition = state["nutrition"]
        recipe = state["recipe"]
        exercise = state["exercise"]
        agent = SupervisorAgent()
        final = agent.integrate(profile, health, nutrition, recipe, exercise)
        return {"final_plan": final}
    except Exception as e:
        return _on_error(state, "supervisor_node", e)


# ---------------------------------------------------------------------------
# 所有节点的注册表（便于按名查找）
# ---------------------------------------------------------------------------

NODES: dict[str, callable] = {
    "health": health_node,
    "nutrition": nutrition_node,
    "recipe": recipe_node,
    "exercise": exercise_node,
    "supervisor": supervisor_node,
}

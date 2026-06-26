"""
LangGraph 工作流的共享状态定义。

什么是 State？
-------------
LangGraph 的 State 是一个 TypedDict（或 Pydantic BaseModel），
所有 Node 都读写它。Node 之间不直接通信，**通过 State 传递数据**。

为什么用 TypedDict 而不是 Pydantic？
------------------------------------
LangGraph 官方推荐 TypedDict，原因：
1. 性能：dict 比 BaseModel 实例化快得多
2. 渐进式更新：Node 只更新部分字段，其他字段保持原值
3. 序列化简单：直接是 JSON 兼容的

但我们用 Pydantic 模型作为字段类型（HealthProfile / HealthAssessment 等），
享受两边的优点：State 容器是 dict，字段值是强类型对象。
"""

from __future__ import annotations

from typing import TypedDict

from app.models.diet import (
    DietPlan,
    ExercisePlan,
    MealPlan,
    NutritionPlan,
)
from app.models.health import HealthAssessment, HealthProfile


class DietPlanState(TypedDict, total=False):
    """多 Agent 工作流的共享状态。

    `total=False` 表示所有字段都是可选的——因为不同 Node 只填一部分字段。
    流程：
        START
          → health_node        填充 health
          → nutrition_node     填充 nutrition
          → recipe_node        填充 recipe
          → exercise_node      填充 exercise
          → supervisor_node    填充 final_plan
        END
    """

    # 输入（初始化时填）
    profile: HealthProfile

    # 各 Agent 的中间结果（对应 Node 依次填充）
    health: HealthAssessment
    nutrition: NutritionPlan
    recipe: MealPlan
    exercise: ExercisePlan

    # 最终结果
    final_plan: DietPlan

    # 错误收集（任何 Node 失败时追加）
    errors: list[str]

"""
Stage 4 多 Agent 工作流的数据模型。

新增 4 个输出结构：
- NutritionPlan：营养规划 Agent 的输出
- MealPlan：菜谱 Agent 的输出（含三餐）
- ExercisePlan：运动建议 Agent 的输出
- DietPlan：Supervisor 整合后的最终方案

设计原则
--------
1. 输入（HealthProfile）已存在，复用 Stage 2 的
2. 中间结果（HealthAssessment）已存在，复用 Stage 2 的
3. 每个新 Agent 都有自己的输出类型，互不耦合
4. 字段都加 description，自动出现在 Swagger 文档
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.health import HealthAssessment, HealthProfile, MacroNutrients


# ---------------------------------------------------------------------------
# 营养规划：把每日宏量目标分配到三餐
# ---------------------------------------------------------------------------

class MealCalories(BaseModel):
    """单餐热量分配。"""
    breakfast: float = Field(..., description="早餐热量（kcal）")
    lunch: float = Field(..., description="午餐热量（kcal）")
    dinner: float = Field(..., description="晚餐热量（kcal）")
    snack: float = Field(default=0, description="加餐热量（kcal）")


class NutritionPlan(BaseModel):
    """营养规划 Agent 的输出。

    在健康评估的"每日总目标"基础上，进一步细化到三餐分配和进食时机。
    """
    daily_target: float = Field(..., description="每日总热量目标（kcal）")
    macros_daily: MacroNutrients = Field(..., description="每日宏量营养素")
    meal_calories: MealCalories = Field(..., description="三餐热量分配")
    hydration_ml: int = Field(..., description="每日饮水量（毫升）")
    timing_tips: list[str] = Field(
        default_factory=list,
        description="进食时机建议（如训练前后、睡前几小时等）",
    )
    llm_used: bool = Field(..., description="本次规划是否调用了真实 LLM")


# ---------------------------------------------------------------------------
# 菜谱：三餐的具体食谱
# ---------------------------------------------------------------------------

class Meal(BaseModel):
    """单餐食谱。"""
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] = Field(
        ..., description="餐别"
    )
    name: str = Field(..., description="餐名，如'希腊酸奶燕麦杯'")
    description: str = Field(..., description="简短描述")
    ingredients: list[str] = Field(
        default_factory=list,
        description="食材清单，如['希腊酸奶 200g', '燕麦 40g']",
    )
    calories: float = Field(..., description="本餐热量（kcal）")
    macros: MacroNutrients = Field(..., description="本餐宏量营养素")
    rag_sources: list[str] = Field(
        default_factory=list,
        description="RAG 检索命中的菜谱/食材名（用于可追溯性）",
    )


class MealPlan(BaseModel):
    """菜谱 Agent 的输出：完整一日三餐（+可选加餐）。"""
    meals: list[Meal] = Field(..., description="三餐列表")
    total_calories: float = Field(..., description="本日总热量")
    variety_note: str = Field(default="", description="食材多样性提示")
    llm_used: bool = Field(..., description="是否调用真实 LLM")


# ---------------------------------------------------------------------------
# 运动计划：每周安排
# ---------------------------------------------------------------------------

class ExerciseSession(BaseModel):
    """单次运动安排。"""
    day: str = Field(..., description="星期几，如'Monday'/'Tuesday'")
    type: Literal["cardio", "strength", "flexibility", "rest"] = Field(
        ..., description="运动类型"
    )
    duration_min: int = Field(..., description="时长（分钟）")
    intensity: Literal["low", "moderate", "high"] = Field(..., description="强度")
    description: str = Field(..., description="具体内容，如'慢跑 5km'")


class ExercisePlan(BaseModel):
    """运动建议 Agent 的输出。"""
    weekly_sessions: list[ExerciseSession] = Field(..., description="每周运动安排")
    weekly_calories_burned: float = Field(
        ..., description="预估每周消耗热量（kcal）"
    )
    tips: list[str] = Field(default_factory=list, description="运动注意事项")
    llm_used: bool = Field(..., description="是否调用真实 LLM")


# ---------------------------------------------------------------------------
# 最终方案：Supervisor 整合的完整饮食方案
# ---------------------------------------------------------------------------

class DietPlan(BaseModel):
    """完整饮食方案。Supervisor 把所有子 Agent 的输出整合成最终结果。"""
    # 输入回显
    profile: HealthProfile = Field(..., description="用户原始画像")

    # 各 Agent 的输出
    health: HealthAssessment = Field(..., description="健康评估结果")
    nutrition: NutritionPlan = Field(..., description="营养规划")
    recipe: MealPlan = Field(..., description="三餐食谱")
    exercise: ExercisePlan = Field(..., description="运动建议")

    # Supervisor 自己生成的
    summary: str = Field(..., description="整体方案摘要（自然语言）")
    key_actions: list[str] = Field(
        default_factory=list,
        description="3-5 条最关键的可执行行动",
    )

    # 元信息
    llm_used: bool = Field(..., description="本次方案是否调用真实 LLM")

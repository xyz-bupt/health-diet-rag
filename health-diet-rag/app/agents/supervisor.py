"""
Supervisor Agent（整合层）。

职责
----
工作流的最后一个节点。读取前面所有 Agent 的输出，生成：
1. 整体方案摘要（自然语言）
2. 3-5 条最关键的可执行行动

为什么需要 Supervisor
--------------------
- 前 4 个 Agent 各管一摊，用户看到 4 块独立内容会很乱
- Supervisor 把所有结果整合成"一段完整方案"
- 这是 LangGraph 多 Agent 编排里 Supervisor 模式的标准做法

在 LangGraph 中的位置
--------------------
位于工作流末端，前面 4 个 Agent 都跑完后才执行。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm, is_llm_available
from app.models.diet import (
    DietPlan,
    ExercisePlan,
    MealPlan,
    NutritionPlan,
)
from app.models.health import HealthAssessment, HealthProfile


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是健康饮食方案的整合专家。根据健康评估、营养规划、菜谱、运动建议，
写一份**面向用户的完整方案摘要**。

要求：
1. summary：3-5 句话，语气专业友好，覆盖"现状 → 目标 → 核心策略"
2. key_actions：3-5 条最关键的可执行行动（不是建议清单，是**最优先**的几件事）
3. 不要重复罗列所有数字，挑重点说
4. 用中文，严格按 JSON 输出
"""

HUMAN_TEMPLATE = """用户目标：{goal}

【健康评估】
- BMI: {bmi} ({bmi_category})
- BMR: {bmr} kcal, TDEE: {tdee} kcal
- 每日目标热量：{target_calories} kcal
- 宏量：蛋白 {protein}g / 碳水 {carbs}g / 脂肪 {fat}g

【营养规划】
- 三餐分配：早 {breakfast} / 午 {lunch} / 晚 {dinner} / 加餐 {snack} kcal
- 每日饮水：{hydration} ml
- 时机建议：{timing_tips}

【菜谱】
{meals_summary}

【运动】
- 每周消耗目标：{weekly_burned} kcal
- 训练频率：{training_days} 天/周
- 注意事项：{exercise_tips}

请输出 JSON：
{{
  "summary": "...",
  "key_actions": ["...", "...", "..."]
}}
"""


def _mock_summary(
    profile: HealthProfile,
    health: HealthAssessment,
    nutrition: NutritionPlan,
    recipe: MealPlan,
    exercise: ExercisePlan,
) -> dict[str, Any]:
    """规则化兜底摘要。"""
    summary = (
        f"基于您 {profile.goal} 的目标，每日目标热量 {health.target_calories:.0f} kcal，"
        f"三餐已规划为 {nutrition.meal_calories.breakfast:.0f}/"
        f"{nutrition.meal_calories.lunch:.0f}/{nutrition.meal_calories.dinner:.0f} kcal，"
        f"搭配每周 {exercise.weekly_calories_burned:.0f} kcal 运动消耗，"
        f"预计可在 8-12 周看到明显效果。"
    )
    actions = [
        f"每日摄入控制在 {health.target_calories:.0f} kcal 左右",
        f"蛋白质摄入 {nutrition.macros_daily.protein_g:.0f}g（每公斤体重 "
        f"{nutrition.macros_daily.protein_g / profile.weight_kg:.1f}g）",
        "每周至少 3 次力量训练，保肌肉不流失",
        "每日饮水 {0} ml，分 6-8 次摄入".format(nutrition.hydration_ml),
        "睡前 3 小时完成晚餐，避免夜宵",
    ]
    return {"summary": summary, "key_actions": actions}


# ---------------------------------------------------------------------------
# Agent 主类
# ---------------------------------------------------------------------------

class SupervisorAgent:
    """Supervisor：整合所有 Agent 输出，生成最终方案。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or get_llm()
        self.llm_available = is_llm_available()

    def integrate(
        self,
        profile: HealthProfile,
        health: HealthAssessment,
        nutrition: NutritionPlan,
        recipe: MealPlan,
        exercise: ExercisePlan,
    ) -> DietPlan:
        """整合所有结果生成最终 DietPlan。"""
        summary_data = self._gen_summary(profile, health, nutrition, recipe, exercise)

        return DietPlan(
            profile=profile,
            health=health,
            nutrition=nutrition,
            recipe=recipe,
            exercise=exercise,
            summary=summary_data["summary"],
            key_actions=summary_data["key_actions"],
            llm_used=self.llm_available,
        )

    def _gen_summary(
        self,
        profile: HealthProfile,
        health: HealthAssessment,
        nutrition: NutritionPlan,
        recipe: MealPlan,
        exercise: ExercisePlan,
    ) -> dict[str, Any]:
        if not self.llm_available:
            return _mock_summary(profile, health, nutrition, recipe, exercise)

        # 拼接菜谱摘要
        meals_summary = "\n".join(
            f"- {m.meal_type}: {m.name} ({m.calories:.0f} kcal)"
            for m in recipe.meals
        )

        prompt_vars = {
            "goal": profile.goal,
            "bmi": health.bmi,
            "bmi_category": health.bmi_category,
            "bmr": health.bmr,
            "tdee": health.tdee,
            "target_calories": health.target_calories,
            "protein": nutrition.macros_daily.protein_g,
            "carbs": nutrition.macros_daily.carbs_g,
            "fat": nutrition.macros_daily.fat_g,
            "breakfast": nutrition.meal_calories.breakfast,
            "lunch": nutrition.meal_calories.lunch,
            "dinner": nutrition.meal_calories.dinner,
            "snack": nutrition.meal_calories.snack,
            "hydration": nutrition.hydration_ml,
            "timing_tips": "; ".join(nutrition.timing_tips),
            "meals_summary": meals_summary,
            "weekly_burned": exercise.weekly_calories_burned,
            "training_days": sum(1 for s in exercise.weekly_sessions if s.type != "rest"),
            "exercise_tips": "; ".join(exercise.tips),
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=HUMAN_TEMPLATE.format(**prompt_vars)),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            from app.agents.health import _parse_json_response
            return _parse_json_response(content)
        except Exception as e:
            print(f"[Supervisor] LLM 失败：{e}")
            return _mock_summary(profile, health, nutrition, recipe, exercise)

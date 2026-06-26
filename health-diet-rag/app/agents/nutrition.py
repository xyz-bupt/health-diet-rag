"""
营养规划 Agent。

职责
----
接收健康评估结果（BMR/TDEE/目标热量/每日宏量），进一步细化：
1. 把每日总热量分配到三餐（早餐 30%、午餐 40%、晚餐 25%、加餐 5%）
2. 计算每日饮水量
3. 用大模型生成进食时机建议（训练前后、睡前等）

为什么单独抽一个 Agent？
------------------------
健康评估只关心"总量"，营养规划关心"分布"：
- 同样 2000 kcal，三餐平均分 vs 早餐重午餐轻，效果天差地别
- 减脂期睡前 3 小时不吃 vs 增肌期训练后立即补，时机完全不同
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.health_calc import calc_macros
from app.core.llm import get_llm, is_llm_available
from app.models.diet import MealCalories, NutritionPlan
from app.models.health import HealthAssessment, HealthProfile


# ---------------------------------------------------------------------------
# 确定性部分：三餐热量分配（标准 30/40/25/5 比例）
# ---------------------------------------------------------------------------

# 不同目标的三餐分配比例（早餐/午餐/晚餐/加餐）
MEAL_DISTRIBUTION = {
    "lose_weight": (0.30, 0.35, 0.25, 0.10),  # 减脂：加餐比例稍高，防饥饿
    "maintain":    (0.30, 0.40, 0.25, 0.05),
    "gain_muscle": (0.25, 0.35, 0.30, 0.10),  # 增肌：训练后加餐重要
}


def calc_meal_calories(
    daily_target: float, goal: str
) -> MealCalories:
    """按目标对应的比例分配三餐热量。"""
    ratios = MEAL_DISTRIBUTION.get(goal, MEAL_DISTRIBUTION["maintain"])
    return MealCalories(
        breakfast=round(daily_target * ratios[0], 1),
        lunch=round(daily_target * ratios[1], 1),
        dinner=round(daily_target * ratios[2], 1),
        snack=round(daily_target * ratios[3], 1),
    )


def calc_hydration(weight_kg: float) -> int:
    """饮水量推荐：体重 × 35 ml（日常），向下取整到 100ml。"""
    ml = int(weight_kg * 35)
    return (ml // 100) * 100


# ---------------------------------------------------------------------------
# Prompt：让 LLM 生成进食时机建议
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是营养时序专家。根据用户的目标和画像，给出 3-5 条**进食时机**相关的建议。

注意：
- 不要重复计算热量（数字已经提供）
- 重点放在"什么时候吃"、"训练前后怎么补"、"睡前几小时不吃"等时机问题
- 每条一句话，具体可执行
- 用中文，严格按 JSON 格式输出
"""

HUMAN_TEMPLATE = """用户目标：{goal}
活动量：{activity_level}
每日目标热量：{target_calories} kcal
三餐分配：早餐 {breakfast} / 午餐 {lunch} / 晚餐 {dinner} / 加餐 {snack} kcal

请输出 JSON：
{{
  "timing_tips": ["建议1", "建议2", ...]
}}
"""


def _mock_timing_tips(goal: str) -> list[str]:
    """Mock 兜底：规则化时机建议。"""
    base = [
        "早餐在起床后 1 小时内吃完，启动代谢",
        "午餐与晚餐间隔 4-5 小时，避免血糖大幅波动",
        "晚餐在睡前 3 小时完成，给消化留时间",
    ]
    if goal == "lose_weight":
        base.append("下午 4 点加餐，防止晚餐过饿暴食")
    elif goal == "gain_muscle":
        base.append("训练后 30 分钟内补充 20-30g 蛋白质")
    return base


# ---------------------------------------------------------------------------
# Agent 主类
# ---------------------------------------------------------------------------

class NutritionPlannerAgent:
    """营养规划 Agent。

    用法：
        agent = NutritionPlannerAgent()
        plan = agent.plan(health_assessment, profile)
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or get_llm()
        self.llm_available = is_llm_available()

    def plan(
        self, health: HealthAssessment, profile: HealthProfile
    ) -> NutritionPlan:
        """根据健康评估结果生成营养规划。"""
        # 1. 确定性计算
        meal_cal = calc_meal_calories(health.target_calories, profile.goal)
        hydration = calc_hydration(profile.weight_kg)

        # 2. LLM 生成时机建议
        timing_tips = self._gen_timing_tips(profile, health, meal_cal)

        return NutritionPlan(
            daily_target=health.target_calories,
            macros_daily=health.macros,
            meal_calories=meal_cal,
            hydration_ml=hydration,
            timing_tips=timing_tips,
            llm_used=self.llm_available,
        )

    def _gen_timing_tips(
        self,
        profile: HealthProfile,
        health: HealthAssessment,
        meal_cal: MealCalories,
    ) -> list[str]:
        if not self.llm_available:
            return _mock_timing_tips(profile.goal)

        prompt_vars = {
            "goal": profile.goal,
            "activity_level": profile.activity_level,
            "target_calories": health.target_calories,
            "breakfast": meal_cal.breakfast,
            "lunch": meal_cal.lunch,
            "dinner": meal_cal.dinner,
            "snack": meal_cal.snack,
        }
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=HUMAN_TEMPLATE.format(**prompt_vars)),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            from app.agents.health import _parse_json_response
            parsed = _parse_json_response(content)
            return parsed.get("timing_tips", _mock_timing_tips(profile.goal))
        except Exception as e:
            print(f"[NutritionPlanner] LLM 失败，降级 Mock：{e}")
            return _mock_timing_tips(profile.goal)

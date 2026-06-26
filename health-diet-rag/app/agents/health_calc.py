"""
健康评估的确定性计算部分。

把"纯计算"从 Agent 里拆出来的原因
--------------------------------
1. 数学公式是确定的，绝不应该交给 LLM（LLM 算数会出错）
2. 拆出来后可以单独单元测试，不需要 LLM 也能验证正确性
3. Agent 类只负责"调度 LLM + 组装结果"，职责清晰

公式来源
--------
- BMR（基础代谢率）：Mifflin-St Jeor 公式，2005 年发表，目前最常用、最准
- TDEE：BMR × 活动系数（Harris-Benedict 改进版系数）
- BMI：体重 / 身高²（米）
- 宏量营养素分配：根据目标调整蛋白/碳水/脂肪的比例
"""

from __future__ import annotations

from app.models.health import (
    ActivityLevel,
    Goal,
    HealthProfile,
    MacroNutrients,
)


# ---------------------------------------------------------------------------
# 活动系数表（不要用 if/elif，用查表更清晰、更易扩展）
# ---------------------------------------------------------------------------

ACTIVITY_FACTORS: dict[ActivityLevel, float] = {
    "sedentary": 1.2,      # 久坐：办公室工作，几乎不运动
    "light": 1.375,        # 轻度：每周 1-3 次轻运动
    "moderate": 1.55,      # 中度：每周 3-5 次中等运动
    "active": 1.725,       # 高度：每周 6-7 次剧烈运动
    "very_active": 1.9,    # 极高：每日双训或体力劳动
}

# 不同目标对应的热量调整系数
GOAL_CALORIE_FACTOR: dict[Goal, float] = {
    "lose_weight": 0.8,    # 减脂：20% 缺口
    "maintain": 1.0,       # 维持：保持 TDEE
    "gain_muscle": 1.1,    # 增肌：10% 盈余
}


# ---------------------------------------------------------------------------
# 各项计算
# ---------------------------------------------------------------------------

def calc_bmi(weight_kg: float, height_cm: float) -> float:
    """计算 BMI = 体重(kg) / 身高(m)²。"""
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def bmi_category(bmi: float) -> str:
    """根据 BMI 返回中文分类（中国标准，比 WHO 更严）。"""
    if bmi < 18.5:
        return "偏瘦"
    if bmi < 24:
        return "正常"
    if bmi < 28:
        return "超重"
    return "肥胖"


def calc_bmr(profile: HealthProfile) -> float:
    """Mifflin-St Jeor 公式计算基础代谢率。

    男性: BMR = 10*体重 + 6.25*身高 - 5*年龄 + 5
    女性: BMR = 10*体重 + 6.25*身高 - 5*年龄 - 161

    这是 2005 年 ADA 推荐公式，比老 Harris-Benedict 更准。
    """
    base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    if profile.gender == "male":
        return round(base + 5, 1)
    return round(base - 161, 1)


def calc_tdee(bmr: float, activity: ActivityLevel) -> float:
    """TDEE = BMR × 活动系数。"""
    return round(bmr * ACTIVITY_FACTORS[activity], 1)


def calc_target_calories(tdee: float, goal: Goal) -> float:
    """根据目标调整每日目标热量。"""
    return round(tdee * GOAL_CALORIE_FACTOR[goal], 1)


def calc_macros(target_calories: float, goal: Goal) -> MacroNutrients:
    """根据目标热量和目标类型分配宏量营养素。

    蛋白质/碳水/脂肪的热量密度都是固定的：
    - 1 g 蛋白质 = 4 kcal
    - 1 g 碳水   = 4 kcal
    - 1 g 脂肪   = 9 kcal

    不同目标的比例参考 ACE（美国运动委员会）：
    - 减脂：    蛋白 40% / 碳水 35% / 脂肪 25%（高蛋白保肌肉）
    - 维持：    蛋白 30% / 碳水 45% / 脂肪 25%
    - 增肌：    蛋白 30% / 碳水 50% / 脂肪 20%（高碳水燃料）
    """
    ratios = {
        "lose_weight": (0.40, 0.35, 0.25),
        "maintain":    (0.30, 0.45, 0.25),
        "gain_muscle": (0.30, 0.50, 0.20),
    }
    p_ratio, c_ratio, f_ratio = ratios[goal]
    return MacroNutrients(
        protein_g=round(target_calories * p_ratio / 4, 1),
        carbs_g=round(target_calories * c_ratio / 4, 1),
        fat_g=round(target_calories * f_ratio / 9, 1),
    )

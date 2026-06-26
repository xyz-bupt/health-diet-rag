"""
健康评估的纯计算测试。

这部分完全不需要 LLM、不需要网络、不需要 API Key，
是数学公式正确性的"金标准"测试。
运行：pytest tests/test_health_calc.py -v
"""

import pytest

from app.agents.health_calc import (
    ACTIVITY_FACTORS,
    GOAL_CALORIE_FACTOR,
    bmi_category,
    calc_bmi,
    calc_bmr,
    calc_macros,
    calc_target_calories,
    calc_tdee,
)
from app.models.health import HealthProfile


# ---------------------------------------------------------------------------
# 辅助：构造测试画像
# ---------------------------------------------------------------------------

def profile(**kwargs) -> HealthProfile:
    """用默认值构造画像，测试时只覆盖关心的字段。"""
    defaults = dict(
        height_cm=175, weight_kg=70, age=28,
        gender="male", activity_level="moderate", goal="maintain",
    )
    defaults.update(kwargs)
    return HealthProfile(**defaults)


# ---------------------------------------------------------------------------
# BMI
# ---------------------------------------------------------------------------

def test_bmi_normal():
    assert calc_bmi(70, 175) == 22.9


def test_bmi_obese():
    assert calc_bmi(100, 170) == 34.6


def test_bmi_category_ranges():
    assert bmi_category(17) == "偏瘦"
    assert bmi_category(20) == "正常"
    assert bmi_category(25) == "超重"
    assert bmi_category(30) == "肥胖"


# ---------------------------------------------------------------------------
# BMR：Mifflin-St Jeor 公式
# ---------------------------------------------------------------------------

def test_bmr_male():
    # 男：10*70 + 6.25*175 - 5*28 + 5 = 1658.75
    p = profile(gender="male")
    assert calc_bmr(p) == 1658.8


def test_bmr_female():
    # 女：10*60 + 6.25*165 - 5*25 - 161 = 1345.25
    p = profile(gender="female", weight_kg=60, height_cm=165, age=25)
    assert calc_bmr(p) == 1345.2


def test_bmr_female_higher_than_male_same_stats_minus_offset():
    """相同身体数据下，女性 BMR 比男性约低 166（5 - (-161) = 166）。"""
    male = profile(gender="male")
    female = profile(gender="female")
    diff = calc_bmr(male) - calc_bmr(female)
    assert abs(diff - 166.0) < 0.1


# ---------------------------------------------------------------------------
# TDEE：活动系数
# ---------------------------------------------------------------------------

def test_tdee_uses_correct_factor():
    bmr = 1658.8
    for level, factor in ACTIVITY_FACTORS.items():
        expected = round(bmr * factor, 1)
        assert calc_tdee(bmr, level) == expected


def test_tdee_increases_with_activity():
    bmr = 1658.8
    values = [calc_tdee(bmr, lvl) for lvl in
              ["sedentary", "light", "moderate", "active", "very_active"]]
    assert values == sorted(values)  # 单调递增


# ---------------------------------------------------------------------------
# 目标热量调整
# ---------------------------------------------------------------------------

def test_target_calories_lose_weight_deficit():
    """减脂：20% 缺口。"""
    assert calc_target_calories(2500, "lose_weight") == 2000.0


def test_target_calories_maintain_same():
    assert calc_target_calories(2500, "maintain") == 2500.0


def test_target_calories_gain_muscle_surplus():
    """增肌：10% 盈余。"""
    assert calc_target_calories(2500, "gain_muscle") == 2750.0


# ---------------------------------------------------------------------------
# 宏量营养素分配
# ---------------------------------------------------------------------------

def test_macros_lose_weight_high_protein():
    """减脂：40/35/25 → 高蛋白保肌肉。"""
    m = calc_macros(2000, "lose_weight")
    # 蛋白 2000*0.4/4 = 200g
    assert m.protein_g == 200.0
    # 碳水 2000*0.35/4 = 175g
    assert m.carbs_g == 175.0
    # 脂肪 2000*0.25/9 ≈ 55.6g
    assert m.fat_g == 55.6


def test_macros_maintain_balanced():
    m = calc_macros(2000, "maintain")
    # 蛋白 2000*0.3/4 = 150g
    assert m.protein_g == 150.0
    # 碳水 2000*0.45/4 = 225g
    assert m.carbs_g == 225.0
    # 脂肪 2000*0.25/9 ≈ 55.6g
    assert m.fat_g == 55.6


def test_macros_gain_muscle_high_carb():
    m = calc_macros(2000, "gain_muscle")
    # 蛋白 2000*0.3/4 = 150g
    assert m.protein_g == 150.0
    # 碳水 2000*0.5/4 = 250g
    assert m.carbs_g == 250.0
    # 脂肪 2000*0.2/9 ≈ 44.4g
    assert m.fat_g == 44.4


def test_macros_total_calories_match():
    """宏量加总的热量应等于目标热量（±1 kcal 容差，因为四舍五入）。"""
    for goal in ["lose_weight", "maintain", "gain_muscle"]:
        target = 2000.0
        m = calc_macros(target, goal)
        total = m.protein_g * 4 + m.carbs_g * 4 + m.fat_g * 9
        assert abs(total - target) < 1.5, f"{goal}: {total} vs {target}"

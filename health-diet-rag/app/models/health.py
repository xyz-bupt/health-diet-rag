"""
健康评估相关的 Pydantic 数据模型。

设计原则
--------
1. 输入/输出分离：HealthProfile 是输入，HealthAssessment 是输出
2. 枚举用 Literal 而非 Enum：更易序列化为 JSON，FastAPI 文档更直观
3. 字段加约束：身高/体重有合理上下限，防止 LLM 拿到离谱数据乱算
4. 加字段说明：description 会出现在 Swagger 文档里，自解释
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 枚举类型（用 Literal 表示有限取值）
# ---------------------------------------------------------------------------

Gender = Literal["male", "female"]
"""性别。male=男，female=女。BMR 公式对男女不同。"""

ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very_active"]
"""
活动量等级，对应 TDEE 系数：
- sedentary    久坐    系数 1.2
- light        轻度    系数 1.375（每周 1-3 次轻运动）
- moderate     中度    系数 1.55（每周 3-5 次中等运动）
- active       高度    系数 1.725（每周 6-7 次剧烈运动）
- very_active  极高    系数 1.9（体力劳动或每日双训）
"""

Goal = Literal["lose_weight", "maintain", "gain_muscle"]
"""
目标：
- lose_weight  减脂：热量缺口 ~20%
- maintain     维持：热量 = TDEE
- gain_muscle  增肌：热量盈余 ~10%
"""


# ---------------------------------------------------------------------------
# 输入：用户健康画像
# ---------------------------------------------------------------------------

class HealthProfile(BaseModel):
    """用户提交的健康基本信息。FastAPI 收到请求后会自动校验。"""

    height_cm: float = Field(
        ..., gt=80, lt=250, description="身高（厘米），范围 80-250"
    )
    weight_kg: float = Field(
        ..., gt=20, lt=300, description="体重（公斤），范围 20-300"
    )
    age: int = Field(
        ..., ge=10, le=120, description="年龄（岁），范围 10-120"
    )
    gender: Gender = Field(..., description="性别：male / female")
    activity_level: ActivityLevel = Field(
        default="moderate",
        description="活动量等级，见 ActivityLevel 注释",
    )
    goal: Goal = Field(
        default="maintain",
        description="目标：lose_weight / maintain / gain_muscle",
    )

    # 选填：饮食偏好，后续菜谱 Agent 会用到，这里只是先收集
    dietary_preference: list[str] = Field(
        default_factory=list,
        description="饮食偏好/限制，如 ['vegetarian', 'low_sodium']",
    )


# ---------------------------------------------------------------------------
# 输出：健康评估结果
# ---------------------------------------------------------------------------

class MacroNutrients(BaseModel):
    """宏量营养素分配（克/天）。"""

    protein_g: float = Field(..., description="蛋白质（克）")
    carbs_g: float = Field(..., description="碳水化合物（克）")
    fat_g: float = Field(..., description="脂肪（克）")


class HealthAssessment(BaseModel):
    """健康评估 Agent 的完整输出。"""

    # ---- 由代码确定性计算 ----
    bmi: float = Field(..., description="体质指数 BMI = 体重 / 身高²")
    bmi_category: str = Field(..., description="BMI 分类：偏瘦/正常/超重/肥胖")
    bmr: float = Field(..., description="基础代谢率 BMR（千卡/天），Mifflin-St Jeor 公式")
    tdee: float = Field(..., description="每日总能量消耗 TDEE（千卡/天）= BMR × 活动系数")
    target_calories: float = Field(..., description="根据目标调整后的每日目标热量")
    macros: MacroNutrients = Field(..., description="宏量营养素分配")

    # ---- 由 LLM 生成（或 Mock 兜底）----
    summary: str = Field(..., description="整体健康评估摘要（自然语言）")
    recommendations: list[str] = Field(
        default_factory=list, description="针对性建议列表"
    )

    # ---- 元信息 ----
    llm_used: bool = Field(..., description="本次评估是否调用了真实 LLM")

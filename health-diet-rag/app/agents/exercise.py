"""
运动建议 Agent。

职责
----
接收健康评估结果 + 用户画像，生成每周运动计划。

确定性部分
----------
- 不同运动的 MET 值（代谢当量）→ 计算热量消耗
- 不同目标的运动频率基线（减脂多做有氧 / 增肌多做力量）
- 每周热量消耗目标

LLM 部分
--------
- 安排到具体星期几
- 生成注意事项（如热身、拉伸、循序渐进）
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm, is_llm_available
from app.models.diet import ExercisePlan, ExerciseSession
from app.models.health import HealthAssessment, HealthProfile


# ---------------------------------------------------------------------------
# 确定性：MET 表 + 热量消耗计算
# ---------------------------------------------------------------------------

# 常见运动的 MET 值（Metabolic Equivalent of Task）
# 1 MET = 静坐时的能量消耗，约 1 kcal/kg/h
MET_VALUES = {
    "walking": 3.5,        # 快走
    "jogging": 7.0,        # 慢跑
    "running": 9.8,        # 跑步（8 km/h）
    "cycling": 7.5,        # 自行车（中等）
    "swimming": 8.0,       # 游泳（自由泳）
    "strength_training": 6.0,  # 力量训练
    "yoga": 3.0,           # 瑜伽
    "hiit": 8.0,           # HIIT
    "hiking": 6.0,         # 徒步
}


def calc_calories_burned(
    met: float, weight_kg: float, duration_min: int
) -> float:
    """计算单次运动消耗：kcal = MET × 体重 × 时长(h)。"""
    return round(met * weight_kg * (duration_min / 60), 1)


# 不同目标的每周安排模板（7 天）
WEEKLY_TEMPLATES = {
    "lose_weight": [
        ("Monday", "cardio", 45, "moderate", "快走或慢跑"),
        ("Tuesday", "strength", 30, "moderate", "全身力量训练"),
        ("Wednesday", "cardio", 30, "moderate", "骑行或游泳"),
        ("Thursday", "flexibility", 30, "low", "瑜伽或拉伸"),
        ("Friday", "cardio", 45, "high", "HIIT 间歇训练"),
        ("Saturday", "cardio", 60, "moderate", "户外徒步"),
        ("Sunday", "rest", 0, "low", "完全休息或散步"),
    ],
    "maintain": [
        ("Monday", "strength", 45, "moderate", "力量训练"),
        ("Tuesday", "cardio", 30, "moderate", "慢跑"),
        ("Wednesday", "flexibility", 30, "low", "瑜伽"),
        ("Thursday", "strength", 45, "moderate", "上下肢力量"),
        ("Friday", "cardio", 30, "moderate", "骑行"),
        ("Saturday", "cardio", 60, "moderate", "户外活动"),
        ("Sunday", "rest", 0, "low", "休息"),
    ],
    "gain_muscle": [
        ("Monday", "strength", 60, "high", "胸+三头"),
        ("Tuesday", "cardio", 20, "low", "轻度有氧恢复"),
        ("Wednesday", "strength", 60, "high", "背+二头"),
        ("Thursday", "flexibility", 20, "low", "拉伸放松"),
        ("Friday", "strength", 60, "high", "腿+肩"),
        ("Saturday", "cardio", 30, "moderate", "中等有氧"),
        ("Sunday", "rest", 0, "low", "完全休息"),
    ],
}


# ---------------------------------------------------------------------------
# Mock 提示（无 LLM 时兜底）
# ---------------------------------------------------------------------------

def _mock_tips(goal: str) -> list[str]:
    base = [
        "运动前热身 5-10 分钟，预防受伤",
        "运动后拉伸 5-10 分钟，促进恢复",
        "循序渐进，每周训练量增幅不超过 10%",
        "训练日注意补充水分和蛋白质",
    ]
    if goal == "lose_weight":
        base.append("力量训练不可少——保肌肉比单纯有氧更重要")
    elif goal == "gain_muscle":
        base.append("增肌期训练量大于有氧，避免能量消耗过度")
    return base


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是运动处方专家。根据用户的目标和身体状况，给出 4-6 条**运动注意事项**。

要求：
- 不要重新生成训练计划（已提供）
- 重点放在热身、拉伸、循序渐进、受伤预防
- 每条一句话，具体可执行
- 用中文，严格按 JSON 输出：{"tips": ["...", "..."]}
"""

HUMAN_TEMPLATE = """用户目标：{goal}
BMI：{bmi}（{bmi_category}）
活动量：{activity_level}
每周训练安排：
{weekly_summary}

请输出 4-6 条注意事项，JSON 格式：{{"tips": ["...", "..."]}}
"""


# ---------------------------------------------------------------------------
# Agent 主类
# ---------------------------------------------------------------------------

class ExerciseAdvisorAgent:
    """运动建议 Agent。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or get_llm()
        self.llm_available = is_llm_available()

    def advise(
        self, health: HealthAssessment, profile: HealthProfile
    ) -> ExercisePlan:
        """生成每周运动计划。"""
        # 1. 取模板
        template = WEEKLY_TEMPLATES.get(profile.goal, WEEKLY_TEMPLATES["maintain"])

        # 2. 计算每次消耗 + 构造 Session
        sessions = []
        total_burned = 0.0
        for day, ex_type, duration, intensity, desc in template:
            if ex_type == "rest":
                sessions.append(ExerciseSession(
                    day=day, type="rest", duration_min=duration,
                    intensity=intensity, description=desc,
                ))
                continue
            # 找该类型对应的 MET（用 hiit/strength_training 等 key 近似）
            met_key_map = {
                "cardio": "jogging" if intensity != "high" else "hiit",
                "strength": "strength_training",
                "flexibility": "yoga",
            }
            met_key = met_key_map.get(ex_type, "walking")
            met = MET_VALUES.get(met_key, 4.0)
            burned = calc_calories_burned(met, profile.weight_kg, duration)
            total_burned += burned
            sessions.append(ExerciseSession(
                day=day, type=ex_type, duration_min=duration,
                intensity=intensity, description=desc,
            ))

        # 3. LLM 生成注意事项
        tips = self._gen_tips(health, profile, sessions)

        return ExercisePlan(
            weekly_sessions=sessions,
            weekly_calories_burned=round(total_burned, 1),
            tips=tips,
            llm_used=self.llm_available,
        )

    def _gen_tips(
        self,
        health: HealthAssessment,
        profile: HealthProfile,
        sessions: list[ExerciseSession],
    ) -> list[str]:
        if not self.llm_available:
            return _mock_tips(profile.goal)

        weekly_summary = "\n".join(
            f"- {s.day}: {s.description} ({s.type}, {s.duration_min}min)"
            for s in sessions
        )
        prompt_vars = {
            "goal": profile.goal,
            "bmi": health.bmi,
            "bmi_category": health.bmi_category,
            "activity_level": profile.activity_level,
            "weekly_summary": weekly_summary,
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
            return parsed.get("tips", _mock_tips(profile.goal))
        except Exception as e:
            print(f"[ExerciseAdvisor] LLM 失败：{e}")
            return _mock_tips(profile.goal)

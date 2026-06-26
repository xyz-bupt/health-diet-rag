"""
健康评估 Agent（Stage 2 版本：单 Agent，尚未编排）。

什么是 "Agent"？
----------------
在 LangChain/LangGraph 语境里，Agent 不是字面意义的"智能体"，
而是 **"LLM + 工具/规则 + 输出格式" 的封装单元**。它接收输入，
决定要不要调工具（这里：调用确定性公式），最后产出结构化结果。

本 Agent 的设计模式：**LLM-as-Interpreter**
- 数学计算交给代码（health_calc.py）
- LLM 只负责解读数字、生成自然语言摘要、给建议
- 这样保证关键数字 100% 正确，LLM 负责它擅长的事

为什么这是好的实践
------------------
让 LLM 直接算 BMR，它经常算错（小数点、单位混淆）。
而让它"解释一个 2571 kcal 的 TDEE 意味着什么"，它做得很好。
**把确定性留给代码，把模糊性留给 LLM**。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.health_calc import (
    ACTIVITY_FACTORS,
    bmi_category,
    calc_bmi,
    calc_bmr,
    calc_macros,
    calc_target_calories,
    calc_tdee,
)
from app.core.llm import get_llm, is_llm_available
from app.models.health import HealthAssessment, HealthProfile


# ---------------------------------------------------------------------------
# Prompt 模板：System 给角色，Human 给任务
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名经过认证的营养师和运动生理学家。你的任务是根据用户的健康数据，
对评估结果做**解读和建议**，而不是重新计算数字（数字已经为你算好）。

输出要求：
1. summary：用 2-3 句话总结用户的当前健康状况（基于 BMI/热量/目标），语气专业且友好。
2. recommendations：3-5 条具体、可操作的建议（饮食/运动/生活习惯），每条一句话。

注意：
- 不要给出医疗诊断，必要时建议咨询医生
- 数字部分以提供的为准，不要自己改
- 用中文回答
"""

HUMAN_TEMPLATE = """用户健康画像：
- 身高: {height_cm} cm
- 体重: {weight_kg} kg
- 年龄: {age}
- 性别: {gender}
- 活动量: {activity_level}（系数 {activity_factor}）
- 目标: {goal}

已计算结果（请基于这些数字解读，不要重新算）：
- BMI: {bmi}（{bmi_category}）
- BMR: {bmr} kcal/天
- TDEE: {tdee} kcal/天
- 每日目标热量: {target_calories} kcal
- 宏量营养素：蛋白质 {protein}g / 碳水 {carbs}g / 脂肪 {fat}g

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "summary": "...",
  "recommendations": ["...", "...", "..."]
}}
"""


# ---------------------------------------------------------------------------
# Mock 输出（LLM 不可用时兜底，保证流程能跑通）
# ---------------------------------------------------------------------------

def _mock_llm_output(profile: HealthProfile, assessment_data: dict[str, Any]) -> dict[str, Any]:
    """根据已有计算结果生成一份规则化的兜底摘要。"""
    summary = (
        f"您的 BMI 为 {assessment_data['bmi']}（{assessment_data['bmi_category']}），"
        f"基础代谢 {assessment_data['bmr']} kcal，日均消耗约 {assessment_data['tdee']} kcal。"
        f"按 {profile.goal} 目标，建议每日摄入 {assessment_data['target_calories']} kcal。"
    )
    recs = [
        f"每日蛋白质摄入约 {assessment_data['protein_g']} g，分配到三餐",
        "保持规律作息，每晚 7-8 小时睡眠有助于代谢",
        "每周至少 150 分钟中等强度运动",
        "注意补充水分，每日 1.5-2 L",
        "（MockLLM：配 DeepSeek key 后将获得个性化建议）",
    ]
    return {"summary": summary, "recommendations": recs}


# ---------------------------------------------------------------------------
# Agent 主类
# ---------------------------------------------------------------------------

class HealthAssessmentAgent:
    """健康评估 Agent。

    用法：
        agent = HealthAssessmentAgent()
        result = agent.assess(profile)
    """

    def __init__(self, llm: Any | None = None) -> None:
        # 允许外部注入 LLM（便于测试）；默认用全局工厂
        self.llm = llm or get_llm()
        self.llm_available = is_llm_available()

    def assess(self, profile: HealthProfile) -> HealthAssessment:
        """对一个用户画像做健康评估，返回结构化结果。"""
        # 步骤 1：确定性计算（永远是代码，不靠 LLM）
        bmi = calc_bmi(profile.weight_kg, profile.height_cm)
        bmr = calc_bmr(profile)
        tdee = calc_tdee(bmr, profile.activity_level)
        target = calc_target_calories(tdee, profile.goal)
        macros = calc_macros(target, profile.goal)

        assessment_data = {
            "bmi": bmi,
            "bmi_category": bmi_category(bmi),
            "bmr": bmr,
            "tdee": tdee,
            "target_calories": target,
            "protein_g": macros.protein_g,
            "carbs_g": macros.carbs_g,
            "fat_g": macros.fat_g,
        }

        # 步骤 2：LLM 解读（或 Mock 兜底）
        llm_result = self._call_llm(profile, assessment_data)

        # 步骤 3：组装最终结构化结果
        return HealthAssessment(
            bmi=bmi,
            bmi_category=bmi_category(bmi),
            bmr=bmr,
            tdee=tdee,
            target_calories=target,
            macros=macros,
            summary=llm_result["summary"],
            recommendations=llm_result["recommendations"],
            llm_used=self.llm_available,
        )

    # ----- 内部：调用 LLM 并解析输出 -----

    def _call_llm(
        self, profile: HealthProfile, data: dict[str, Any]
    ) -> dict[str, Any]:
        """调 LLM 生成解读；失败时降级到 Mock 输出（不让接口挂掉）。"""
        if not self.llm_available:
            return _mock_llm_output(profile, data)

        prompt_vars = {
            **data,
            "height_cm": profile.height_cm,
            "weight_kg": profile.weight_kg,
            "age": profile.age,
            "gender": profile.gender,
            "activity_level": profile.activity_level,
            "activity_factor": ACTIVITY_FACTORS[profile.activity_level],
            "goal": profile.goal,
            "protein": data["protein_g"],
            "carbs": data["carbs_g"],
            "fat": data["fat_g"],
        }
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=HUMAN_TEMPLATE.format(**prompt_vars)),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return _parse_json_response(content)
        except Exception as e:
            # 关键工程实践：LLM 调用失败时降级到 Mock，不让接口 500
            # 真实生产里这里会接监控/告警
            print(f"[HealthAssessmentAgent] LLM 调用失败，降级到 Mock：{e}")
            return _mock_llm_output(profile, data)


def _parse_json_response(content: str) -> dict[str, Any]:
    """从 LLM 输出里解析 JSON。

    LLM 经常会在 JSON 外面包 ```json ... ``` 或加额外文字，
    这里做容错：找到第一个 { 到最后一个 } 之间的内容。
    """
    # 去掉 markdown 代码块标记
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # 用 split 取出 fence 内的内容
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            # 去掉开头的语言标识（json / python 等）
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            elif cleaned.split("\n", 1)[0].strip().isalpha():
                # 兜底：第一行是单个语言名也去掉
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM 输出中找不到 JSON：{content[:100]}")

    json_str = cleaned[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 二次清理：有时 LLM 会输出尾随逗号或单引号
        # 用更宽松的解析（json5 风格的简化版）
        import re
        # 去掉尾随逗号（JSON5 允许但 JSON 不允许）
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        # 单引号 → 双引号
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON 解析失败: {e}; 原始内容前 200 字符: {content[:200]}"
            )

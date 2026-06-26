"""
菜谱生成 Agent（Stage 3 的 RAG 第一次真正被用上）。

职责
----
接收营养规划（三餐目标热量 + 宏量）和用户画像，生成具体的三餐食谱。

工作流
------
1. 用 RAG（小模型）检索：根据用户目标检索相关菜谱/食材
2. 把检索结果作为"参考资料"塞进 LLM 的 prompt
3. 用大模型组合出符合热量/宏量要求的三餐食谱

这就是**完整的 RAG 闭环**：检索 → 增强 → 生成。
- R: Retrieval（小模型检索）
- A: Augmented（检索结果增强 prompt）
- G: Generation（大模型生成）

为什么菜谱必须用 RAG
--------------------
- LLM 不知道你的食材库有什么
- LLM 会胡编营养数据（说"鸡胸肉每 100g 含 50g 蛋白"）
- RAG 把"事实"塞进 prompt，让 LLM 基于事实生成（大幅减少幻觉）
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.health_calc import calc_macros
from app.core.llm import get_llm, is_llm_available
from app.models.diet import Meal, MealPlan
from app.models.health import HealthProfile
from app.models.diet import NutritionPlan
from app.rag.retriever import get_retriever


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是营养菜谱设计师。根据用户的目标和营养需求，**优先使用参考资料中的食材和菜谱**设计一日三餐。

约束：
1. 每餐热量必须接近给定目标（误差 ±10%）
2. 食材优先从参考资料选，但可根据需要调整克数
3. 三餐的宏量营养素加总应接近每日目标
4. 食材具体到克数
5. 用中文输出，严格按 JSON 格式

输出 JSON 格式：
{
  "meals": [
    {
      "meal_type": "breakfast",
      "name": "餐名",
      "description": "简短描述",
      "ingredients": ["食材1 100g", "食材2 50g"],
      "calories": 数字,
      "macros": {"protein_g": 数字, "carbs_g": 数字, "fat_g": 数字}
    },
    ... (lunch, dinner, 可选 snack)
  ],
  "variety_note": "食材多样性提示"
}
"""

HUMAN_TEMPLATE = """用户画像：
- 目标：{goal}
- 性别：{gender}
- 活动量：{activity_level}

每日营养目标：
- 总热量：{daily_target} kcal
- 蛋白质 {protein}g / 碳水 {carbs}g / 脂肪 {fat}g

三餐热量目标：
- 早餐：{breakfast} kcal
- 午餐：{lunch} kcal
- 晚餐：{dinner} kcal
- 加餐：{snack} kcal

参考资料（来自知识库检索）：
{context}

请基于以上资料设计一日三餐。每餐的食材名称出现在参考资料中时，在 rag_sources 字段列出。"""

# 不同目标对应的检索查询
RECIPE_SEARCH_QUERIES = {
    "lose_weight": ["低脂高蛋白菜谱", "减脂期适合的早餐", "低热量饱腹食物"],
    "maintain": ["健康均衡菜谱", "日常营养餐", "家常健康菜"],
    "gain_muscle": ["高蛋白增肌菜谱", "训练后恢复餐", "高热量健康食物"],
}


# ---------------------------------------------------------------------------
# Mock 菜谱（无 LLM 时兜底）
# ---------------------------------------------------------------------------

def _mock_meal_plan(nutrition: NutritionPlan, profile: HealthProfile) -> MealPlan:
    """规则化兜底菜谱：固定模板 + 检索到的食材填充。"""
    # 用 RAG 检索一些食材当填充（即使没 LLM 也用上小模型）
    retriever = get_retriever()
    try:
        results = retriever.search(
            f"{profile.goal} 高蛋白食材", k=3, source_filter="ingredients"
        )
    except Exception:
        results = []

    picked_names = []
    for r in results[:3]:
        name = r.metadata.get("name", "")
        if name:
            picked_names.append(name)
    if not picked_names:
        picked_names = ["鸡胸肉", "糙米", "西兰花"]

    breakfast_macros = calc_macros(nutrition.meal_calories.breakfast, profile.goal)
    lunch_macros = calc_macros(nutrition.meal_calories.lunch, profile.goal)
    dinner_macros = calc_macros(nutrition.meal_calories.dinner, profile.goal)

    meals = [
        Meal(
            meal_type="breakfast",
            name="高蛋白燕麦杯（规则版）",
            description="希腊酸奶+燕麦+蓝莓组合，富含蛋白和纤维",
            ingredients=["希腊酸奶 200g", "燕麦 40g", "蓝莓 50g"],
            calories=nutrition.meal_calories.breakfast,
            macros=breakfast_macros,
            rag_sources=[r.metadata.get("name", "") for r in results[:2]],
        ),
        Meal(
            meal_type="lunch",
            name=f"{picked_names[0]} 杂粮饭（规则版）",
            description=f"{picked_names[0]} 搭配糙米和蔬菜",
            ingredients=[f"{picked_names[0]} 150g", "糙米 80g", "西兰花 100g"],
            calories=nutrition.meal_calories.lunch,
            macros=lunch_macros,
            rag_sources=picked_names,
        ),
        Meal(
            meal_type="dinner",
            name="轻食沙拉（规则版）",
            description="低脂高蛋白晚餐",
            ingredients=[f"{picked_names[-1]} 120g", "生菜 100g", "番茄 80g"],
            calories=nutrition.meal_calories.dinner,
            macros=dinner_macros,
            rag_sources=[picked_names[-1]],
        ),
    ]
    total = sum(m.calories for m in meals)
    return MealPlan(
        meals=meals,
        total_calories=round(total, 1),
        variety_note=f"本餐使用食材：{', '.join(picked_names)}（来自知识库）",
        llm_used=False,
    )


# ---------------------------------------------------------------------------
# Agent 主类
# ---------------------------------------------------------------------------

class RecipeAgent:
    """菜谱生成 Agent。

    用法：
        agent = RecipeAgent()
        plan = agent.generate(nutrition_plan, profile)
    """

    def __init__(self, llm: Any | None = None, retriever=None) -> None:
        self.llm = llm or get_llm()
        # 注入 retriever 便于测试；默认用全局单例
        self._retriever = retriever or get_retriever()
        self.llm_available = is_llm_available()

    def generate(
        self, nutrition: NutritionPlan, profile: HealthProfile
    ) -> MealPlan:
        """生成一日三餐食谱。"""
        # 1. RAG 检索（小模型 + ChromaDB）
        context, rag_sources = self._retrieve_context(profile.goal)

        # 2. 调 LLM 生成（或 Mock 兜底）
        if not self.llm_available:
            return _mock_meal_plan(nutrition, profile)

        # 3. 拼接 prompt 调 LLM
        meals_data = self._call_llm(nutrition, profile, context)
        if meals_data is None:
            return _mock_meal_plan(nutrition, profile)

        # 4. 解析 LLM 输出为 Meal 列表
        meals = []
        for m in meals_data.get("meals", []):
            try:
                meals.append(Meal(
                    meal_type=m["meal_type"],
                    name=m["name"],
                    description=m.get("description", ""),
                    ingredients=m.get("ingredients", []),
                    calories=float(m["calories"]),
                    macros={  # 兼容大小写键
                        "protein_g": float(m["macros"].get("protein_g", 0)),
                        "carbs_g": float(m["macros"].get("carbs_g", 0)),
                        "fat_g": float(m["macros"].get("fat_g", 0)),
                    },
                    rag_sources=rag_sources,
                ))
            except (KeyError, ValueError) as e:
                print(f"[RecipeAgent] 跳过格式错误的 meal：{e}")

        if not meals:
            return _mock_meal_plan(nutrition, profile)

        total = sum(m.calories for m in meals)
        return MealPlan(
            meals=meals,
            total_calories=round(total, 1),
            variety_note=meals_data.get("variety_note", ""),
            llm_used=True,
        )

    # ----- 内部：RAG 检索 -----

    def _retrieve_context(self, goal: str) -> tuple[str, list[str]]:
        """用 RAG 检索相关菜谱和食材，拼接成 prompt 上下文。"""
        queries = RECIPE_SEARCH_QUERIES.get(goal, RECIPE_SEARCH_QUERIES["maintain"])
        all_results = []
        for q in queries:
            try:
                results = self._retriever.search(q, k=2)
                all_results.extend(results)
            except Exception as e:
                print(f"[RecipeAgent] RAG 检索失败 '{q}'：{e}")

        if not all_results:
            return "（知识库为空或未建索引）", []

        # 去重
        seen = set()
        unique = []
        for r in all_results:
            key = r.document[:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # 拼接上下文（控制总长度，避免 prompt 过长）
        context_parts = []
        sources = []
        for i, r in enumerate(unique[:6], 1):  # 取前 6 条
            name = r.metadata.get("name", r.metadata.get("title", f"片段{i}"))
            sources.append(name)
            # 截断每条到 200 字符，避免 prompt 过长
            snippet = r.document[:200].replace("\n", " ")
            context_parts.append(f"[{i}] {name}：{snippet}")

        return "\n\n".join(context_parts), sources

    # ----- 内部：调 LLM -----

    def _call_llm(
        self, nutrition: NutritionPlan, profile: HealthProfile, context: str
    ) -> dict[str, Any] | None:
        prompt_vars = {
            "goal": profile.goal,
            "gender": profile.gender,
            "activity_level": profile.activity_level,
            "daily_target": nutrition.daily_target,
            "protein": nutrition.macros_daily.protein_g,
            "carbs": nutrition.macros_daily.carbs_g,
            "fat": nutrition.macros_daily.fat_g,
            "breakfast": nutrition.meal_calories.breakfast,
            "lunch": nutrition.meal_calories.lunch,
            "dinner": nutrition.meal_calories.dinner,
            "snack": nutrition.meal_calories.snack,
            "context": context,
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
            print(f"[RecipeAgent] LLM 失败：{e}")
            return None

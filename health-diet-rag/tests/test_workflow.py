"""
Stage 4 多 Agent 工作流测试。

策略：
- 各 Agent 单独测：确定性逻辑必须正确
- 工作流端到端测：5 个 Agent 串联不出错
- Mock LLM 路径全跑通（不依赖 API key）
"""

import pytest

from app.agents.exercise import (
    MET_VALUES,
    ExerciseAdvisorAgent,
    WEEKLY_TEMPLATES,
    calc_calories_burned,
)
from app.agents.health import HealthAssessmentAgent
from app.agents.nutrition import (
    MEAL_DISTRIBUTION,
    NutritionPlannerAgent,
    calc_hydration,
    calc_meal_calories,
)
from app.agents.recipe import RecipeAgent
from app.agents.supervisor import SupervisorAgent
from app.graph.workflow import build_diet_plan_graph, run_diet_plan, stream_diet_plan
from app.models.health import HealthProfile


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def profile(**kwargs) -> HealthProfile:
    defaults = dict(
        height_cm=175, weight_kg=70, age=28,
        gender="male", activity_level="moderate", goal="maintain",
    )
    defaults.update(kwargs)
    return HealthProfile(**defaults)


# ---------------------------------------------------------------------------
# 营养规划：纯计算
# ---------------------------------------------------------------------------

class TestNutritionCalc:
    def test_meal_distribution_lose_weight(self):
        cal = calc_meal_calories(2000, "lose_weight")
        b, l, d, s = cal.breakfast, cal.lunch, cal.dinner, cal.snack
        # 减脂：30/35/25/10
        assert abs(b - 600) < 1
        assert abs(l - 700) < 1
        assert abs(d - 500) < 1
        assert abs(s - 200) < 1
        # 加总应等于每日总热量
        assert abs(b + l + d + s - 2000) < 2

    def test_meal_distribution_maintain(self):
        cal = calc_meal_calories(2000, "maintain")
        b, l, d, s = cal.breakfast, cal.lunch, cal.dinner, cal.snack
        # 维持：30/40/25/5
        assert abs(b + l + d + s - 2000) < 2

    def test_meal_distribution_gain_muscle(self):
        cal = calc_meal_calories(2000, "gain_muscle")
        # 增肌加餐比例更高
        assert cal.snack > calc_meal_calories(2000, "maintain").snack

    def test_hydration_rounds_to_100(self):
        # 70kg × 35 = 2450 → 取整到 2400
        assert calc_hydration(70) == 2400
        # 60kg × 35 = 2100
        assert calc_hydration(60) == 2100
        # 65kg × 35 = 2275 → 2200
        assert calc_hydration(65) == 2200

    def test_all_goals_have_distribution(self):
        for goal in ["lose_weight", "maintain", "gain_muscle"]:
            assert goal in MEAL_DISTRIBUTION


# ---------------------------------------------------------------------------
# 营养规划 Agent
# ---------------------------------------------------------------------------

class TestNutritionPlannerAgent:
    def test_plan_returns_complete_structure(self):
        p = profile()
        h = HealthAssessmentAgent().assess(p)
        plan = NutritionPlannerAgent().plan(h, p)
        assert plan.daily_target > 0
        assert plan.macros_daily.protein_g > 0
        assert plan.meal_calories.breakfast > 0
        assert plan.hydration_ml > 0
        assert len(plan.timing_tips) >= 3
        assert isinstance(plan.llm_used, bool)


# ---------------------------------------------------------------------------
# 菜谱 Agent（依赖 RAG，需要建索引）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensure_index():
    """确保 RAG 索引已建（模块级共享）。"""
    from app.rag.indexer import get_indexer
    indexer = get_indexer()
    if not indexer.is_indexed():
        indexer.index_all()
    return indexer


class TestRecipeAgent:
    def test_generate_returns_three_meals(self, ensure_index):
        p = profile(goal="lose_weight")
        h = HealthAssessmentAgent().assess(p)
        nutrition = NutritionPlannerAgent().plan(h, p)
        plan = RecipeAgent().generate(nutrition, p)
        assert len(plan.meals) >= 3  # 至少三餐
        types = {m.meal_type for m in plan.meals}
        assert "breakfast" in types
        assert "lunch" in types
        assert "dinner" in types
        assert plan.total_calories > 0
        # 每餐都有热量和宏量
        for meal in plan.meals:
            assert meal.calories > 0
            assert meal.macros.protein_g >= 0
            assert len(meal.ingredients) > 0

    def test_recipe_rag_sources_populated(self, ensure_index):
        """菜谱应携带 RAG 召回的食材名（追溯性）。"""
        p = profile(goal="maintain")
        h = HealthAssessmentAgent().assess(p)
        nutrition = NutritionPlannerAgent().plan(h, p)
        plan = RecipeAgent().generate(nutrition, p)
        all_sources = [s for m in plan.meals for s in m.rag_sources]
        # Mock 模式下也会有 RAG 检索结果
        assert len(all_sources) >= 1


# ---------------------------------------------------------------------------
# 运动 Agent：纯计算
# ---------------------------------------------------------------------------

class TestExerciseCalc:
    def test_met_values_completeness(self):
        # 所有模板里用到的运动类型都能映射到 MET
        for goal, template in WEEKLY_TEMPLATES.items():
            for _, ex_type, _, _, _ in template:
                # rest 不需要 MET
                if ex_type == "rest":
                    continue
                # 检查 met_key_map 能处理
                met_key_map = {
                    "cardio": "jogging",
                    "strength": "strength_training",
                    "flexibility": "yoga",
                }
                assert met_key_map.get(ex_type, "walking") in MET_VALUES

    def test_calories_burned_formula(self):
        # 70kg 人 慢跑（MET=7）30 分钟：7 × 70 × 0.5 = 245
        assert calc_calories_burned(7.0, 70, 30) == 245.0

    def test_weekly_template_completeness(self):
        for goal in ["lose_weight", "maintain", "gain_muscle"]:
            assert goal in WEEKLY_TEMPLATES
            assert len(WEEKLY_TEMPLATES[goal]) == 7  # 一周 7 天


class TestExerciseAgent:
    def test_advise_returns_seven_sessions(self):
        p = profile()
        h = HealthAssessmentAgent().assess(p)
        plan = ExerciseAdvisorAgent().advise(h, p)
        assert len(plan.weekly_sessions) == 7
        assert plan.weekly_calories_burned > 0
        assert len(plan.tips) >= 3
        # 至少有一个休息日
        rest_days = [s for s in plan.weekly_sessions if s.type == "rest"]
        assert len(rest_days) >= 1


# ---------------------------------------------------------------------------
# Supervisor Agent
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_integrate_returns_diet_plan(self):
        p = profile()
        h = HealthAssessmentAgent().assess(p)
        nutrition = NutritionPlannerAgent().plan(h, p)
        recipe = RecipeAgent().generate(nutrition, p)
        exercise = ExerciseAdvisorAgent().advise(h, p)
        final = SupervisorAgent().integrate(p, h, nutrition, recipe, exercise)

        assert final.profile == p
        assert final.health.bmi > 0
        assert final.nutrition.daily_target > 0
        assert len(final.recipe.meals) >= 3
        assert len(final.exercise.weekly_sessions) == 7
        assert len(final.summary) > 50
        assert len(final.key_actions) >= 3


# ---------------------------------------------------------------------------
# LangGraph 工作流端到端
# ---------------------------------------------------------------------------

class TestWorkflow:
    def test_workflow_returns_all_fields(self, ensure_index):
        p = profile(goal="lose_weight")
        state = run_diet_plan(p)
        # 所有字段都被填充
        for key in ["profile", "health", "nutrition", "recipe", "exercise", "final_plan"]:
            assert key in state, f"缺少字段 {key}"
        # 没有错误
        assert state.get("errors", []) == []

    def test_workflow_final_plan_is_complete(self, ensure_index):
        p = profile(goal="gain_muscle")
        state = run_diet_plan(p)
        fp = state["final_plan"]
        assert fp.profile == p
        assert fp.health.tdee > 0
        assert len(fp.recipe.meals) >= 3
        assert len(fp.summary) > 50

    def test_workflow_streaming_yields_each_node(self, ensure_index):
        p = profile()
        events = list(stream_diet_plan(p))
        # 应该有 5 个事件（5 个节点）
        node_names = [list(e.keys())[0] for e in events]
        assert "health_node" in node_names
        assert "supervisor_node" in node_names
        # supervisor 最后产出
        last_event = events[-1]
        assert "final_plan" in last_event["supervisor_node"]

    def test_workflow_handles_different_goals(self, ensure_index):
        """3 种目标都能跑通。"""
        for goal in ["lose_weight", "maintain", "gain_muscle"]:
            p = profile(goal=goal)
            state = run_diet_plan(p)
            assert "final_plan" in state
            assert state.get("errors", []) == []


# ---------------------------------------------------------------------------
# API 接口测试
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestDietPlanAPI:
    def test_diet_plan_endpoint_success(self):
        # 先建索引（保证 RAG 可用）
        client.post("/api/v1/index")
        resp = client.post("/api/v1/diet-plan", json={
            "height_cm": 175, "weight_kg": 70, "age": 28,
            "gender": "male", "activity_level": "moderate", "goal": "maintain",
        })
        assert resp.status_code == 200
        data = resp.json()
        # 完整字段
        assert "health" in data
        assert "nutrition" in data
        assert "recipe" in data
        assert "exercise" in data
        assert "summary" in data
        assert "key_actions" in data

    def test_diet_plan_endpoint_rejects_invalid_input(self):
        resp = client.post("/api/v1/diet-plan", json={
            "height_cm": 50,  # 非法
            "weight_kg": 70, "age": 28, "gender": "male",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Stage 1-3 回归测试
# ---------------------------------------------------------------------------

class TestRegression:
    def test_stage1_health_still_works(self):
        assert client.get("/health").status_code == 200

    def test_stage2_assess_still_works(self):
        resp = client.post("/api/v1/assess", json={
            "height_cm": 175, "weight_kg": 70, "age": 28, "gender": "male",
        })
        assert resp.status_code == 200

    def test_stage3_search_still_works(self):
        resp = client.get("/api/v1/foods/search", params={"q": "鸡胸肉", "k": 2})
        assert resp.status_code == 200

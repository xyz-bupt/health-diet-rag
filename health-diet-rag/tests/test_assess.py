"""
健康评估 Agent + /api/v1/assess 接口的集成测试。

策略：
- 计算：精确断言
- LLM 摘要：在没有真实 key 时用 MockLLM，断言结构而非具体措辞
"""

import pytest
from fastapi.testclient import TestClient

from app.agents.health import HealthAssessmentAgent, _parse_json_response
from app.core.llm import MockLLM, is_llm_available, get_llm
from app.main import app
from app.models.health import HealthProfile

client = TestClient(app)


# ---------------------------------------------------------------------------
# Agent 测试
# ---------------------------------------------------------------------------

def test_agent_returns_complete_assessment():
    p = HealthProfile(
        height_cm=175, weight_kg=70, age=28,
        gender="male", activity_level="moderate", goal="maintain",
    )
    agent = HealthAssessmentAgent()
    result = agent.assess(p)

    # 所有关键字段都被填充
    assert result.bmi > 0
    assert result.bmr > 0
    assert result.tdee > result.bmr  # TDEE 必大于 BMR
    assert result.target_calories > 0
    assert result.macros.protein_g > 0
    assert result.macros.carbs_g > 0
    assert result.macros.fat_g > 0
    assert len(result.summary) > 10
    assert len(result.recommendations) >= 1


def test_agent_lose_weight_creates_deficit():
    p = HealthProfile(
        height_cm=170, weight_kg=80, age=30,
        gender="female", activity_level="light", goal="lose_weight",
    )
    agent = HealthAssessmentAgent()
    result = agent.assess(p)
    # 减脂目标下，目标热量应该 < TDEE
    assert result.target_calories < result.tdee
    # 缺口比例约为 20%
    ratio = result.target_calories / result.tdee
    assert 0.75 < ratio < 0.85


def test_agent_gain_muscle_creates_surplus():
    p = HealthProfile(
        height_cm=180, weight_kg=75, age=25,
        gender="male", activity_level="active", goal="gain_muscle",
    )
    agent = HealthAssessmentAgent()
    result = agent.assess(p)
    assert result.target_calories > result.tdee


def test_agent_llm_used_flag_reflects_availability():
    """MockLLM 时 llm_used=False；真实 LLM 时 llm_used=True（仅在有 key 时验证）。"""
    agent = HealthAssessmentAgent()
    p = HealthProfile(
        height_cm=175, weight_kg=70, age=28,
        gender="male", activity_level="moderate", goal="maintain",
    )
    result = agent.assess(p)
    assert result.llm_used == is_llm_available()


# ---------------------------------------------------------------------------
# JSON 解析容错
# ---------------------------------------------------------------------------

def test_parse_json_plain():
    out = _parse_json_response('{"summary": "hi", "recommendations": ["a"]}')
    assert out["summary"] == "hi"


def test_parse_json_with_markdown_fence():
    content = '```json\n{"summary": "hi", "recommendations": ["a", "b"]}\n```'
    out = _parse_json_response(content)
    assert out["recommendations"] == ["a", "b"]


def test_parse_json_with_surrounding_text():
    content = '好的，以下是结果：\n{"summary": "hi", "recommendations": ["a"]}\n希望对你有帮助。'
    out = _parse_json_response(content)
    assert out["summary"] == "hi"


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError):
        _parse_json_response("no json here")


# ---------------------------------------------------------------------------
# API 接口测试
# ---------------------------------------------------------------------------

def test_assess_endpoint_success():
    resp = client.post("/api/v1/assess", json={
        "height_cm": 175, "weight_kg": 70, "age": 28,
        "gender": "male", "activity_level": "moderate", "goal": "maintain",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["bmi"] == 22.9
    assert data["bmi_category"] == "正常"
    assert data["bmr"] == 1658.8
    assert data["tdee"] == 2571.1
    assert data["target_calories"] == 2571.1
    assert "protein_g" in data["macros"]
    assert "summary" in data
    assert "recommendations" in data


def test_assess_endpoint_rejects_invalid_height():
    resp = client.post("/api/v1/assess", json={
        "height_cm": 50,  # 太矮
        "weight_kg": 70, "age": 28, "gender": "male",
    })
    assert resp.status_code == 422


def test_assess_endpoint_rejects_missing_required():
    resp = client.post("/api/v1/assess", json={"height_cm": 175})
    assert resp.status_code == 422


def test_assess_endpoint_rejects_invalid_gender():
    resp = client.post("/api/v1/assess", json={
        "height_cm": 175, "weight_kg": 70, "age": 28,
        "gender": "alien",
    })
    assert resp.status_code == 422


def test_assess_endpoint_uses_defaults():
    """不传可选字段（activity_level, goal）时用默认值，不报错。"""
    resp = client.post("/api/v1/assess", json={
        "height_cm": 175, "weight_kg": 70, "age": 28, "gender": "male",
    })
    assert resp.status_code == 200
    data = resp.json()
    # 默认 moderate × maintain → target = tdee
    assert data["target_calories"] == data["tdee"]


def test_assess_endpoint_female_profile():
    resp = client.post("/api/v1/assess", json={
        "height_cm": 165, "weight_kg": 60,
        "age": 25, "gender": "female",
        "activity_level": "light", "goal": "lose_weight",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_calories"] < data["tdee"]


# ---------------------------------------------------------------------------
# Stage 1 路由回归（确保新增代码没破坏老接口）
# ---------------------------------------------------------------------------

def test_stage1_health_still_works():
    """健康检查路由应仍可访问。"""
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200

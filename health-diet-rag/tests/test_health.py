"""
健康检查端点的测试。

用 FastAPI 的 TestClient（基于 httpx），可以不发真实 HTTP 请求测试路由。
运行：pytest tests/test_health.py -v
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    """根路径应返回前端 HTML 页面（Stage 5 起改为返回前端）。

    如果 static/index.html 不存在，则回退到 JSON。
    """
    response = client.get("/")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        # 返回前端页面
        assert "健康饮食" in response.text
    else:
        # 回退：JSON 接口
        data = response.json()
        assert "message" in data
        assert "version" in data


def test_health_check():
    """/health 应返回 healthy 状态。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "env" in data

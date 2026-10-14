"""冒烟测试：验证 CareerPilot 服务能启动且健康检查接口可用。"""
from fastapi.testclient import TestClient

from main import app


def test_health():
    """/api/careerpilot/health 应返回 200 且 status=ok。"""
    r = TestClient(app).get("/api/careerpilot/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
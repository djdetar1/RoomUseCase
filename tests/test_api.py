from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_requires_fields():
    # Missing numeric fields -> 422 from pydantic validation.
    resp = client.post("/predict", json={"date": "2015-02-04 17:51:00"})
    assert resp.status_code == 422

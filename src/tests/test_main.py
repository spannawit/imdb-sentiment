from fastapi.testclient import TestClient
from apps.main import app
from unittest.mock import patch


client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    sample_review = {"review": "This movie was absolutely fantastic!"}
    response = client.post("/predict", json=sample_review)
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()
    assert isinstance(response.json()["confidence"], float)

def test_predict_invalid():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # validation error

def test_predict_empty_review():
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 200  # model should still return a prediction
    assert "label" in response.json()
    assert "confidence" in response.json()

def test_predict_internal_error():
    with patch("apps.main.predict", side_effect=ValueError("Forced error")):
        response = client.post("/predict", json={"review": "This will crash"})
        assert response.status_code == 500
        assert "Forced error" in response.json()["detail"]

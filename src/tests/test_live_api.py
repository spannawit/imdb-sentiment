import requests

BASE_URL = "http://localhost:8000"

def test_live_health_check():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_live_predict():
    payload = {"review": "I really loved this film, it was great!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

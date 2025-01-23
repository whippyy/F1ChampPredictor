# app/tests/test_predict.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predictions/predict", json={"driver_id": 1, "team_id": 1, "race_id": 1, "historical_data": [1.5, 2.0]})
    assert response.status_code == 200
    assert "prediction" in response.json()


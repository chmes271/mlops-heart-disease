from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_endpoint():
    payload = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 130, 
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 140, 
        "exang": 0, "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
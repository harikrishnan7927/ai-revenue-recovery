from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "monthly_revenue": 1000,
            "tenure_months": 12,
            "login_frequency": 15,
            "payment_delay_days": 5,
            "support_tickets": 2,
            "usage_score": 70,
            "discount_used": 0
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "revenue_at_risk" in result
    assert "risk_level" in result
    assert "recovery_action" in result

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert result["revenue_at_risk"] >= 0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_invalid_prediction_request():
    response = client.post(
        "/predict",
        json={
            "monthly_revenue": 1000
        }
    )

    assert response.status_code == 422
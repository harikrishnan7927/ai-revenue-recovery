import os
import joblib
import pandas as pd


def test_dataset_exists():
    assert os.path.exists("data/customer_revenue_data_large.csv")


def test_model_exists():
    assert os.path.exists("models/churn_model_large.pkl")


def test_dataset_columns():
    df = pd.read_csv("data/customer_revenue_data_large.csv")

    required_columns = [
        "customer_id",
        "monthly_revenue",
        "tenure_months",
        "login_frequency",
        "payment_delay_days",
        "support_tickets",
        "usage_score",
        "discount_used",
        "churn_risk"
    ]

    for column in required_columns:
        assert column in df.columns


def test_dataset_size():
    df = pd.read_csv("data/customer_revenue_data_large.csv")
    assert len(df) == 1000


def test_model_loads():
    model = joblib.load("models/churn_model_large.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("models/churn_model_large.pkl")

    sample = pd.DataFrame([{
        "monthly_revenue": 1000,
        "tenure_months": 12,
        "login_frequency": 15,
        "payment_delay_days": 5,
        "support_tickets": 2,
        "usage_score": 70,
        "discount_used": 0
    }])

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]
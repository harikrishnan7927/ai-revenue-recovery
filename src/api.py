from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="AI Revenue Recovery API",
    version="1.0.0",
    description="API for customer churn prediction and revenue risk analysis"
)

MODEL_PATH = "models/churn_model_large.pkl"
model = joblib.load(MODEL_PATH)


FEATURES = [
    "monthly_revenue",
    "tenure_months",
    "login_frequency",
    "payment_delay_days",
    "support_tickets",
    "usage_score",
    "discount_used"
]


class CustomerData(BaseModel):
    monthly_revenue: float
    tenure_months: int
    login_frequency: float
    payment_delay_days: float
    support_tickets: int
    usage_score: float
    discount_used: int


@app.get("/")
def home():
    return {
        "message": "AI Revenue Recovery API is running",
        "status": "success"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    customer_data = {
        "monthly_revenue": customer.monthly_revenue,
        "tenure_months": customer.tenure_months,
        "login_frequency": customer.login_frequency,
        "payment_delay_days": customer.payment_delay_days,
        "support_tickets": customer.support_tickets,
        "usage_score": customer.usage_score,
        "discount_used": customer.discount_used
    }

    data = pd.DataFrame([customer_data], columns=FEATURES)

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    revenue_at_risk = customer.monthly_revenue * probability

    if probability >= 0.70:
        risk_level = "HIGH"
        recovery_action = "Payment recovery campaign"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
        recovery_action = "Customer re-engagement campaign"
    else:
        risk_level = "LOW"
        recovery_action = "No immediate action"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "risk_level": risk_level,
        "recovery_action": recovery_action
    }
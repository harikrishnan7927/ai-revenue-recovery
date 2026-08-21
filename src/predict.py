import pandas as pd
import joblib


# Load the trained model
model = joblib.load("models/churn_model.pkl")


# Example customer
customer = pd.DataFrame([{
    "monthly_revenue": 500,
    "tenure_months": 8,
    "login_frequency": 7,
    "payment_delay_days": 18,
    "support_tickets": 5,
    "usage_score": 40,
    "discount_used": 1
}])


# Predict churn risk
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]


# Display result
monthly_revenue = customer["monthly_revenue"].iloc[0]
estimated_revenue_at_risk = monthly_revenue * probability

print("Churn Risk Prediction:", prediction)
print("Churn Probability:", round(probability * 100, 2), "%")
print("Estimated Monthly Revenue at Risk:", round(estimated_revenue_at_risk, 2))


if prediction == 1:
    print("Risk Level: HIGH")
    print("Recommended Action: Customer retention/recovery campaign")
else:
    print("Risk Level: LOW")
    print("Recommended Action: Continue normal customer engagement")
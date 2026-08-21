import pandas as pd
import joblib


# Load dataset and trained model
data = pd.read_csv("data/customer_revenue_data.csv")
model = joblib.load("models/churn_model.pkl")


# Features used by the model
features = [
    "monthly_revenue",
    "tenure_months",
    "login_frequency",
    "payment_delay_days",
    "support_tickets",
    "usage_score",
    "discount_used"
]


# Predict churn probability for every customer
data["churn_probability"] = model.predict_proba(
    data[features]
)[:, 1]


# Calculate estimated revenue at risk
data["revenue_at_risk"] = (
    data["monthly_revenue"] * data["churn_probability"]
)


# Determine risk level
def get_risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


data["risk_level"] = data["churn_probability"].apply(
    get_risk_level
)


# Determine recommended recovery action
def get_recovery_action(row):

    if row["risk_level"] == "HIGH":

        # Customers with serious payment delays
        if row["payment_delay_days"] >= 15:
            return "Payment recovery campaign"

        # Customers with very low product usage
        elif row["usage_score"] < 50:
            return "Customer re-engagement campaign"

        # Customers requiring significant support
        elif row["support_tickets"] >= 5:
            return "Priority customer support"

        # Other high-risk customers
        else:
            return "Retention offer"

    elif row["risk_level"] == "MEDIUM":
        return "Engagement reminder"

    else:
        return "No immediate action"


data["recovery_action"] = data.apply(
    get_recovery_action,
    axis=1
)


# Sort customers by highest revenue at risk
data = data.sort_values(
    by="revenue_at_risk",
    ascending=False
)


# Display report
print("\n===== AI REVENUE RECOVERY REPORT =====\n")

print(
    data[
        [
            "customer_id",
            "monthly_revenue",
            "churn_probability",
            "revenue_at_risk",
            "risk_level",
            "recovery_action"
        ]
    ].to_string(index=False)
)


# Calculate summary statistics
total_revenue_at_risk = data["revenue_at_risk"].sum()

high_risk_count = (
    data["risk_level"] == "HIGH"
).sum()

medium_risk_count = (
    data["risk_level"] == "MEDIUM"
).sum()

low_risk_count = (
    data["risk_level"] == "LOW"
).sum()


# Display summary
print("\n======================================")
print("AI REVENUE RECOVERY SUMMARY")
print("======================================")

print("Total customers:", len(data))
print("High-risk customers:", high_risk_count)
print("Medium-risk customers:", medium_risk_count)
print("Low-risk customers:", low_risk_count)

print(
    "Total estimated revenue at risk:",
    round(total_revenue_at_risk, 2)
)

print("======================================")
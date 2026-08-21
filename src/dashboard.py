import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💰",
    layout="wide"
)


# Load dataset
data = pd.read_csv("data/customer_revenue_data.csv")

# Load trained model
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


# Predict churn probability
data["churn_probability"] = model.predict_proba(
    data[features]
)[:, 1]


# Calculate revenue at risk
data["revenue_at_risk"] = (
    data["monthly_revenue"] *
    data["churn_probability"]
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


# Determine recovery action
def get_recovery_action(row):

    if row["risk_level"] == "HIGH":

        if row["payment_delay_days"] >= 15:
            return "Payment recovery campaign"

        elif row["usage_score"] < 50:
            return "Customer re-engagement campaign"

        elif row["support_tickets"] >= 5:
            return "Priority customer support"

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


# Sort customers by revenue at risk
data = data.sort_values(
    by="revenue_at_risk",
    ascending=False
)


# Dashboard title
st.title("AI Revenue Recovery System")

st.write(
    "Machine Learning powered customer churn "
    "and revenue risk analysis."
)


# Calculate dashboard metrics
total_customers = len(data)

high_risk = (
    data["risk_level"] == "HIGH"
).sum()

medium_risk = (
    data["risk_level"] == "MEDIUM"
).sum()

low_risk = (
    data["risk_level"] == "LOW"
).sum()

total_revenue_at_risk = data["revenue_at_risk"].sum()


# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "High Risk",
        high_risk
    )

with col3:
    st.metric(
        "Medium Risk",
        medium_risk
    )

with col4:
    st.metric(
        "Revenue at Risk",
        f"₹{total_revenue_at_risk:,.2f}"
    )


# Risk distribution
st.subheader("Customer Risk Distribution")

risk_counts = data["risk_level"].value_counts()

st.bar_chart(risk_counts)


# Customer risk table
st.subheader("Customer Revenue Risk")

display_data = data[
    [
        "customer_id",
        "monthly_revenue",
        "churn_probability",
        "revenue_at_risk",
        "risk_level",
        "recovery_action"
    ]
].copy()


# Convert probability to percentage
display_data["churn_probability"] = (
    display_data["churn_probability"] * 100
).round(2)


# Round revenue
display_data["revenue_at_risk"] = (
    display_data["revenue_at_risk"]
).round(2)


# Rename columns
display_data = display_data.rename(
    columns={
        "customer_id": "Customer",
        "monthly_revenue": "Monthly Revenue",
        "churn_probability": "Churn Probability (%)",
        "revenue_at_risk": "Revenue at Risk",
        "risk_level": "Risk Level",
        "recovery_action": "Recovery Action"
    }
)


# Display complete table
st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# Priority customers
st.subheader("Priority Recovery Customers")

high_risk_customers = display_data[
    display_data["Risk Level"] == "HIGH"
]


st.dataframe(
    high_risk_customers,
    use_container_width=True,
    hide_index=True
)


# Footer
st.divider()

st.caption(
    "AI Revenue Recovery | "
    "Machine Learning + Business Risk Analytics"
)
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# Load dataset and trained model
# --------------------------------------------------

data = pd.read_csv("data/customer_revenue_data.csv")

model = joblib.load("models/churn_model.pkl")


# --------------------------------------------------
# Model features
# --------------------------------------------------

features = [
    "monthly_revenue",
    "tenure_months",
    "login_frequency",
    "payment_delay_days",
    "support_tickets",
    "usage_score",
    "discount_used"
]


# --------------------------------------------------
# Functions
# --------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.70:
        return "HIGH"

    elif probability >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


def get_recovery_action(
    risk_level,
    payment_delay_days,
    usage_score,
    support_tickets
):

    if risk_level == "HIGH":

        if payment_delay_days >= 15:
            return "Payment recovery campaign"

        elif usage_score < 50:
            return "Customer re-engagement campaign"

        elif support_tickets >= 5:
            return "Priority customer support"

        else:
            return "Retention offer"

    elif risk_level == "MEDIUM":
        return "Engagement reminder"

    else:
        return "No immediate action"


# --------------------------------------------------
# Existing customer predictions
# --------------------------------------------------

data["churn_probability"] = model.predict_proba(
    data[features]
)[:, 1]


data["revenue_at_risk"] = (
    data["monthly_revenue"]
    * data["churn_probability"]
)


data["risk_level"] = data["churn_probability"].apply(
    get_risk_level
)


data["recovery_action"] = data.apply(
    lambda row: get_recovery_action(
        row["risk_level"],
        row["payment_delay_days"],
        row["usage_score"],
        row["support_tickets"]
    ),
    axis=1
)


data = data.sort_values(
    by="revenue_at_risk",
    ascending=False
)


# --------------------------------------------------
# Dashboard title
# --------------------------------------------------

st.title("💰 AI Revenue Recovery System")

st.write(
    "Machine Learning powered customer churn "
    "and revenue risk analysis."
)


# --------------------------------------------------
# Dashboard metrics
# --------------------------------------------------

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


# --------------------------------------------------
# Risk distribution
# --------------------------------------------------

st.subheader("Customer Risk Distribution")

risk_counts = data["risk_level"].value_counts()

st.bar_chart(risk_counts)


# --------------------------------------------------
# Customer revenue risk table
# --------------------------------------------------

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


display_data["churn_probability"] = (
    display_data["churn_probability"] * 100
).round(2)


display_data["revenue_at_risk"] = (
    display_data["revenue_at_risk"]
).round(2)


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


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Priority recovery customers
# --------------------------------------------------

st.subheader("🚨 Priority Recovery Customers")


high_risk_customers = display_data[
    display_data["Risk Level"] == "HIGH"
]


st.dataframe(
    high_risk_customers,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

st.subheader("🔍 ML Feature Importance")


importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


st.bar_chart(
    importance.set_index("Feature")["Importance"]
)


st.dataframe(
    importance,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Business interpretation
# --------------------------------------------------

st.subheader("💡 Business Interpretation")


top_feature = importance.iloc[0]["Feature"]


st.write(
    f"The model identifies **{top_feature}** as the "
    "most influential feature among the available "
    "customer attributes."
)


# --------------------------------------------------
# NEW CUSTOMER PREDICTION
# --------------------------------------------------

st.divider()

st.header("🧑‍💼 New Customer Churn Prediction")

st.write(
    "Enter customer information below to estimate "
    "churn risk and potential revenue loss."
)


with st.form("customer_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        monthly_revenue = st.number_input(
            "Monthly Revenue (₹)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        tenure_months = st.number_input(
            "Tenure (Months)",
            min_value=0,
            value=12,
            step=1
        )

        login_frequency = st.number_input(
            "Login Frequency",
            min_value=0,
            value=10,
            step=1
        )

        payment_delay_days = st.number_input(
            "Payment Delay (Days)",
            min_value=0,
            value=5,
            step=1
        )

    with col2:

        support_tickets = st.number_input(
            "Support Tickets",
            min_value=0,
            value=2,
            step=1
        )

        usage_score = st.slider(
            "Usage Score",
            min_value=0,
            max_value=100,
            value=70
        )

        discount_used = st.selectbox(
            "Discount Used",
            options=[0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    predict_button = st.form_submit_button(
        "🔮 Predict Churn Risk"
    )


# --------------------------------------------------
# New customer prediction result
# --------------------------------------------------

if predict_button:

    new_customer = pd.DataFrame([{
        "monthly_revenue": monthly_revenue,
        "tenure_months": tenure_months,
        "login_frequency": login_frequency,
        "payment_delay_days": payment_delay_days,
        "support_tickets": support_tickets,
        "usage_score": usage_score,
        "discount_used": discount_used
    }])


    probability = model.predict_proba(
        new_customer[features]
    )[0][1]


    prediction = model.predict(
        new_customer[features]
    )[0]


    risk_level = get_risk_level(
        probability
    )


    revenue_at_risk = (
        monthly_revenue * probability
    )


    recovery_action = get_recovery_action(
        risk_level,
        payment_delay_days,
        usage_score,
        support_tickets
    )


    st.subheader("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Estimated Revenue at Risk",
            f"₹{revenue_at_risk:,.2f}"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    st.success(
        f"Recommended Action: {recovery_action}"
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Revenue Recovery | "
    "Machine Learning + Business Risk Analytics"
)
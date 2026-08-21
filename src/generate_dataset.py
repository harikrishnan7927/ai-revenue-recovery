import numpy as np
import pandas as pd


# --------------------------------------------------
# Settings
# --------------------------------------------------

np.random.seed(42)

NUMBER_OF_CUSTOMERS = 1000


# --------------------------------------------------
# Generate customer IDs
# --------------------------------------------------

customer_ids = [
    f"C{i:04d}"
    for i in range(1, NUMBER_OF_CUSTOMERS + 1)
]


# --------------------------------------------------
# Generate customer features
# --------------------------------------------------

monthly_revenue = np.random.randint(
    200,
    3001,
    NUMBER_OF_CUSTOMERS
)

tenure_months = np.random.randint(
    1,
    61,
    NUMBER_OF_CUSTOMERS
)

login_frequency = np.random.randint(
    1,
    41,
    NUMBER_OF_CUSTOMERS
)

payment_delay_days = np.random.randint(
    0,
    31,
    NUMBER_OF_CUSTOMERS
)

support_tickets = np.random.randint(
    0,
    11,
    NUMBER_OF_CUSTOMERS
)

usage_score = np.random.randint(
    20,
    101,
    NUMBER_OF_CUSTOMERS
)

discount_used = np.random.choice(
    [0, 1],
    size=NUMBER_OF_CUSTOMERS,
    p=[0.65, 0.35]
)


# --------------------------------------------------
# Create churn score
# --------------------------------------------------

churn_score = (
    0.10 * payment_delay_days
    + 0.15 * support_tickets
    - 0.06 * usage_score
    - 0.05 * login_frequency
    - 0.025 * tenure_months
    + 0.00005 * monthly_revenue
    + 0.15 * discount_used
)


# Add random variation
churn_score += np.random.normal(
    0,
    0.80,
    NUMBER_OF_CUSTOMERS
)


# --------------------------------------------------
# Convert score to probability
# --------------------------------------------------

churn_probability = (
    1 / (1 + np.exp(-churn_score))
)


# --------------------------------------------------
# Generate churn labels
# --------------------------------------------------

churn_risk = (
    churn_probability >= 0.50
).astype(int)


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

data = pd.DataFrame({
    "customer_id": customer_ids,
    "monthly_revenue": monthly_revenue,
    "tenure_months": tenure_months,
    "login_frequency": login_frequency,
    "payment_delay_days": payment_delay_days,
    "support_tickets": support_tickets,
    "usage_score": usage_score,
    "discount_used": discount_used,
    "churn_risk": churn_risk
})


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

output_path = "data/customer_revenue_data_large.csv"

data.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

high_risk = int(data["churn_risk"].sum())
low_risk = int((data["churn_risk"] == 0).sum())


print("\n======================================")
print("   BALANCED CUSTOMER DATASET")
print("======================================")

print("Total customers:", len(data))

print(
    "High-risk customers:",
    high_risk,
    f"({high_risk / len(data) * 100:.1f}%)"
)

print(
    "Low-risk customers:",
    low_risk,
    f"({low_risk / len(data) * 100:.1f}%)"
)

print(
    "Average monthly revenue:",
    round(data["monthly_revenue"].mean(), 2)
)

print(
    "Average usage score:",
    round(data["usage_score"].mean(), 2)
)

print(
    "Average payment delay:",
    round(data["payment_delay_days"].mean(), 2)
)

print("\nDataset saved to:")
print(output_path)

print("\n======================================")
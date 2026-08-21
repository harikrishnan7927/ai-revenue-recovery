import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# Load large dataset
# --------------------------------------------------

data = pd.read_csv(
    "data/customer_revenue_data_large.csv"
)


# --------------------------------------------------
# Features
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


X = data[features]

y = data["churn_risk"]


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = "models/churn_model_large.pkl"

joblib.dump(
    model,
    model_path
)


# --------------------------------------------------
# Display information
# --------------------------------------------------

print("\n======================================")
print("      MODEL TRAINING COMPLETE")
print("======================================")

print(
    "Total customers:",
    len(data)
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "High-risk customers:",
    int(y.sum())
)

print(
    "Low-risk customers:",
    int((y == 0).sum())
)

print("\nModel:")
print("Random Forest Classifier")

print("\nModel saved to:")
print(model_path)

print("\n======================================")
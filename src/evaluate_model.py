import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Load large dataset and model
# --------------------------------------------------

data = pd.read_csv(
    "data/customer_revenue_data_large.csv"
)

model = joblib.load(
    "models/churn_model_large.pkl"
)


# --------------------------------------------------
# Features and target
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
# Same train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# --------------------------------------------------
# Display metrics
# --------------------------------------------------

print("\n======================================")
print("       LARGE MODEL EVALUATION")
print("======================================")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("======================================")
print("CONFUSION MATRIX")
print("======================================")

print(matrix)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n======================================")
print("FEATURE IMPORTANCE")
print("======================================")

print(
    importance.to_string(
        index=False
    )
)

print("\n======================================")
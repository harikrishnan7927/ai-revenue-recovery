import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load dataset
data = pd.read_csv("data/customer_revenue_data.csv")


# 2. Select input features
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


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train model
model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# 8. Save trained model
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved successfully to models/churn_model.pkl")
AI Revenue Recovery System

An end-to-end Machine Learning system that predicts customer churn risk, estimates potential revenue loss, and recommends customer recovery actions.

🎯 Project Overview

Customer churn can cause significant recurring revenue loss for businesses. This project uses Machine Learning to identify customers who are likely to churn and estimate the revenue that could be lost.

The system analyzes customer behavior such as:

- Monthly revenue
- Customer tenure
- Login frequency
- Payment delays
- Support tickets
- Product usage
- Discount usage

The predicted churn probability is then used to calculate the estimated monthly revenue at risk.

🔄 System Workflow

Customer Data
      ↓
Data Generation / Collection
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Random Forest Classifier
      ↓
Churn Probability
      ↓
Risk Classification
      ↓
Revenue-at-Risk Calculation
      ↓
Recovery Recommendation
      ↓
Streamlit Dashboard

🚀 Key Features

1. Customer Churn Prediction

The Random Forest model predicts whether a customer is at risk of churn.

2. Churn Probability

The system provides a probability score representing the estimated likelihood of customer churn.

3. Revenue-at-Risk

The system estimates potential monthly revenue loss using:

Revenue at Risk =
Monthly Revenue × Churn Probability

4. Risk Classification

Customers are categorized into:

- HIGH
- MEDIUM
- LOW

5. Recovery Recommendations

The system recommends actions such as:

- Payment recovery campaign
- Customer re-engagement campaign
- Priority customer support
- Retention offer
- Engagement reminder
- No immediate action

6. New Customer Prediction

The Streamlit dashboard allows users to enter information for a new customer and receive an instant churn prediction.

7. Model Evaluation

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

8. Feature Importance

The system identifies which customer features contribute most to the model's predictions.

📊 Dataset

The current development dataset contains:

- 1,000 customers
- 9 columns
- 7 customer behavior/features
- 1 customer identifier
- 1 churn target

Features

Feature| Description
customer_id| Unique customer identifier
monthly_revenue| Monthly customer revenue
tenure_months| Number of months as a customer
login_frequency| Frequency of customer logins
payment_delay_days| Average payment delay
support_tickets| Number of support tickets
usage_score| Product/service usage score
discount_used| Whether a discount was used
churn_risk| Churn target variable

🤖 Machine Learning Model

The project uses a:

Random Forest Classifier

Configuration includes:

- 200 decision trees
- Maximum tree depth: 8
- Minimum samples split: 5
- Minimum samples leaf: 2
- Balanced class weights
- 80/20 train-test split
- Stratified sampling

📈 Model Performance

Evaluation was performed on a held-out test set of 200 customers.

Metric| Score
Accuracy| 94.00%
Precision| 57.89%
Recall| 73.33%
F1 Score| 64.71%

Confusion Matrix

[[177   8]
 [  4  11]]

The model correctly identified 11 of the 15 churn cases in the test set.

Because churn prediction is a business-risk problem, recall is particularly important because missing a genuinely high-risk customer can result in lost revenue.

🔍 Feature Importance

The model identified the following features as the most important:

Feature| Importance
Usage Score| 38.19%
Payment Delay Days| 28.07%
Login Frequency| 9.85%
Support Tickets| 9.58%
Tenure Months| 7.40%
Monthly Revenue| 5.80%
Discount Used| 1.12%

These values describe the features the trained model relied on most heavily; they should not be interpreted as proof of causal relationships.

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard providing:

- Customer risk overview
- Risk distribution
- Revenue-at-risk analysis
- Priority recovery customers
- Feature importance
- Model performance
- New customer churn prediction
- Recovery recommendations

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Git
- GitHub

📁 Project Structure

ai-revenue-recovery/
│
├── data/
│   ├── customer_revenue_data.csv
│   └── customer_revenue_data_large.csv
│
├── models/
│   ├── churn_model.pkl
│   └── churn_model_large.pkl
│
├── notebooks/
│
├── src/
│   ├── dashboard.py
│   ├── evaluate_model.py
│   ├── generate_dataset.py
│   ├── predict.py
│   ├── risk_report.py
│   └── train_model.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore

⚙️ Installation

Clone the repository and enter the project directory.

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

▶️ Run the Project

Generate Dataset

python src/generate_dataset.py

Train the Model

python src/train_model.py

Evaluate the Model

python src/evaluate_model.py

Generate Risk Report

python src/risk_report.py

Run Individual Prediction

python src/predict.py

Launch Dashboard

python -m streamlit run src/dashboard.py

The dashboard will normally be available at:

http://localhost:8501

💼 Business Value

The system can help businesses:

1. Identify customers with high churn probability.
2. Prioritize customers requiring immediate attention.
3. Estimate potential recurring revenue loss.
4. Recommend appropriate recovery actions.
5. Support data-driven customer retention strategies.

🔮 Future Improvements

Potential future development includes:

- Larger real-world datasets
- Advanced feature engineering
- XGBoost / LightGBM comparison
- Hyperparameter optimization
- SHAP-based explainable AI
- Customer segmentation
- Automated recovery campaigns
- Real-time IoT/transaction data integration
- Database integration
- Cloud deployment
- Authentication and role-based access
- Model monitoring and retraining
- API deployment using FastAPI

⚠️ Disclaimer

This project uses a generated development dataset for demonstration and learning purposes. The model's current performance should not be treated as evidence of performance on real-world production data.

👨‍💻 Project Goal

The goal of this project is to demonstrate an end-to-end Machine Learning workflow that connects predictive analytics with a practical business problem: reducing customer churn and recovering potentially lost revenue.
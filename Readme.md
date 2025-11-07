# 📊 Telco Customer Churn Analytics Dashboard

Interactive Streamlit dashboard for analyzing and predicting customer churn in telecom. Combines **data cleaning, EDA, machine learning, and visualization** for actionable insights.

---

## 🚀 Project Overview
- Cleans and preprocesses raw customer data
- Performs exploratory data analysis (EDA)
- Trains ML models: Logistic Regression, Random Forest, XGBoost
- Interactive Streamlit dashboard for KPI tracking, insights, and predictions

---

## 🛠 Features
- **KPI Cards:** Total Customers, Churn Count, Churn Rate, Avg Monthly Charge
- **Insights & Recommendations:** Month-to-month churn, high charges, short tenure trends
- **Visualizations:** Churn distribution, monthly charges vs churn, tenure vs churn, top 5 feature importances
- **Predictive Analytics:** Top 10 customers most likely to churn, XGBoost model (Accuracy: 96.10%, ROC AUC: 99.24%)

---

## 💻 How to Run
```bash
git clone <https://github.com/JYOTIRADITYA-web-bit/Telco_Customer_Churn_Analysis>
cd Customer_Churn
pip install -r requirements.txt
streamlit run App.py


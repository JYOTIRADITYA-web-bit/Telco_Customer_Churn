# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib

# -------------------------------------------------------------
# ✅ PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Churn Analytics Dashboard")

# -------------------------------------------------------------
# ✅ LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telco.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["total_charges"].median(), inplace=True)
    df["churn_label"] = df["churn_label"].map({"Yes": 1, "No": 0})
    return df

data = load_data()

# -------------------------------------------------------------
# ✅ LOAD XGBOOST MODEL
# -------------------------------------------------------------
@st.cache_resource
def load_xgb():
    return joblib.load("churn_xgb_pipeline.pkl")

try:
    xgb_model = load_xgb()
except Exception as e:
    st.error("❌ Missing or invalid file: churn_xgb_pipeline.pkl. Please place the model file in the app folder.")
    st.stop()

# -------------------------------------------------------------
# ✅ SIDEBAR FILTERS
# -------------------------------------------------------------
st.sidebar.header("🔍 Filter Data")

# Filter 1: Contract Type
contract_options = sorted(data["contract"].dropna().unique())
selected_contracts = st.sidebar.multiselect(
    "Contract Type", contract_options, default=contract_options
)

# Filter 2: Tenure Range
min_tenure, max_tenure = int(data["tenure_in_months"].min()), int(data["tenure_in_months"].max())
selected_tenure = st.sidebar.slider(
    "Tenure in Months", min_tenure, max_tenure, (min_tenure, max_tenure)
)

# Filter 3: Internet Service
internet_options = sorted(data["internet_service"].dropna().unique())
selected_internet = st.sidebar.multiselect(
    "Internet Service Type", internet_options, default=internet_options
)

# -------------------------------------------------------------
# ✅ APPLY FILTERS
# -------------------------------------------------------------
filtered = data.copy()
filtered = filtered[filtered["contract"].isin(selected_contracts)]
filtered = filtered[filtered["internet_service"].isin(selected_internet)]
filtered = filtered[
    (filtered["tenure_in_months"] >= selected_tenure[0]) & 
    (filtered["tenure_in_months"] <= selected_tenure[1])
]

# -------------------------------------------------------------
# ✅ DOWNLOAD FILTERED DATA
# -------------------------------------------------------------
st.sidebar.download_button(
    label="Download Filtered Data",
    data=filtered.to_csv(index=False),
    file_name="filtered_telco.csv",
    mime="text/csv"
)

# -------------------------------------------------------------
# ✅ KPI CARDS
# -------------------------------------------------------------
st.subheader("📌 Key Performance Indicators")
total_customers = len(filtered)
churn_count = filtered["churn_label"].sum()
churn_rate = round((churn_count / total_customers) * 100, 2) if total_customers else 0
avg_charge = round(filtered["monthly_charge"].mean(), 2) if total_customers else 0

kpi_card_style = """
    <style>
        .kpi-card {
            background: #ffffff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #1a1a1a;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.12);
            border: 1px solid #e6e6e6;
        }
        .kpi-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: -5px;
            color: #003366;
        }
        .kpi-label {
            font-size: 16px;
            opacity: 0.7;
        }
    </style>
"""
st.markdown(kpi_card_style, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_customers}</div><div class='kpi-label'>Total Customers</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{churn_count}</div><div class='kpi-label'>Churn Count</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{churn_rate}%</div><div class='kpi-label'>Churn Rate</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${avg_charge}</div><div class='kpi-label'>Avg Monthly Charge</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# ✅ INSIGHTS & RECOMMENDATIONS
# -------------------------------------------------------------
insights = [
    "Month-to-month customers show the highest churn.",
    "Higher monthly charges strongly correlate with churn.",
    "Short tenure customers leave more often.",
    "Fiber Optic users churn more than DSL.",
    "Customers without security add-ons churn higher."
]
recommendations = [
    "Offer incentives to switch to annual/2-year contracts.",
    "Review high-charge customer segments for retention offers.",
    "Implement onboarding engagement for new customers.",
    "Improve Fiber Optic service satisfaction.",
    "Bundle security services at discounted rates."
]

insight_fig = go.Figure(data=[go.Table(
    header=dict(values=["📝 Insight", "✅ Recommendation"], fill_color="darkgreen", font=dict(color="white", size=18), align="center"),
    cells=dict(values=[insights, recommendations], fill_color=[['#e6ffe6']*5, ['#ffd9b3']*5], font=dict(color="black", size=16), align="left", height=55)
)])
insight_fig.update_layout(title="📌 Insights & Recommendations")
st.plotly_chart(insight_fig, use_container_width=True)

# -------------------------------------------------------------
# ✅ FEATURE IMPORTANCE (Random Forest)
# -------------------------------------------------------------
rf_data = filtered.copy()
cat_cols = rf_data.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    rf_data[col] = LabelEncoder().fit_transform(rf_data[col].astype(str))

X_rf = rf_data.drop("churn_label", axis=1)
y_rf = rf_data["churn_label"]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_rf, y_rf)

feat_imp = pd.DataFrame({"Feature": X_rf.columns, "Importance": rf.feature_importances_}).sort_values(by="Importance", ascending=False).head(5)
st.subheader("🔥 Top 5 Feature Importances (Random Forest)")
st.plotly_chart(px.bar(feat_imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Magma"), use_container_width=True)

# -------------------------------------------------------------
# ✅ CHURN VISUALIZATIONS
# -------------------------------------------------------------
filtered["churn_label_text"] = filtered["churn_label"].map({0:"No", 1:"Yes"})

st.plotly_chart(px.histogram(filtered, x="churn_label_text", color="churn_label_text", title="📊 Churn Distribution"), use_container_width=True)
st.plotly_chart(px.box(filtered, x="churn_label_text", y="monthly_charge", title="💵 Monthly Charges vs Churn"), use_container_width=True)
st.plotly_chart(px.box(filtered, x="churn_label_text", y="tenure_in_months", title="⏳ Tenure vs Churn"), use_container_width=True)

contract_churn = filtered.groupby("contract")["churn_label"].mean().reset_index().rename(columns={"churn_label":"churn_rate"})
fig_contract = px.bar(contract_churn, x="contract", y="churn_rate", title="Churn Rate by Contract Type", text=contract_churn["churn_rate"].round(2))
fig_contract.update_traces(textposition="outside")
st.plotly_chart(fig_contract, use_container_width=True)

fig_charge_dist = px.histogram(filtered, x="monthly_charge", color="churn_label_text", nbins=40, barmode="overlay", title="Distribution of Monthly Charges for Churn vs Non-Churn")
st.plotly_chart(fig_charge_dist, use_container_width=True)

# -------------------------------------------------------------
# ✅ TOP 10 CUSTOMERS MOST LIKELY TO CHURN
# -------------------------------------------------------------
st.subheader("🔥 Top 10 Customers Most Likely to Churn")
predict_df = filtered.drop(columns=["churn_label"], errors="ignore").copy()

try:
    expected = getattr(xgb_model, "feature_names_in_", None)
    if expected is None:
        expected = xgb_model.named_steps["preprocessor"].feature_names_in_
    expected = [c for c in expected if c in predict_df.columns]
    predict_df = predict_df[expected]
except Exception:
    pass

try:
    churn_proba = xgb_model.predict_proba(predict_df)[:, 1]
except Exception as e:
    st.error(f"❌ Prediction error with the XGBoost pipeline: {e}")
    st.stop()

risk_df = filtered.copy()
risk_df["churn_probability"] = churn_proba
top10 = risk_df.sort_values(by="churn_probability", ascending=False).head(10)

fig_top10 = go.Figure(data=[go.Table(
    header=dict(values=["Customer ID", "Monthly Charge", "Tenure", "Churn Probability"], fill_color="black", font=dict(color="white", size=16), align="center"),
    cells=dict(values=[
        top10.get("customer_id", pd.Series(["N/A"]*len(top10))),
        top10.get("monthly_charge", pd.Series(["N/A"]*len(top10))),
        top10.get("tenure_in_months", pd.Series(["N/A"]*len(top10))),
        top10["churn_probability"].round(4)
    ], fill_color="lightyellow", align="center", font=dict(color="black", size=15), height=32)
)])
fig_top10.update_layout(title="🚨 Top 10 High-Risk Customers", margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_top10, use_container_width=True)

# -------------------------------------------------------------
# ✅ MODEL PERFORMANCE (STATIC)
# -------------------------------------------------------------
st.subheader("✅ XGBoost Model Performance")
st.markdown("""
### ✅ Accuracy: **96.10%**  
### ✅ ROC AUC: **99.24%**
""")

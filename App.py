# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
<<<<<<< HEAD
import joblib

# -------------------------------------------------------------
# ✅ PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Churn Analytics Dashboard")

# -------------------------------------------------------------
# ✅ LOAD DATA
# Bringing customer data into the dashboard and   cleaning it so everything works smoothly without missing or invalid values.
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telco.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["total_charges"].median(), inplace=True)
    df["churn_label"] = df["churn_label"].map({"Yes": 1, "No": 0})
    return df

# -------------------------------------------------------------
# ✅ LOAD XGBOOST MODEL
# Loading our pre-trained churn prediction model that helps us identify customers likely to leave.
# -------------------------------------------------------------
@st.cache_resource
def load_xgb():
    return joblib.load("churn_xgb_pipeline.pkl")

data = load_data()

try:
    xgb_model = load_xgb()
except Exception as e:
    st.error("❌ Missing or invalid file: churn_xgb_pipeline.pkl. Please place the model file in the app folder.")
    st.stop()

# -------------------------------------------------------------
# ✅ SIDEBAR FILTERS (3 FILTERS)
# -------------------------------------------------------------
st.sidebar.header("🔍 Filter Data")

# ✅ Filter 1: Contract Type
contract_options = sorted(data["contract"].dropna().unique())
selected_contracts = st.sidebar.multiselect(
    "Contract Type", contract_options, default=contract_options
)

# ✅ Filter 2: Tenure Range
min_tenure, max_tenure = int(data["tenure_in_months"].min()), int(data["tenure_in_months"].max())
selected_tenure = st.sidebar.slider(
    "Tenure in Months", min_tenure, max_tenure, (min_tenure, max_tenure)
)

# ✅ Filter 3: Internet Service
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
# ✅ KPI CARDS  (Key Performance Indicators)
# -------------------------------------------------------------
st.subheader("📌 Key Performance Indicators")

# KPI values
total_customers = len(filtered)
churn_count = filtered["churn_label"].sum()
churn_rate = round((churn_count / total_customers) * 100, 2) if total_customers else 0
avg_charge = round(filtered["monthly_charge"].mean(), 2) if total_customers else 0

# KPI Card Style (Light Theme)
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

# Show KPI Cards in 4 Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_customers}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{churn_count}</div>
            <div class="kpi-label">Churn Count</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{churn_rate}%</div>
            <div class="kpi-label">Churn Rate</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${avg_charge}</div>
            <div class="kpi-label">Avg Monthly Charge</div>
        </div>
    """, unsafe_allow_html=True)


st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# ✅ INSIGHTS & RECOMMENDATIONS 
# A simple table outlining what the data is telling us and actions the team can take to reduce churn.
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

insight_fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=["📝 Insight", "✅ Recommendation"],
                fill_color="darkgreen",
                font=dict(color="white", size=18),
                align="center"
            ),
            cells=dict(
                values=[insights, recommendations],
                fill_color=[['#e6ffe6'] * 5, ['#ffd9b3'] * 5],
                font=dict(color="black", size=16),   # ✅ FIXED
                align="left",
                height=55
            )
        )
    ]
)

insight_fig.update_layout(title="📌 Insights & Recommendations")
st.plotly_chart(insight_fig, use_container_width=True)

# -------------------------------------------------------------
# ✅ FEATURE IMPORTANCE
# Shows which customer attributes matter most when predicting churn, helping the team understand the root causes.
# -------------------------------------------------------------
rf_data = filtered.copy()

# Encode categorical columns for RF
cat_cols = rf_data.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    rf_data[col] = LabelEncoder().fit_transform(rf_data[col].astype(str))

X_rf = rf_data.drop("churn_label", axis=1)
y_rf = rf_data["churn_label"]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_rf, y_rf)

feat_imp = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).head(5)

st.subheader("🔥 Top 5 Feature Importances (Random Forest)")
st.plotly_chart(px.bar(
    feat_imp, x="Importance", y="Feature", orientation="h",
    color="Importance", color_continuous_scale="Magma"
), use_container_width=True)

# -------------------------------------------------------------
# ✅ CHURN VISUALIZATIONS
# User-friendly charts that break down churn behavior to help identify patterns among customers.
# -------------------------------------------------------------
filtered["churn_label_text"] = filtered["churn_label"].map({0: "No", 1: "Yes"})

st.plotly_chart(px.histogram(
    filtered, x="churn_label_text", color="churn_label_text",
    title="📊 Churn Distribution"
), use_container_width=True)

st.plotly_chart(px.box(
    filtered, x="churn_label_text", y="monthly_charge",
    title="💵 Monthly Charges vs Churn"
), use_container_width=True)

st.plotly_chart(px.box(
    filtered, x="churn_label_text", y="tenure_in_months",
    title="⏳ Tenure vs Churn"
), use_container_width=True)

# -------------------------------------------------------------
# ✅ VISUAL 4: Churn Rate by Contract Type
# -------------------------------------------------------------
st.subheader("📈 Churn Rate by Contract Type")

contract_churn = (
    filtered.groupby("contract")["churn_label"]
    .mean()
    .reset_index()
    .rename(columns={"churn_label": "churn_rate"})
)

fig_contract = px.bar(
    contract_churn,
    x="contract",
    y="churn_rate",
    title="Churn Rate by Contract Type",
    text=contract_churn["churn_rate"].round(2),
)

fig_contract.update_traces(textposition="outside")
st.plotly_chart(fig_contract, use_container_width=True)

# -------------------------------------------------------------
# ✅ VISUAL 5: Monthly Charges Distribution by Churn Status
# -------------------------------------------------------------
st.subheader("📉 Monthly Charges Distribution (Churn vs Non-Churn)")

fig_charge_dist = px.histogram(
    filtered,
    x="monthly_charge",
    color="churn_label_text",
    nbins=40,
    barmode="overlay",
    title="Distribution of Monthly Charges for Churn vs Non-Churn"
)

st.plotly_chart(fig_charge_dist, use_container_width=True)

# ================================================================
# 🔥 Top 10 Customers Most Likely to Churn
# ================================================================
st.subheader("🔥 Top 10 Customers Most Likely to Churn")

predict_df = filtered.drop(columns=["churn_label"], errors="ignore").copy()

try:
    expected = getattr(xgb_model, "feature_names_in_", None)
    if expected is None:
        try:
            expected = xgb_model.named_steps["preprocessor"].feature_names_in_
        except Exception:
            expected = None
    if expected is not None:
        expected = [c for c in expected if c in predict_df.columns]
        predict_df = predict_df[expected]
except Exception:
    predict_df = predict_df

try:
    churn_proba = xgb_model.predict_proba(predict_df)[:, 1]
except Exception as e:
    st.error(f"❌ Prediction error with the XGBoost pipeline: {e}")
    st.stop()

risk_df = filtered.copy()
risk_df["churn_probability"] = churn_proba

top10 = risk_df.sort_values(by="churn_probability", ascending=False).head(10)

fig_top10 = go.Figure(
    data=[go.Table(
        header=dict(
            values=["Customer ID", "Monthly Charge", "Tenure", "Churn Probability"],
            fill_color="black",
            font=dict(color="white", size=16),
            align="center"
        ),
        cells=dict(
            values=[
                top10.get("customer_id", pd.Series(["N/A"]*len(top10))),
                top10.get("monthly_charge", pd.Series(["N/A"]*len(top10))),
                top10.get("tenure_in_months", pd.Series(["N/A"]*len(top10))),
                top10["churn_probability"].round(4)
            ],
            fill_color="lightyellow",
            align="center",
            font=dict(color="black", size=15),
            height=32
        )
    )]
)

fig_top10.update_layout(
    title="🚨 Top 10 High-Risk Customers",
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig_top10, use_container_width=True)



# -------------------------------------------------------------
# ✅ MODEL PERFORMANCE (STATIC FROM YOUR NOTEBOOK)
# -------------------------------------------------------------
st.subheader("✅ XGBoost Model")

st.markdown("""
### ✅ Accuracy: **96.10%**  
### ✅ ROC AUC: **99.24%**
""")  
=======
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------
try:
    data = pd.read_csv("telco.csv")
except FileNotFoundError:
    st.error("Default dataset 'telco.csv' not found.")
    st.stop()

data.columns = data.columns.str.lower().str.replace(' ', '_')
data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')
data['total_charges'].fillna(data['total_charges'].median(), inplace=True)
data['churn_label'] = data['churn_label'].map({'Yes':1,'No':0})

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Data")
contract_options = data['contract'].unique().tolist()
selected_contracts = st.sidebar.multiselect("Contract Type", contract_options, default=contract_options)
min_tenure, max_tenure = int(data['tenure_in_months'].min()), int(data['tenure_in_months'].max())
selected_tenure = st.sidebar.slider("Tenure in Months", min_tenure, max_tenure, (min_tenure, max_tenure))

filtered_data = data[data['contract'].isin(selected_contracts)]
filtered_data = filtered_data[(filtered_data['tenure_in_months'] >= selected_tenure[0]) &
                              (filtered_data['tenure_in_months'] <= selected_tenure[1])]

# -------------------------------
# 1️⃣ KPI Metrics
# -------------------------------
total_customers = len(filtered_data)
churn_count = filtered_data['churn_label'].sum()
churn_rate = round(churn_count / total_customers * 100, 2) if total_customers>0 else 0
avg_monthly_charge = round(filtered_data['monthly_charge'].mean(),2) if total_customers>0 else 0

kpi_names = ["💼 Total Customers", "📉 Churn Count", "📊 Churn Rate", "💰 Avg Monthly Charge"]
kpi_values = [total_customers, churn_count, f"{churn_rate}%", f"${avg_monthly_charge}"]

kpi_table = go.Figure(data=[go.Table(
    header=dict(values=["KPI","Value"], fill_color='midnightblue', font=dict(color='white',size=18), align='center'),
    cells=dict(values=[kpi_names, kpi_values], fill_color='lavender', font=dict(color='black',size=16), align='center', height=40)
)])
kpi_table.update_layout(title=dict(text="📌 Key Performance Indicators", font=dict(size=20)), margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(kpi_table, use_container_width=True)

# -------------------------------
# 2️⃣ Insights & Recommendations
# -------------------------------
insights = [
    "Month-to-month contracts have higher churn.",
    "Higher monthly charges increase churn risk.",
    "No device protection plan correlates with higher churn.",
    "Fiber optic internet users churn more than DSL.",
    "Longer tenure and long-term contracts reduce churn."
]
recommendations = [
    "Incentivize month-to-month customers to switch to annual contracts.",
    "Review pricing for high monthly charges to reduce churn.",
    "Promote device protection plans to improve retention.",
    "Provide special support or perks for Fiber optic users.",
    "Encourage longer contract durations through loyalty programs."
]

insight_table = go.Figure(data=[go.Table(
    header=dict(values=["📝 Insight", "✅ Recommendation"], fill_color='darkgreen', font=dict(color='white', size=18), align='center'),
    cells=dict(values=[insights, recommendations],
               fill_color=[['#e6ffe6','#ccffcc','#e6ffe6','#ccffcc','#e6ffe6'],
                           ['#ffd9b3','#ffb84d','#ffd9b3','#ffb84d','#ffd9b3']],
               font=dict(color='black', size=16), align='left', height=60)
)])
insight_table.update_layout(title=dict(text="📌 Insights & Recommendations", font=dict(size=22)), margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(insight_table, use_container_width=True)

# -------------------------------
# 3️⃣ Top 5 Feature Importances
# -------------------------------
categorical_cols = filtered_data.select_dtypes(include='object').columns.tolist()
categorical_cols = [c for c in categorical_cols if c != 'churn_label']
for col in categorical_cols:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col].astype(str))

X = filtered_data.drop('churn_label', axis=1)
y = filtered_data['churn_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
top5_feat = feat_imp.sort_values(by='Importance', ascending=False).head(5)

fig_feat = px.bar(top5_feat, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Magma', title='🔥 Top 5 Feature Importances')
fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_feat, use_container_width=True)

# -------------------------------
# 4️⃣ Key Visualizations
# -------------------------------
filtered_data['churn_label_text'] = filtered_data['churn_label'].map({0:'No',1:'Yes'})

fig1 = px.histogram(filtered_data, x='churn_label_text', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='📊 Churn Distribution')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(filtered_data, x='churn_label_text', y='monthly_charge', color='churn_label_text',
              color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='💵 Monthly Charge vs Churn')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(filtered_data, x='churn_label_text', y='tenure_in_months', color='churn_label_text',
              color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='⏳ Tenure vs Churn')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.histogram(filtered_data, x='avg_monthly_gb_download', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='📶 Avg Monthly GB Download vs Churn')
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.histogram(filtered_data, x='number_of_referrals', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='🤝 Number of Referrals vs Churn')
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# 5️⃣ ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                             line=dict(color='blue', width=3)))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Guess',
                             line=dict(color='red', width=2, dash='dash')))
roc_fig.update_layout(title=f"📈 ROC Curve - AUC: {roc_auc:.2f}",
                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(roc_fig, use_container_width=True)

# -------------------------------
# 6️⃣ Confusion Matrix & Classification Report
# -------------------------------
st.subheader("🧮 Model Evaluation (Holdout Set)")
st.text("Confusion Matrix:")
st.text(confusion_matrix(y_test, y_pred))
st.text("\nClassification Report:")
st.text(classification_report(y_test, y_pred))
>>>>>>> fce4222d2912f9ea465be103898e8846d35de55b

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1A1A2E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #7C3AED;
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-high { color: #E53E3E; font-weight: bold; }
    .risk-medium { color: #DD6B20; font-weight: bold; }
    .risk-low { color: #38A169; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model_and_data():
    try:
        with open('models/xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        df_raw = pd.read_csv('data/customer_churn.csv')
        return model, feature_cols, df_raw
    except Exception:
        # Generate data and train model on the fly if files are missing
        import xgboost as xgb
        
        np.random.seed(42)
        n_samples = 2000
        data = pd.DataFrame()
        data['customerID'] = [f"CUST-{i:05d}" for i in range(n_samples)]
        data['SeniorCitizen'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        data['Partner'] = np.random.choice(['Yes', 'No'], size=n_samples)
        data['Dependents'] = np.random.choice(['Yes', 'No'], size=n_samples)
        data['tenure'] = np.random.randint(1, 73, size=n_samples)
        data['PhoneService'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
        data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.3, 0.5, 0.2])
        data['OnlineSecurity'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
        data['TechSupport'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
        data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2])
        data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], size=n_samples)
        data['PaymentMethod'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], size=n_samples)
        
        base_charge = 20
        data['MonthlyCharges'] = base_charge + data['tenure'] * 0.5 + (data['InternetService'] == 'Fiber optic') * 40 + (data['InternetService'] == 'DSL') * 20
        data['MonthlyCharges'] += np.random.normal(0, 5, size=n_samples)
        
        churn_prob = np.zeros(n_samples)
        churn_prob += (data['Contract'] == 'Month-to-month') * 0.3
        churn_prob += (data['InternetService'] == 'Fiber optic') * 0.2
        churn_prob += (data['TechSupport'] == 'No') * 0.1
        churn_prob -= (data['tenure'] > 24) * 0.2
        churn_prob += data['SeniorCitizen'] * 0.1
        churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, size=n_samples), 0, 1)
        data['Churn'] = (churn_prob > 0.5).astype(int)
        
        df_processed = data.copy().set_index('customerID').drop('Churn', axis=1)
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        df_processed = pd.get_dummies(df_processed, columns=cat_cols)
        
        feature_cols = list(df_processed.columns)
        X = df_processed
        y = data['Churn']
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)
        
        return model, feature_cols, data

def preprocess_input(df, feature_cols):
    df_processed = df.copy()
    if 'customerID' in df_processed.columns:
        df_processed.set_index('customerID', inplace=True)
    if 'Churn' in df_processed.columns:
        df_processed.drop('Churn', axis=1, inplace=True)
        
    cat_cols = df_processed.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_processed, columns=cat_cols)
    
    for col in feature_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[feature_cols]
    return df_processed

# --- App Header ---
st.title("🛡️ ChurnGuard AI")
st.markdown("**Predict, Explain, and Prevent Customer Churn using XAI.**")

with st.spinner("Loading AI Models and Data..."):
    model, feature_cols, df_raw = get_model_and_data()
    df_processed = preprocess_input(df_raw, feature_cols)

# Calculate global metrics
probs = model.predict_proba(df_processed)[:, 1]
df_raw['Risk_Score'] = probs

# --- Vibrant Tabs Navigation ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🔍 Customer Analysis", "⚙️ What-If Simulator"])

with tab1:
    st.header("Executive Dashboard")
    avg_risk = df_raw['Risk_Score'].mean()
    high_risk_count = (df_raw['Risk_Score'] > 0.7).sum()
    total_customers = len(df_raw)
    revenue_at_risk = df_raw[df_raw['Risk_Score'] > 0.7]['MonthlyCharges'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_customers:,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_risk:.1%}</div><div class="metric-label">Avg Churn Risk</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{high_risk_count:,}</div><div class="metric-label">High Risk Customers</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${revenue_at_risk:,.0f}</div><div class="metric-label">Monthly Rev at Risk</div></div>', unsafe_allow_html=True)
    
    st.write("---")
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.subheader("Risk Distribution")
        # Use Streamlit native chart instead of Plotly to avoid dependency issues
        risk_hist, bin_edges = np.histogram(df_raw['Risk_Score'], bins=20, range=(0, 1))
        chart_data = pd.DataFrame({
            "Churn Risk Score": bin_edges[:-1],
            "Count": risk_hist
        }).set_index("Churn Risk Score")
        st.bar_chart(chart_data, color="#7C3AED")
        
    with col_plot2:
        st.subheader("Global Feature Importance (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_processed.sample(min(500, len(df_processed)), random_state=42))
        
        fig_shap, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        fig_shap.patch.set_facecolor('#0F0F1A')
        ax.set_facecolor('#0F0F1A')
        shap.summary_plot(shap_values, df_processed.sample(min(500, len(df_processed)), random_state=42), show=False, plot_size=(6,4))
        st.pyplot(fig_shap)

with tab2:
    st.header("Individual Customer Analysis")
    selected_customer = st.selectbox("Select Customer ID:", df_raw['customerID'].head(100))
    cust_data = df_raw[df_raw['customerID'] == selected_customer].iloc[0]
    cust_processed = df_processed.loc[[selected_customer]]
    
    prob = model.predict_proba(cust_processed)[0][1]
    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
    risk_class = f"risk-{risk_level.lower()}"
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Customer Profile")
        st.write(f"**Tenure:** {cust_data['tenure']} months")
        st.write(f"**Contract:** {cust_data['Contract']}")
        st.write(f"**Internet Service:** {cust_data['InternetService']}")
        st.write(f"**Monthly Charges:** ${cust_data['MonthlyCharges']:.2f}")
        st.markdown(f"### Churn Risk: <span class='{risk_class}'>{prob:.1%} ({risk_level})</span>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Why? (Local SHAP Explanation)")
        explainer = shap.TreeExplainer(model)
        shap_values_local = explainer(cust_processed)
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0F0F1A')
        ax.set_facecolor('#0F0F1A')
        shap.plots.waterfall(shap_values_local[0], max_display=10, show=False)
        st.pyplot(fig)

with tab3:
    st.header("What-If Simulator")
    st.write("Modify customer attributes to see how it affects churn probability.")
    
    selected_customer_sim = st.selectbox("Select Customer as Base (Sim):", df_raw['customerID'].head(100))
    cust_data_sim = df_raw[df_raw['customerID'] == selected_customer_sim].iloc[0].copy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_tenure = st.slider("Tenure (months)", 1, 72, int(cust_data_sim['tenure']))
        new_monthly = st.number_input("Monthly Charges ($)", 15.0, 150.0, float(cust_data_sim['MonthlyCharges']))
        
    with col2:
        contract_options = ['Month-to-month', 'One year', 'Two year']
        new_contract = st.selectbox("Contract", contract_options, index=contract_options.index(cust_data_sim['Contract']))
        internet_options = ['DSL', 'Fiber optic', 'No']
        new_internet = st.selectbox("Internet Service", internet_options, index=internet_options.index(cust_data_sim['InternetService']))
        
    with col3:
        tech_options = ['Yes', 'No', 'No internet service']
        new_tech = st.selectbox("Tech Support", tech_options, index=tech_options.index(cust_data_sim['TechSupport']))
    
    sim_data = cust_data_sim.copy()
    sim_data['tenure'] = new_tenure
    sim_data['MonthlyCharges'] = new_monthly
    sim_data['Contract'] = new_contract
    sim_data['InternetService'] = new_internet
    sim_data['TechSupport'] = new_tech
    
    sim_df = pd.DataFrame([sim_data])
    sim_processed = preprocess_input(sim_df, feature_cols)
    
    orig_prob = model.predict_proba(df_processed.loc[[selected_customer_sim]])[0][1]
    new_prob = model.predict_proba(sim_processed)[0][1]
    
    st.write("---")
    st.subheader("Simulation Results")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Original Risk", f"{orig_prob:.1%}")
    with col_r2:
        st.metric("Simulated Risk", f"{new_prob:.1%}", delta=f"{(new_prob - orig_prob)*100:.1f}%", delta_color="inverse")

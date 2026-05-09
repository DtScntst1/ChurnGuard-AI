import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

st.title("ChurnGuard AI: Predict, Explain and Prevent")
st.markdown("Predict and understand customer churn using Explainable AI (XAI).")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Customer Analysis", "What-If Simulator"])

# Synthetic data generation
@st.cache_data
def load_data():
          np.random.seed(42)
          n = 1000
          data = pd.DataFrame({
              'Tenure': np.random.randint(1, 72, n),
              'MonthlyCharges': np.random.uniform(20, 120, n),
              'TotalCharges': np.random.uniform(100, 8000, n),
              'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
              'TechSupport': np.random.choice(['Yes', 'No'], n),
              'OnlineSecurity': np.random.choice(['Yes', 'No'], n),
              'Churn': np.random.choice([0, 1], n, p=[0.8, 0.2])
          })
          return data

df = load_data()

# Simple preprocessing
df_model = pd.get_dummies(df.drop('Churn', axis=1))
y = df['Churn']

# Model
@st.cache_resource
def get_model():
          m = xgb.XGBClassifier(n_estimators=100, max_depth=4)
          m.fit(df_model, y)
          return m

model = get_model()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_model)

if page == "Executive Dashboard":
          st.header("Executive Dashboard")
          col1, col2, col3, col4 = st.columns(4)
          col1.metric("Total Customers", "5,000")
          col2.metric("Avg Churn Risk", "8.4%", "-1.2%")
          col3.metric("High Risk Customers", "233")
          col4.metric("Monthly Rev at Risk", "$14.2K")

    st.subheader("Global Feature Importance (SHAP)")
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, df_model, show=False)
    st.pyplot(plt.gcf())

elif page == "Customer Analysis":
    st.header("Individual Customer Analysis")
    cust_id = st.selectbox("Select Customer ID", range(len(df)))
    st.write(df.iloc[cust_id])

    prob = model.predict_proba(df_model.iloc[[cust_id]])[0][1]
    st.write(f"Churn Risk: {prob:.2%}")
    st.progress(prob)

elif page == "What-If Simulator":
    st.header("What-If Simulator")
    tenure = st.slider("Tenure (months)", 1, 72, 24)
    charges = st.slider("Monthly Charges ($)", 20, 150, 70)

    new_data = df_model.iloc[[0]].copy()
    new_data['Tenure'] = tenure
    new_data['MonthlyCharges'] = charges
    new_prob = model.predict_proba(new_data)[0][1]
    st.metric("Simulated Churn Risk", f"{new_prob:.2%}")

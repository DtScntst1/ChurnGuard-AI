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

st.title("ChurnGuard AI: Predict, Explain & Prevent")

# Synthetic data for demo
@st.cache_data
def load_data():
      np.random.seed(42)
      n = 1000
      data = pd.DataFrame({
          'Tenure': np.random.randint(1, 72, n),
          'MonthlyCharges': np.random.uniform(20, 120, n),
          'Churn': np.random.choice([0, 1], n, p=[0.7, 0.3])
      })
      return data

data = load_data()
st.write("Current Data Sample:", data.head())

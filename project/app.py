# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------
# Setup paths
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ---------------------------
# Load model and datasets
# ---------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')

# Load
daily_model = load_model(MODEL_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ---------------------------
# Streamlit Layout
# ---------------------------
st.set_page_config(page_title="Power Demand Dashboard", layout="wide")

st.title("⚡ Power Demand Dashboard")
st.markdown("### Forecast vs Actual Power Demand (Daily)")

# ---------------------------
# Forecast (Next 30 Days)
# ---------------------------
n_forecast_days = 30

# Generate forecast
forecast = daily_model.predict(n_periods=n_forecast_days)
forecast_index = pd.date_range(
    start=df_daily.index[-1] + pd.Timedelta(days=1),
    periods=n_forecast_days,
    freq='D'
)

# ---------------------------
# Plotly Chart
# ---------------------------
fig = go.Figure()

# Actual Data
fig.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily['power_demand'],
    mode='lines',
    name='Actual',
    line=dict(color='royalblue', width=2)
))

# Forecast Data
fig.add_trace(go.Scatter(
    x=forecast_index,
    y=forecast,
    mode='lines',
    name='Forecast',
    line=dict(color='orange', dash='dash', width=2)
))

fig.update_layout(
    title="Daily Power Demand Forecast (Next 30 Days)",
    xaxis_title="Date",
    yaxis_title="Power Demand (MW)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Performance Metrics
# ---------------------------
st.markdown("### 📊 Model Performance (Last 30 Days)")
try:
    # Compare last actuals to forecast length
    actual_recent = df_daily['power_demand'][-n_forecast_days:]
    mae = mean_absolute_error(actual_recent, forecast[:len(actual_recent)])
    mse = mean_squared_error(actual_recent, forecast[:len(actual_recent)])
    rmse = np.sqrt(mse)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
except Exception as e:
    st.warning("Metrics could not be computed — insufficient recent actual data.")
    st.write(e)

# ---------------------------
# Optional Data Preview
# ---------------------------
with st.expander("📂 View Latest Daily Data"):
    st.dataframe(df_daily.tail(10))

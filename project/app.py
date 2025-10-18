# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go

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

daily_model = load_model(MODEL_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ---------------------------
# Streamlit Layout
# ---------------------------
st.title("Power Demand Dashboard")
st.markdown("Forecast vs Actual Power Demand (Daily)")

# ---------------------------
# Forecast
# ---------------------------
n_forecast_days = 30
forecast = daily_model.predict(n_periods=n_forecast_days)
forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=n_forecast_days, freq='D')

# ---------------------------
# Plotly Chart
# ---------------------------
fig = go.Figure()
# Actual
fig.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily['power_demand'],
    mode='lines',
    name='Actual'
))
# Forecast
fig.add_trace(go.Scatter(
    x=forecast_index,
    y=forecast,
    mode='lines',
    name='Forecast'
))
fig.update_layout(
    title="Daily Power Demand Forecast",
    xaxis_title="Date",
    yaxis_title="Power Demand",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Show metrics
# ---------------------------
st.markdown("### Metrics (Daily)")
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(df_daily['power_demand'][-n_forecast_days:], forecast)
mse = mean_squared_error(df_daily['power_demand'][-n_forecast_days:], forecast)
rmse = np.sqrt(mse)

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

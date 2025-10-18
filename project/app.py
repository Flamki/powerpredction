# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go

# --------------------------
# Set base directory
# --------------------------
BASE_DIR = os.path.dirname(__file__)

# --------------------------
# Load model and data
# --------------------------
MODEL_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")

daily_model = joblib.load(MODEL_PATH)
df_daily = pd.read_csv(DAILY_CSV, parse_dates=['datetime'], index_col='datetime')

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Power Demand Dashboard", layout="wide")
st.title("⚡ Power Demand Dashboard")
st.markdown("This dashboard shows the **daily power demand forecast** using ARIMA model.")

# --------------------------
# Forecast
# --------------------------
n_days = st.slider("Select forecast horizon (days):", min_value=7, max_value=60, value=30)

forecast = daily_model.predict(n_periods=n_days)
forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1),
                               periods=n_days, freq='D')

# --------------------------
# Plotly Figure
# --------------------------
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
    mode='lines+markers',
    name='Forecast'
))

fig.update_layout(
    title="Daily Power Demand Forecast",
    xaxis_title="Date",
    yaxis_title="Power Demand",
    template="plotly_dark"
)

# --------------------------
# Display in Streamlit
# --------------------------
st.plotly_chart(fig, use_container_width=True)

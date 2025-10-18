# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

daily_model = load_model(MODEL_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Dashboard Settings")

forecast_days = st.sidebar.slider(
    "Select Forecast Range (Days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

show_hourly = st.sidebar.checkbox("Show Hourly Trends", value=False)
show_diagnostics = st.sidebar.checkbox("Show Model Diagnostics", value=True)
show_download = st.sidebar.checkbox("Enable Forecast Download", value=True)

# ---------------------------
# Forecast
# ---------------------------
forecast = daily_model.predict(n_periods=forecast_days)
forecast_index = pd.date_range(
    start=df_daily.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)
forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

# ---------------------------
# Title
# ---------------------------
st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("Visualize, analyze, and forecast power demand trends using ARIMA modeling.")

# ---------------------------
# Plot Actual vs Forecast
# ---------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily["power_demand"],
    mode="lines",
    name="Actual Demand",
    line=dict(color="#636EFA", width=2)
))
fig.add_trace(go.Scatter(
    x=forecast_index,
    y=forecast,
    mode="lines",
    name="Forecast",
    line=dict(color="#EF553B", width=2, dash="dot")
))
fig.update_layout(
    title="📈 Daily Power Demand Forecast",
    xaxis_title="Date",
    yaxis_title="Power Demand",
    template="plotly_white",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# KPIs / Metrics
# ---------------------------
st.markdown("### 📊 Performance Metrics")

if len(df_daily) > forecast_days:
    test_actual = df_daily["power_demand"][-forecast_days:]
    mae = mean_absolute_error(test_actual, forecast)
    rmse = np.sqrt(mean_squared_error(test_actual, forecast))
else:
    mae, rmse = np.nan, np.nan

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# ---------------------------
# Trend Insights
# ---------------------------
st.markdown("### 🔍 Trend Analysis")
st.line_chart(df_daily["power_demand"].rolling(7).mean(), use_container_width=True)

# ---------------------------
# Hourly View
# ---------------------------
if show_hourly:
    st.markdown("### ⏱️ Hourly Power Demand")
    st.line_chart(df_hourly["power_demand"].tail(500))

# ---------------------------
# Diagnostics
# ---------------------------
if show_diagnostics:
    st.markdown("### 🧠 Model Diagnostics")

    residuals = daily_model.arima_res_.resid
    diag_fig = go.Figure()
    diag_fig.add_trace(go.Scatter(
        x=np.arange(len(residuals)),
        y=residuals,
        mode="lines",
        name="Residuals",
        line=dict(color="#00CC96")
    ))
    diag_fig.update_layout(
        title="Residuals (Model Fit Quality)",
        xaxis_title="Time",
        yaxis_title="Residuals",
        template="plotly_white"
    )
    st.plotly_chart(diag_fig, use_container_width=True)

# ---------------------------
# Forecast Table & Download
# ---------------------------
st.markdown("### 🧾 Forecast Data Table")
st.dataframe(forecast_df.tail(15))

if show_download:
    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name="power_forecast.csv",
        mime="text/csv"
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
**Developed by ⚡ bleu.x ML Studio**  
Powered by ARIMA | Streamlit | Plotly
""")

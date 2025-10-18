# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Power Demand Dashboard", layout="wide")

st.title("⚡ Power Demand Forecast Dashboard")

# --- Load Model & Data Safely ---
if not os.path.exists("arima_power_model_daily.pkl"):
    st.error("❌ Daily ARIMA model file not found! Upload 'arima_power_model_daily.pkl'")
else:
    daily_model = joblib.load("arima_power_model_daily.pkl")

if not os.path.exists("power_demand_daily.csv") or not os.path.exists("power_demand_processed.csv"):
    st.error("❌ Dataset files not found! Upload 'power_demand_daily.csv' and 'power_demand_processed.csv'")
else:
    df_daily = pd.read_csv("power_demand_daily.csv", parse_dates=['datetime'], index_col='datetime')
    df_hourly = pd.read_csv("power_demand_processed.csv", parse_dates=['datetime'], index_col='datetime')

# --- Sidebar for Options ---
st.sidebar.header("Options")
forecast_days = st.sidebar.slider("Number of days to forecast", 7, 60, 30)
show_hourly = st.sidebar.checkbox("Show hourly data", value=False)

# --- Forecasting ---
if 'daily_model' in locals() and 'df_daily' in locals():
    forecast = daily_model.predict(n_periods=forecast_days)
    forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

    # --- Plot Daily Forecast ---
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=df_daily.index, y=df_daily['power_demand'], mode='lines', name='Actual'))
    fig_daily.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))

    fig_daily.update_layout(
        title="Daily Power Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Power Demand",
        template="plotly_white"
    )

    st.plotly_chart(fig_daily, use_container_width=True)

    # --- Show Accuracy Metrics if past data available ---
    if len(df_daily) >= forecast_days:
        y_true = df_daily['power_demand'][-forecast_days:]
        mae = mean_absolute_error(y_true, forecast[:len(y_true)])
        rmse = mean_squared_error(y_true, forecast[:len(y_true)], squared=False)
        st.subheader("Model Accuracy Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Accuracy Approximation: {100 - (mae / y_true.mean() * 100):.2f}%")

# --- Hourly Data ---
if show_hourly and 'df_hourly' in locals():
    st.subheader("Hourly Data Preview")
    st.dataframe(df_hourly.tail(50))

    # Optional: Plot last 7 days hourly
    last_7d = df_hourly.last('7D')
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(x=last_7d.index, y=last_7d['power_demand'], mode='lines', name='Hourly Demand'))
    fig_hourly.update_layout(title="Last 7 Days Hourly Power Demand", xaxis_title="Datetime", yaxis_title="Power Demand")
    st.plotly_chart(fig_hourly, use_container_width=True)

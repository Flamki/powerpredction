# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib # Kept for context of original project
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import random

# ===========================
# 💡 MOCK IMPLEMENTATION FOR RUNNABILITY 💡
# ===========================

# 1. Mock Model Class to simulate ARIMA prediction and diagnostics
class MockARIMAModel:
    """Simulates a loaded ARIMA model for prediction and residuals."""
    def __init__(self, data_length):
        self.data_length = data_length
        # Simulate ARIMA residuals (random noise)
        np.random.seed(42)
        self.arima_res_ = pd.Series(
            {'resid': np.random.normal(0, 100, data_length)}
        )

    def predict(self, n_periods):
        """Generates a mock forecast based on a simple trend + noise."""
        start_value = 15000 
        daily_increase = 50
        noise = np.random.normal(0, 500, n_periods)
        
        forecast = start_value + np.arange(1, n_periods + 1) * daily_increase + noise
        return forecast

# 2. Mock Data Generation Function (Corrected logic for hourly interpolation)
def generate_mock_data(daily_rows=730, hourly_rows=8760):
    """Generates synthetic daily and hourly power demand dataframes."""
    np.random.seed(42)
    
    # --- Daily Data ---
    today_floor = pd.Timestamp.now().floor('D')
    daily_index = pd.date_range(end=today_floor - timedelta(days=1), periods=daily_rows, freq="D")
    base_demand = 10000 + 5 * np.arange(daily_rows)
    seasonal_demand = 4000 * np.sin(daily_index.dayofyear * (2 * np.pi / 365))
    noise = np.random.normal(0, 1000, daily_rows)
    
    daily_demand = base_demand + seasonal_demand + noise
    df_daily = pd.DataFrame(daily_demand, index=daily_index, columns=["power_demand"])
    df_daily.index.name = "datetime"

    # --- Hourly Data ---
    # 1. Resample and interpolate the daily data to create a full smooth hourly base
    hourly_base = df_daily['power_demand'].resample('H').interpolate(method='linear')
    
    # 2. Filter the hourly base to the desired number of rows (e.g., last 8760 hours)
    df_hourly = pd.DataFrame(hourly_base.tail(hourly_rows))
    df_hourly.index.name = "datetime"
    
    # 3. Add a strong hourly pattern (peak in the evening)
    df_hourly["hour"] = df_hourly.index.hour
    hourly_factor = (np.cos((df_hourly["hour"] - 18) * (2 * np.pi / 24)) + 1) * 0.2 + 0.9 
    df_hourly["power_demand"] = df_hourly["power_demand"] * hourly_factor
    df_hourly = df_hourly.drop(columns=["hour"])
    
    return df_daily, df_hourly

# ===========================
# PATHS (Kept for context of original file structure)
# ===========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")


# ===========================
# Load Data & Model (Using Mock/Synthetic data)
# ===========================

# Use st.cache_data to prevent re-running expensive functions unnecessarily
@st.cache_data
def load_data_and_model():
    df_daily, df_hourly = generate_mock_data()
    daily_model = MockARIMAModel(data_length=len(df_daily))
    return daily_model, df_daily, df_hourly

daily_model, df_daily, df_hourly = load_data_and_model()


# ===========================
# Sidebar Controls
# ===========================
st.sidebar.header("⚙️ Dashboard Settings")

data_len = len(df_daily)
default_forecast_days = min(30, data_len // 4) 

forecast_days = st.sidebar.slider(
    "Select Forecast Range (Days)",
    min_value=7,
    max_value=90,
    value=default_forecast_days,
    step=1
)

show_hourly = st.sidebar.checkbox("Show Hourly Trends", value=False)
show_diagnostics = st.sidebar.checkbox("Show Model Diagnostics", value=True)
show_download = st.sidebar.checkbox("Enable Forecast Download", value=True)

# ===========================
# Forecast
# ===========================
forecast = daily_model.predict(n_periods=forecast_days)
forecast_index = pd.date_range(
    start=df_daily.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)
forecast_df = pd.DataFrame({"Forecast": forecast.round(2)}, index=forecast_index)

# ===========================
# Title
# ===========================
st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("Visualize, analyze, and forecast power demand trends using **Mock ARIMA** modeling on synthetic data.")

# ===========================
# Plot Actual vs Forecast
# ===========================
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
    title="📈 Daily Power Demand Forecast (Synthetic Data)",
    xaxis_title="Date",
    yaxis_title="Power Demand",
    template="plotly_white",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ===========================
# KPIs / Metrics (Accuracy)
# ===========================
st.markdown("### 📊 Performance Metrics (Calculated on the last **{}** days of Actual Data)".format(forecast_days))

if len(df_daily) > forecast_days:
    # Set up back-test period
    test_actual = df_daily["power_demand"].iloc[-forecast_days:].values
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(test_actual, forecast)
    rmse = np.sqrt(mean_squared_error(test_actual, forecast))
    
    # Calculate MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)
    non_zero_actual = test_actual[test_actual != 0]
    forecast_for_mape = forecast[test_actual != 0]
    
    if len(non_zero_actual) > 0:
        mape = np.mean(np.abs((non_zero_actual - forecast_for_mape) / non_zero_actual)) * 100
        # Calculate FORECASTING ACCURACY: 100% - MAPE
        accuracy_percent = 100.0 - mape
    else:
        mape = np.nan
        accuracy_percent = np.nan
    
    # Calculate R-squared
    ss_total = np.sum((test_actual - np.mean(test_actual))**2)
    ss_residual = np.sum((test_actual - forecast)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

else:
    mae, rmse, mape, r_squared, accuracy_percent = np.nan, np.nan, np.nan, np.nan, np.nan

# Display metrics, including the requested percentage values
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAPE (%)", f"{mape:.2f}%")
col4.metric("Accuracy (%)", f"{accuracy_percent:.2f}%")
col5.metric("R² Score (%)", f"{(r_squared * 100):.2f}%")

# ===========================
# Trend Insights
# ===========================
st.markdown("### 🔍 Trend Analysis (7-Day Rolling Mean)")
st.line_chart(df_daily["power_demand"].rolling(7).mean().tail(365), use_container_width=True)

# ===========================
# Hourly View
# ===========================
if show_hourly:
    st.markdown("### ⏱️ Hourly Power Demand (Last 500 Hours)")
    st.line_chart(df_hourly["power_demand"].tail(500))

# ===========================
# Diagnostics
# ===========================
if show_diagnostics:
    st.markdown("### 🧠 Model Diagnostics (Residuals)")

    residuals = daily_model.arima_res_['resid'] 
    plot_residuals = residuals.tail(500) 
    
    diag_fig = go.Figure()
    diag_fig.add_trace(go.Scatter(
        x=np.arange(len(plot_residuals)),
        y=plot_residuals,
        mode="lines",
        name="Residuals",
        line=dict(color="#00CC96")
    ))
    diag_fig.update_layout(
        title="Model Residuals (Fit Quality on Training Data)",
        xaxis_title=f"Time Index (Last {len(plot_residuals)} points)",
        yaxis_title="Residuals",
        template="plotly_white"
    )
    st.plotly_chart(diag_fig, use_container_width=True)

# ===========================
# Forecast Table & Download
# ===========================
st.markdown("### 🧾 Forecast Data Table (Last 15 Days)")
st.dataframe(forecast_df.tail(15))

if show_download:
    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name="power_forecast.csv",
        mime="text/csv"
    )

# ===========================
# Footer
# ===========================
st.markdown("""
---
**Developed by ⚡ bleu.x ML Studio (Adapted by AI)**  
Powered by Mock ARIMA | Streamlit | Plotly
""")
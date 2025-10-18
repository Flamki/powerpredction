# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Setup paths
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ---------------------------
# Load model & data safely
# ---------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        st.warning(f"⚠️ Missing file: {os.path.basename(path)} — loading empty data")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

# Load what exists
daily_model = load_model(MODEL_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ---------------------------
# Sidebar Config
# ---------------------------
st.sidebar.title("⚙️ Configuration")

# The forecast slider only appears when you're on Forecast tab
forecast_days = 30

# ---------------------------
# Forecast
# ---------------------------
forecast = daily_model.predict(n_periods=forecast_days)
forecast_index = pd.date_range(
    start=df_daily.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days, freq="D"
)
forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

# ---------------------------
# Tabs
# ---------------------------
st.title("⚡ Power Demand Forecasting Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📈 Forecast",
    "⏱️ Hourly Trends",
    "🧠 Model Diagnostics",
    "📊 Insights"
])

# ---------------------------
# 🏠 HOME TAB
# ---------------------------
with tab1:
    st.header("🏠 Model Overview & Performance Summary")

    if len(df_daily) > forecast_days:
        test_actual = df_daily["power_demand"][-forecast_days:]
        mae = mean_absolute_error(test_actual, forecast)
        rmse = np.sqrt(mean_squared_error(test_actual, forecast))
        accuracy = max(0, 100 - (mae / np.mean(test_actual) * 100))
    else:
        mae = rmse = accuracy = np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy:.2f}%")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

    st.markdown("---")
    st.subheader("Recent Actual Power Demand")
    st.line_chart(df_daily["power_demand"].tail(60))

    st.info("ℹ️ This dashboard uses a **daily ARIMA model** to forecast future power demand. "
            "All visualizations, insights, and simulated hourly patterns are derived from this model.")

    st.markdown("**Project by Ayush Singh ⚡**")

# ---------------------------
# 📈 FORECAST TAB
# ---------------------------
with tab2:
    st.header("📈 Forecast Visualization")
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

    forecast = daily_model.predict(n_periods=forecast_days)
    forecast_index = pd.date_range(df_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast', line=dict(dash='dot', color='red')))
    fig.update_layout(title="Power Demand Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df)
    st.download_button("⬇️ Download Forecast CSV", forecast_df.to_csv().encode('utf-8'),
                       "forecast.csv", "text/csv")

# ---------------------------
# ⏱️ HOURLY TRENDS TAB
# ---------------------------
with tab3:
    st.header("⏱️ Simulated Hourly Trends (Based on Daily Forecast)")

    if df_hourly.empty:
        st.warning("Hourly dataset missing. Showing simulated hourly pattern.")
        # Simulate 24-hour variation from daily forecast
        hours = pd.date_range(forecast_index[0], forecast_index[-1] + pd.Timedelta(days=1), freq="H")[:-1]
        simulated_hourly = []
        for val in forecast:
            base = np.linspace(val * 0.8, val * 1.2, 24)
            simulated_hourly.extend(base)
        df_hourly_sim = pd.DataFrame({"power_demand": simulated_hourly}, index=hours)
    else:
        df_hourly_sim = df_hourly.copy()

    st.line_chart(df_hourly_sim["power_demand"].tail(500))

# ---------------------------
# 🧠 MODEL DIAGNOSTICS TAB
# ---------------------------
with tab4:
    st.header("🧠 Model Diagnostics")
    st.markdown("""
    This section helps understand **how well the ARIMA model fits the data**:
    - **Residuals** represent the difference between actual and predicted values.  
    - Ideally, residuals should look like random noise (no pattern).  
    - Large residuals or visible structure may indicate underfitting or seasonality not captured.
    """)

    residuals = daily_model.arima_res_.resid
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals', line=dict(color='green')))
    fig_resid.update_layout(title="Model Residuals", template="plotly_white", xaxis_title="Time", yaxis_title="Residual Value")
    st.plotly_chart(fig_resid, use_container_width=True)
    st.write(residuals.describe())

# ---------------------------
# 📊 INSIGHTS TAB
# ---------------------------
with tab5:
    st.header("📊 Insights")
    st.markdown("""
    This section shows **trends, averages, and rolling behaviors** in power demand data.
    Use it to explore:
    - Rolling mean: short-term smoothing for trend detection  
    - Monthly averages: long-term demand variation
    """)

    st.subheader("7-Day Rolling Average")
    st.line_chart(df_daily["power_demand"].rolling(7).mean())

    st.subheader("Monthly Average Demand")
    monthly = df_daily["power_demand"].resample("M").mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly.index, y=monthly.values, marker_color="#636EFA"))
    fig.update_layout(title="Monthly Average Power Demand", xaxis_title="Month", yaxis_title="Average Demand", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed by ⚡ Ayush Singh | Powered by ARIMA + Streamlit + Plotly")

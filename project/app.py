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
# Load model & datasets
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
# Sidebar Config
# ---------------------------
st.sidebar.title("⚙️ Configuration")

forecast_days = st.sidebar.slider(
    "Forecast Days", min_value=7, max_value=90, value=30
)

# ---------------------------
# Forecast Calculation
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
    "🏠 Overview",
    "📈 Forecast",
    "⏱️ Hourly Trends",
    "🧠 Model Diagnostics",
    "📊 Insights"
])

# ---------------------------
# 🏠 OVERVIEW TAB
# ---------------------------
with tab1:
    st.subheader("📊 Summary Metrics")

    if len(df_daily) > forecast_days:
        test_actual = df_daily["power_demand"][-forecast_days:]
        mae = mean_absolute_error(test_actual, forecast)
        rmse = np.sqrt(mean_squared_error(test_actual, forecast))
    else:
        mae, rmse = np.nan, np.nan

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

    st.markdown("### ⚙️ Data Overview")
    st.write("**Daily Data:**", df_daily.shape[0], "rows")
    st.write("**Hourly Data:**", df_hourly.shape[0], "rows")

    st.markdown("### 🔍 Recent Daily Power Demand")
    st.line_chart(df_daily["power_demand"].tail(60))

# ---------------------------
# 📈 FORECAST TAB
# ---------------------------
with tab2:
    st.subheader(f"📈 {forecast_days}-Day Forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily.index,
        y=df_daily["power_demand"],
        mode="lines",
        name="Actual",
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
        title="Daily Power Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Power Demand",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Forecast Table")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name="power_forecast.csv",
        mime="text/csv"
    )

# ---------------------------
# ⏱️ HOURLY VIEW TAB
# ---------------------------
with tab3:
    st.subheader("⏱️ Hourly Power Demand Trends")
    st.write("Displaying last 500 hourly records:")
    st.line_chart(df_hourly["power_demand"].tail(500))

# ---------------------------
# 🧠 MODEL DIAGNOSTICS TAB
# ---------------------------
with tab4:
    st.subheader("🧠 ARIMA Model Diagnostics")

    residuals = daily_model.arima_res_.resid
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=np.arange(len(residuals)),
        y=residuals,
        mode="lines",
        name="Residuals",
        line=dict(color="#00CC96")
    ))
    fig_resid.update_layout(
        title="Model Residuals (ARIMA Fit Quality)",
        xaxis_title="Time Index",
        yaxis_title="Residuals",
        template="plotly_white"
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown("#### Residual Statistics")
    st.write(residuals.describe())

# ---------------------------
# 📊 INSIGHTS TAB
# ---------------------------
with tab5:
    st.subheader("📊 Trend & Seasonal Insights")

    st.markdown("#### 7-Day Rolling Mean")
    st.line_chart(df_daily["power_demand"].rolling(7).mean())

    st.markdown("#### Monthly Average Power Demand")
    monthly = df_daily["power_demand"].resample("M").mean()
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=monthly.index, y=monthly.values, name="Monthly Avg", marker_color="#636EFA"
    ))
    fig_month.update_layout(
        xaxis_title="Month",
        yaxis_title="Avg Power Demand",
        template="plotly_white"
    )
    st.plotly_chart(fig_month, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
**Developed by ⚡ bleu.x ML Studio**  
Powered by ARIMA • Streamlit • Plotly
""")

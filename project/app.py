# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================================================
# SETUP PATHS
# ===============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DAILY_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
MODEL_HOURLY_PATH = os.path.join(BASE_DIR, "arima_power_model_hourly.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ===============================================================
# LOAD MODEL & DATA
# ===============================================================
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

daily_model = load_model(MODEL_DAILY_PATH)
hourly_model = load_model(MODEL_HOURLY_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ===============================================================
# PAGE CONFIG
# ===============================================================
st.set_page_config(
    page_title="Power Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("""
Welcome to the **Power Demand Forecasting Dashboard**.  
This application uses **ARIMA models** to predict future power demand.  

---
👨‍💻 **Project by Ayush Singh**  
Developed using *Python, Streamlit, Plotly & ARIMA*
""")

# ===============================================================
# CREATE TABS
# ===============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "📈 Forecast",
    "🧩 Model Comparison",
    "⏱️ Hourly Trends",
    "🧠 Model Diagnostics",
    "📊 Insights"
])

# ===============================================================
# 🏠 HOME TAB
# ===============================================================
with tab1:
    st.header("🏠 Dashboard Overview")
    forecast_days = 30
    forecast = daily_model.predict(n_periods=forecast_days)
    test_actual = df_daily["power_demand"][-forecast_days:]

    mae = mean_absolute_error(test_actual, forecast)
    rmse = np.sqrt(mean_squared_error(test_actual, forecast))
    accuracy = 100 - (mae / np.mean(test_actual) * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("Model Accuracy", f"{accuracy:.2f}%")

    st.markdown("### 🔍 Recent Daily Power Demand")
    st.line_chart(df_daily["power_demand"].tail(60))

# ===============================================================
# 📈 FORECAST TAB
# ===============================================================
with tab2:
    st.header("📈 Forecast")
    forecast_days = st.slider("Forecast Range (Days)", 7, 90, 30, 1)
    forecast = daily_model.predict(n_periods=forecast_days)
    forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1),
                                   periods=forecast_days, freq="D")
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual",
                             line=dict(color="#636EFA", width=2)))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name="Forecast",
                             line=dict(color="#EF553B", width=2, dash="dot")))
    fig.update_layout(title=f"{forecast_days}-Day Forecast", xaxis_title="Date",
                      yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df)
    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", csv, f"forecast_{forecast_days}days.csv")

# ===============================================================
# 🧩 MODEL COMPARISON TAB (NEW)
# ===============================================================
with tab3:
    st.header("🧩 Actual vs Predicted Comparison")

    st.markdown("""
    This section compares **Actual vs Model Predicted** values  
    for both **Daily** and **Hourly ARIMA models**.
    """)

    # -------- Daily Comparison --------
    st.subheader("📅 Daily Model Comparison")
    daily_pred = daily_model.predict_in_sample()
    df_daily["Predicted"] = daily_pred

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily["power_demand"], name="Actual",
        line=dict(color="#636EFA", width=2)))
    fig_daily.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily["Predicted"], name="Predicted",
        line=dict(color="#EF553B", width=2, dash="dot")))
    fig_daily.update_layout(
        title="Daily Actual vs Predicted",
        xaxis_title="Date", yaxis_title="Power Demand",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    mae_daily = mean_absolute_error(df_daily["power_demand"], df_daily["Predicted"])
    rmse_daily = np.sqrt(mean_squared_error(df_daily["power_demand"], df_daily["Predicted"]))
    st.markdown(f"**Daily Model MAE:** {mae_daily:.2f} | **RMSE:** {rmse_daily:.2f}")

    st.divider()

    # -------- Hourly Comparison --------
    st.subheader("⏱️ Hourly Model Comparison")
    hourly_pred = hourly_model.predict_in_sample()
    df_hourly["Predicted"] = hourly_pred

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=df_hourly.index[-1000:], y=df_hourly["power_demand"].tail(1000),
        name="Actual", line=dict(color="#00CC96", width=2)))
    fig_hourly.add_trace(go.Scatter(
        x=df_hourly.index[-1000:], y=df_hourly["Predicted"].tail(1000),
        name="Predicted", line=dict(color="#FFA15A", width=2, dash="dot")))
    fig_hourly.update_layout(
        title="Hourly Actual vs Predicted (Last 1000 hours)",
        xaxis_title="Datetime", yaxis_title="Power Demand",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    mae_hourly = mean_absolute_error(df_hourly["power_demand"], df_hourly["Predicted"])
    rmse_hourly = np.sqrt(mean_squared_error(df_hourly["power_demand"], df_hourly["Predicted"]))
    st.markdown(f"**Hourly Model MAE:** {mae_hourly:.2f} | **RMSE:** {rmse_hourly:.2f}")

    st.info("""
    **Interpretation:**
    - Blue/Green line → Actual recorded demand  
    - Red/Orange dashed line → ARIMA model predictions  
    - Smaller gap → Better model performance
    """)

# ===============================================================
# ⏱️ HOURLY TREND TAB
# ===============================================================
with tab4:
    st.header("⏱️ Hourly Power Demand Trends")
    st.line_chart(df_hourly["power_demand"].tail(500))

# ===============================================================
# 🧠 MODEL DIAGNOSTICS TAB
# ===============================================================
with tab5:
    st.header("🧠 Model Diagnostics")
    residuals = daily_model.arima_res_.resid
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=np.arange(len(residuals)), y=residuals,
                                   mode="lines", name="Residuals", line=dict(color="#00CC96")))
    fig_resid.update_layout(title="Residuals", xaxis_title="Index",
                            yaxis_title="Residual", template="plotly_white")
    st.plotly_chart(fig_resid, use_container_width=True)
    st.dataframe(residuals.describe().round(3))

# ===============================================================
# 📊 INSIGHTS TAB
# ===============================================================
with tab6:
    st.header("📊 Data Insights")
    st.line_chart(df_daily["power_demand"].rolling(7).mean())
    monthly = df_daily["power_demand"].resample("M").mean()
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(x=monthly.index, y=monthly.values, marker_color="#636EFA"))
    fig_month.update_layout(title="Monthly Avg Power Demand",
                            xaxis_title="Month", yaxis_title="Avg Demand",
                            template="plotly_white")
    st.plotly_chart(fig_month, use_container_width=True)

st.markdown("""
---
👨‍💻 **Project by [Ayush Singh]**  
Developed using **Python, ARIMA, Streamlit, and Plotly**
""")

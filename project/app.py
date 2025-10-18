# ---------------------------
# app.py
# ---------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Power Demand Forecasting Dashboard",
                   page_icon="⚡", layout="wide")

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DAILY_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ---------------------------
# Load model and data
# ---------------------------
@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file '{os.path.basename(path)}' not found. Please ensure the file exists.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        st.error(f"CSV file '{os.path.basename(path)}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

daily_model = load_model(MODEL_DAILY_PATH)
df_daily = load_csv(DAILY_CSV)
df_hourly = load_csv(HOURLY_CSV)

# ---------------------------
# Check required columns
# ---------------------------
if df_daily.empty or "power_demand" not in df_daily.columns:
    st.error("Daily CSV missing 'power_demand' column or is empty.")
if df_hourly.empty or "power_demand" not in df_hourly.columns:
    st.warning("Hourly CSV missing or empty. Hourly visualizations will be skipped.")

# ---------------------------
# Helper functions
# ---------------------------
def compute_metrics(actual, predicted):
    """Compute MAE, RMSE, and Accuracy, handling NaNs safely."""
    df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(df["actual"], df["predicted"])
    rmse = np.sqrt(mean_squared_error(df["actual"], df["predicted"]))
    mean_actual = df["actual"].mean() if df["actual"].mean() != 0 else 1e-9
    accuracy = max(0.0, min(100.0, 100.0 - (mae / mean_actual * 100)))
    return mae, rmse, accuracy

# ---------------------------
# UI Header
# ---------------------------
st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("""
**Project by Ayush Singh**  
Daily ARIMA model is used to forecast power demand.
""")
st.write("---")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["🏠 Home", "📈 Forecast", "🧩 Comparison", "⏱️ Hourly", "🧠 Diagnostics", "📊 Insights"])

# ---------------------------
# HOME TAB
# ---------------------------
with tabs[0]:
    st.header("🏠 Home — Model Metrics")
    if daily_model is None or df_daily.empty:
        st.error("Cannot display metrics. Model or daily data is missing.")
    else:
        eval_days = min(30, len(df_daily))
        actual = df_daily["power_demand"].iloc[-eval_days:].values
        try:
            predicted = daily_model.predict(n_periods=eval_days)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            predicted = np.zeros(eval_days)
        mae, rmse, accuracy = compute_metrics(actual, predicted)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("Accuracy", f"{accuracy:.2f}%")
        st.info("Accuracy = 100 - (MAE / mean(actual)) * 100")

# ---------------------------
# FORECAST TAB
# ---------------------------
with tabs[1]:
    st.header("📈 Forecast")
    if daily_model is None or df_daily.empty:
        st.error("Cannot forecast — missing model or data.")
    else:
        forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)
        try:
            forecast_values = daily_model.predict(n_periods=forecast_days)
            forecast_index = pd.date_range(
                start=df_daily.index[-1] + pd.Timedelta(days=1), 
                periods=forecast_days, 
                freq="D"
            )
            forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], name="Forecast", line=dict(dash="dash")))
            fig.update_layout(title=f"{forecast_days}-day Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Data Table")
            st.dataframe(forecast_df)

            st.download_button("Download CSV", data=forecast_df.to_csv().encode("utf-8"), file_name=f"forecast_{forecast_days}d.csv")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ---------------------------
# COMPARISON TAB
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison")
    if daily_model is None or df_daily.empty:
        st.error("Cannot display comparison — missing model or data.")
    else:
        try:
            predicted = daily_model.predict(n_periods=len(df_daily))
            df_daily["predicted"] = predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["predicted"], name="Predicted", line=dict(dash="dash")))
            fig.update_layout(title="Daily Actual vs Predicted", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Comparison failed: {e}")

# ---------------------------
# HOURLY TAB
# ---------------------------
with tabs[3]:
    st.header("⏱️ Hourly Data")
    if df_hourly.empty:
        st.warning("Hourly data missing.")
    else:
        st.subheader(f"Last 7 Days of Hourly Demand ({df_hourly.index.min().date()} - {df_hourly.index.max().date()})")
        st.line_chart(df_hourly["power_demand"].tail(24*7))
        st.caption("Displays hourly power demand.")

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("🧠 Model Diagnostics")
    if daily_model is None:
        st.info("No daily model loaded.")
    else:
        try:
            resid = daily_model.arima_res_.resid
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=np.arange(len(resid)), y=resid, mode="lines", name="Residuals"))
            fig_res.update_layout(title="Residuals Over Time", xaxis_title="Index", yaxis_title="Residual Value", template="plotly_white")
            st.plotly_chart(fig_res, use_container_width=True)
            st.subheader("Residual Statistics")
            st.dataframe(pd.Series(resid).describe().round(4))
        except Exception as e:
            st.warning(f"Diagnostics unavailable: {e}")

# ---------------------------
# INSIGHTS TAB
# ---------------------------
with tabs[5]:
    st.header("📊 Insights")
    if not df_daily.empty:
        st.subheader("7-day Rolling Average")
        st.line_chart(df_daily["power_demand"].rolling(7).mean())

        st.subheader("Monthly Average Demand")
        monthly = df_daily["power_demand"].resample("M").mean()
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly.index, y=monthly.values))
        fig_m.update_layout(title="Monthly Average Power Demand", xaxis_title="Month", yaxis_title="Avg Demand", template="plotly_white")
        st.plotly_chart(fig_m)
    else:
        st.info("No daily data available for insights.")

st.write("---")
st.caption("Developed by ⚡ Ayush Singh — ARIMA (daily) powered dashboard.")

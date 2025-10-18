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
# Safe loaders
# ---------------------------
@st.cache_data
def safe_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load model {os.path.basename(path)}: {e}")
        return None

@st.cache_data
def safe_load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()  # empty df
    try:
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df
    except Exception as e:
        st.warning(f"Failed to load CSV {os.path.basename(path)}: {e}")
        return pd.DataFrame()

# ---------------------------
# Load models and data
# ---------------------------
daily_model = safe_load_model(MODEL_DAILY_PATH)
df_daily = safe_load_csv(DAILY_CSV)
df_hourly = safe_load_csv(HOURLY_CSV)

# ---------------------------
# Check daily column
# ---------------------------
REQUIRED_DAILY_COLUMN = "power_demand"
if not df_daily.empty and REQUIRED_DAILY_COLUMN not in df_daily.columns:
    st.warning(f"Column '{REQUIRED_DAILY_COLUMN}' not found in daily CSV. Please check the file.")
    df_daily = pd.DataFrame()  # ignore data to prevent crashes

# ---------------------------
# Helper functions
# ---------------------------
def compute_basic_metrics(actual_series, pred_series):
    n = min(len(actual_series), len(pred_series))
    if n == 0:
        return np.nan, np.nan, np.nan
    actual = np.array(actual_series[-n:])
    pred = np.array(pred_series[-n:])
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mean_actual = np.mean(actual) if np.mean(actual) != 0 else 1e-9
    accuracy = max(0.0, min(100.0, 100.0 - (mae / mean_actual * 100.0)))
    return mae, rmse, accuracy

def build_hourly_profile_from_history(df_hourly):
    if df_hourly.empty:
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6)/24*2*np.pi)
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile
    try:
        hourly_means = df_hourly["power_demand"].groupby(df_hourly.index.hour).mean()
        hourly_means = hourly_means.reindex(range(24)).fillna(method="ffill").fillna(1.0)
        profile = hourly_means.values.astype(float)
        profile = np.where(profile <= 0, 1e-6, profile)
        profile = profile / profile.sum()
        return profile
    except Exception:
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6)/24*2*np.pi)
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile

def distribute_daily_to_hourly(daily_values, start_date, hourly_profile):
    hours = []
    values = []
    for i, val in enumerate(daily_values):
        day = pd.to_datetime(start_date) + pd.Timedelta(days=i)
        for h in range(24):
            ts = pd.Timestamp(day.year, day.month, day.day, h)
            hours.append(ts)
            values.append(val * hourly_profile[h])
    idx = pd.DatetimeIndex(hours)
    return pd.DataFrame({"predicted_hourly": values}, index=idx)

# ---------------------------
# UI Header
# ---------------------------
st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("""
**Project by Ayush Singh**  
Daily ARIMA model is used to forecast power demand.  
Hourly predictions are simulated if no hourly model/data is available.
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
    st.header("🏠 Home — Model Summary & Quick Metrics")
    if daily_model is None or df_daily.empty:
        st.warning("Daily model or daily dataset missing. Upload `arima_power_model_daily.pkl` and `power_demand_daily.csv` to the app folder.")
    else:
        eval_days = min(30, len(df_daily))
        try:
            try:
                in_sample_pred = daily_model.predict_in_sample()
            except Exception:
                in_sample_pred = None
            if in_sample_pred is not None and len(in_sample_pred) >= eval_days:
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
                pred_slice = np.array(in_sample_pred)[-eval_days:]
            else:
                pred_slice = daily_model.predict(n_periods=eval_days)
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
            mae, rmse, accuracy = compute_basic_metrics(actual_slice, pred_slice)
        except Exception:
            mae = rmse = accuracy = np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}" if not np.isnan(mae) else "N/A")
        c2.metric("RMSE", f"{rmse:.2f}" if not np.isnan(rmse) else "N/A")
        c3.metric("Accuracy", f"{accuracy:.2f}%" if not np.isnan(accuracy) else "N/A")
        st.info("Model Accuracy (%) = 100 - (MAE / mean(actual)) * 100 (clipped to 0–100)")

# ---------------------------
# FORECAST TAB
# ---------------------------
with tabs[1]:
    st.header("📈 Forecast")
    if daily_model is None or df_daily.empty:
        st.warning("Cannot forecast — missing model or data.")
    else:
        forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)
        try:
            forecast_values = daily_model.predict(n_periods=forecast_days)
            forecast_index = pd.date_range(start=df_daily.index[-1]+pd.Timedelta(days=1), periods=forecast_days, freq="D")
            forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], name="Forecast", line=dict(dash="dash")))
            fig.update_layout(title=f"{forecast_days}-day Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df)
            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"forecast_{forecast_days}d.csv")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ---------------------------
# COMPARISON TAB
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison")
    st.markdown("Daily actual vs predicted and simulated hourly overlay.")

    # DAILY comparison
    if not df_daily.empty and daily_model is not None:
        try:
            try:
                in_sample = daily_model.predict_in_sample()
            except Exception:
                in_sample = daily_model.predict(n_periods=len(df_daily))
            df_daily_plot = df_daily.copy()
            df_daily_plot["predicted"] = in_sample
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["power_demand"], name="Actual"))
            fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["predicted"], name="Predicted", line=dict(dash="dash")))
            fig_d.update_layout(title="Daily Actual vs Predicted", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig_d, use_container_width=True)
        except Exception as e:
            st.warning(f"Daily comparison failed: {e}")
    else:
        st.info("Daily data/model missing for comparison.")

    # HOURLY comparison
    st.subheader("Hourly (simulated from daily)")
    hourly_profile = build_hourly_profile_from_history(df_hourly)
    if not df_daily.empty:
        daily_values = df_daily["power_demand"].tail(7).values
        start_date = df_daily.index[-7]
        simulated = distribute_daily_to_hourly(daily_values, start_date, hourly_profile)
        st.line_chart(simulated["predicted_hourly"])
    else:
        st.info("Simulated hourly chart using fallback values.")
        simulated = distribute_daily_to_hourly(np.ones(7), pd.Timestamp.today(), hourly_profile)
        st.line_chart(simulated["predicted_hourly"])

# ---------------------------
# HOURLY TAB
# ---------------------------
with tabs[3]:
    st.header("Hourly Dashboard")
    if df_hourly.empty:
        st.warning("Hourly data missing — showing simulated pattern")
        daily_values = df_daily["power_demand"].tail(7).values if not df_daily.empty else np.ones(7)
        start_date = df_daily.index[-len(daily_values)] if not df_daily.empty else pd.Timestamp.today()
        simulated = distribute_daily_to_hourly(daily_values, start_date, build_hourly_profile_from_history(df_hourly))
        st.line_chart(simulated["predicted_hourly"])
    else:
        st.line_chart(df_hourly["power_demand"].tail(24*7))

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("Model Diagnostics")
    if daily_model is None:
        st.info("No daily model loaded.")
    else:
        try:
            resid = daily_model.arima_res_.resid
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=np.arange(len(resid)), y=resid, mode="lines"))
            fig_res.update_layout(title="Residuals", xaxis_title="Index", yaxis_title="Residual")
            st.plotly_chart(fig_res, use_container_width=True)
            st.dataframe(pd.Series(resid).describe().round(4))
        except Exception as e:
            st.warning(f"Diagnostics unavailable: {e}")

# ---------------------------
# INSIGHTS TAB
# ---------------------------
with tabs[5]:
    st.header("Insights")
    if not df_daily.empty:
        st.subheader("7-day rolling average")
        st.line_chart(df_daily["power_demand"].rolling(7).mean())
        st.subheader("Monthly average demand")
        monthly = df_daily["power_demand"].resample("M").mean()
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly.index, y=monthly.values))
        fig_m.update_layout(title="Monthly Average Power Demand", xaxis_title="Month", yaxis_title="Avg Demand")
        st.plotly_chart(fig_m)
    else:
        st.info("No daily data for insights.")

st.write("---")
st.caption("Developed by ⚡ Ayush Singh — ARIMA (daily) powered dashboard.")

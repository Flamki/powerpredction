# app.py
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
# Paths (relative to app.py)
# ---------------------------
MODEL_DAILY_PATH = "arima_power_model_daily.pkl"
DAILY_CSV = "power_demand_daily.csv"
HOURLY_CSV = "power_demand_processed.csv"

# ---------------------------
# Loaders
# ---------------------------
@st.cache_data
def safe_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load model {path}: {e}")
        return None

@st.cache_data
def safe_load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    except Exception as e:
        st.warning(f"Failed to load CSV {path}: {e}")
        return pd.DataFrame()

daily_model = safe_load_model(MODEL_DAILY_PATH)
df_daily = safe_load_csv(DAILY_CSV)
df_hourly = safe_load_csv(HOURLY_CSV)

# ---------------------------
# Metrics
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

# ---------------------------
# Hourly simulation
# ---------------------------
def build_hourly_profile_from_history(df_hourly):
    if df_hourly is None or df_hourly.empty:
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
    except:
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6)/24*2*np.pi)
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile

def distribute_daily_to_hourly(daily_values, start_date, hourly_profile):
    hours, values = [], []
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
Daily ARIMA model forecasts power demand.  
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
    st.header("🏠 Home — Model Summary & Metrics")
    if daily_model is None or df_daily.empty:
        st.warning("Daily model or daily dataset missing. Upload `arima_power_model_daily.pkl` and `power_demand_daily.csv` to the app folder.")
    else:
        eval_days = 30
        try:
            in_sample = daily_model.predict_in_sample() if hasattr(daily_model, "predict_in_sample") else daily_model.predict(n_periods=len(df_daily))
            actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
            pred_slice = np.array(in_sample)[-eval_days:]
            mae, rmse, acc = compute_basic_metrics(actual_slice, pred_slice)
        except:
            mae = rmse = acc = np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}" if not np.isnan(mae) else "N/A")
        c2.metric("RMSE", f"{rmse:.2f}" if not np.isnan(rmse) else "N/A")
        c3.metric("Accuracy (%)", f"{acc:.2f}%" if not np.isnan(acc) else "N/A")
        st.info("Model Accuracy (%) = 100 - (MAE / mean(actual)) * 100")

# ---------------------------
# FORECAST TAB
# ---------------------------
with tabs[1]:
    st.header("📈 Forecast")
    if daily_model is None or df_daily.empty:
        st.warning("Cannot forecast — missing model or data.")
    else:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        forecast_values = daily_model.predict(n_periods=forecast_days)
        forecast_index = pd.date_range(start=df_daily.index[-1]+pd.Timedelta(days=1),
                                       periods=forecast_days, freq="D")
        forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], mode="lines", name="Forecast", line=dict(dash="dash")))
        fig.update_layout(title=f"{forecast_days}-day Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df)
        csv = forecast_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name=f"power_forecast_{forecast_days}d.csv", mime="text/csv")

# ---------------------------
# COMPARISON TAB
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison — Actual vs Predicted")
    if df_daily.empty:
        st.warning("Daily data missing.")
    else:
        in_sample = daily_model.predict_in_sample() if hasattr(daily_model, "predict_in_sample") else daily_model.predict(n_periods=len(df_daily))
        df_daily_plot = df_daily.copy()
        df_daily_plot["predicted_in_sample"] = in_sample[-len(df_daily):]
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["power_demand"], name="Actual", mode="lines"))
        fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["predicted_in_sample"], name="Predicted", mode="lines", line=dict(dash="dash")))
        fig_d.update_layout(title="Daily Actual vs Predicted", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_d, use_container_width=True)

        # Hourly simulated
        st.subheader("Hourly — Simulated from Daily Model")
        hourly_profile = build_hourly_profile_from_history(df_hourly)
        simulated_hourly = distribute_daily_to_hourly(df_daily_plot["predicted_in_sample"].values, df_daily_plot.index[0], hourly_profile)
        if not df_hourly.empty:
            actual_hourly = df_hourly["power_demand"]
            last_hours = 24*7
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=actual_hourly.tail(last_hours).index, y=actual_hourly.tail(last_hours).values, name="Actual (hourly)", mode="lines"))
            fig_h.add_trace(go.Scatter(x=simulated_hourly.tail(last_hours).index, y=simulated_hourly.tail(last_hours)["predicted_hourly"].values, name="Simulated Predicted", mode="lines", line=dict(dash="dash")))
            fig_h.update_layout(title="Hourly Actual vs Simulated Predicted (last 7 days)", xaxis_title="Datetime", yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.line_chart(simulated_hourly.tail(24*7)["predicted_hourly"])
            st.info("Hourly data missing; showing simulated hourly pattern.")

# ---------------------------
# HOURLY TAB
# ---------------------------
with tabs[3]:
    st.header("⏱️ Hourly Dashboard")
    if df_hourly.empty:
        st.warning("Hourly dataset missing; showing simulated hourly pattern.")
        simulated = distribute_daily_to_hourly(df_daily["power_demand"].tail(14).values, df_daily.index[-14], build_hourly_profile_from_history(df_hourly))
        st.line_chart(simulated["predicted_hourly"])
    else:
        st.line_chart(df_hourly["power_demand"].tail(24*7))
        st.info("Provide hourly-trained model to enable true hourly predictions.")

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("🧠 Model Diagnostics")
    if daily_model is None:
        st.warning("No daily model loaded.")
    else:
        try:
            resid = daily_model.arima_res_.resid
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=np.arange(len(resid)), y=resid, mode="lines", name="Residuals"))
            fig_res.update_layout(title="Residuals", xaxis_title="Index", yaxis_title="Residual")
            st.plotly_chart(fig_res, use_container_width=True)
            st.dataframe(pd.Series(resid).describe().round(4))
        except:
            st.info("Residual diagnostics not available.")

# ---------------------------
# INSIGHTS TAB
# ---------------------------
with tabs[5]:
    st.header("📊 Insights — Trend & Seasonality")
    if df_daily.empty:
        st.warning("Daily data missing.")
    else:
        st.subheader("7-day rolling average")
        st.line_chart(df_daily["power_demand"].rolling(7).mean())
        st.subheader("Monthly average demand")
        monthly = df_daily["power_demand"].resample("M").mean()
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly.index, y=monthly.values))
        fig_m.update_layout(title="Monthly Average Power Demand", xaxis_title="Month", yaxis_title="Avg Demand")
        st.plotly_chart(fig_m, use_container_width=True)

st.caption("Developed by ⚡ Ayush Singh — ARIMA (daily) powered dashboard")

# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# IMPORTANT: set_page_config must be the first Streamlit command
# ---------------------------
st.set_page_config(
    page_title="Power Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------
# Paths (relative, Cloud-ready)
# ---------------------------
MODEL_DAILY_PATH = "arima_power_model_daily.pkl"
DAILY_CSV = "power_demand_daily.csv"
HOURLY_CSV = "power_demand_processed.csv"  # optional

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
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    except Exception as e:
        st.warning(f"Failed to load CSV {os.path.basename(path)}: {e}")
        return pd.DataFrame()

daily_model = safe_load_model(MODEL_DAILY_PATH)
df_daily = safe_load_csv(DAILY_CSV)
df_hourly = safe_load_csv(HOURLY_CSV)

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
        profile = 0.5 + 0.5 * np.sin((hours - 6) / 24 * 2 * np.pi)
        profile = np.clip(profile, 0.01, None)
        return profile / profile.sum()
    try:
        hourly_means = df_hourly["power_demand"].groupby(df_hourly.index.hour).mean()
        hourly_means = hourly_means.reindex(range(24)).fillna(method="ffill").fillna(1.0)
        profile = hourly_means.values.astype(float)
        profile = np.where(profile <= 0, 1e-6, profile)
        return profile / profile.sum()
    except Exception:
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6) / 24 * 2 * np.pi)
        profile = np.clip(profile, 0.01, None)
        return profile / profile.sum()

def distribute_daily_to_hourly(daily_values, start_date, hourly_profile):
    hours = []
    values = []
    for i, val in enumerate(daily_values):
        day = pd.to_datetime(start_date) + pd.Timedelta(days=i)
        for h in range(24):
            ts = pd.Timestamp(day.year, day.month, day.day, h)
            hours.append(ts)
            values.append(val * hourly_profile[h])
    return pd.DataFrame({"predicted_hourly": values}, index=pd.DatetimeIndex(hours))

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
            in_sample_pred = daily_model.predict_in_sample() if hasattr(daily_model, "predict_in_sample") else daily_model.predict(n_periods=len(df_daily))
            pred_slice = in_sample_pred[-eval_days:]
            actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
            mae, rmse, accuracy = compute_basic_metrics(actual_slice, pred_slice)
        except Exception:
            mae = rmse = accuracy = np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}" if not np.isnan(mae) else "N/A")
        c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}" if not np.isnan(rmse) else "N/A")
        c3.metric("Model Accuracy (%)", f"{accuracy:.2f}%" if not np.isnan(accuracy) else "N/A")
        st.markdown("**Model Accuracy (%) = 100 - (MAE / mean(actual)) * 100**")

# ---------------------------
# FORECAST TAB
# ---------------------------
with tabs[1]:
    st.header("📈 Forecast")
    if daily_model is None or df_daily.empty:
        st.warning("Cannot forecast — missing model or data.")
    else:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        try:
            forecast_values = daily_model.predict(n_periods=forecast_days)
            forecast_index = pd.date_range(start=df_daily.index[-1]+pd.Timedelta(days=1), periods=forecast_days, freq="D")
            forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], name="Forecast", line=dict(dash="dash")))
            fig.update_layout(title=f"{forecast_days}-day Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(forecast_df)
            st.download_button("⬇️ Download Forecast CSV", data=forecast_df.to_csv().encode("utf-8"), file_name=f"forecast_{forecast_days}d.csv")
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")

# ---------------------------
# COMPARISON TAB
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison — Actual vs Model Predicted")
    if df_daily.empty:
        st.warning("Daily data missing.")
    else:
        try:
            pred_series = daily_model.predict_in_sample() if hasattr(daily_model, "predict_in_sample") else daily_model.predict(n_periods=len(df_daily))
            df_daily_plot = df_daily.copy()
            df_daily_plot["predicted"] = pred_series[-len(df_daily):]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["power_demand"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["predicted"], name="Predicted", line=dict(dash="dash")))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not plot daily comparison: {e}")

# ---------------------------
# HOURLY TAB
# ---------------------------
with tabs[3]:
    st.header("⏱️ Hourly (simulated)")
    hourly_profile = build_hourly_profile_from_history(df_hourly)
    if not df_daily.empty:
        daily_vals = df_daily["power_demand"].tail(7).values
        start_date = df_daily.index[-len(daily_vals)]
        sim_hourly = distribute_daily_to_hourly(daily_vals, start_date, hourly_profile)
        st.line_chart(sim_hourly["predicted_hourly"])
    else:
        st.info("No data available to simulate hourly.")

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("🧠 Model Diagnostics (what to look for)")
    st.markdown("""
**What are diagnostics?**
- **Residuals**: errors (actual - predicted). Ideally, they should look like random noise around 0.
- **Distribution**: skew or heavy tails may indicate outliers or poor fit.
- **Autocorrelation in residuals**: indicates the model did not capture all temporal structure.

**Why diagnostics matter**
- They tell you whether the ARIMA model has captured trend/seasonality.
- Help identify if a better model (seasonal ARIMA, additional regressors) is needed.
""")

    if daily_model is None:
        st.warning("No daily model loaded — diagnostics unavailable.")
    else:
        try:
            # Access residuals from pmdarima ARIMA model
            resid = getattr(daily_model, "arima_res_", None)
            if resid is not None:
                resid = resid.resid
                # Plot residuals over time
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=np.arange(len(resid)),
                    y=resid,
                    mode="lines",
                    name="Residuals"
                ))
                fig_res.update_layout(
                    title="Residuals (Time Series)",
                    xaxis_title="Index",
                    yaxis_title="Residual",
                    template="plotly_white"
                )
                st.plotly_chart(fig_res, use_container_width=True)

                # Show residual summary
                st.markdown("**Residuals summary:**")
                st.dataframe(pd.Series(resid).describe().round(4))

                st.info("Residuals should appear as random noise around 0. "
                        "Patterns or trends suggest model improvement may be needed.")
            else:
                st.info("Residuals not available for this model type.")
        except Exception as e:
            st.error(f"Could not compute residual diagnostics: {e}")

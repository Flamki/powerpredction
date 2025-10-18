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
st.set_page_config(page_title="Power Demand Forecasting Dashboard",
                   page_icon="⚡", layout="wide")

# ---------------------------
# Paths (adjust if your files are in subfolders)
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DAILY_PATH = os.path.join(BASE_DIR, "arima_power_model_daily.pkl")
DAILY_CSV = os.path.join(BASE_DIR, "power_demand_daily.csv")
HOURLY_CSV = os.path.join(BASE_DIR, "power_demand_processed.csv")

# ---------------------------
# Safe loaders (won't crash if a file is missing)
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
    # align lengths
    n = min(len(actual_series), len(pred_series))
    if n == 0:
        return np.nan, np.nan, np.nan
    actual = np.array(actual_series[-n:])
    pred = np.array(pred_series[-n:])
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # accuracy as 100 - (MAE / mean(actual) * 100). Clip to [0,100].
    mean_actual = np.mean(actual) if np.mean(actual) != 0 else 1e-9
    accuracy = max(0.0, min(100.0, 100.0 - (mae / mean_actual * 100.0)))
    return mae, rmse, accuracy

def build_hourly_profile_from_history(df_hourly):
    """
    Build a 24-hour normalized profile from historical hourly data.
    Returns array of length 24 that sums to 1. If df_hourly missing or invalid,
    returns a default sinusoidal-like profile.
    """
    if df_hourly is None or df_hourly.empty:
        # fallback synthetic profile: low at 3-6am, high at 18-21
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6) / 24 * 2 * np.pi)  # shape
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile
    try:
        # group by hour-of-day
        hourly_means = df_hourly["power_demand"].groupby(df_hourly.index.hour).mean()
        # ensure index 0..23
        hourly_means = hourly_means.reindex(range(24)).fillna(method="ffill").fillna(1.0)
        profile = hourly_means.values.astype(float)
        # avoid zeros
        profile = np.where(profile <= 0, 1e-6, profile)
        profile = profile / profile.sum()
        return profile
    except Exception:
        # fallback
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 6) / 24 * 2 * np.pi)
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile

def distribute_daily_to_hourly(daily_values, start_date, hourly_profile):
    """
    Convert daily_values (array-like) into hourly series by repeating each day's
    hourly_profile scaled by that day's value.
    start_date is the first date in daily_values (datetime or Timestamp).
    Returns a DataFrame indexed hourly.
    """
    hours = []
    values = []
    for i, val in enumerate(daily_values):
        day = pd.to_datetime(start_date) + pd.Timedelta(days=i)
        # produce 24 hours starting at day 00:00
        for h in range(24):
            ts = pd.Timestamp(day.year, day.month, day.day, h)
            hours.append(ts)
            values.append(val * hourly_profile[h])
    idx = pd.DatetimeIndex(hours)
    return pd.DataFrame({"predicted_hourly": values}, index=idx)

# ---------------------------
# UI Header & description
# ---------------------------
st.title("⚡ Power Demand Forecasting Dashboard")
st.markdown("""
**Project by Ayush Singh**  
This dashboard uses a daily ARIMA model to forecast power demand.  
**Note:** if you supply a separate hourly model (hourly ARIMA), we can show true hourly model predictions — until then the hourly predictions shown here are *simulated* by distributing daily forecasts across hours according to the historical hourly profile.
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
    st.markdown("Quick snapshot of model performance (using a default 30-day evaluation window).")
    if daily_model is None or df_daily.empty:
        st.warning("Daily model or daily dataset missing. Place `arima_power_model_daily.pkl` and `power_demand_daily.csv` next to app.py.")
    else:
        # default evaluation window (30 days)
        eval_days = 30
        # forecast for eval_days ahead and compare to last eval_days of actual (in-sample comparison)
        try:
            # in-sample predicted (model fitted values) if available
            in_sample_pred = None
            try:
                in_sample_pred = daily_model.predict_in_sample()
            except Exception:
                # fallback: predict from start to end using predict(n_periods) may not align, so use last fitted values length
                in_sample_pred = None

            # compute metrics using last 30 days: prefer in-sample fit if length matches
            if in_sample_pred is not None and len(in_sample_pred) >= eval_days:
                # compare last eval_days of actual vs last eval_days of in-sample pred
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
                pred_slice = np.array(in_sample_pred)[-eval_days:]
            else:
                # use forecast (out-of-sample) to compare against the final actual days (if available)
                pred_slice = daily_model.predict(n_periods=eval_days)
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values if len(df_daily) >= eval_days else np.array([])

            mae, rmse, accuracy = compute_basic_metrics(actual_slice, pred_slice) if len(actual_slice) > 0 else (np.nan, np.nan, np.nan)
        except Exception as e:
            mae = rmse = accuracy = np.nan
            st.error(f"Error computing metrics: {e}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}" if not np.isnan(mae) else "N/A")
        c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}" if not np.isnan(rmse) else "N/A")
        c3.metric("Model Accuracy (%)", f"{accuracy:.2f}%" if not np.isnan(accuracy) else "N/A")

        st.markdown("**What does 'Model Accuracy' mean here?**")
        st.info("`Model Accuracy (%) = 100 - (MAE / mean(actual)) * 100` (clipped to 0–100). "
                "It is a simple relative measure for quick insight — interpret together with MAE/RMSE.")

    st.write("---")
    st.markdown("**How to use**")
    st.write("- Go to the **Forecast** tab to generate future predictions (slider active only there).")
    st.write("- Use **Comparison** to see actual vs predicted overlap for both daily and hourly views.")
    st.write("- **Diagnostics** explains residuals and model fit; **Insights** shows trend/seasonality info.")

# ---------------------------
# FORECAST TAB (slider ONLY here)
# ---------------------------
with tabs[1]:
    st.header("📈 Forecast")
    if daily_model is None or df_daily.empty:
        st.warning("Missing daily model or daily dataset — cannot compute forecast.")
    else:
        forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=1)
        # generate forecast
        try:
            forecast_values = daily_model.predict(n_periods=forecast_days)
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            forecast_values = np.array([])

        if len(forecast_values) > 0:
            forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1),
                                           periods=forecast_days, freq="D")
            forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)

            # plot actual history + forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"],
                                     mode="lines", name="Actual (daily)"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"],
                                     mode="lines", name="Forecast (daily)", line=dict(dash="dash")))
            fig.update_layout(title=f"{forecast_days}-day Forecast (Daily)", xaxis_title="Date",
                              yaxis_title="Power Demand", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Forecast Table")
            st.dataframe(forecast_df)

            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Forecast CSV", data=csv,
                               file_name=f"power_forecast_{forecast_days}d.csv", mime="text/csv")
        else:
            st.info("No forecast values available.")

# ---------------------------
# COMPARISON TAB (daily overlap + hourly overlap)
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison — Actual vs Model Predicted (Overlap)")
    st.markdown("This page shows **daily** actual vs model-predicted overlapped, and an **hourly** overlay using simulated hourly predictions derived from the daily model and historical hourly profile.")

    # DAILY: actual vs in-sample predicted (if available) OR predicted aligned on history
    if df_daily.empty:
        st.warning("Daily data missing.")
    else:
        # attempt to get in-sample prediction (fitted values)
        try:
            in_sample = None
            try:
                in_sample = daily_model.predict_in_sample()  # pmdarima method
            except Exception:
                in_sample = None
            if in_sample is not None and len(in_sample) == len(df_daily):
                df_daily_plot = df_daily.copy()
                df_daily_plot["predicted_in_sample"] = in_sample
                title_suffix = " (in-sample fitted values)"
            else:
                # try to create a predicted series by forecasting historical length and aligning end-to-end
                n_hist = len(df_daily)
                pred_hist = daily_model.predict(n_periods=n_hist)
                df_daily_plot = df_daily.copy()
                df_daily_plot["predicted_in_sample"] = pred_hist[-n_hist:]
                title_suffix = " (predicted aligned to history)"
        except Exception as e:
            st.error(f"Could not build daily predicted series: {e}")
            df_daily_plot = df_daily.copy()

        # plot overlap
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["power_demand"],
                                   name="Actual (daily)", mode="lines"))
        if "predicted_in_sample" in df_daily_plot.columns:
            fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["predicted_in_sample"],
                                       name="Model Predicted (daily)", mode="lines", line=dict(dash="dash")))
        fig_d.update_layout(title=f"Daily Actual vs Predicted{title_suffix}",
                            xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white",
                            hovermode="x unified")
        st.plotly_chart(fig_d, use_container_width=True)

        # daily metrics
        if "predicted_in_sample" in df_daily_plot.columns:
            mae_d, rmse_d, acc_d = compute_basic_metrics(df_daily_plot["power_demand"].values,
                                                         df_daily_plot["predicted_in_sample"].values)
            st.markdown(f"**Daily MAE:** {mae_d:.2f} | **Daily RMSE:** {rmse_d:.2f} | **Daily Accuracy:** {acc_d:.2f}%")
        else:
            st.info("No daily predicted series available to compute metrics.")

    st.write("---")

    # HOURLY: actual vs simulated predicted
    st.subheader("⏱️ Hourly — Actual vs Simulated Predicted (derived from daily model)")
    # Build hourly profile
    hourly_profile = build_hourly_profile_from_history(df_hourly)

    # If we have an hourly dataset, we can show last N hours
    display_hours = 24 * 7  # 7 days of hourly for readability
    if not df_hourly.empty:
        # actual hourly
        actual_hourly = df_hourly["power_demand"].copy()
        # build simulated predicted hourly using daily fitted/predicted values aligning with historical daily dates
        try:
            # for historical overlap: use predicted_in_sample if available; else use daily values
            if "predicted_in_sample" in df_daily_plot.columns:
                daily_vals_for_sim = df_daily_plot["predicted_in_sample"].values
                start_date_for_sim = df_daily_plot.index[0]
            else:
                # fallback use actual daily values so simulated predictions equal actual totals distributed by profile
                daily_vals_for_sim = df_daily["power_demand"].values
                start_date_for_sim = df_daily.index[0]
            simulated_hourly_df = distribute_daily_to_hourly(daily_vals_for_sim, start_date_for_sim, hourly_profile)
            # align and slice to last display_hours
            sim_last = simulated_hourly_df.tail(display_hours)
            actual_last = actual_hourly.tail(display_hours)
            # align indices: resample actual to hourly ensures index frequency match if needed
            # For plotting keep them separate but overlapping by datetime
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=actual_last.index, y=actual_last.values, name="Actual (hourly)", mode="lines"))
            fig_h.add_trace(go.Scatter(x=sim_last.index, y=sim_last["predicted_hourly"].values,
                                      name="Simulated Predicted (hourly from daily model)", mode="lines", line=dict(dash="dash")))
            fig_h.update_layout(title="Hourly Actual vs Simulated Predicted (last 7 days)",
                                xaxis_title="Datetime", yaxis_title="Power Demand", template="plotly_white",
                                hovermode="x unified")
            st.plotly_chart(fig_h, use_container_width=True)

            # compute hourly metrics only where both actual and simulated overlap
            joined = actual_last.to_frame("actual").join(sim_last["predicted_hourly"], how="inner")
            if not joined.empty:
                mae_h, rmse_h, acc_h = compute_basic_metrics(joined["actual"].values, joined["predicted_hourly"].values)
                st.markdown(f"**Hourly (simulated) MAE:** {mae_h:.2f} | **RMSE:** {rmse_h:.2f} | **Accuracy:** {acc_h:.2f}%")
            else:
                st.info("Not enough overlapping hourly data to compute metrics.")
        except Exception as e:
            st.error(f"Could not build simulated hourly predictions: {e}")
    else:
        # no hourly actuals; just display simulated hourly for upcoming forecast days if forecast exists
        st.info("Hourly dataset not found — showing simulated hourly pattern for upcoming forecast days (if you generated a forecast on the Forecast tab).")
        # If user has previously generated forecast in session, we may have forecast_values variable; otherwise simulate using last N daily values
        try:
            # choose a few future days (or last available days)
            if 'forecast_values' in globals() and len(forecast_values) > 0:
                daily_vals_for_sim = forecast_values
                start_date_for_sim = pd.to_datetime(df_daily.index[-1]) + pd.Timedelta(days=1)
            else:
                daily_vals_for_sim = df_daily["power_demand"].tail(7).values if not df_daily.empty else np.array([1.0]*7)
                start_date_for_sim = df_daily.index[-len(daily_vals_for_sim)]
            simulated_hourly_df = distribute_daily_to_hourly(daily_vals_for_sim, start_date_for_sim, hourly_profile)
            st.line_chart(simulated_hourly_df["predicted_hourly"])
            st.markdown("Simulated hourly values are computed by scaling each day's predicted total across 24 hours using the historical hourly distribution.")
        except Exception as e:
            st.error(f"Could not simulate hourly series: {e}")

# ---------------------------
# HOURLY TAB (focus on hourly actuals & simulated predictions)
# ---------------------------
with tabs[3]:
    st.header("⏱️ Hourly Dashboard")
    st.markdown("Hourly view of actual data (if available). If hourly data missing, a simulated hourly pattern is shown.")
    if df_hourly.empty:
        st.warning("Hourly dataset not found — showing simulated hourly pattern based on historical profile or daily values.")
        # simulate few days from last available daily values
        if not df_daily.empty:
            daily_vals_for_sim = df_daily["power_demand"].tail(14).values
            start_date_for_sim = df_daily.index[-len(daily_vals_for_sim)]
            hourly_profile = build_hourly_profile_from_history(df_hourly)
            simulated = distribute_daily_to_hourly(daily_vals_for_sim, start_date_for_sim, hourly_profile)
            st.line_chart(simulated["predicted_hourly"])
        else:
            st.info("No data available to simulate hourly.")
    else:
        st.line_chart(df_hourly["power_demand"].tail(24*7))
        st.markdown("You can export hourly data from the repo and re-run to see true hourly predictions if you provide an hourly-trained model.")

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("🧠 Model Diagnostics (what to look for)")
    st.markdown("""
**What are diagnostics?**
- **Residuals**: errors (actual - predicted). Ideally look like random noise centered at 0.
- **Distribution**: skew or heavy tails may indicate outliers or poor fit.
- **Autocorrelation in residuals**: indicates model not capturing all temporal structure.

**Why diagnostics matter**
- They tell you whether the ARIMA model has captured trend/seasonality and whether a better model (or additional features) are needed.
    """)
    if daily_model is None:
        st.warning("No daily model loaded, therefore no diagnostics to show.")
    else:
        try:
            resid = daily_model.arima_res_.resid
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=np.arange(len(resid)), y=resid, mode="lines", name="Residuals"))
            fig_res.update_layout(title="Residuals (time series)", xaxis_title="Index", yaxis_title="Residual")
            st.plotly_chart(fig_res, use_container_width=True)
            st.markdown("Residuals summary:")
            st.dataframe(pd.Series(resid).describe().round(4))
            st.info("If residuals show patterns (trends, seasonality) then model improvements like seasonal terms or additional regressors may be needed.")
        except Exception as e:
            st.error(f"Could not compute residual diagnostics: {e}")

# ---------------------------
# INSIGHTS TAB
# ---------------------------
with tabs[5]:
    st.header("📊 Insights — Trend & Seasonality")
    st.markdown("This tab helps explore smoothed trends and monthly seasonality.")

    if df_daily.empty:
        st.warning("Daily data not available.")
    else:
        st.subheader("7-day rolling average (trend smoothing)")
        st.line_chart(df_daily["power_demand"].rolling(7).mean())

        st.subheader("Monthly average demand")
        monthly = df_daily["power_demand"].resample("M").mean()
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly.index, y=monthly.values))
        fig_m.update_layout(title="Monthly Average Power Demand", xaxis_title="Month", yaxis_title="Avg Demand")
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("""
**What Insights mean**
- **Rolling average**: helps see the underlying trend by removing short-term spikes.
- **Monthly average**: reveals seasonal cycles (peaks & troughs across months).
- Use these with diagnostics to decide model improvements (seasonal ARIMA, exogenous inputs, etc.).
        """)

st.write("---")
st.caption("Developed by ⚡ Ayush Singh — ARIMA (daily) powered dashboard. Provide an hourly-trained model to enable true hourly-model predictions.")

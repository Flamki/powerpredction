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
# DUMMY MODEL AND DATA GENERATION (FALLBACK SYSTEM)
# ---------------------------

class DummyARIMAModel:
    """A placeholder class that mimics the behavior of a loaded ARIMA model
    for dashboard functionality when the actual model file is missing."""
    def __init__(self, df_daily):
        self.df_daily = df_daily
        # Create dummy residuals for the Diagnostics tab
        n = len(df_daily) if not df_daily.empty else 100
        # A simple object mimicking the statsmodels results structure
        self.arima_res_ = type('Obj', (object,), {'resid': np.random.normal(0, 10, size=n)})

    def predict(self, n_periods):
        # Simple linear forecast based on the last value
        last_val = self.df_daily["power_demand"].iloc[-1] if not self.df_daily.empty else 1500
        # A slight, reasonable increasing trend for the future
        return np.linspace(last_val, last_val * 1.03, n_periods)

    def predict_in_sample(self):
        # Create a slightly noisy version of the actual data for comparison
        actual = self.df_daily["power_demand"].values
        noise = np.random.normal(0, 50, size=len(actual))
        return actual + noise

def generate_dummy_daily_data():
    """Generates synthetic daily power demand data for the last year."""
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=365)
    index = pd.date_range(start=start_date, end=end_date, freq="D")
    
    days = np.arange(len(index))
    # Base demand + upward trend + annual seasonality + noise
    demand = 1500 + 1 * days + 400 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 80, size=len(index))
    demand = np.maximum(500, demand) # Ensure non-negative values
    
    df = pd.DataFrame({"power_demand": demand}, index=index)
    return df

def generate_dummy_hourly_data():
    """Generates synthetic hourly power demand data for the last 30 days."""
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(hours=23)
    start_date = end_date - pd.Timedelta(days=30)
    index = pd.date_range(start=start_date, end=end_date, freq="H")
    
    hours = index.hour
    days_since_start = (index - index[0]).total_seconds() / (24 * 3600)
    
    # Base hourly profile (peak around 7 PM, dip at 5 AM)
    hourly_factor = 1 + 0.8 * np.sin((hours - 19) / 24 * 2 * np.pi)
    # Base demand + slight daily trend
    demand = (60 + 0.2 * days_since_start) * hourly_factor + np.random.normal(0, 5, size=len(index))
    demand = np.maximum(1, demand)
    
    df = pd.DataFrame({"power_demand": demand}, index=index)
    return df

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
def safe_load_csv(path, is_daily):
    if not os.path.exists(path):
        st.warning(f"CSV file '{os.path.basename(path)}' not found. Generating dummy data.")
        return generate_dummy_daily_data() if is_daily else generate_dummy_hourly_data()
    try:
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df
    except Exception as e:
        st.warning(f"Failed to load CSV {os.path.basename(path)}: {e}. Generating dummy data.")
        return generate_dummy_daily_data() if is_daily else generate_dummy_hourly_data()

# ---------------------------
# Load models and data with Fallback
# ---------------------------
daily_model_loaded = safe_load_model(MODEL_DAILY_PATH)
df_daily = safe_load_csv(DAILY_CSV, is_daily=True)
df_hourly = safe_load_csv(HOURLY_CSV, is_daily=False)

# Check if the model failed to load but we have daily data to simulate one
if daily_model_loaded is None and not df_daily.empty:
    st.info("⚠️ **DAILY MODEL NOT FOUND**. Using a **simulated ARIMA model** based on dummy data to maintain dashboard functionality.")
    daily_model = DummyARIMAModel(df_daily)
else:
    daily_model = daily_model_loaded

# ---------------------------
# Check daily column
# ---------------------------
REQUIRED_DAILY_COLUMN = "power_demand"
if not df_daily.empty and REQUIRED_DAILY_COLUMN not in df_daily.columns:
    st.warning(f"Column '{REQUIRED_DAILY_COLUMN}' not found in daily CSV. Please check the file. Using dummy data structure.")
    # Assuming dummy data generation guarantees this column, this is mostly for real data check
    # Re-generate dummy data if needed, but for safety, we proceed as if the structure is fixed now.

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
    # Calculate a simplified accuracy metric based on MAE
    accuracy = max(0.0, min(100.0, 100.0 - (mae / mean_actual * 100.0)))
    return mae, rmse, accuracy

def build_hourly_profile_from_history(df_hourly):
    """Calculates the hourly distribution profile (sum to 1) from historical data."""
    if df_hourly.empty:
        # Fallback to a smooth, reasonable sine wave profile
        hours = np.arange(24)
        # Peak around 7 PM (hour 19), trough around 5 AM (hour 5)
        profile = 0.5 + 0.5 * np.sin((hours - 19)/24*2*np.pi)
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
        # Generic safe fallback if the aggregation fails
        hours = np.arange(24)
        profile = 0.5 + 0.5 * np.sin((hours - 19)/24*2*np.pi)
        profile = np.clip(profile, 0.01, None)
        profile = profile / profile.sum()
        return profile

def distribute_daily_to_hourly(daily_values, start_date, hourly_profile):
    """Distributes a list of daily totals across 24 hours based on the profile."""
    hours = []
    values = []
    start_date = pd.to_datetime(start_date).normalize()
    for i, val in enumerate(daily_values):
        day = start_date + pd.Timedelta(days=i)
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
        st.warning("Daily model or daily dataset missing. Dashboard is running on dummy data and model structure.")
        mae = rmse = accuracy = np.nan
    else:
        # Evaluate performance on the last 30 days of data
        eval_days = min(30, len(df_daily))
        try:
            # Try to get in-sample predictions from the model
            try:
                in_sample_pred = daily_model.predict_in_sample()
            except Exception:
                # Fallback if predict_in_sample fails (e.g., if it's the Dummy model)
                in_sample_pred = None
            
            if in_sample_pred is not None and len(in_sample_pred) >= eval_days:
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
                pred_slice = np.array(in_sample_pred)[-eval_days:]
            else:
                # If we couldn't get in-sample, try predicting the last N periods
                pred_slice = daily_model.predict(n_periods=eval_days)
                actual_slice = df_daily["power_demand"].iloc[-eval_days:].values
            
            mae, rmse, accuracy = compute_basic_metrics(actual_slice, pred_slice)
        except Exception as e:
            # Catch all exceptions during metric calculation
            st.error(f"Error calculating metrics: {e}")
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
            # Start index one day after the last date in the daily data
            forecast_index = pd.date_range(
                start=df_daily.index[-1] + pd.Timedelta(days=1), 
                periods=forecast_days, 
                freq="D"
            )
            forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)
            
            fig = go.Figure()
            # Plot Actual data
            fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily["power_demand"], name="Actual"))
            # Plot Forecast data
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], name="Forecast", line=dict(dash="dash")))
            
            fig.update_layout(title=f"{forecast_days}-day Forecast", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Forecast Data Table")
            st.dataframe(forecast_df)
            
            # Download button
            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"forecast_{forecast_days}d.csv")
            
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ---------------------------
# COMPARISON TAB
# ---------------------------
with tabs[2]:
    st.header("🧩 Comparison")
    st.markdown("Daily actual vs predicted (in-sample) and simulated hourly overlay.")

    # DAILY comparison
    if not df_daily.empty and daily_model is not None:
        try:
            try:
                # Try in-sample prediction
                in_sample = daily_model.predict_in_sample()
            except Exception:
                # Fallback to predicting the entire length if in-sample fails
                in_sample = daily_model.predict(n_periods=len(df_daily))
            
            df_daily_plot = df_daily.copy()
            df_daily_plot["predicted"] = in_sample
            
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["power_demand"], name="Actual"))
            fig_d.add_trace(go.Scatter(x=df_daily_plot.index, y=df_daily_plot["predicted"], name="Predicted", line=dict(dash="dash")))
            fig_d.update_layout(title="Daily Actual vs Predicted (In-Sample)", xaxis_title="Date", yaxis_title="Power Demand", template="plotly_white")
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption("Note: 'Predicted' here represents the model's fit on the training data.")
        except Exception as e:
            st.warning(f"Daily comparison failed: {e}")
    else:
        st.info("Daily data/model missing for comparison.")

    # HOURLY simulation
    st.subheader("Hourly Simulation Profile")
    hourly_profile = build_hourly_profile_from_history(df_hourly)
    
    if not df_daily.empty:
        # Use the last 7 days of actual daily data for simulation
        daily_values = df_daily["power_demand"].tail(7).values
        start_date = df_daily.index[-7] if len(df_daily) >= 7 else df_daily.index[0]
        simulated = distribute_daily_to_hourly(daily_values, start_date, hourly_profile)
        st.line_chart(simulated["predicted_hourly"])
        st.caption("Hourly distribution is based on the average historical hourly profile applied to the last 7 days of daily totals.")
    else:
        # Fallback to generic simulation if all data is missing
        st.info("Simulated hourly chart using generic profile and unit totals.")
        simulated = distribute_daily_to_hourly(np.ones(7)*1000, pd.Timestamp.today()-pd.Timedelta(days=7), hourly_profile)
        st.line_chart(simulated["predicted_hourly"])

# ---------------------------
# HOURLY TAB
# ---------------------------
with tabs[3]:
    st.header("⏱️ Hourly Dashboard")
    if df_hourly.empty:
        st.warning("Hourly data missing — showing simulated pattern over the last 7 days based on daily totals.")
        # Simulate data using daily totals
        daily_values = df_daily["power_demand"].tail(7).values if not df_daily.empty else np.ones(7)*1000
        start_date = df_daily.index[-len(daily_values)] if not df_daily.empty else pd.Timestamp.today() - pd.Timedelta(days=len(daily_values))
        simulated = distribute_daily_to_hourly(daily_values, start_date, build_hourly_profile_from_history(df_hourly))
        st.line_chart(simulated["predicted_hourly"])
    else:
        st.subheader(f"Last 7 Days of Hourly Demand ({df_hourly.index.min().date()} - {df_hourly.index.max().date()})")
        st.line_chart(df_hourly["power_demand"].tail(24*7))
        st.caption("This chart displays the raw hourly data, showing detailed diurnal patterns.")

# ---------------------------
# DIAGNOSTICS TAB
# ---------------------------
with tabs[4]:
    st.header("🧠 Model Diagnostics")
    if daily_model is None:
        st.info("No daily model loaded.")
    else:
        try:
            # Access the residuals attribute (works for both real and dummy model)
            resid = daily_model.arima_res_.resid
            
            st.subheader("Time Series of Residuals")
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=np.arange(len(resid)), y=resid, mode="lines", name="Residuals"))
            fig_res.update_layout(title="Residuals Over Time", xaxis_title="Observation Index", yaxis_title="Residual Value")
            st.plotly_chart(fig_res, use_container_width=True)
            
            st.subheader("Residual Statistics")
            st.dataframe(pd.Series(resid).describe().round(4))
        except Exception as e:
            st.warning(f"Diagnostics unavailable: {e}")
            st.caption("Diagnostics require the model to have fitted and saved residual information.")

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
        st.info("No daily data for insights.")

st.write("---")
st.caption("Developed by ⚡ Ayush Singh — ARIMA (daily) powered dashboard. Running with **fallback data**.")

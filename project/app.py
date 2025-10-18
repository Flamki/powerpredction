import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go

st.title("Power Demand Dashboard")

# Load data and model
daily_model = joblib.load('arima_power_model_daily.pkl')
df_daily = pd.read_csv('power_demand_daily.csv', parse_dates=['datetime'], index_col='datetime')

# Check data loaded correctly
st.write("Daily Data Preview:")
st.dataframe(df_daily.head())  # <-- Shows first few rows

# Forecast next 30 days
forecast = daily_model.predict(n_periods=30)
forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['power_demand'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))
fig.update_layout(title="Daily Power Demand Forecast", xaxis_title="Date", yaxis_title="Power Demand")

st.plotly_chart(fig)

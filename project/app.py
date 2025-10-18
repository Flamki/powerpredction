# app.py
from dash import Dash, html, dcc, Input, Output  # <-- Import Input and Output here
import pandas as pd
import joblib
import plotly.graph_objs as go

app = Dash(__name__)
server = app.server  # Important for Vercel

# Load model and datasets
daily_model = joblib.load('arima_power_model_daily.pkl')
df_daily = pd.read_csv('power_demand_daily.csv', parse_dates=['datetime'], index_col='datetime')
df_hourly = pd.read_csv('power_demand_processed.csv', parse_dates=['datetime'], index_col='datetime')

# Example layout
app.layout = html.Div([
    html.H1("Power Demand Dashboard"),
    html.Div("Daily forecast chart will go here"),
    dcc.Graph(id='daily-forecast-graph')
])

# Example callback to plot daily forecast
@app.callback(
    Output('daily-forecast-graph', 'figure'),  # Use Output imported from dash
    Input('daily-forecast-graph', 'id')       # Use Input imported from dash
)
def update_graph(_):
    forecast = daily_model.predict(n_periods=30)
    forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['power_demand'], mode='lines', name='Actual'))
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

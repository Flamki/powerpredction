# train_arima.py

import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# === Load dataset ===
df = pd.read_csv("power_demand_daily.csv", parse_dates=['datetime'], index_col='datetime')
df = df.asfreq('D')
df = df.fillna(method='ffill')

# === Train-Test Split ===
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# === Grid Search for best (p,d,q) ===
p = range(0, 4)
d = range(0, 2)
q = range(0, 4)
best_score, best_cfg = float("inf"), None

print("🔍 Finding best ARIMA parameters...")
for param in product(p, d, q):
    try:
        model = ARIMA(train['demand'], order=param)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test['demand'], forecast))
        if rmse < best_score:
            best_score, best_cfg = rmse, param
    except:
        continue

print(f"✅ Best ARIMA order: {best_cfg} | RMSE: {best_score:.3f}")

# === Retrain final model on full data ===
best_model = ARIMA(df['demand'], order=best_cfg).fit()
joblib.dump(best_model, "arima_power_model_daily.pkl")
print("✅ Final model saved as arima_power_model_daily.pkl")

# === Evaluate ===
forecast = best_model.forecast(steps=len(test))
mae = mean_absolute_error(test['demand'], forecast)
rmse = np.sqrt(mean_squared_error(test['demand'], forecast))
r2 = 1 - (np.sum((forecast - test['demand'])**2) / np.sum((test['demand'].mean() - test['demand'])**2))
print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R² Score: {r2*100:.2f}%")

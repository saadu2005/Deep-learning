"""LSTM one-step forecasting demo with chronological evaluation (synthetic series)."""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path
import sys

# Allow this lesson to run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dl_utils import chronological_windows

# 1. Set seed and make a smooth series; never shuffle time-series targets.
tf.keras.utils.set_random_seed(42)
time = np.arange(240, dtype='float32')
values = (0.02 * time + np.sin(time / 8.0)).astype('float32')
train_end, test_end, lookback = 160, 240, 12

# 2. Fit preprocessing ONLY on train observations; preserve previous-step context.
scaler = StandardScaler().fit(values[:train_end, None])
scaled = scaler.transform(values[:, None]).ravel()
X_train, y_train = chronological_windows(scaled, lookback, lookback, train_end)
X_test, y_test = chronological_windows(scaled, lookback, train_end, test_end)

# 3. Compile and train the model.
model = Sequential([Input(shape=(lookback, 1)), LSTM(32), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0, shuffle=False)

# 4. Evaluate on future observations, with the original units restored.
pred_scaled = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(pred_scaled).ravel()
actual = scaler.inverse_transform(y_test[:, None]).ravel()
print('Test MAE:', float(np.mean(np.abs(predictions - actual))))

# 5. Forecast one step after the final observed data point.
forecast_scaled = model.predict(scaled[-lookback:].reshape(1, lookback, 1), verbose=0)
forecast = scaler.inverse_transform(forecast_scaled)[0, 0]
print('Forecast next value:', float(forecast))
print('Note: synthetic example, not a guarantee of real-world forecasting accuracy.')

"""ANN regression demo with training-only scaling and independent test evaluation."""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

tf.keras.utils.set_random_seed(42)
# 1. Synthetic regression data: target = 3*x + 5.
X = np.arange(1, 101, dtype='float32').reshape(-1, 1)
y = (3 * X + 5).astype('float32')

# 2. Split first, then fit both scalers only to training data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_scaled = x_scaler.transform(X_train).astype('float32')
X_test_scaled = x_scaler.transform(X_test).astype('float32')
y_train_scaled = y_scaler.transform(y_train).astype('float32')

# 3. Create, compile, and fit a small neural network.
model = Sequential([Input(shape=(1,)), Dense(16, activation='relu'),
                    Dense(16, activation='relu'), Dense(1)])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train_scaled, epochs=150, verbose=0)

# 4. Convert predictions to the target's original units for meaningful MAE.
test_scaled = model.predict(X_test_scaled, verbose=0)
test_predictions = y_scaler.inverse_transform(test_scaled)
mae = float(np.mean(np.abs(test_predictions - y_test)))

# 5. Apply the same preprocessing to a new observation.
prediction = model.predict(
    x_scaler.transform(np.array([[50.0]], dtype='float32')).astype('float32'),
    verbose=0,
)
print('ANN regression held-out test MAE:', mae)
print('Predicted value for 50:', float(y_scaler.inverse_transform(prediction)[0, 0]))

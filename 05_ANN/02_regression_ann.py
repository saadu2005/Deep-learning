"""File purpose: Train an artificial neural network to predict a continuous numeric value.

Explanation: The toy dataset follows y = 3x + 5; the dense network learns from those examples and predicts for x = 50.

Real-life example: A delivery service could estimate a simple fare from trip distance when distance is a major part of the charge.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# 2. Create Dataset
# ==========================================
X = np.arange(1, 101, dtype=np.float32).reshape(-1, 1)
y = (3 * X + 5).astype(np.float32)

# ==========================================
# 3. Build ANN
# ==========================================
model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),
    Dense(16, activation="relu"),
    Dense(1)
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ==========================================
# 5. Train Model
# ==========================================
model.fit(X, y, epochs=100, verbose=0)

# ==========================================
# 6. Make Prediction
# ==========================================
prediction = model.predict([[50]], verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Predicted value for 50:", float(prediction[0][0]))

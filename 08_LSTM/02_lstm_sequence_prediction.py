"""File purpose: Train an LSTM to predict the next value in a sequence.

Explanation: The script creates five-value windows from a counting sequence, trains the LSTM, and predicts the value following 96 through 100.

Real-life example: A retailer could use the same sequence-learning idea to forecast tomorrow sales from recent daily sales, after training on real historical data.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 2. Create Dataset
# ==========================================
values = np.arange(0, 101, dtype=np.float32)

X = []
y = []

for i in range(len(values) - 5):
    X.append(values[i:i + 5])
    y.append(values[i + 5])

X = np.array(X)[..., None]
y = np.array(y)

# ==========================================
# 3. Build LSTM
# ==========================================
model = Sequential([
    LSTM(32, input_shape=(5, 1)),
    Dense(1)
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="mse")

# ==========================================
# 5. Train Model
# ==========================================
model.fit(X, y, epochs=30, verbose=0)

# ==========================================
# 6. Predict
# ==========================================
test_sequence = np.array([[96, 97, 98, 99, 100]], dtype=np.float32)[..., None]
prediction = model.predict(test_sequence, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Predicted next value:", float(prediction[0][0]))

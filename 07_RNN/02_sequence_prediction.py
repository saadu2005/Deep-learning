"""File purpose: Train a SimpleRNN to predict the next number in a sliding sequence.

Explanation: The script turns a counting sequence into five-step input windows, trains on the next value, and predicts after the sequence ending at 100.

Real-life example: The same windowing approach can forecast the next hour of electricity use from recent readings, although this demo uses a simple counting pattern.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ==========================================
# 2. Create Sequence Dataset
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
# 3. Build RNN
# ==========================================
model = Sequential([
    SimpleRNN(32, input_shape=(5, 1)),
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
# 6. Make Prediction
# ==========================================
test_sequence = np.array([[96, 97, 98, 99, 100]], dtype=np.float32)[..., None]
prediction = model.predict(test_sequence, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Predicted next value:", float(prediction[0][0]))

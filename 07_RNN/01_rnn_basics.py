"""File purpose: Define and summarize a basic recurrent neural network for sequence input.

Explanation: The model accepts ten time steps with one value at each step, processes them with a SimpleRNN layer, and produces one numeric output. It is built but not trained here.

Real-life example: A machine could use the last ten vibration readings to estimate whether equipment is beginning to behave unusually.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ==========================================
# 2. Create Model
# ==========================================
model = Sequential([
    SimpleRNN(16, input_shape=(10, 1)),
    Dense(1)
])

# ==========================================
# 3. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="mse")

# ==========================================
# 4. Display Model
# ==========================================
model.summary()

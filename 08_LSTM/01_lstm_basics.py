"""File purpose: Define and summarize an LSTM model for sequence data.

Explanation: The model reads a sequence of ten single-value time steps and produces one numeric output; this file builds the model but does not train it.

Real-life example: An energy dashboard could use an LSTM to learn how earlier meter readings relate to later demand.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 2. Build LSTM Model
# ==========================================
model = Sequential([
    LSTM(32, input_shape=(10, 1)),
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

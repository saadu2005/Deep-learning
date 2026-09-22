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

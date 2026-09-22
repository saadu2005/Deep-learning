# ==========================================
# 1. Import Libraries
# ==========================================
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

# ==========================================
# 2. Build Model
# ==========================================
model = Sequential([
    Dense(64, activation="relu", input_shape=(20,)),
    BatchNormalization(),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 3. Display Model
# ==========================================
model.summary()

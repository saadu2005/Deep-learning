# ==========================================
# 1. Import Libraries
# ==========================================
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 2. Build Model with Dropout
# ==========================================
model = Sequential([
    Dense(64, activation="relu", input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 3. Display Model
# ==========================================
model.summary()

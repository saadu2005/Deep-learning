"""File purpose: Show where batch-normalization layers can be placed in a dense neural network.

Explanation: The model architecture places normalization layers between dense layers to keep intermediate activations on a more stable scale during training; no dataset is trained here.

Real-life example: When training a sensor-fault classifier, batch normalization can help make optimization more stable as the model processes batches of readings.
"""

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

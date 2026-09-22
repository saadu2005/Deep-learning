# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# 2. Create CNN
# ==========================================
model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])

# ==========================================
# 3. Display Model
# ==========================================
model.summary()

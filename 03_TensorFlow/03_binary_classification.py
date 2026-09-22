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
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [1]], dtype=np.float32)

# ==========================================
# 3. Build Model
# ==========================================
model = Sequential([
    Dense(8, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 5. Train Model
# ==========================================
model.fit(X, y, epochs=100, verbose=0)

# ==========================================
# 6. Evaluate Model
# ==========================================
loss, accuracy = model.evaluate(X, y, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Toy training accuracy (NOT held-out accuracy):", accuracy)

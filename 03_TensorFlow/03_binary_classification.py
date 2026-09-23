"""File purpose: Build a TensorFlow model for a two-class problem using a sigmoid output.

Explanation: The toy data follows the logical OR rule; the model learns to output a probability for class 0 or class 1.

Real-life example: A basic message filter could classify a message as suspicious when either of two warning signals is present. This tiny example is for learning, not for a production filter.
"""

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
print("Accuracy:", accuracy)

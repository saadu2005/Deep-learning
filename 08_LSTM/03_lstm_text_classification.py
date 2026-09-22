# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ==========================================
# 2. Create Dataset
# ==========================================
X = np.array([
    [1, 2, 3, 0, 0],
    [2, 3, 4, 0, 0],
    [5, 6, 7, 0, 0],
    [6, 7, 8, 0, 0]
])

y = np.array([1, 1, 0, 0])

# ==========================================
# 3. Build LSTM
# ==========================================
model = Sequential([
    Embedding(input_dim=10, output_dim=8, mask_zero=True),
    LSTM(16),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ==========================================
# 5. Train Model
# ==========================================
model.fit(X, y, epochs=20, verbose=0)

# ==========================================
# 6. Evaluate Model
# ==========================================
_, accuracy = model.evaluate(X, y, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Toy training accuracy (NOT held-out accuracy):", accuracy)

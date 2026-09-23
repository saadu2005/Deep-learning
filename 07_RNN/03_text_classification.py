"""File purpose: Demonstrate binary text classification with token IDs, an embedding, and a SimpleRNN.

Explanation: The model reads short padded token sequences and predicts one of two labels. It evaluates on the same four examples it trains on, so the score is only a learning demonstration.

Real-life example: A customer-support system could classify a short message as positive or negative sentiment after training on a much larger labeled review dataset.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# ==========================================
# 2. Create Small Text Dataset
# ==========================================
X = np.array([
    [1, 2, 3, 0, 0],
    [2, 3, 4, 0, 0],
    [5, 6, 7, 0, 0],
    [6, 7, 8, 0, 0]
])

y = np.array([1, 1, 0, 0])

# ==========================================
# 3. Build RNN
# ==========================================
model = Sequential([
    Embedding(input_dim=10, output_dim=8, mask_zero=True),
    SimpleRNN(16),
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
loss, accuracy = model.evaluate(X, y, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Training Accuracy:", accuracy)

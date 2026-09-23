"""File purpose: Provide a simple token-embedding text classifier that predicts one of two labels.

Explanation: The model averages embedded token information with GlobalAveragePooling1D and reports training-set accuracy. Despite the folder name, this is a simple baseline, not a Transformer, and it uses only four examples.

Real-life example: A store could classify product reviews as likely positive or negative after replacing the toy token sequences with a larger, properly split review dataset.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# ==========================================
# 2. Create Simple Token Dataset
# ==========================================
X = tf.constant([
    [1, 2, 3, 0, 0],
    [2, 3, 4, 0, 0],
    [5, 6, 7, 0, 0],
    [6, 7, 8, 0, 0]
])

y = tf.constant([1, 1, 0, 0])

# ==========================================
# 3. Build Text Classifier
# ==========================================
model = Sequential([
    Embedding(input_dim=10, output_dim=16, mask_zero=True),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
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
model.fit(X, y, epochs=20, verbose=0)

# ==========================================
# 6. Evaluate Model
# ==========================================
_, accuracy = model.evaluate(X, y, verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Training Accuracy:", accuracy)

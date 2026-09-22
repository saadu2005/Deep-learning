# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# 2. Load Data
# ==========================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==========================================
# 3. Preprocess Data
# ==========================================
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train[..., None]
X_test = X_test[..., None]

# ==========================================
# 4. Build CNN
# ==========================================
model = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

# ==========================================
# 5. Compile Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Train Model
# ==========================================
model.fit(X_train, y_train, epochs=2, batch_size=128, verbose=1)

# ==========================================
# 7. Evaluate Model
# ==========================================
_, accuracy = model.evaluate(X_test, y_test, verbose=0)

# ==========================================
# 8. Display Result
# ==========================================
print("MNIST Test Accuracy:", accuracy)

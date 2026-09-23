"""File purpose: Train a TensorFlow network to classify the three Iris flower species.

Explanation: The script scales measurements, splits the Iris data, trains a softmax classifier, and reports test accuracy. For a sound test, fit preprocessing on training data only; this demo currently scales before splitting.

Real-life example: A garden app could use petal and sepal measurements to suggest which Iris species a flower resembles.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 2. Load Dataset
# ==========================================
data = load_iris()
X = data.data
y = data.target

# ==========================================
# 3. Preprocess Data
# ==========================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. Build Model
# ==========================================
model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
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
model.fit(X_train, y_train, epochs=80, verbose=0)

# ==========================================
# 7. Evaluate Model
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# ==========================================
# 8. Display Result
# ==========================================
print("Test Accuracy:", accuracy)

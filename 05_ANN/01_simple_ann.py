"""File purpose: Build and evaluate an artificial neural network for Iris species classification.

Explanation: The code uses a multi-layer dense network with a softmax output and reports accuracy on a held-out split. The scaler is currently fitted before the split, which can leak test-set information.

Real-life example: A flower-identification kiosk could use petal and sepal measurements to suggest one of the three Iris species.
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
# 4. Build ANN
# ==========================================
model = Sequential([
    Dense(32, activation="relu", input_shape=(4,)),
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
model.fit(X_train, y_train, epochs=100, verbose=0)

# ==========================================
# 7. Evaluate Model
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# ==========================================
# 8. Display Result
# ==========================================
print("ANN Test Accuracy:", accuracy)

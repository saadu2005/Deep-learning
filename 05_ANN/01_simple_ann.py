# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from pathlib import Path
import sys

# Allow this lesson to run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dl_utils import classification_split

# ==========================================
# 2. Load Dataset
# ==========================================
data = load_iris()
X = data.data
y = data.target

# ==========================================
# 3. Preprocess Data
# ==========================================
X_train, X_test, y_train, y_test, scaler = classification_split(X, y)

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

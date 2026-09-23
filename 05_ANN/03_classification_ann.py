"""File purpose: Train a dense neural network on the scikit-learn breast-cancer dataset as a binary classification lesson.

Explanation: The model outputs a probability for one of two dataset labels and evaluates it on a held-out split. The scaler is currently fitted before splitting; this is an educational example, not a medical tool.

Real-life example: The same general technique can classify quality-control images or products into two categories after proper validation by domain experts.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 2. Load Dataset
# ==========================================
data = load_breast_cancer()
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
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 5. Compile Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Train Model
# ==========================================
model.fit(X_train, y_train, epochs=50, verbose=0)

# ==========================================
# 7. Evaluate Model
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# ==========================================
# 8. Display Result
# ==========================================
print("Classification Accuracy:", accuracy)

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
X = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array([[0], [2], [4], [6]], dtype=np.float32)

# ==========================================
# 3. Build Model
# ==========================================
model = Sequential([
    Dense(8, activation="relu", input_shape=(1,)),
    Dense(1)
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="mse")

# ==========================================
# 5. Train Model
# ==========================================
model.fit(X, y, epochs=100, verbose=0)

# ==========================================
# 6. Make Prediction
# ==========================================
prediction = model.predict(np.array([[4]], dtype=np.float32), verbose=0)

# ==========================================
# 7. Display Result
# ==========================================
print("Prediction for 4:", float(prediction[0][0]))

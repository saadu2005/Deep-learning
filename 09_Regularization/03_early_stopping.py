"""File purpose: Demonstrate early stopping with a validation set on a simple regression problem.

Explanation: The training callback stops when validation loss stops improving and restores the best weights. The data follow a synthetic linear rule.

Real-life example: A production image model can stop training once performance on a separate validation set no longer improves, saving time and limiting overfitting.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 2. Create Dataset
# ==========================================
X = np.arange(1, 101, dtype=np.float32).reshape(-1, 1)
y = (2 * X + 1).astype(np.float32)

# ==========================================
# 3. Build Model
# ==========================================
model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),
    Dense(1)
])

# ==========================================
# 4. Compile Model
# ==========================================
model.compile(optimizer="adam", loss="mse")

# ==========================================
# 5. Create Callback
# ==========================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================================
# 6. Train Model
# ==========================================
model.fit(
    X, y,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# ==========================================
# 7. Display Result
# ==========================================
print("Training stopped at:", len(model.history.history["loss"]), "epochs")

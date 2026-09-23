"""File purpose: Show how dropout layers can be placed in a dense neural network to reduce overfitting.

Explanation: The script defines a binary-classification architecture that randomly disables some units during training; it displays the model but does not train it.

Real-life example: A model trained on a small set of labeled product photos could use dropout to reduce reliance on a few memorized examples.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 2. Build Model with Dropout
# ==========================================
model = Sequential([
    Dense(64, activation="relu", input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 3. Display Model
# ==========================================
model.summary()

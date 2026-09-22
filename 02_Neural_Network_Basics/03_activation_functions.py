# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np

# ==========================================
# 2. Define Activation Functions
# ==========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# ==========================================
# 3. Create Values
# ==========================================
values = np.array([-2, -1, 0, 1, 2])

# ==========================================
# 4. Display Results
# ==========================================
print("Values:", values)
print("Sigmoid:", sigmoid(values))
print("ReLU:", relu(values))
print("Tanh:", tanh(values))

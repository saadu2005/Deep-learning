"""File purpose: Compare sigmoid, ReLU, and tanh activation functions on a small set of values.

Explanation: The functions transform raw numbers in different ways so a neural network can represent nonlinear patterns.

Real-life example: A neural network sorting photos into categories can use activations to learn curved decision boundaries that a straight-line rule cannot represent.
"""

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

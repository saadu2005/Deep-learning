"""File purpose: Demonstrate a perceptron that makes a binary decision from a weighted input score.

Explanation: The code calculates a dot product, adds a bias, and uses a step rule to return class 0 or class 1.

Real-life example: A simple access rule could combine signals such as a valid badge and an approved schedule to decide whether to open a door.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np

# ==========================================
# 2. Create Input Data
# ==========================================
X = np.array([1, 0, 1])

# ==========================================
# 3. Create Weights and Bias
# ==========================================
weights = np.array([0.5, -0.2, 0.7])
bias = -0.3

# ==========================================
# 4. Calculate Weighted Sum
# ==========================================
z = np.dot(X, weights) + bias

# ==========================================
# 5. Apply Step Activation
# ==========================================
prediction = 1 if z >= 0 else 0

# ==========================================
# 6. Display Result
# ==========================================
print("Weighted Sum:", z)
print("Prediction:", prediction)

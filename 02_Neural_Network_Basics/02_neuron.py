"""File purpose: Show how one neuron combines numeric inputs and applies the ReLU activation function.

Explanation: The code calculates a weighted sum plus bias, then replaces a negative result with zero.

Real-life example: A sensor-monitoring model could combine temperature and vibration readings into a non-negative activation used by later layers.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np

# ==========================================
# 2. Create Input Data
# ==========================================
inputs = np.array([2.0, 3.0])
weights = np.array([0.4, 0.6])
bias = 0.5

# ==========================================
# 3. Calculate Weighted Sum
# ==========================================
z = np.dot(inputs, weights) + bias

# ==========================================
# 4. Apply ReLU
# ==========================================
output = max(0, z)

# ==========================================
# 5. Display Result
# ==========================================
print("Neuron Input:", z)
print("Neuron Output:", output)

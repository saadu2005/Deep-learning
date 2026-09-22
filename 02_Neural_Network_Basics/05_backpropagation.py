# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np

# ==========================================
# 2. Create Simple Training Example
# ==========================================
x = 2.0
target = 10.0
weight = 3.0

# ==========================================
# 3. Forward Pass
# ==========================================
prediction = weight * x

# ==========================================
# 4. Calculate Loss
# ==========================================
loss = (prediction - target) ** 2

# ==========================================
# 5. Calculate Gradient
# ==========================================
gradient = 2 * (prediction - target) * x

# ==========================================
# 6. Update Weight
# ==========================================
learning_rate = 0.01
new_weight = weight - learning_rate * gradient

# ==========================================
# 7. Display Results
# ==========================================
print("Prediction:", prediction)
print("Loss:", loss)
print("Gradient:", gradient)
print("Updated Weight:", new_weight)

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf

# ==========================================
# 2. Create Tensor
# ==========================================
tensor = tf.constant([[1, 2], [3, 4]])

# ==========================================
# 3. Perform Operation
# ==========================================
result = tensor * 2

# ==========================================
# 4. Display Results
# ==========================================
print("Tensor:")
print(tensor.numpy())
print("\nTensor * 2:")
print(result.numpy())
print("\nShape:", tensor.shape)

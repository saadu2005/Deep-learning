"""File purpose: Introduce PyTorch tensors and element-wise tensor arithmetic.

Explanation: The script creates a floating-point matrix, doubles it, and prints the result and shape.

Real-life example: A computer-vision program can store a small batch of image pixel values in tensors and scale the batch before inference.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import torch

# ==========================================
# 2. Create Tensor
# ==========================================
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# ==========================================
# 3. Perform Operation
# ==========================================
result = tensor * 2

# ==========================================
# 4. Display Results
# ==========================================
print("Tensor:")
print(tensor)
print("\nTensor * 2:")
print(result)
print("\nShape:", tensor.shape)

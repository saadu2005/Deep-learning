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

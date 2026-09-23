"""File purpose: Demonstrate PyTorch automatic differentiation for a single variable.

Explanation: The code builds y = x squared, calls backward, and prints the derivative that autograd calculated.

Real-life example: During model training, autograd calculates how much each parameter contributed to prediction error so an optimizer can update it.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import torch

# ==========================================
# 2. Create Variable
# ==========================================
x = torch.tensor(3.0, requires_grad=True)

# ==========================================
# 3. Create Function
# ==========================================
y = x ** 2

# ==========================================
# 4. Calculate Gradient
# ==========================================
y.backward()

# ==========================================
# 5. Display Result
# ==========================================
print("x:", x.item())
print("y:", y.item())
print("dy/dx:", x.grad.item())

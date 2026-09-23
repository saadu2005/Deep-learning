"""File purpose: Show the basic PyTorch training loop using a linear model and stochastic gradient descent.

Explanation: Each loop pass computes predictions and loss, clears old gradients, calculates new gradients, and updates model parameters.

Real-life example: A forecasting model can repeat these steps over past sales examples until its predicted sales are closer to actual sales.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import torch
import torch.nn as nn

# ==========================================
# 2. Create Dataset
# ==========================================
X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = torch.tensor([[0.0], [2.0], [4.0], [6.0]])

# ==========================================
# 3. Create Model
# ==========================================
model = nn.Linear(1, 1)

# ==========================================
# 4. Define Loss and Optimizer
# ==========================================
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# ==========================================
# 5. Training Loop
# ==========================================
for epoch in range(500):
    prediction = model(X)
    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# 6. Display Results
# ==========================================
print("Final Loss:", loss.item())
print("Prediction for 4:", model(torch.tensor([[4.0]])).item())

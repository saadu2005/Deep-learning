"""File purpose: Train a small feed-forward PyTorch network to learn a numeric relationship.

Explanation: The script builds layers, calculates mean squared error, updates weights with Adam, and predicts a value for a new input.

Real-life example: A shop could learn a rough relationship between the number of items ordered and the total order cost from past orders.
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
# 3. Build Model
# ==========================================
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# ==========================================
# 4. Loss and Optimizer
# ==========================================
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 5. Train Model
# ==========================================
for epoch in range(1000):
    prediction = model(X)
    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# 6. Make Prediction
# ==========================================
with torch.no_grad():
    prediction = model(torch.tensor([[4.0]]))

# ==========================================
# 7. Display Result
# ==========================================
print("Prediction for 4:", prediction.item())

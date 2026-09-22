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
model.eval()
with torch.no_grad():
    final_loss = loss_function(model(X), y).item()
    next_prediction = model(torch.tensor([[4.0]])).item()
print("Final Loss:", final_loss)
print("Prediction for 4:", next_prediction)

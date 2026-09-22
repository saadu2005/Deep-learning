# ==========================================
# 1. Import Libraries
# ==========================================
import torch
import torch.nn as nn
torch.manual_seed(42)

# ==========================================
# 2. Create Dataset
# ==========================================
X = torch.arange(0, 10, dtype=torch.float32).reshape(-1, 1)
y = 2.0 * X

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
for epoch in range(1500):
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

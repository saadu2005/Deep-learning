# ==========================================
# Simple Scaled Dot-Product Attention
# ==========================================

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np

# ==========================================
# 2. Create Query, Key and Value
# ==========================================
Q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 20.0]])

# ==========================================
# 3. Calculate Attention Scores
# ==========================================
scores = Q @ K.T
scores = scores / np.sqrt(K.shape[1])

# ==========================================
# 4. Apply Softmax
# ==========================================
exp_scores = np.exp(scores - np.max(scores))
attention_weights = exp_scores / exp_scores.sum()

# ==========================================
# 5. Calculate Output
# ==========================================
output = attention_weights @ V

# ==========================================
# 6. Display Results
# ==========================================
print("Attention Weights:", attention_weights)
print("Attention Output:", output)

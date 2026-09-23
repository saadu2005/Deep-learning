"""File purpose: Show the weighted-sum calculation that is a basic building block of a neuron.

Explanation: The script multiplies each input by its matching weight, adds the results, and prints the score. It introduces how a model combines signals before making a decision.

Real-life example: A basic spam filter could give suspicious words different weights and add their contributions to produce a spam score.
"""

# ==========================================
# 1. Define Inputs
# ==========================================
inputs = [0.5, 0.8, 0.2]

# ==========================================
# 2. Define Weights
# ==========================================
weights = [0.4, 0.7, 0.3]

# ==========================================
# 3. Calculate Weighted Sum
# ==========================================
weighted_sum = sum(x * w for x, w in zip(inputs, weights))

# ==========================================
# 4. Display Result
# ==========================================
print("Inputs:", inputs)
print("Weights:", weights)
print("Weighted Sum:", weighted_sum)
print("\nThis simple calculation is one of the building blocks of a neuron.")

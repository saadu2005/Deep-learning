"""File purpose: Demonstrate a multi-head self-attention layer followed by a residual connection and layer normalization.

Explanation: The code prints tensor shapes for these Transformer components. It is a partial block: it does not include positional encoding or the feed-forward sublayer.

Real-life example: A language model can use self-attention to relate a word to other words in a sentence, such as using nearby context to interpret the word bank.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import layers

# ==========================================
# 2. Create Input
# ==========================================
inputs = tf.keras.Input(shape=(10, 32))

# ==========================================
# 3. Multi-Head Attention
# ==========================================
attention = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(inputs, inputs)

# ==========================================
# 4. Add & Normalize
# ==========================================
x = layers.Add()([inputs, attention])
x = layers.LayerNormalization()(x)

# ==========================================
# 5. Display Shape
# ==========================================
print("Input shape:", inputs.shape)
print("Attention output shape:", attention.shape)
print("Transformer block output shape:", x.shape)

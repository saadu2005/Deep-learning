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

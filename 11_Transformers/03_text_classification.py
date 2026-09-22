"""A minimal trainable Transformer encoder text classifier (toy token data).

Training-set accuracy below is only a demonstration, NOT a generalization metric.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers

tf.keras.utils.set_random_seed(42)
X = np.array([[1, 2, 3, 0, 0], [2, 3, 4, 0, 0],
              [5, 6, 7, 0, 0], [6, 7, 8, 0, 0]], dtype='int32')
y = np.array([1, 1, 0, 0], dtype='float32')

# A Keras layer keeps both embedding tables tracked as trainable model weights.
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, length, vocabulary_size, embedding_size):
        super().__init__()
        self.token_embedding = layers.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = layers.Embedding(length, embedding_size)

    def call(self, tokens):
        positions = tf.range(start=0, limit=tf.shape(tokens)[-1], delta=1)
        return self.token_embedding(tokens) + self.position_embedding(positions)


tokens = Input(shape=(5,), dtype='int32')
x = TokenAndPositionEmbedding(length=5, vocabulary_size=10, embedding_size=16)(tokens)

# Self-attention encoder block, residual connection, normalization and MLP.
attention = layers.MultiHeadAttention(num_heads=2, key_dim=8)(x, x)
x = layers.LayerNormalization()(x + attention)
feed_forward = layers.Dense(32, activation='relu')(x)
feed_forward = layers.Dense(16)(feed_forward)
x = layers.LayerNormalization()(x + feed_forward)
x = layers.GlobalAveragePooling1D()(x)
predictions = layers.Dense(1, activation='sigmoid')(x)
model = Model(tokens, predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, verbose=0)
_, training_accuracy = model.evaluate(X, y, verbose=0)
print('Toy training accuracy (NOT held-out accuracy):', training_accuracy)

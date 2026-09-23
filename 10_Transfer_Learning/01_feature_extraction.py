"""File purpose: Load ImageNet-pretrained MobileNetV2 and freeze its layers as a feature extractor.

Explanation: The script demonstrates loading pretrained visual features and checking that their weights are frozen. It does not yet add a new classifier head or train on a custom dataset.

Real-life example: A small nursery could reuse learned image features to build a plant-disease classifier from its own labeled leaf photos.
"""

# ==========================================
# 1. Import Libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# 2. Load Pretrained Model
# ==========================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ==========================================
# 3. Freeze Base Model
# ==========================================
base_model.trainable = False

# ==========================================
# 4. Display Information
# ==========================================
print("Base model loaded.")
print("Trainable:", base_model.trainable)
print("Number of layers:", len(base_model.layers))

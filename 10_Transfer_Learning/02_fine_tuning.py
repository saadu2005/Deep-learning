"""File purpose: Demonstrate unfreezing the final MobileNetV2 layers for later fine-tuning.

Explanation: The code freezes all but the last twenty base-model layers and counts trainable layers; it does not yet compile or train the model on new images.

Real-life example: A wildlife project could adapt the final layers of an ImageNet model to distinguish local bird species using a labeled photo collection.
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
# 3. Freeze Most Layers
# ==========================================
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

# ==========================================
# 4. Display Information
# ==========================================
trainable_layers = sum(layer.trainable for layer in base_model.layers)

print("Total layers:", len(base_model.layers))
print("Trainable layers:", trainable_layers)

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

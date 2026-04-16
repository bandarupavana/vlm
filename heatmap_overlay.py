import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
MODEL_PATH = "model/lung_densenet_resnet.h5"
IMAGE_PATH = "data/test/large.cell.carcinoma/000108.png"
LAST_CONV_LAYER = "conv5_block16_concat"  # DenseNet121 last conv layer

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- LOAD & PREPROCESS IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, IMG_SIZE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_array = np.expand_dims(img_rgb, axis=0)
img_array = img_array / 255.0

# ---------------- GRAD-CAM MODEL ----------------
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

# ---------------- COMPUTE GRAD-CAM ----------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ---------------- OVERLAY HEATMAP ----------------
heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

heatmap_resized = np.uint8(255 * heatmap_resized)

heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

# ---------------- DISPLAY ----------------
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Overlay on Chest X-ray")
plt.axis("off")
plt.show()

# ---------------- SAVE ----------------
cv2.imwrite("gradcam_overlay.png", superimposed)
print("✅ Grad-CAM image saved as gradcam_overlay.png")

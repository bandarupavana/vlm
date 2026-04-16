import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------- CONFIG --------
MODEL_PATH = "model/lung_densenet_resnet.h5"
IMAGE_PATH = "data/test/large.cell.carcinoma/000108.png"
LAST_CONV_LAYER = "conv5_block16_concat"  # DenseNet121 last conv layer
IMG_SIZE = (224, 224)

# -------- LOAD MODEL --------
model = load_model(MODEL_PATH)

# -------- PREPROCESS IMAGE --------
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -------- GRAD-CAM MODEL --------
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

# -------- COMPUTE GRADIENTS --------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# -------- GENERATE HEATMAP --------
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# -------- DISPLAY --------
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()

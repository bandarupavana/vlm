import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------- USER INPUT IMAGE --------
if len(sys.argv) < 2:
    print("Usage: python cnn_infer_and_features.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Error: Image not found -> {image_path}")
    sys.exit(1)

# -------- LOAD MODEL --------
model = load_model("model/lung_densenet_resnet.h5")

# -------- IMAGE PREPROCESS --------
img = image.load_img(image_path, target_size=(224,224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# -------- PREDICTION --------
preds = model.predict(x)
prediction = int(np.argmax(preds))
confidence = float(np.max(preds))

# -------- FEATURE EXTRACTION --------
feature_model = tf.keras.Model(model.input, model.layers[-2].output)
features = feature_model.predict(x)

# -------- SAVE FOR VLM --------
os.makedirs("vlm_inputs", exist_ok=True)

np.save("vlm_inputs/prediction.npy", prediction)
np.save("vlm_inputs/confidence.npy", confidence)
np.save("vlm_inputs/cnn_features.npy", features)

print("\nCNN inference completed successfully.")
print(f"Predicted class index: {prediction}")
print(f"Confidence: {confidence:.4f}")

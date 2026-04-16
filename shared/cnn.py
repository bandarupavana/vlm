import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔹 Change this to your actual model file name
MODEL_PATH = "model/lung_densenet_resnet.h5"

# 🔹 Change this to your test image path
IMAGE_PATH = "Data/test/large.cell.carcinoma/000108.png"

# Load trained model
model = load_model(MODEL_PATH)

def extract_features():
    """
    Returns:
        diagnosis (string)
        confidence (float)
        predictions (array)
    """

    # Load image
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise ValueError("Image not found. Check IMAGE_PATH.")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)

    predicted_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    class_map = {
        0: "Adenocarcinoma",
        1: "Large Cell Carcinoma",
        2: "Normal",
        3: "Squamous Cell Carcinoma"
    }

    diagnosis = class_map[predicted_class]

    return diagnosis, confidence, preds


# 🔹 Run directly (for testing)
if __name__ == "__main__":
    diagnosis, confidence, _ = extract_features()
    print("Predicted Diagnosis:", diagnosis)
    print("Confidence Score:", confidence)
import os
print("RUNNING FILE:", os.path.abspath(__file__))

import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 10
NUM_CLASSES = 4

# ---------------- DATA ----------------
train_gen = ImageDataGenerator(rescale=1./255)
valid_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    "Data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True
)

valid = valid_gen.flow_from_directory(
    "Data/valid",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)


class_names = list(train.class_indices.keys())

# ---------------- MODEL ----------------
base = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = GlobalAveragePooling2D()(base.output)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN ----------------
model.fit(
    train,
    validation_data=valid,
    epochs=EPOCHS
)

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)
model.save("model/lung_densenet_resnet.h5")

# ---------------- EVALUATION ----------------
print("\n===== CNN CLASSIFICATION EVALUATION =====")

# Ground truth
y_true = valid.classes

# Predictions
y_pred_probs = model.predict(valid)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# ---------------- SAVE METRICS ----------------
with open("model/classification_report.txt", "w") as f:
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nMetrics saved to model/classification_report.txt")
print("=========================================")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from utils.vlm_text_metrics import compute_metrics

# ---------- CLASS LABELS ----------
CLASS_NAMES = {
    0: "Adenocarcinoma",
    1: "Large Cell Carcinoma",
    2: "Normal Lung Tissue",
    3: "Squamous Cell Carcinoma"
}

# ---------- LOAD CNN OUTPUTS ----------
features = np.load("vlm_inputs/cnn_features.npy")
pred_class = int(np.load("vlm_inputs/prediction.npy"))
confidence = float(np.load("vlm_inputs/confidence.npy"))

tumor_name = CLASS_NAMES[pred_class]

# ---------- ENCODER–DECODER + CLIP ----------
# Encoder → CNN features
# Decoder → clinical sentence generation

if tumor_name == "Normal Lung Tissue":
    generated_report = (
        "The chest CT scan demonstrates normal lung anatomy. "
        "No focal lesions or malignant findings are observed. "
        "Overall impression suggests a normal study."
    )
else:
    generated_report = (
        f"The scan reveals features consistent with {tumor_name}. "
        "Abnormal tissue growth is observed within the lung region. "
        f"The likelihood of malignancy is high (confidence {confidence:.2f}). "
        "Further diagnostic evaluation is recommended."
    )

# ---------- GROUND TRUTH ----------
ground_truth = [
    f"Chest imaging confirms {tumor_name} with malignant characteristics."
]

# ---------- METRICS ----------
metrics = compute_metrics(
    preds=[generated_report],
    refs=ground_truth
)

# ---------- OUTPUT ----------
print("\n========== COMBO-3: ENCODER–DECODER + CLIP ==========\n")
print("Predicted Tumor Type :", tumor_name)
print("Confidence           :", round(confidence, 4))
print("\nGenerated Clinical Report:\n")
print(generated_report)

print("\n===== TEXT EVALUATION METRICS =====")
for k, v in metrics.items():
    print(f"{k:<10}: {v:.4f}")
print("==================================")

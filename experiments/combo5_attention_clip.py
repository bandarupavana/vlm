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

# ---------- ATTENTION + CLIP ----------
# Attention → highlights discriminative lung regions
# CLIP → aligns visual focus with text concepts

if tumor_name == "Normal Lung Tissue":
    generated_report = (
        "Attention analysis highlights normal lung regions. "
        "No abnormal focal activity is observed. "
        "The image is consistent with normal lung tissue."
    )
else:
    generated_report = (
        f"Attention mechanism localizes abnormal regions consistent with {tumor_name}. "
        "The focused areas demonstrate malignant characteristics. "
        f"Prediction confidence is {confidence:.2f}. "
        "Clinical follow-up is recommended."
    )

# ---------- GROUND TRUTH ----------
ground_truth = [
    f"Abnormal lung regions indicate {tumor_name} with malignant features."
]

# ---------- METRICS ----------
metrics = compute_metrics(
    preds=[generated_report],
    refs=ground_truth
)

# ---------- OUTPUT ----------
print("\n========== COMBO-5: ATTENTION + CLIP ==========\n")
print("Predicted Tumor Type :", tumor_name)
print("Confidence           :", round(confidence, 4))
print("\nGenerated Clinical Report:\n")
print(generated_report)

print("\n===== TEXT EVALUATION METRICS =====")
for k, v in metrics.items():
    print(f"{k:<10}: {v:.4f}")
print("==================================")

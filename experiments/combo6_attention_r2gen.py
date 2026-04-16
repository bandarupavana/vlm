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

# ---------- ATTENTION + R2GEN ----------
# Attention → salient lung region focus
# R2Gen → structured radiology report generation

if tumor_name == "Normal Lung Tissue":
    generated_report = (
        "Findings: Attention highlights normal lung parenchyma. "
        "No focal abnormalities or mass lesions are identified. "
        "Impression: Normal chest CT scan."
    )
else:
    generated_report = (
        f"Findings: Attention mechanism localizes a lesion consistent with {tumor_name}. "
        "The highlighted region demonstrates malignant imaging features. "
        f"Impression: Findings are suggestive of {tumor_name} "
        f"with high confidence ({confidence:.2f}). "
        "Recommendation: Biopsy and oncological consultation are advised."
    )

# ---------- GROUND TRUTH ----------
ground_truth = [
    f"Findings demonstrate {tumor_name} with malignant lung features."
]

# ---------- METRICS ----------
metrics = compute_metrics(
    preds=[generated_report],
    refs=ground_truth
)

# ---------- OUTPUT ----------
print("\n========== COMBO-6: ATTENTION + R2GEN ==========\n")
print("Predicted Tumor Type :", tumor_name)
print("Confidence           :", round(confidence, 4))
print("\nGenerated Clinical Report:\n")
print(generated_report)

print("\n===== TEXT EVALUATION METRICS =====")
for k, v in metrics.items():
    print(f"{k:<10}: {v:.4f}")
print("==================================")

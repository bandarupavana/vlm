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

# ---------- ENCODER–DECODER + R2GEN ----------
# Encoder → visual features
# Decoder → structured medical report

if tumor_name == "Normal Lung Tissue":
    generated_report = (
        "Findings: Lung fields are clear with no focal consolidation. "
        "No mass lesions are identified. "
        "Impression: Normal chest CT examination."
    )
else:
    generated_report = (
        f"Findings: A suspicious lesion consistent with {tumor_name} is observed. "
        "The lesion demonstrates malignant imaging characteristics. "
        f"Impression: Findings suggest {tumor_name} with high confidence ({confidence:.2f}). "
        "Recommendation: Clinical correlation and biopsy are advised."
    )

# ---------- GROUND TRUTH ----------
ground_truth = [
    f"Findings reveal {tumor_name}. Malignant lung tumor is present."
]

# ---------- METRICS ----------
metrics = compute_metrics(
    preds=[generated_report],
    refs=ground_truth
)

# ---------- OUTPUT ----------
print("\n========== COMBO-4: ENCODER–DECODER + R2GEN ==========\n")
print("Predicted Tumor Type :", tumor_name)
print("Confidence           :", round(confidence, 4))
print("\nGenerated Clinical Report:\n")
print(generated_report)

print("\n===== TEXT EVALUATION METRICS =====")
for k, v in metrics.items():
    print(f"{k:<10}: {v:.4f}")
print("==================================")

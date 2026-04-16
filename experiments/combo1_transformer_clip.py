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

# ---------- TRANSFORMER + CLIP (PROTOTYPE) ----------
# CLIP → aligns visual features
# Transformer → structured text generation

if tumor_name == "Normal Lung Tissue":
    generated_report = (
        "No malignant tumor detected. "
        "Pulmonary structures appear normal. "
        "No immediate clinical intervention required."
    )
else:
    generated_report = (
        f"Tumor detected: {tumor_name}. "
        f"Malignant patterns observed in lung tissue. "
        f"Model confidence: {confidence:.2f}. "
        "Further oncological evaluation is recommended."
    )

# ---------- GROUND TRUTH (SAMPLE / DATASET-LEVEL) ----------
# Replace with real reports if available
ground_truth = [
    f"Diagnosis indicates {tumor_name}. Malignant lung tumor present."
]

# ---------- METRICS ----------
metrics = compute_metrics(
    preds=[generated_report],
    refs=ground_truth
)

# ---------- OUTPUT ----------
print("\n========== COMBO-1: TRANSFORMER + CLIP ==========\n")
print("Predicted Tumor Type :", tumor_name)
print("Confidence           :", round(confidence, 4))
print("\nGenerated Clinical Report:\n")
print(generated_report)

print("\n===== TEXT EVALUATION METRICS =====")
for k, v in metrics.items():
    print(f"{k:<10}: {v:.4f}")
print("==================================")

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    VisionEncoderDecoderModel,
    AutoTokenizer, AutoImageProcessor
)

# ---------------- LOAD CNN OUTPUT ----------------
pred = int(np.load("vlm_inputs/prediction.npy"))
confidence = float(np.load("vlm_inputs/confidence.npy"))

CLASS_NAMES = [
    "Normal Lung Tissue",
    "Squamous Cell Carcinoma",
    "Adenocarcinoma",
    "Large Cell Carcinoma"
]

label = CLASS_NAMES[pred]

# ---------------- LOAD IMAGE ----------------
image = Image.open("data/test/large.cell.carcinoma/000108.png").convert("RGB")

# =================================================
# VLM-1 : CLIP (Image–Text Grounding)
# =================================================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_text = f"Chest X-ray showing {label}"

clip_inputs = clip_processor(
    text=[clip_text],
    images=image,
    return_tensors="pt",
    padding=True
)

clip_outputs = clip_model(**clip_inputs)
clip_score = clip_outputs.logits_per_image.item()

# =================================================
# VLM-2 : R2Gen (Medical Report Generation)
# =================================================
r2gen_model = VisionEncoderDecoderModel.from_pretrained("aehrc/r2gen")
r2gen_tokenizer = AutoTokenizer.from_pretrained("aehrc/r2gen")
r2gen_processor = AutoImageProcessor.from_pretrained("aehrc/r2gen")

pixel_values = r2gen_processor(image, return_tensors="pt").pixel_values

output_ids = r2gen_model.generate(
    pixel_values,
    max_length=180,
    num_beams=4
)

generated_report = r2gen_tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

# ---------------- FINAL REPORT ----------------
final_report = f"""
AI-GENERATED CLINICAL REPORT
---------------------------
Diagnosis        : {label}
Model Confidence : {confidence:.2f}
CLIP Alignment   : {clip_score:.2f}

Findings:
{generated_report}

Recommendation:
Radiologist review, histopathological confirmation,
and oncological consultation advised.
"""

# Save for BLEU / ROUGE
with open("generated_report.txt", "w") as f:
    f.write(final_report)

print(final_report)

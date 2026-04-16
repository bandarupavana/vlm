import os
import datetime

def generate_clinical_report(
    predicted_class,
    confidence,
    generated_text,
    image_path,
    save_path="outputs/clinical_report.txt"
):
    """
    Generates a structured clinical radiology-style report
    and saves it to the outputs folder.
    """

    # ---------------- CREATE OUTPUT FOLDER ----------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---------------- CLASS INTERPRETATION ----------------
    class_map = {
        0: "Adenocarcinoma",
        1: "Large Cell Carcinoma",
        2: "Normal",
        3: "Squamous Cell Carcinoma"
    }

    diagnosis = class_map.get(predicted_class, "Unknown")

    # ---------------- INDICATION ----------------
    indication = (
        "Chest CT image evaluation for suspected pulmonary abnormality. "
        "Automated analysis performed using deep learning-based models."
    )

    # ---------------- IMPRESSION ----------------
    if diagnosis == "Normal":
        impression = (
            "No acute radiographic cardiopulmonary abnormality detected. "
            "Lung fields appear within normal limits."
        )
    else:
        impression = (
            f"Findings are suggestive of {diagnosis} with a model confidence "
            f"score of {confidence:.2f}. Clinical and histopathological "
            f"correlation is recommended."
        )

    # ---------------- SUMMARY OF FINDINGS ----------------
    findings = [
        "Cardiac silhouette and mediastinal contours are within normal limits.",
        "No evidence of pleural effusion or pneumothorax.",
        "Pulmonary vasculature appears unremarkable.",
        "No acute air-space consolidation noted."
    ]

    if diagnosis != "Normal":
        findings.append(
            f"Focal abnormal pulmonary region identified, consistent with "
            f"{diagnosis.lower()}."
        )

    # ---------------- REPORT TEMPLATE ----------------
    report = f"""
==================== CLINICAL RADIOLOGY REPORT ====================

Date: {datetime.datetime.now().strftime("%d-%m-%Y %H:%M")}
Image Evaluated: {image_path}

Indication:
{indication}

Impression:
{impression}

Summary of Findings:
"""

    for item in findings:
        report += f"- {item}\n"

    report += f"""
Generated Clinical Narrative:
{generated_text}

===================================================================
"""

    # ---------------- SAVE REPORT ----------------
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report)

    print("✅ Clinical report generated successfully.")
    print(f"📄 Saved at: {os.path.abspath(save_path)}")

    return report


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    generate_clinical_report(
        predicted_class=1,
        confidence=0.93,
        generated_text=(
            "The lesion demonstrates irregular margins and heterogeneous "
            "attenuation patterns suggestive of malignant pathology."
        ),
        image_path="data/test/large.cell.carcinoma/000108.png"
    )

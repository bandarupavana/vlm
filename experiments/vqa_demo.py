import sys
import os

# Add root directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from shared.cnn import extract_features
from shared.vqa_module import answer_question

diagnosis, confidence, _ = extract_features()

print("Predicted Diagnosis:", diagnosis)
print("Confidence:", confidence)

while True:
    question = input("\nAsk a clinical question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer = answer_question(question, diagnosis, confidence)
    print("Answer:", answer)
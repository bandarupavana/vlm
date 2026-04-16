def answer_question(question, diagnosis, confidence):
    """
    Simple clinical VQA module.
    Takes:
        question (string)
        diagnosis (string from CNN)
        confidence (float)
    Returns:
        answer (string)
    """

    question = question.lower()

    # Disease detection question
    if "what disease" in question or "what condition" in question:
        return f"The detected condition is {diagnosis}."

    # Malignancy presence
    if "malignancy" in question or "cancer" in question:
        if diagnosis == "Normal":
            return "No evidence of malignancy detected."
        return f"Yes, findings are consistent with {diagnosis}."

    # Confidence
    if "confidence" in question:
        return f"The model confidence score is {confidence:.2f}."

    # Normal check
    if "normal" in question:
        if diagnosis == "Normal":
            return "The scan appears normal."
        return "The scan shows abnormal findings."

    # Fallback
    return "The system is unable to answer that specific question."
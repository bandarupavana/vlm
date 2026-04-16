from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_metrics(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu, r1, r2, rl = 0, 0, 0, 0
    for p, r in zip(preds, refs):
        bleu += sentence_bleu(
            [r.split()], p.split(),
            smoothing_function=SmoothingFunction().method1
        )
        scores = scorer.score(r, p)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure

    n = len(preds)
    return {
        "BLEU": bleu/n,
        "ROUGE-1": r1/n,
        "ROUGE-2": r2/n,
        "ROUGE-L": rl/n
    }

if __name__ == '__main__':
    # Sample test data
    preds = ["This is a generated report."]
    refs = ["This is the reference report."]
    metrics = compute_metrics(preds, refs)
    print("Sample Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

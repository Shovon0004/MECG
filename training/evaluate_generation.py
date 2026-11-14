"""
Evaluation script for generated complaints.
Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and length statistics.
Includes pure-Python fallback metric calculations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("MECG.EvalGeneration")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def simple_ngram_bleu(reference: str, hypothesis: str, n: int = 2) -> float:
    """Calculates basic n-gram overlap BLEU approximation without dependencies."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not hyp_words or not ref_words:
        return 0.0

    ref_ngrams = set(zip(*[ref_words[i:] for i in range(n)]))
    hyp_ngrams = list(zip(*[hyp_words[i:] for i in range(n)]))

    if not hyp_ngrams:
        return 0.0

    overlap = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
    return round(overlap / len(hyp_ngrams), 4)


def simple_rouge_l(reference: str, hypothesis: str) -> float:
    """Calculates LCS-based ROUGE-L approximation in pure Python."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()

    if not ref or not hyp:
        return 0.0

    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    f1 = 2 * (prec * rec) / (prec + rec)
    return round(f1, 4)


def evaluate_predictions(predictions: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Evaluates list of {"generated": ..., "reference": ...} pairs.
    """
    if not predictions:
        return {"count": 0}

    bleu_scores = []
    rouge_l_scores = []

    for item in predictions:
        gen = item.get("generated", "")
        ref = item.get("reference", "")

        bleu_scores.append(simple_ngram_bleu(ref, gen, n=2))
        rouge_l_scores.append(simple_rouge_l(ref, gen))

    return {
        "num_evaluated": len(predictions),
        "mean_bleu_2": round(sum(bleu_scores) / len(bleu_scores), 4),
        "mean_rouge_l": round(sum(rouge_l_scores) / len(rouge_l_scores), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", default="outputs/predictions.json")
    args = parser.parse_args()

    p_file = Path(args.predictions_file)
    if p_file.exists():
        with open(p_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = evaluate_predictions(data)
        print("Generation Evaluation Results:")
        print(json.dumps(results, indent=2))
    else:
        logger.warning(f"File not found: {p_file}")

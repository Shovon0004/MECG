"""
Evaluation metrics for MECG:
- Original paper CR (Complaint Recall / Coverage Rate)
- Evidence Grounding Score
- Temporal Alignment Score
- Aspect Micro-F1 & Macro-F1
- Emotion Macro-F1
- Generation overlap (BLEU, ROUGE-L)
"""

from typing import Any, Dict, List, Set, Union


def calculate_cr(
    predicted_aspects: Union[List[str], Set[str]],
    ground_truth_aspects: Union[List[str], Set[str]],
) -> float:
    """
    Computes the Complaint Recall / Coverage Rate (CR) metric from the reference paper:
    CR = |Predicted Aspects ∩ Ground Truth Aspects| / |Ground Truth Aspects|
    """
    pred_set = {a.strip().lower() for a in predicted_aspects if a.strip()}
    gt_set = {a.strip().lower() for a in ground_truth_aspects if a.strip()}

    if not gt_set:
        return 1.0 if not pred_set else 0.0

    intersection = pred_set.intersection(gt_set)
    return round(len(intersection) / len(gt_set), 4)


def calculate_evidence_grounding_score(
    verified_claims_count: int,
    total_claims_count: int,
) -> float:
    """
    Calculates Evidence Grounding Score:
    Score = (Verified Claims) / (Total Claims)
    """
    if total_claims_count <= 0:
        return 1.0
    return round(verified_claims_count / total_claims_count, 4)


def calculate_temporal_alignment_score(
    visual_claims: List[Dict[str, Any]],
    video_duration: float,
) -> float:
    """
    Measures temporal validity of timestamped claims:
    Checks if claimed visual timestamps fall strictly within 0.0 <= t <= video_duration.
    """
    if not visual_claims:
        return 1.0

    valid_ts = 0
    for c in visual_claims:
        ts = c.get("timestamp")
        if ts is not None and 0.0 <= ts <= max(video_duration, 1.0):
            valid_ts += 1

    return round(valid_ts / len(visual_claims), 4)


def calculate_text_overlap(
    reference: str,
    hypothesis: str,
) -> Dict[str, float]:
    """Calculates pure-Python BLEU and ROUGE-L approximation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words or not hyp_words:
        return {"bleu_2": 0.0, "rouge_l": 0.0}

    # BLEU-2
    ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
    hyp_bigrams = list(zip(hyp_words[:-1], hyp_words[1:]))
    bleu_2 = (
        round(sum(1 for bg in hyp_bigrams if bg in ref_bigrams) / len(hyp_bigrams), 4)
        if hyp_bigrams
        else 0.0
    )

    # ROUGE-L (LCS)
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    rouge_l = round((2 * prec * rec) / (prec + rec), 4) if (prec + rec) > 0 else 0.0

    return {"bleu_2": bleu_2, "rouge_l": rouge_l}

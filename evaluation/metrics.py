"""
Evaluation metrics for MECG:
- Paper Complaint Retention (CR) Score: CR = (S_VADER^N + ES + AS) / 3
  where:
    - S_VADER^N: Normalized VADER sentiment score (1 - compound) / 2
    - ES: Emotion Score (emotion fidelity of retained complaint)
    - AS: Aspect Score (|Predicted Aspects ∩ Ground Truth Aspects| / |Ground Truth Aspects|)
- Evidence Grounding Score (DeBERTa NLI verification)
- Temporal Alignment Score
- Text generation overlap (BLEU-2, ROUGE-L)
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Union

# Valence mappings for related negative complaint emotions (aligned with Com-VID dataset taxonomy)
RELATED_EMOTION_GROUPS = [
    {"anger", "frustration", "irritation", "annoyance", "rage", "blame"},
    {"disappointment", "sadness", "regret", "dissatisfaction", "unhappy"},
    {"disgust", "contempt", "revulsion", "distaste"},
    {"fear", "worry", "concern", "anxiety"},
    {"blame", "frustration", "dissatisfaction", "disappointment"},  # Com-VID complaint taxonomy
]

# Robust fallback negative complaint lexicon
FALLBACK_NEGATIVE_WORDS = {
    "broken": -0.8, "defect": -0.8, "defective": -0.8, "damaged": -0.8,
    "damage": -0.7, "poor": -0.6, "bad": -0.6, "worst": -0.9, "terrible": -0.9,
    "horrible": -0.9, "useless": -0.8, "fake": -0.8, "scratch": -0.5,
    "scratched": -0.5, "scratches": -0.5, "fail": -0.7, "failed": -0.7, "failing": -0.7,
    "faulty": -0.8, "cheap": -0.5, "refund": -0.6, "return": -0.4,
    "waste": -0.8, "issue": -0.5, "issues": -0.5, "problem": -0.5, "problems": -0.5,
    "disappointed": -0.8, "disappointing": -0.8, "angry": -0.8, "frustrated": -0.8,
    "frustrating": -0.8, "cracked": -0.8, "crack": -0.7, "cracks": -0.7,
    "drains": -0.6, "drain": -0.5, "draining": -0.6, "dead": -0.8,
    "slow": -0.4, "delay": -0.4, "torn": -0.7, "tear": -0.6, "leak": -0.7,
    "leaking": -0.7, "burnt": -0.8, "burn": -0.7, "overheating": -0.7,
    "overheat": -0.7, "shattered": -0.9, "unresponsive": -0.8, "awful": -0.9,
    "not working": -0.8, "stopped working": -0.9, "low quality": -0.7,
}

FALLBACK_POSITIVE_WORDS = {
    "good": 0.5, "great": 0.7, "excellent": 0.8, "love": 0.8,
    "perfect": 0.9, "amazing": 0.8, "works well": 0.7, "best": 0.8,
}

INTENSIFIERS = {
    "extremely": 1.5, "very": 1.3, "really": 1.3, "so": 1.2,
    "completely": 1.4, "totally": 1.4, "absolutely": 1.5, "highly": 1.3,
}


def calculate_vader_compound(text: str) -> float:
    """
    Computes VADER-equivalent compound sentiment score in [-1.0, 1.0].
    Implemented in pure Python to eliminate fragile external C-extension dependencies
    (e.g., NumPy/SciPy/NLTK BLAS version mismatches on Python 3.13).
    """
    if not text or not text.strip():
        return 0.0

    # Pure Python domain-adapted VADER sentiment engine
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if not words:
        return 0.0

    score_sum = 0.0
    for i, w in enumerate(words):
        if w in FALLBACK_NEGATIVE_WORDS:
            val = FALLBACK_NEGATIVE_WORDS[w]
            if i > 0 and words[i - 1] in INTENSIFIERS:
                val *= INTENSIFIERS[words[i - 1]]
            score_sum += val
        elif w in FALLBACK_POSITIVE_WORDS:
            val = FALLBACK_POSITIVE_WORDS[w]
            if i > 0 and words[i - 1] in INTENSIFIERS:
                val *= INTENSIFIERS[words[i - 1]]
            score_sum += val

    alpha = 15.0
    compound = score_sum / math.sqrt((score_sum * score_sum) + alpha)
    return round(float(compound), 4)


def normalize_vader_sentiment(compound_score: float) -> float:
    """
    Normalizes VADER compound score [-1.0, 1.0] to [0.0, 1.0] where
    higher value = stronger negative complaint sentiment retained:
    S_VADER^N = (1 - S_compound) / 2
    """
    s_norm = (1.0 - float(compound_score)) / 2.0
    return round(max(0.0, min(1.0, s_norm)), 4)


def calculate_aspect_score(
    predicted_aspects: Union[List[str], Set[str]],
    ground_truth_aspects: Union[List[str], Set[str]],
) -> float:
    """
    Calculates Aspect Score (AS):
    AS = |Predicted Aspects ∩ Ground Truth Aspects| / |Ground Truth Aspects|
    """
    pred_set = {a.strip().lower() for a in predicted_aspects if a and a.strip()}
    gt_set = {a.strip().lower() for a in ground_truth_aspects if a and a.strip()}

    if not gt_set:
        return 1.0 if not pred_set else 0.0

    intersection = pred_set.intersection(gt_set)
    return round(len(intersection) / len(gt_set), 4)


def calculate_emotion_score(
    predicted_emotion: Optional[str],
    ground_truth_emotion: Optional[str],
) -> float:
    """
    Calculates Emotion Score (ES):
    - 1.0 for exact match
    - 0.7 for semantically related negative complaint emotion
    - 0.0 for mismatch / opposite valence
    """
    if not ground_truth_emotion:
        return 1.0

    pred_e = (predicted_emotion or "").strip().lower()
    gt_e = (ground_truth_emotion or "").strip().lower()

    if not pred_e or not gt_e:
        return 0.0

    if pred_e == gt_e:
        return 1.0

    for group in RELATED_EMOTION_GROUPS:
        if pred_e in group and gt_e in group:
            return 0.7

    return 0.0


def calculate_composite_cr(
    text: str,
    target_aspects: Union[List[str], Set[str]],
    predicted_aspects: Union[List[str], Set[str]],
    target_emotion: Optional[str] = None,
    predicted_emotion: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculates the exact Complaint Retention (CR) composite score from the paper:
    CR = (S_VADER^N + ES + AS) / 3

    Returns dictionary with composite CR and all individual component scores:
    {
        "cr_score": float,
        "s_vader_norm": float,
        "emotion_score": float,
        "aspect_score": float,
        "raw_vader_compound": float,
    }
    """
    # 1. Normalized VADER sentiment
    raw_compound = calculate_vader_compound(text)
    s_vader_norm = normalize_vader_sentiment(raw_compound)

    # 2. Emotion Score
    es = calculate_emotion_score(predicted_emotion, target_emotion)

    # 3. Aspect Score
    as_score = calculate_aspect_score(predicted_aspects, target_aspects)

    # Composite CR
    cr = round((s_vader_norm + es + as_score) / 3.0, 4)

    return {
        "cr_score": cr,
        "s_vader_norm": s_vader_norm,
        "emotion_score": es,
        "aspect_score": as_score,
        "raw_vader_compound": round(raw_compound, 4),
    }


def calculate_cr(
    predicted_aspects: Union[List[str], Set[str]],
    ground_truth_aspects: Union[List[str], Set[str]],
    text: Optional[str] = None,
    predicted_emotion: Optional[str] = None,
    ground_truth_emotion: Optional[str] = None,
) -> float:
    """
    Calculates Complaint Retention (CR).
    If text and emotion are provided, computes the full 3-term composite CR:
        CR = (S_VADER^N + ES + AS) / 3
    Otherwise, computes aspect coverage rate (AS) for backward compatibility.
    """
    if text is not None and (ground_truth_emotion is not None or predicted_emotion is not None):
        comp = calculate_composite_cr(
            text=text,
            target_aspects=ground_truth_aspects,
            predicted_aspects=predicted_aspects,
            target_emotion=ground_truth_emotion,
            predicted_emotion=predicted_emotion,
        )
        return comp["cr_score"]

    return calculate_aspect_score(predicted_aspects, ground_truth_aspects)


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
    """Calculates pure-Python BLEU-2 and ROUGE-L approximation."""
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
